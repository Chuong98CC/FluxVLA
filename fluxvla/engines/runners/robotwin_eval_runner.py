# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RoboTwin 2.0 closed-loop evaluation runner.

The policy runs here, in the FluxVLA environment, while the simulator runs in
its own RoboTwin environment behind ``tools/robotwin_sim_worker.py``. The
runner owns replanning (``action_horizon`` actions per call, one simulator
``take_action`` per loop iteration), instruction delivery, z-score
normalization from the checkpoint's saved statistics, and the results JSON.

Replanning and observation cadence follow the RoboTwin direct task API: a new
observation is requested only when the action queue is empty, so the simulator
advances exactly one ``take_action`` per loop iteration.
"""

import base64
import copy
import gc
import json
import os
import select
import subprocess
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from fluxvla.engines.utils import initialize_overwatch
from fluxvla.engines.utils.name_map import str_to_dtype
from fluxvla.engines.utils.torch_utils import set_seed_everywhere

from ..utils.root import RUNNERS
from .base_eval_runner import BaseEvalRunner

overwatch = initialize_overwatch(__name__)

__all__ = ['RoboTwinEvalRunner']

# Instruction template the released RoboTwin C2R checkpoint was trained with.
ROBOTWIN_PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following "
    'instruction: {task}')

# Compact camera mosaic used by the released recipe: head camera on top,
# left/right wrist cameras below, resized with PIL bilinear. The trained
# geometry is 288x256 (height x width).
ROBOTWIN_HEAD_SIZE_WH = (256, 192)
ROBOTWIN_WRIST_SIZE_WH = (128, 96)
ROBOTWIN_MOSAIC_SIZE_HW = (288, 256)

_NORMALIZATION_EPSILON = 1e-8
_NORMALIZATION_CLAMP = 5.0


class _SimBridge:
    """Line-delimited JSON client for ``tools/robotwin_sim_worker.py``."""

    def __init__(self, python: str, worker_script: str, root: str,
                 response_timeout: float = 1800.0,
                 close_timeout: float = 60.0):
        worker_script = str(Path(worker_script).resolve())
        if not Path(worker_script).is_file():
            raise FileNotFoundError(worker_script)
        if not Path(python).is_file():
            raise FileNotFoundError(
                f'RoboTwin simulator interpreter not found: {python}')
        if response_timeout <= 0 or close_timeout <= 0:
            raise ValueError('Bridge timeouts must be positive')
        self.response_timeout = float(response_timeout)
        self.close_timeout = float(close_timeout)
        self.process = subprocess.Popen(
            [python, worker_script, '--root', str(root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            cwd=str(root),
            text=True,
            bufsize=1,
        )

    def request_raw(self, payload: Dict[str, Any],
                    timeout: Optional[float] = None) -> Dict[str, Any]:
        if self.process.poll() is not None:
            raise RuntimeError(
                'RoboTwin simulator worker exited before the request '
                f'(returncode={self.process.returncode})')
        assert self.process.stdin is not None and self.process.stdout is not None
        limit = self.response_timeout if timeout is None else float(timeout)
        self.process.stdin.write(json.dumps(payload) + '\n')
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], limit)
        if not ready:
            self.terminate()
            raise RuntimeError(
                'RoboTwin simulator worker did not respond within '
                f'{limit:.1f}s')
        line = self.process.stdout.readline()
        if not line:
            self.terminate()
            raise RuntimeError(
                'RoboTwin simulator worker closed its output stream '
                f'(returncode={self.process.returncode})')
        return json.loads(line)

    def request(self, payload: Dict[str, Any],
                timeout: Optional[float] = None) -> Dict[str, Any]:
        response = self.request_raw(payload, timeout=timeout)
        if not response.get('ok', False):
            raise RuntimeError(
                f'RoboTwin simulator worker failed: {response.get("error")}')
        return response

    def terminate(self, grace: float = 10.0, exit_grace: float = 2.0) -> None:
        """Stop the worker with bounded waits (never an unbounded join)."""
        if self.process.poll() is None:
            # A worker that just acknowledged ``close`` only has to unwind its
            # own interpreter; give it a short chance to exit on its own.
            try:
                self.process.wait(timeout=exit_grace)
            except subprocess.TimeoutExpired:
                pass
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                self.process.kill()
                try:
                    self.process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    pass

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.request({'op': 'close'}, timeout=self.close_timeout)
            except (RuntimeError, ValueError, OSError):
                pass
        self.terminate()
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()


def _decode_raster(entry: Dict[str, Any]) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(entry['data']))
    array = np.frombuffer(raw, dtype=np.uint8).reshape(entry['shape'])
    return np.ascontiguousarray(array)


def compose_compact_mosaic(head: np.ndarray, left: np.ndarray,
                           right: np.ndarray) -> np.ndarray:
    """Build the 288x256 three-camera mosaic in the released layout."""
    from PIL import Image

    head_image = Image.fromarray(head).resize(ROBOTWIN_HEAD_SIZE_WH,
                                              Image.BILINEAR)
    left_image = Image.fromarray(left).resize(ROBOTWIN_WRIST_SIZE_WH,
                                              Image.BILINEAR)
    right_image = Image.fromarray(right).resize(ROBOTWIN_WRIST_SIZE_WH,
                                                Image.BILINEAR)
    top = np.asarray(head_image, dtype=np.uint8)
    bottom = np.concatenate(
        [np.asarray(left_image, dtype=np.uint8),
         np.asarray(right_image, dtype=np.uint8)],
        axis=1)
    mosaic = np.concatenate([top, bottom], axis=0)
    if mosaic.shape != (*ROBOTWIN_MOSAIC_SIZE_HW, 3):
        raise ValueError(
            f'Composed RoboTwin mosaic has shape {mosaic.shape}, expected '
            f'{(*ROBOTWIN_MOSAIC_SIZE_HW, 3)}')
    return mosaic


class _FieldNormalizer:
    """z-score codec matching the released ``SingleFieldLinearNormalizer``."""

    def __init__(self, stats: Dict[str, Any], key: str):
        if 'global_mean' not in stats or 'global_std' not in stats:
            raise KeyError(
                f'Statistics for {key} need global_mean and global_std')
        mean = np.asarray(stats['global_mean'], dtype=np.float64)
        std = np.asarray(stats['global_std'], dtype=np.float64)
        self.key = key
        self.mean = mean
        self.std = std
        self.scale = 1.0 / (std + _NORMALIZATION_EPSILON)
        self.offset = -mean / (std + _NORMALIZATION_EPSILON)

    def normalize(self, value: np.ndarray) -> np.ndarray:
        normalized = np.asarray(value, dtype=np.float64) * self.scale + self.offset
        return np.clip(normalized, -_NORMALIZATION_CLAMP,
                       _NORMALIZATION_CLAMP)

    def denormalize(self, value: np.ndarray) -> np.ndarray:
        return (np.asarray(value, dtype=np.float64) - self.offset) / self.scale

    def signature(self) -> Dict[str, float]:
        return {
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
        }


def load_robotwin_stats(path: str,
                        state_key: str = 'default',
                        action_key: str = 'default'):
    """Load the released ``dataset_stats.json`` z-score codecs."""
    with open(path, 'r', encoding='utf-8') as stream:
        payload = json.load(stream)
    for field, key in (('state', state_key), ('action', action_key)):
        if field not in payload or key not in payload[field]:
            raise KeyError(
                f'RoboTwin statistics {path} has no {field}.{key}')
    return (_FieldNormalizer(payload['state'][state_key], 'state'),
            _FieldNormalizer(payload['action'][action_key], 'action'))


@RUNNERS.register_module()
class RoboTwinEvalRunner(BaseEvalRunner):
    """Run the native PatchWAM policy in RoboTwin 2.0.

    Args:
        cfg: Full training/evaluation configuration (the runner reads
            ``cfg.model``/``cfg.inference_model``).
        seed: Base seed. Candidate seeds come from one monotonic cursor per run
            starting at ``100000 * (1 + seed)``; every attempt, accepted or
            rejected, advances it by one, so no two episodes reuse a candidate.
        ckpt_path: Optional checkpoint override; when given it replaces the
            model config's ``pretrained_name_or_path``.
        task_name: RoboTwin task module name (for example ``adjust_bottle``).
        task_config: RoboTwin task configuration (``demo_clean`` or
            ``demo_randomized``).
        robotwin_root: RoboTwin 2.0 checkout root (simulator + assets).
        sim_python: Interpreter of the RoboTwin simulation environment.
        dataset_stats_path: Released ``dataset_stats.json`` with the z-score
            statistics the checkpoint was trained with.
        instruction: Optional fixed instruction. When omitted the worker draws
            one from the RoboTwin description generator.
        instruction_type: RoboTwin instruction split (``seen``/``unseen``).
        expert_check: Validate the episode seed with the expert plan before the
            rollout, as the RoboTwin evaluation entrypoint does.
        episodes_per_task: Rollouts per task for this run.
        action_horizon: Predicted action chunk length.
        replan_steps: Actions executed before the next observation/prediction.
        num_inference_steps: Denoising steps per prediction.
        max_episodes: Optional cap on executed episodes (useful for smoke runs).
        max_seed_attempts: Upper bound on candidate seeds tried for one episode
            when RoboTwin rejects a scene or the expert plan fails.
        response_timeout: Seconds to wait for each simulator worker response
            before terminating the worker.
        output_dir: Results directory; defaults to ``runs/evaluation/robotwin``.
        run_id: Optional explicit run directory name.
        model_family: Label written into the results JSON.
        worker_script: Override for ``tools/robotwin_sim_worker.py``.
        setup_overrides: Optional RoboTwin task-setup overrides.
    """

    def __init__(
        self,
        cfg: Dict,
        seed: int = 0,
        ckpt_path: Optional[str] = None,
        task_name: str = 'adjust_bottle',
        task_config: str = 'demo_clean',
        robotwin_root: Optional[str] = None,
        sim_python: Optional[str] = None,
        dataset_stats_path: Optional[str] = None,
        instruction: Optional[str] = None,
        instruction_type: str = 'unseen',
        expert_check: bool = True,
        episodes_per_task: int = 1,
        action_horizon: int = 16,
        replan_steps: int = 16,
        num_inference_steps: int = 10,
        max_episodes: Optional[int] = None,
        max_seed_attempts: int = 50,
        output_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        model_family: str = 'patchwam',
        worker_script: Optional[str] = None,
        setup_overrides: Optional[Dict[str, Any]] = None,
        response_timeout: float = 1800.0,
        mixed_precision_dtype: str = 'bf16',
        enable_mixed_precision_training: bool = True,
        model_build_device: Optional[str] = None,
        model_build_dtype=None,
        **kwargs,
    ) -> None:
        if kwargs:
            raise TypeError(
                f'Unexpected RoboTwinEvalRunner arguments: {sorted(kwargs)}')
        self.cfg = cfg
        self.seed = int(seed)
        self.ckpt_path = ckpt_path
        self.task_name = str(task_name)
        self.task_config = str(task_config)
        self.robotwin_root = robotwin_root
        self.sim_python = sim_python
        self.dataset_stats_path = dataset_stats_path
        self.instruction = instruction
        self.instruction_type = str(instruction_type)
        self.expert_check = bool(expert_check)
        self.episodes_per_task = int(episodes_per_task)
        if self.episodes_per_task < 1:
            raise ValueError('episodes_per_task must be at least one')
        self.action_horizon = int(action_horizon)
        if self.action_horizon < 1:
            raise ValueError('action_horizon must be positive')
        self.replan_steps = int(replan_steps)
        if not 1 <= self.replan_steps <= self.action_horizon:
            raise ValueError(
                'replan_steps must be in [1, action_horizon], got '
                f'{self.replan_steps}')
        self.num_inference_steps = int(num_inference_steps)
        self.max_episodes = (None if max_episodes is None else
                             int(max_episodes))
        self.max_seed_attempts = int(max_seed_attempts)
        if self.max_seed_attempts < 1:
            raise ValueError('max_seed_attempts must be positive')
        self.output_dir = Path(output_dir or 'runs/evaluation/robotwin')
        self.run_id = run_id
        self.model_family = str(model_family)
        self.worker_script = worker_script or str(
            Path(__file__).resolve().parents[3] / 'tools' /
            'robotwin_sim_worker.py')
        self.setup_overrides = dict(setup_overrides or {})
        self.response_timeout = float(response_timeout)
        if self.response_timeout <= 0:
            raise ValueError('response_timeout must be positive')
        self.mixed_precision_dtype = str_to_dtype(mixed_precision_dtype)
        self.enable_mixed_precision_training = bool(
            enable_mixed_precision_training)
        self.model_build_device = model_build_device
        self.model_build_dtype = model_build_dtype
        # ``PureOverwatch`` has no rank API, so read the launcher's local rank.
        self.device_id = int(os.environ.get('LOCAL_RANK', 0))
        self.vla = None
        self.resolved_checkpoint: Optional[str] = None
        self.config_path: Optional[str] = None
        self._seed_cursor: Optional[int] = None
        self.bridge: Optional[_SimBridge] = None
        self.state_normalizer: Optional[_FieldNormalizer] = None
        self.action_normalizer: Optional[_FieldNormalizer] = None
        self.run_dir: Optional[Path] = None
        self.results: Dict[str, Any] = {}

    # -------------------------------------------------------------- lifecycle
    @staticmethod
    def _cfg_value(cfg_obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(cfg_obj, dict):
            return cfg_obj.get(key, default)
        return getattr(cfg_obj, key, default)

    def _resolve_checkpoint_path(self, model_cfg: Any) -> Optional[str]:
        """Run checkpoint recorded in the results, independent of the model."""
        if self.ckpt_path:
            return str(self.ckpt_path)
        configured = self._cfg_value(model_cfg, 'pretrained_name_or_path')
        return None if not configured else str(configured)

    def _validate_prerequisites(self) -> None:
        for name, value in (
            ('robotwin_root', self.robotwin_root),
            ('sim_python', self.sim_python),
            ('dataset_stats_path', self.dataset_stats_path),
        ):
            if not value:
                raise ValueError(
                    f'RoboTwinEvalRunner requires `{name}` to run the simulator '
                    'evaluation')
        if not Path(self.robotwin_root).is_dir():
            raise FileNotFoundError(self.robotwin_root)
        if not Path(self.dataset_stats_path).is_file():
            raise FileNotFoundError(self.dataset_stats_path)

    def run_setup(self) -> None:
        self._validate_prerequisites()
        set_seed_everywhere(self.seed)
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device_id)

        model_cfg = self.prepare_eval_model_cfg(
            self.cfg,
            model_build_device=self.model_build_device,
            model_build_dtype=self.model_build_dtype)
        if self.ckpt_path is not None:
            self._set_model_cfg_value(model_cfg, 'pretrained_name_or_path',
                                      str(self.ckpt_path))
        # Capture the run identity from the configuration rather than from the
        # built model: attribute availability on the model object is not part
        # of the evaluation contract.
        self.resolved_checkpoint = self._resolve_checkpoint_path(model_cfg)
        self.config_path = self._cfg_value(self.cfg, 'filename')
        from fluxvla.engines import build_vla_from_cfg

        self.vla = build_vla_from_cfg(model_cfg)
        pretrained = getattr(self.vla, 'pretrained_name_or_path', None)
        if self.resolved_checkpoint is None and pretrained is not None:
            self.resolved_checkpoint = str(pretrained)
        if pretrained:
            self.vla.from_pretrained()
        self.vla.eval()
        device = ('cpu' if self.model_build_device == 'cpu' else
                  f'cuda:{self.device_id}')
        if self.model_build_device == 'cpu':
            self.vla.to(device='cpu')
        elif self.enable_mixed_precision_training:
            self.vla.to(device=device, dtype=self.mixed_precision_dtype)
        else:
            self.vla.to(device=device)

        self.state_normalizer, self.action_normalizer = load_robotwin_stats(
            self.dataset_stats_path)
        self.bridge = _SimBridge(self.sim_python, self.worker_script,
                                 self.robotwin_root,
                                 response_timeout=self.response_timeout)
        if self._seed_cursor is None:
            self._seed_cursor = 100000 * (1 + self.seed)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.output_dir / (self.run_id or
                                          f'{self.task_name}_{timestamp}')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        overwatch.info(
            f'RoboTwin evaluation ready: task={self.task_name} '
            f'config={self.task_config} episodes={self.episodes_per_task} '
            f'horizon={self.action_horizon} replan={self.replan_steps} '
            f'steps={self.num_inference_steps} run_dir={self.run_dir}')

    def cleanup(self) -> None:
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None
        self.vla = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------------- inference
    @staticmethod
    def _stack_observation(response: Dict[str, Any]) -> np.ndarray:
        images = response['images']
        mosaic = compose_compact_mosaic(
            _decode_raster(images['head_camera']),
            _decode_raster(images['left_camera']),
            _decode_raster(images['right_camera']))
        return mosaic

    def _predict(self, response: Dict[str, Any], episode_seed: int,
                 call_index: int) -> np.ndarray:
        mosaic = self._stack_observation(response)
        image = torch.from_numpy(mosaic).permute(2, 0, 1).float()
        parameter = next(self.vla.parameters())
        image = image.to(device=parameter.device, dtype=parameter.dtype)
        # [3, H, W] -> [B=1, 3, H, W]; the backbone adds the view/time axes.
        image = (image * (2.0 / 255.0) - 1.0).unsqueeze(0)
        state = self.state_normalizer.normalize(
            np.asarray(response['state'], dtype=np.float32))
        states = torch.from_numpy(state).float().to(
            device=parameter.device, dtype=image.dtype).unsqueeze(0)
        instruction = response.get('instruction') or self.instruction or ''
        prompt = [ROBOTWIN_PROMPT_TEMPLATE.format(task=instruction)]
        prediction_seed = (episode_seed * 1000003 + call_index) % (2**31 - 1)
        with torch.no_grad():
            actions = self.vla.predict_action(
                images=image,
                states=states,
                task_description=prompt,
                action_horizon=self.action_horizon,
                num_inference_steps=self.num_inference_steps,
                seed=prediction_seed)
        actions = actions.detach().to(device='cpu',
                                      dtype=torch.float32).numpy()
        if actions.ndim != 3 or actions.shape[1] != self.action_horizon:
            raise ValueError(
                f'Policy returned action shape {actions.shape}, expected '
                f'[B, {self.action_horizon}, A]')
        physical = self.action_normalizer.denormalize(actions[0])
        return np.asarray(physical, dtype=np.float32)[:self.replan_steps]

    # -------------------------------------------------------------- episodes
    def _run_episode(self, episode_index: int) -> Dict[str, Any]:
        # One monotonic candidate cursor for the whole run: every attempt,
        # accepted or rejected, consumes one seed so episodes never repeat a
        # seed that an earlier episode already tried.
        if self._seed_cursor is None:
            self._seed_cursor = 100000 * (1 + self.seed)
        base_seed = self._seed_cursor
        request = {
            'op': 'reset',
            'task': self.task_name,
            'config': self.task_config,
            'episode': episode_index,
            'instruction_type': self.instruction_type,
            'expert_check': self.expert_check,
            'setup_overrides': copy.deepcopy(self.setup_overrides),
        }
        if self.instruction:
            request['instruction'] = self.instruction
        # RoboTwin rejects individual seeds whose scene is unstable or whose
        # expert plan fails; the official entrypoint advances the seed.
        response, episode_seed, attempts = None, None, 0
        for attempt in range(self.max_seed_attempts):
            candidate = self._seed_cursor
            self._seed_cursor += 1
            response = self.bridge.request_raw({**request, 'seed': candidate})
            attempts = attempt + 1
            if response.get('ok'):
                episode_seed = candidate
                break
            if not response.get('retryable'):
                raise RuntimeError(
                    f'RoboTwin reset failed for seed {candidate}: '
                    f'{response.get("error")}')
            overwatch.warning(
                f'RoboTwin seed {candidate} rejected ({response.get("error")}); '
                'trying the next seed')
        if episode_seed is None:
            raise RuntimeError(
                f'No usable RoboTwin seed within {self.max_seed_attempts} '
                f'attempts starting at {base_seed}')
        step_limit = int(response.get('step_limit') or 0)
        steps = 0
        predictions = 0
        success = bool(response.get('success'))
        queue: List[np.ndarray] = []
        while not success and (step_limit <= 0 or steps < step_limit):
            if not queue:
                queue = list(self._predict(response, episode_seed, predictions))
                predictions += 1
            action = queue.pop(0)
            response = self.bridge.request(
                {'op': 'step', 'action': action.tolist()})
            steps = int(response.get('step', steps))
            success = bool(response.get('success'))
            if response.get('done'):
                break
        return {
            'episode_index': episode_index,
            'seed': episode_seed,
            'seed_candidates_start': base_seed,
            'seed_attempts': attempts,
            'instruction': response.get('instruction'),
            'success': bool(success),
            'steps': steps,
            'step_limit': step_limit,
            'predictions': predictions,
        }

    def run(self) -> None:
        if self.bridge is None or self.vla is None:
            raise RuntimeError('run_setup() must be called before run()')
        episodes: List[Dict[str, Any]] = []
        total = self.episodes_per_task
        if self.max_episodes is not None:
            total = min(total, self.max_episodes)
        for episode_index in range(total):
            record = self._run_episode(episode_index)
            episodes.append(record)
            overwatch.info(
                f'RoboTwin episode {episode_index}: '
                f'success={record["success"]} steps={record["steps"]}/'
                f'{record["step_limit"]} predictions={record["predictions"]} '
                f'instruction={record["instruction"]!r}')
            self._write_results(episodes, total)

    # ----------------------------------------------------------------- output
    def _write_results(self, episodes: List[Dict[str, Any]],
                       requested: int) -> None:
        completed = len(episodes)
        successes = sum(1 for record in episodes if record['success'])
        payload = {
            'benchmark': 'RoboTwin 2.0',
            'task': self.task_name,
            'task_config': self.task_config,
            'protocol': {
                'action_horizon': self.action_horizon,
                'replan_steps': self.replan_steps,
                'num_inference_steps': self.num_inference_steps,
                'instruction_type': self.instruction_type,
                'expert_check': self.expert_check,
                'base_seed': self.seed,
                'seed_scheme': (
                    'monotonic cursor starting at 100000 * (1 + base_seed); '
                    'every attempt (accepted or rejected) advances it by one'),
            },
            'seeds': {
                'initial': 100000 * (1 + self.seed),
                'next_candidate': (self._seed_cursor if self._seed_cursor
                                   is not None else 100000 * (1 + self.seed)),
            },
            'checkpoint': self.resolved_checkpoint,
            'config_path': self.config_path,
            'dataset_stats': str(self.dataset_stats_path),
            'normalization': {
                'state': self.state_normalizer.signature(),
                'action': self.action_normalizer.signature(),
            },
            'robotwin_root': str(self.robotwin_root),
            'model_family': self.model_family,
            'requested_episodes': requested,
            'completed_episodes': completed,
            'successes': successes,
            'success_rate': (successes / completed) if completed else None,
            'episodes': episodes,
        }
        self.results = payload
        if self.run_dir is not None:
            with open(self.run_dir / 'results.json', 'w',
                      encoding='utf-8') as stream:
                json.dump(payload, stream, indent=2)
