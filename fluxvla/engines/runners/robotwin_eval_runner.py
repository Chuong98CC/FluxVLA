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

"""RoboTwin evaluation runners.

This module hosts two independent closed-loop RoboTwin evaluators.

``RoboTwinEvalRunner`` keeps the policy in the FluxVLA environment while the
simulator runs in its own RoboTwin environment behind
``tools/robotwin_sim_worker.py``. The runner owns replanning
(``action_horizon`` actions per call, one simulator ``take_action`` per loop
iteration), instruction delivery, z-score normalization from the checkpoint's
saved statistics, and the results JSON. Replanning and observation cadence
follow the RoboTwin direct task API: a new observation is requested only when
the action queue is empty, so the simulator advances exactly one
``take_action`` per loop iteration.

``RobotwinEvalRunner`` embeds the RoboTwin evaluation harness in-process and
implements its ``eval_policy.py`` policy contract (``get_model`` / ``eval`` /
``reset_model`` / ``close_model``) so the simulator drives the policy
directly. It adds per-task progress tracking, summary/report artifacts, and
distributed worker support.
"""

import base64
import copy
import csv
import gc
import hashlib
import importlib.util
import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import yaml
from safetensors.torch import load_file

from fluxvla.engines.utils import initialize_overwatch
from fluxvla.engines.utils.name_map import str_to_dtype
from fluxvla.engines.utils.torch_utils import set_seed_everywhere

from ..utils.root import RUNNERS
from .base_eval_runner import BaseEvalRunner

overwatch = initialize_overwatch(__name__)

__all__ = ['RoboTwinEvalRunner', 'RobotwinEvalRunner']

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
    """Load z-score codecs from a released asset or native training statistics.

    The released ``dataset_stats.json`` schema is
    ``{state: {default: {global_mean, global_std}}, action: {...}}``. A newly
    trained run writes the native ``dataset_statistics.json`` schema
    (``{statistic_name: {proprio: {mean, std, ...}, action: {...}}}``) instead;
    that payload is converted on load with
    :func:`~fluxvla.datasets.utils.statistics.robotwin_statistics_from_native`
    so trained weights evaluate with their own training statistics. The released
    schema path is unchanged and keeps its original validation errors.
    """
    with open(path, 'r', encoding='utf-8') as stream:
        payload = json.load(stream)
    if not (isinstance(payload, dict)
            and ('state' in payload or 'action' in payload)):
        # Imported here to keep the runner module import-light and free of a
        # dataset <-> engines cycle.
        from fluxvla.datasets.utils.statistics import \
            robotwin_statistics_from_native
        payload = robotwin_statistics_from_native(payload)
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
        instruction_selection: How the worker picks one instruction out of the
            generated variants. ``seeded`` (default) keeps the native
            per-episode deterministic ``np.random.default_rng`` choice;
            ``official`` reproduces the released entrypoint's global
            ``np.random.choice`` over the variant pool.
        instruction_pool_size: Number of instruction variants the worker asks
            the RoboTwin generator for; defaults to ``episodes_per_task`` (the
            released ``test_num`` pool size) and is only used by
            ``instruction_selection='official'``.
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
        instruction_selection: str = 'seeded',
        instruction_pool_size: Optional[int] = None,
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
        self.instruction_selection = str(instruction_selection)
        if self.instruction_selection not in ('seeded', 'official'):
            raise ValueError(
                "instruction_selection must be 'seeded' or 'official', got "
                f'{self.instruction_selection!r}')
        self.expert_check = bool(expert_check)
        self.episodes_per_task = int(episodes_per_task)
        if self.episodes_per_task < 1:
            raise ValueError('episodes_per_task must be at least one')
        self.instruction_pool_size = (
            self.episodes_per_task if instruction_pool_size is None else
            int(instruction_pool_size))
        if self.instruction_pool_size < 1:
            raise ValueError('instruction_pool_size must be positive')
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
            'instruction_selection': self.instruction_selection,
            'instruction_pool_size': self.instruction_pool_size,
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
                'instruction_selection': self.instruction_selection,
                'instruction_pool_size': self.instruction_pool_size,
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


# Map RoboTwin observation cameras to the FluxVLA inference image keys.
# FluxVLA converted data order: cam_high + cam_left_wrist + cam_right_wrist.
ROBOTWIN_CAMERA_KEYS = (
    ('cam_high', 'head_camera'),
    ('cam_left_wrist', 'left_camera'),
    ('cam_right_wrist', 'right_camera'),
)

# RoboTwin policy contract. Upstream ``script/eval_policy.py`` resolves the
# policy with ``importlib.import_module(policy_name)`` and reads the
# module-level ``get_model`` / ``eval`` / ``reset_model`` / ``close_model``
# attributes, so ``RobotwinEvalRunner`` passes its own module path as
# ``policy_name`` and the functions below implement that contract in-process.


def _robotwin_print(*args, **kwargs) -> None:
    """Suppress only RoboTwin's per-action progress output."""
    if (args and isinstance(args[0], str) and args[0].startswith('step:')
            and kwargs.get('end') == '\r'):
        return
    print(*args, **kwargs)


def _build_observation(observation: Dict[str, Any], instruction: str) -> Dict:
    """Translate a RoboTwin observation into FluxVLA inference fields."""
    cameras = observation['observation']
    result = {
        target: np.ascontiguousarray(cameras[source]['rgb'], dtype=np.uint8)
        for target, source in ROBOTWIN_CAMERA_KEYS
    }
    result['qpos'] = np.asarray(
        observation['joint_action']['vector'], dtype=np.float32)
    if result['qpos'].shape != (14, ):
        raise ValueError(f"Expected 14-D qpos, got {result['qpos'].shape}")
    result['task_description'] = instruction
    return result


def _validate_actions(actions: np.ndarray, eval_chunk_size: int) -> np.ndarray:
    """Validate and truncate denormalized RoboTwin actions."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(
            f'Expected FluxVLA actions shaped [T, 14], got {actions.shape}')
    if actions.shape[0] < eval_chunk_size:
        raise ValueError(
            f'FluxVLA returned {actions.shape[0]} actions, fewer than '
            f'eval_chunk_size={eval_chunk_size}')
    if not np.isfinite(actions).all():
        raise ValueError('FluxVLA actions must be finite')
    return actions[:eval_chunk_size]


def _load_state_dict(ckpt_path: Path) -> Dict:
    """Load a single safetensors or PyTorch checkpoint onto CPU."""
    ckpt_path = str(ckpt_path)
    if ckpt_path.endswith('.safetensors'):
        state_dict = load_file(ckpt_path, device='cpu')
    else:
        # A sibling .safetensors is preferred when available because
        # the .pt file also contains the optimizer/scheduler state
        # which is unnecessary for inference and quickly exhausts
        # CPU RAM when loaded on every rank (SIGKILL / exit -9).
        sf_candidate = (
            ckpt_path[:-len('.pt')] +
            '.safetensors' if ckpt_path.endswith('.pt') else None)
        if sf_candidate is not None and os.path.exists(sf_candidate):
            state_dict = load_file(sf_candidate, device='cpu')
        else:
            # mmap=True avoids copying the whole checkpoint into RAM
            # on every rank.
            try:
                checkpoint = torch.load(
                    ckpt_path, map_location='cpu', mmap=True)
            except TypeError:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
                # Drop optimizer/scheduler state ASAP to reclaim RAM.
                checkpoint.pop('optimizer_state_dict', None)
                checkpoint.pop('scheduler_state_dict', None)
                checkpoint.pop('optimizer_state_index_to_name', None)
            else:
                state_dict = checkpoint
            del checkpoint
            gc.collect()
    return state_dict


def _inject_checkpoint_tokenizer(dataset: Dict, ckpt_path: str) -> None:
    model_path = Path(ckpt_path).resolve().parent.parent
    tokenizer_path = model_path / 'tokenizer'
    if not tokenizer_path.is_dir():
        return

    for transform in dataset.get('transforms', []):
        tokenizer = transform.get('tokenizer')
        if isinstance(tokenizer, dict):
            tokenizer['model_path'] = tokenizer_path.as_posix()


class LocalFluxVLAPolicy:
    """FluxVLA model, preprocessing and denormalization in one process.

    Args:
        cfg (Dict): Full config object containing the model section.
        ckpt_path (str): Path to the model checkpoint.
        dataset (Dict): Evaluation dataset config.
        denormalize_action (Dict): Action denormalization transform config.
        eval_chunk_size (int): Number of predicted actions executed per
            prediction.
        unnorm_key (str): Top-level key in the dataset statistics.
        device (str): Torch device the model runs on.
        mixed_precision_dtype (str): Mixed precision dtype name.
        enable_mixed_precision_training (bool): Whether to run inference
            under mixed precision autocast. Default is True.
        norm_stats_path (str): Optional explicit dataset statistics path.
        max_episode_steps (int): Optional cap on actions per evaluation
            episode.
    """

    def __init__(self,
                 cfg: Dict,
                 ckpt_path: str,
                 dataset: Dict[str, Any],
                 denormalize_action: Dict[str, Any],
                 eval_chunk_size: int = 32,
                 unnorm_key: str = 'private',
                 device: str = 'cuda:0',
                 mixed_precision_dtype: str = 'bf16',
                 enable_mixed_precision_training: bool = True,
                 norm_stats_path: Optional[str] = None,
                 max_episode_steps: Optional[int] = None) -> None:
        from fluxvla.engines import (build_dataset_from_cfg,
                                     build_transform_from_cfg,
                                     build_vla_from_cfg)

        self.ckpt_path = Path(ckpt_path).expanduser().resolve()
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(
                f'Checkpoint path {self.ckpt_path} does not exist!')
        self.norm_stats_path = (
            norm_stats_path if norm_stats_path is not None else
            BaseEvalRunner.default_stats_path(self.ckpt_path))
        if not Path(self.norm_stats_path).is_file():
            raise FileNotFoundError(f'Dataset statistics file not found at '
                                    f'{self.norm_stats_path}!')

        self.max_episode_steps = max_episode_steps
        self.eval_chunk_size = eval_chunk_size
        if self.eval_chunk_size <= 0:
            raise ValueError('eval_chunk_size must be positive')
        self.unnorm_key = unnorm_key
        self.device = torch.device(device)
        self.mixed_precision_dtype = str_to_dtype(mixed_precision_dtype)
        self.enable_mixed_precision_training = enable_mixed_precision_training

        dataset_cfg = copy.deepcopy(dataset)
        _inject_checkpoint_tokenizer(dataset_cfg, str(self.ckpt_path))
        dataset_cfg['norm_stats'] = self.norm_stats_path
        dataset_cfg.setdefault('model_path', None)
        dataset_cfg.setdefault('statistic_name', self.unnorm_key)
        self.dataset = build_dataset_from_cfg(dataset_cfg)

        denormalize_cfg = copy.deepcopy(denormalize_action)
        denormalize_cfg['norm_stats'] = self.norm_stats_path
        denormalize_cfg.setdefault('statistic_name', self.unnorm_key)
        self.denormalize_action = build_transform_from_cfg(denormalize_cfg)

        # Build model and load checkpoint weights.
        model_cfg = BaseEvalRunner.prepare_eval_model_cfg(cfg)
        self.vla = build_vla_from_cfg(model_cfg).eval()
        self.vla.load_state_dict(_load_state_dict(self.ckpt_path), strict=True)

        # Attach norm_stats to the model for heads that consume them.
        with open(self.norm_stats_path, 'r', encoding='utf-8') as f:
            self.vla.norm_stats = json.load(f)
        self.vla.to(self.device)

    def predict(self, observation: Dict[str, Any],
                instruction: str) -> np.ndarray:
        """Predict one denormalized action chunk for a RoboTwin observation."""
        # Build input dict for the dataset transform pipeline.
        batch = self.dataset(_build_observation(observation, instruction))
        if isinstance(batch, tuple):
            batch = batch[0]
        batch['unnorm_key'] = self.unnorm_key
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)

        # Model inference.
        with torch.autocast(
                'cuda',
                dtype=self.mixed_precision_dtype,
                enabled=self.enable_mixed_precision_training
                and self.device.type == 'cuda'):
            with torch.no_grad():
                raw_actions = self.vla.predict_action(**batch)

        # Denormalize to raw joint positions. Delta-action recipes add the
        # current raw state back, so pass it through when the dataset
        # exposes it (the RoboCasa convention).
        denorm_input = dict(action=raw_actions.float().cpu().numpy())
        raw_state = getattr(self.dataset, 'last_raw_state', None)
        if raw_state is not None:
            denorm_input['state'] = raw_state
        actions = self.denormalize_action(denorm_input)
        return _validate_actions(actions, self.eval_chunk_size)

    def reset(self) -> None:
        """Reset model-side episode state when the model supports it."""
        reset = getattr(self.vla, 'reset', None)
        if callable(reset):
            reset()

    def close(self) -> None:
        """Release the model and its CUDA allocations."""
        vla = getattr(self, 'vla', None)
        if vla is not None:
            vla.cpu()
        self.vla = None
        self.dataset = None
        self.denormalize_action = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# One model is shared across tasks; upstream calls ``get_model`` once per task.
_ACTIVE_POLICY: Optional[LocalFluxVLAPolicy] = None
_ACTIVE_KEY: Optional[tuple] = None


def get_model(usr_args: Dict[str, Any]) -> LocalFluxVLAPolicy:
    """RoboTwin policy entrypoint with one model shared across tasks."""
    global _ACTIVE_POLICY, _ACTIVE_KEY

    cfg = usr_args.get('_fluxvla_cfg')
    if cfg is None:
        config_path = usr_args.get('fluxvla_config')
        if not config_path:
            raise ValueError(
                'Unified RoboTwin evaluation requires _fluxvla_cfg or '
                'fluxvla_config')
        from mmengine import Config
        cfg = Config.fromfile(config_path)

    dataset = usr_args.get('dataset')
    denormalize_action = usr_args.get('denormalize_action')
    if dataset is None or denormalize_action is None:
        # Standalone use through ``fluxvla_config``: read the runner section.
        eval_cfg = getattr(cfg, 'eval', None) or {}
        dataset = dataset if dataset is not None else eval_cfg.get('dataset')
        denormalize_action = (
            denormalize_action if denormalize_action is not None else
            eval_cfg.get('denormalize_action'))

    eval_chunk_size = usr_args.get('eval_chunk_size', 32)

    # Rebuild only when the checkpoint or inference settings change.
    key = (
        id(cfg),
        str(Path(usr_args['ckpt_path']).expanduser().resolve()),
        eval_chunk_size,
        usr_args.get('device', 'cuda:0'),
        usr_args.get('mixed_precision_dtype', 'bf16'),
        usr_args.get('enable_mixed_precision_training', True),
        usr_args.get('norm_stats_path') or '',
        usr_args.get('max_episode_steps'),
    )
    if _ACTIVE_POLICY is not None and _ACTIVE_KEY != key:
        close_model()
    if _ACTIVE_POLICY is None:
        _ACTIVE_POLICY = LocalFluxVLAPolicy(
            cfg=cfg,
            ckpt_path=key[1],
            dataset=dataset,
            denormalize_action=denormalize_action,
            eval_chunk_size=key[2],
            unnorm_key=usr_args.get('unnorm_key', 'private'),
            device=key[3],
            mixed_precision_dtype=key[4],
            enable_mixed_precision_training=key[5],
            norm_stats_path=usr_args.get('norm_stats_path'),
            max_episode_steps=key[7],
        )
        _ACTIVE_KEY = key
    return _ACTIVE_POLICY


def eval(task_env: Any, model: LocalFluxVLAPolicy,
         observation: Dict[str, Any]) -> None:
    """Predict and execute one action chunk in RoboTwin."""
    if model.max_episode_steps is not None:
        task_env.step_lim = min(task_env.step_lim, model.max_episode_steps)
    actions = model.predict(observation, task_env.get_instruction())
    for action in actions:
        if (task_env.eval_success
                or task_env.take_action_cnt >= task_env.step_lim):
            break
        task_env.take_action(action, action_type='qpos')


def reset_model(model: LocalFluxVLAPolicy) -> None:
    """RoboTwin policy hook called before every episode."""
    model.reset()


def close_model() -> None:
    """Release the cached model and its CUDA allocations."""
    global _ACTIVE_POLICY, _ACTIVE_KEY
    if _ACTIVE_POLICY is not None:
        _ACTIVE_POLICY.close()
    _ACTIVE_POLICY = None
    _ACTIVE_KEY = None


@RUNNERS.register_module()
class RobotwinEvalRunner(BaseEvalRunner):
    """Runner for evaluating VLA models on RoboTwin simulation tasks.

    The runner drives an unmodified upstream RoboTwin checkout in-process:
    it loads ``script/eval_policy.py`` by path and calls its ``main`` per
    task, so task setup, seed selection and success accounting are exactly
    upstream's.

    Under torchrun, complete tasks are assigned by global rank. Each task
    retains the upstream trial loop. All ranks share a run directory and
    rank zero writes the merged summary. Per-task progress supports resuming
    with a different world size. Waiting for other ranks to finish their
    tasks has no time limit; result exchange starts only after all ranks
    finish.

    Args:
        cfg (Dict): Full config object containing the model section.
        seed (int): Random seed.
        ckpt_path (str): Path to the model checkpoint.
        model_family (str): Model family name, such as ``pi0`` or ``groot``.
        task_list (list or str): RoboTwin task names to evaluate.
        dataset (Dict): Evaluation dataset config.
        denormalize_action (Dict): Action denormalization transform config.
        robotwin_root (str): RoboTwin checkout root. Defaults to the
            ``ROBOTWIN_ROOT`` environment variable or ``src/RoboTwin``.
        task_suite_name (str): RoboTwin evaluation suite: ``clean`` or
            ``random``. Mapped to upstream task configuration names.
        eval_chunk_size (int): Number of predicted actions executed per
            prediction. Must not exceed the trained action chunk.
        num_trials_per_task (int): Number of trials for each task.
        max_episode_steps (int): Optional cap on actions per evaluation
            episode. Defaults to the upstream task limit.
        eval_shard_strategy (str): Must be ``task``. Each rank evaluates all
            trials for its assigned tasks.
        instruction_type (str): RoboTwin instruction type, such as ``seen``
            or ``unseen``.
        checkpoint_label (str): RoboTwin ``ckpt_setting`` label. Defaults to
            the checkpoint stem.
        resume (bool): Whether to reuse the newest unfinished run directory
            for the same checkpoint and ``task_suite_name``.
        save_video (bool): Whether to save rollout videos.
        unnorm_key (str): Top-level key in the dataset statistics.
        norm_stats_path (str): Optional explicit dataset statistics path.
        vulkan_icd (str): Optional explicit Vulkan ICD JSON path exported as
            ``VK_ICD_FILENAMES``. When unset, a fallback manifest is written
            automatically on images without a system NVIDIA ICD.
        run_id_suffix (str): Optional suffix appended to the eval run id.
        result_output_dir (str): Optional output root. Eval artifacts are
            written under ``<result_output_dir>/eval_runs/<ckpt>/<run_id>``
            to match LIBERO.
        mixed_precision_dtype (str): Mixed precision dtype name.
        enable_mixed_precision_training (bool): Whether to run inference
            under mixed precision autocast. Default is True.
    """

    # Upstream imports the policy by module name; this module is the policy.
    POLICY_MODULE = __name__
    # Default checkout location provisioned by ``scripts/install_env.sh``
    # (``FLUXVLA_ROBOTWIN_ROOT``); ``ROBOTWIN_ROOT`` overrides it at runtime.
    DEFAULT_ROBOTWIN_ROOT = 'src/RoboTwin'
    ROBOTWIN_ROOT_ENV = 'ROBOTWIN_ROOT'
    RUN_ID_PREFIX = 'EVAL-robotwin'
    # Keep upstream YAML filenames and API names inside the adapter.
    UPSTREAM_TASK_CONFIGS = {
        'clean': 'demo_clean',
        'random': 'demo_randomized',
    }
    # Name tag of the per-task derived task_config YAML (see
    # ``_effective_task_config_file``).
    DERIVED_TASK_CONFIG_TAG = '__fluxvla_video_'
    # SAPIEN renders through Vulkan; stripped container images often lack
    # the NVIDIA ICD manifest the Vulkan loader needs to find the driver.
    NVIDIA_ICD_MANIFEST = {
        'file_format_version': '1.0.0',
        'ICD': {
            'library_path': 'libGLX_nvidia.so.0',
            'api_version': '1.2.175',
        },
    }
    VULKAN_ICD_SEARCH_DIRS = ('/usr/share/vulkan/icd.d', '/etc/vulkan/icd.d')

    def __init__(self,
                 cfg: Dict,
                 seed: int,
                 ckpt_path: str,
                 model_family: str,
                 task_list: Union[Iterable[str], str],
                 dataset: Dict,
                 denormalize_action: Dict,
                 robotwin_root: Optional[str] = None,
                 task_suite_name: str = 'clean',
                 eval_chunk_size: int = 32,
                 num_trials_per_task: int = 100,
                 instruction_type: str = 'unseen',
                 checkpoint_label: Optional[str] = None,
                 resume: bool = True,
                 save_video: bool = False,
                 unnorm_key: str = 'private',
                 norm_stats_path: Optional[str] = None,
                 vulkan_icd: Optional[str] = None,
                 run_id_suffix: Optional[str] = None,
                 result_output_dir: Optional[str] = None,
                 mixed_precision_dtype: str = 'bf16',
                 enable_mixed_precision_training: bool = True,
                 eval_shard_strategy: str = 'task',
                 max_episode_steps: Optional[int] = None,
                 **kwargs) -> None:
        # Manager workers run as plain single-process python, where overwatch
        # is the pure logger without ``local_rank``.
        local_rank = getattr(overwatch, 'local_rank', None)
        self.device_id = local_rank() if callable(local_rank) else 0
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        if self.world_size > 1 and torch.cuda.is_available():
            # eval.py exposes one GPU per process. Use the CUDA ordinal
            # selected during distributed initialization.
            self.device_id = torch.cuda.current_device()

        self.cfg = cfg
        self.seed = seed
        self.ckpt_path = str(Path(ckpt_path).expanduser().resolve())
        self.model_family = model_family
        self.robotwin_root = self._resolve_robotwin_root(robotwin_root)
        self.dataset_cfg = dataset
        self.denormalize_action_cfg = denormalize_action
        self.task_list = self._normalize_task_list(task_list)
        if eval_shard_strategy != 'task':
            raise ValueError('RoboTwin supports only eval_shard_strategy=task')
        self.local_task_list = self.task_list[self.rank::self.world_size]
        if 'task_config' in kwargs:
            raise TypeError(
                'Use task_suite_name=clean or random instead of task_config')
        if task_suite_name not in self.UPSTREAM_TASK_CONFIGS:
            raise ValueError(
                'RoboTwin task_suite_name must be clean or random')
        self.task_suite_name = task_suite_name
        self._upstream_task_config = self.UPSTREAM_TASK_CONFIGS[
            task_suite_name]
        if max_episode_steps is not None and (
                isinstance(max_episode_steps, bool)
                or not isinstance(max_episode_steps, int)
                or max_episode_steps <= 0):
            raise ValueError('max_episode_steps must be a positive integer')
        self.max_episode_steps = max_episode_steps
        self.num_trials_per_task = num_trials_per_task
        self.eval_chunk_size = eval_chunk_size
        self.instruction_type = instruction_type
        self.checkpoint_label = (
            checkpoint_label
            if checkpoint_label else Path(self.ckpt_path).stem)
        self.resume = resume
        self.save_video = save_video
        self.unnorm_key = unnorm_key
        self.mixed_precision_dtype = mixed_precision_dtype
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.norm_stats_path = norm_stats_path
        self.vulkan_icd = vulkan_icd
        self.run_id_suffix = run_id_suffix
        self.result_output_dir = result_output_dir
        run_dir = self._resolve_run_dir() if self.rank == 0 else None
        self._completion_store = None
        if self.world_size > 1:
            run_dir_holder = [
                run_dir, uuid.uuid4().hex if self.rank == 0 else None
            ]
            dist.broadcast_object_list(run_dir_holder, src=0)
            run_dir, sync_id = run_dir_holder
            # A fresh namespace prevents resumed runs from reading old state.
            self._completion_store = dist.PrefixStore(
                f'robotwin_completion/{sync_id}',
                dist.distributed_c10d._get_default_store())
        self.run_dir = run_dir
        self.summary_path = Path(self.run_dir) / 'summary.json'
        self._warp_cache = None
        self._previous_warp_cache = None
        self._previous_warp_config = None

        # Per-task state set while upstream ``main`` runs.
        self._effective_task_config = self._upstream_task_config
        self._policy_module = None
        self._robotwin_eval_module = None
        self._last_success_count = None
        self._last_episode_count = None

    @classmethod
    def _resolve_robotwin_root(cls, robotwin_root: Optional[str]) -> Path:
        """Explicit argument > ``ROBOTWIN_ROOT`` env > ``src/RoboTwin``."""
        root = robotwin_root or os.environ.get(
            cls.ROBOTWIN_ROOT_ENV) or cls.DEFAULT_ROBOTWIN_ROOT
        return Path(root).expanduser().resolve()

    @staticmethod
    def _normalize_task_list(
            task_list: Union[Iterable[str], str]) -> List[str]:
        """Normalize the task filter into an ordered, duplicate-free list."""
        if isinstance(task_list, str):
            task_list = [task_list]
        resolved = [task.strip() for task in task_list if task.strip()]
        if len(resolved) == 0:
            raise ValueError('task_list must contain at least one task')
        if len(set(resolved)) != len(resolved):
            raise ValueError('task_list must not contain duplicates')
        return resolved

    @classmethod
    def _build_run_id(cls,
                      model_family: str,
                      timestamp: str,
                      suffix: Optional[str] = None) -> str:
        run_id = f'{cls.RUN_ID_PREFIX}-{model_family}-{timestamp}'
        if suffix:
            run_id = f'{run_id}-{suffix}'
        return run_id

    @staticmethod
    def _build_ckpt_tag(ckpt_path: str) -> str:
        """Stable per-checkpoint folder name for grouping eval runs."""
        return Path(ckpt_path).resolve().stem

    @classmethod
    def _build_ckpt_root(cls,
                         ckpt_path: str,
                         output_dir: Optional[str] = None) -> Path:
        """Per-checkpoint directory holding all eval runs."""
        if output_dir is not None:
            root = Path(output_dir).expanduser().resolve()
        else:
            root = Path(ckpt_path).resolve().parent.parent
        return root / 'eval_runs' / cls._build_ckpt_tag(ckpt_path)

    def _resolve_run_dir(self) -> str:
        """Reuse the newest matching unfinished run, or create a new run."""
        ckpt_root = self._build_ckpt_root(self.ckpt_path,
                                          self.result_output_dir)
        if self.resume and ckpt_root.is_dir():
            pattern = f'{self.RUN_ID_PREFIX}-{self.model_family}-*'
            if self.run_id_suffix:
                pattern = f'{pattern}-{self.run_id_suffix}'
            candidates = sorted(
                (d for d in ckpt_root.glob(pattern) if d.is_dir()),
                key=lambda d: d.stat().st_mtime,
                reverse=True)
            for candidate in candidates:
                summary = candidate / 'summary.json'
                if not summary.is_file():
                    continue
                try:
                    with open(summary, 'r', encoding='utf-8') as f:
                        payload = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                same_config = (
                    payload['ckpt'] == self.ckpt_path
                    and payload.get('task_suite_name') == self.task_suite_name
                    and payload['trials_per_task'] == self.num_trials_per_task
                    and payload.get('max_episode_steps')
                    == self.max_episode_steps and payload['seed'] == self.seed
                    and payload['instruction_type'] == self.instruction_type
                    and payload['eval_chunk_size'] == self.eval_chunk_size and
                    payload.get('task_list', self.task_list) == self.task_list)
                unfinished = (
                    payload['overall']['completed_tasks'] <
                    payload['overall']['total_tasks'])
                if same_config and unfinished:
                    return str(candidate)
        timestamp = time.strftime('%Y_%m_%d-%H_%M_%S')
        run_id = self._build_run_id(self.model_family, timestamp,
                                    self.run_id_suffix)
        run_dir = ckpt_root / run_id
        # Second-resolution timestamps can collide (e.g. two runs launched by
        # a manager in the same second); disambiguate with a counter.
        counter = 1
        while run_dir.exists():
            run_dir = ckpt_root / f'{run_id}-{counter}'
            counter += 1
        return str(run_dir)

    def _base_policy_args(self) -> Dict:
        """``usr_args`` fields consumed by the module-level policy."""
        return {
            '_fluxvla_cfg': self.cfg,
            'dataset': self.dataset_cfg,
            'denormalize_action': self.denormalize_action_cfg,
            'ckpt_path': self.ckpt_path,
            'eval_chunk_size': self.eval_chunk_size,
            'max_episode_steps': self.max_episode_steps,
            'unnorm_key': self.unnorm_key,
            'device': f'cuda:{self.device_id}',
            'mixed_precision_dtype': self.mixed_precision_dtype,
            'enable_mixed_precision_training':
            self.enable_mixed_precision_training,
            'norm_stats_path': self.norm_stats_path,
        }

    def _setup_vulkan_icd(self) -> None:
        """Point Vulkan at the NVIDIA driver on stripped container images."""
        # An explicit path always wins.
        if self.vulkan_icd:
            icd_path = Path(self.vulkan_icd).expanduser().resolve()
            if not icd_path.is_file():
                raise FileNotFoundError(f'Vulkan ICD not found: {icd_path}')
            os.environ['VK_ICD_FILENAMES'] = str(icd_path)
            return
        # Nothing to do when the loader is already configured or a system
        # NVIDIA ICD manifest exists.
        if os.environ.get('VK_ICD_FILENAMES'):
            return
        for icd_dir in self.VULKAN_ICD_SEARCH_DIRS:
            if any(Path(icd_dir).glob('nvidia*.json')):
                return
        # Otherwise write a fallback manifest under run_dir so SAPIEN's
        # headless Vulkan rendering works out of the box.
        icd_path = Path(self.run_dir) / f'nvidia_icd_rank{self.rank}.json'
        icd_path.write_text(
            json.dumps(self.NVIDIA_ICD_MANIFEST, indent=2) + '\n',
            encoding='utf-8')
        os.environ['VK_ICD_FILENAMES'] = str(icd_path)
        self._log_rank(f'Wrote fallback Vulkan ICD manifest: {icd_path}')

    def _log_rank(self, message: str) -> None:
        """Write runner messages to the console and this rank's log."""
        overwatch.info(message)
        with (Path(self.run_dir) / f'rank{self.rank}.txt').open(
                'a', encoding='utf-8') as log_file:
            log_file.write(message + '\n')

    def run_setup(self) -> None:
        """Validate inputs, prepare the run directory and build the model."""
        Path(self.run_dir).mkdir(parents=True, exist_ok=True)
        self._log_rank(
            f'RoboTwin setup: {time.strftime("%Y-%m-%d %H:%M:%S")}, '
            f'rank={self.rank}, world_size={self.world_size}, '
            f'checkpoint={self.ckpt_path}, '
            f'task_suite_name={self.task_suite_name}, '
            f'seed={self.seed}, local_tasks={self.local_task_list}')
        checkpoint = Path(self.ckpt_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f'Checkpoint path {checkpoint} does not exist!')
        eval_script = self.robotwin_root / 'script' / 'eval_policy.py'
        if not eval_script.is_file():
            raise FileNotFoundError(
                f'RoboTwin eval entrypoint not found: {eval_script}')
        if self.num_trials_per_task <= 0:
            raise ValueError('num_trials_per_task must be positive')
        if self.rank == 0:
            self._write_summary_json(self._load_task_progress())
        if not self.local_task_list:
            return
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device_id)
        if self.world_size > 1:
            self._warp_cache = tempfile.TemporaryDirectory(
                prefix=f'fluxvla_robotwin_rank{self.rank}_')
            self._previous_warp_cache = os.environ.get('WARP_CACHE_PATH')
            os.environ['WARP_CACHE_PATH'] = self._warp_cache.name
            # Warp may already have been imported by another runner.
            import warp
            self._previous_warp_config = warp.config.kernel_cache_dir
            warp.config.kernel_cache_dir = self._warp_cache.name
        self._setup_vulkan_icd()
        set_seed_everywhere(self.seed)

        # Build model through the policy contract so upstream's per-task
        # ``get_model`` call finds it already loaded.
        self._policy_module = sys.modules[__name__]
        get_model(self._base_policy_args())
        self._log_rank(
            f'RoboTwin local model ready: checkpoint={self.ckpt_path}, '
            f'eval_chunk_size={self.eval_chunk_size}, '
            f'tasks={len(self.task_list)}')

    @contextmanager
    def _robotwin_context(self):
        """Run with the checkout as cwd and on ``sys.path``, then restore."""
        previous_cwd = Path.cwd()
        inserted = []
        original_create_scene = None
        if self.world_size > 1:
            import sapien
            original_create_scene = sapien.Engine.create_scene
            device = f'cuda:{self.device_id}'

            def create_scene(engine, config=None):
                # SAPIEN 3's legacy Engine otherwise constructs RenderSystem
                # without a device, which can render all ranks on GPU zero.
                if config is not None:
                    sapien.physx.set_scene_config(config)
                return sapien.Scene([
                    sapien.physx.PhysxCpuSystem(),
                    sapien.render.RenderSystem(device),
                ])

            sapien.Engine.create_scene = create_scene
        for path in (self.robotwin_root, self.robotwin_root / 'policy',
                     self.robotwin_root / 'description' / 'utils'):
            path_text = str(path)
            if path_text not in sys.path:
                sys.path.insert(0, path_text)
                inserted.append(path_text)
        os.chdir(self.robotwin_root)
        try:
            yield
        finally:
            if original_create_scene is not None:
                sapien.Engine.create_scene = original_create_scene
            os.chdir(previous_cwd)
            for path_text in inserted:
                if path_text in sys.path:
                    sys.path.remove(path_text)

    def _load_robotwin_eval_module(self):
        """Load upstream ``script/eval_policy.py`` by path (cached)."""
        if self._robotwin_eval_module is not None:
            return self._robotwin_eval_module
        module_path = self.robotwin_root / 'script' / 'eval_policy.py'
        with self._robotwin_context():
            spec = importlib.util.spec_from_file_location(
                'fluxvla_robotwin_eval_policy', module_path)
            if spec is None or spec.loader is None:
                raise ImportError(
                    f'Cannot load RoboTwin eval module: {module_path}')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            base_task = importlib.import_module('envs._base_task')
            base_task.print = _robotwin_print
        self._install_trial_hook(module)
        self._robotwin_eval_module = module
        return module

    def _install_trial_hook(self, module) -> None:
        """Wrap upstream ``eval_policy`` to set and observe the trial count."""
        original = module.eval_policy
        runner = self

        def eval_policy_hook(*args, **kwargs):
            kwargs['test_num'] = runner.num_trials_per_task
            result = original(*args, **kwargs)
            task_env = args[1] if len(args) > 1 else kwargs['TASK_ENV']
            episodes = int(task_env.test_num)
            successes = int(result[1])
            if (episodes != runner.num_trials_per_task
                    or not 0 <= successes <= episodes):
                raise RuntimeError(
                    f'RoboTwin returned {successes}/{episodes} episodes; '
                    f'expected {runner.num_trials_per_task} completed episodes'
                )
            runner._last_success_count = successes
            runner._last_episode_count = episodes
            return result

        module.eval_policy = eval_policy_hook

    @contextmanager
    def _effective_task_config_file(self):
        """Yield the ``task_config`` name upstream should load for one task."""
        config_dir = self.robotwin_root / 'task_config'
        source = config_dir / f'{self._upstream_task_config}.yml'
        if not source.is_file():
            raise FileNotFoundError(
                f'RoboTwin task_config not found: {source}')
        payload = yaml.safe_load(source.read_text(encoding='utf-8')) or {}
        if payload.get('eval_video_log', False) == self.save_video:
            self._effective_task_config = self._upstream_task_config
            yield self._upstream_task_config
            return
        payload['eval_video_log'] = self.save_video
        derived_name = (
            f'{self._upstream_task_config}{self.DERIVED_TASK_CONFIG_TAG}'
            f"{'on' if self.save_video else 'off'}_"
            f'{self.rank}_{os.getpid()}')
        derived = config_dir / f'{derived_name}.yml'
        derived.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding='utf-8')
        self._effective_task_config = derived_name
        try:
            yield derived_name
        finally:
            derived.unlink(missing_ok=True)
            self._effective_task_config = self._upstream_task_config

    def _task_args(self, task_name: str) -> Dict:
        """``usr_args`` passed to upstream ``main`` for one task."""
        args = self._base_policy_args()
        args.update({
            'task_name': task_name,
            'task_config': self._effective_task_config,
            'ckpt_setting': self._upstream_checkpoint_label(),
            'policy_name': self.POLICY_MODULE,
            'instruction_type': self.instruction_type,
            'seed': self.seed,
            'test_num': self.num_trials_per_task,
        })
        return args

    def _upstream_checkpoint_label(self) -> str:
        """Isolate upstream artifacts by distributed run and rank."""
        if self.world_size == 1:
            return self.checkpoint_label
        run_tag = hashlib.sha256(self.run_dir.encode()).hexdigest()[:12]
        return f'{self.checkpoint_label}__fluxvla_{run_tag}_rank{self.rank}'

    def _upstream_result_root(self, task_name: str) -> Path:
        """Directory where upstream writes this task's results."""
        return (self.robotwin_root / 'eval_result' / task_name /
                self.POLICY_MODULE / self._effective_task_config /
                self._upstream_checkpoint_label())

    @staticmethod
    def _correct_result_file(path, successes, total_episodes):
        """Correct upstream's final rate line using observed episode counts.

        Upstream main divides by 100. Preserve its timestamp and instruction
        metadata, and replace the final rate using observed episode counts.
        Reject unexpected formats before changing the content.
        """
        if total_episodes <= 0 or not 0 <= successes <= total_episodes:
            raise ValueError('Invalid RoboTwin episode counts')
        path = Path(path)
        lines = path.read_text(encoding='utf-8').rstrip().splitlines()
        if not lines:
            raise ValueError(f'Empty RoboTwin result: {path}')
        float(lines[-1])
        lines[-1] = str(successes / total_episodes)
        tmp_path = f'{path}.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        os.replace(tmp_path, path)

    def _collect_upstream_result(self, task_name: str, task_dir: Path,
                                 started_at: float) -> Path:
        """Move upstream's newest ``_result.txt`` for this task into task_dir.

        Upstream names the leaf directory with a wall-clock timestamp, so the
        newest leaf created after ``started_at`` belongs to this run. The
        upstream tree for this task is removed afterwards to keep the checkout
        clean and to make later runs unambiguous.
        """
        result_root = self._upstream_result_root(task_name)
        candidates = [
            leaf for leaf in result_root.glob('*/_result.txt')
            if leaf.stat().st_mtime >= started_at - 1.0
        ] if result_root.is_dir() else []
        if not candidates:
            raise FileNotFoundError(
                f'RoboTwin wrote no result for {task_name} under {result_root}'
            )
        newest = max(candidates, key=lambda leaf: leaf.stat().st_mtime)
        task_dir.mkdir(parents=True, exist_ok=True)
        target = task_dir / '_result.txt'
        shutil.move(str(newest), str(target))
        # Upstream main still divides by 100 even when the hook changes the
        # trial count. Normalize its artifact using observed episode counts.
        self._correct_result_file(target, self._last_success_count,
                                  self._last_episode_count)
        # Recorded ``episode*.mp4`` files live next to ``_result.txt``.
        for extra in newest.parent.iterdir():
            shutil.move(str(extra), str(task_dir / extra.name))
        shutil.rmtree(result_root, ignore_errors=True)
        parent = result_root.parent
        while parent != self.robotwin_root:
            try:
                parent.rmdir()
            except OSError:
                # Another rank may have created/removed a sibling directory.
                break
            parent = parent.parent
        return target

    def _load_task_progress(self) -> Dict[str, Dict]:
        """Recover completed tasks independently of the previous rank count."""
        previous = {}
        if self.resume and self.summary_path.is_file():
            with open(self.summary_path, 'r', encoding='utf-8') as f:
                previous = json.load(f)['task_results']
        task_results = {}
        for task in self.task_list:
            progress_path = Path(self.run_dir) / 'tasks' / task / 'result.json'
            stats = previous.get(task)
            if self.resume and progress_path.is_file():
                with open(progress_path, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
            task_results[task] = stats or {
                'status': 'MISSING',
                'successes': 0,
                'total_episodes': 0,
                'success_rate': None,
                'duration': 0.0,
                'gpu_id': None,
            }
        return task_results

    def _write_task_progress(self, task: str, stats: Dict) -> None:
        """Each task has one writer; atomic records survive worker exits."""
        progress_path = Path(self.run_dir) / 'tasks' / task / 'result.json'
        tmp_path = f'{progress_path}.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4)
            f.write('\n')
        os.replace(tmp_path, progress_path)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Human-readable duration for summary files."""
        seconds = int(round(seconds))
        if seconds < 60:
            return f'{seconds:02d}s'
        if seconds < 3600:
            return f'{seconds // 60:02d}m{seconds % 60:02d}s'
        hours, rem = divmod(seconds, 3600)
        return f'{hours:02d}h{rem // 60:02d}m{rem % 60:02d}s'

    @staticmethod
    def _summarize_completed_tasks(task_results: Dict[str, Dict]) -> Dict:
        """Aggregate completed tasks, including expert validation time."""
        completed = [
            stats for stats in task_results.values()
            if stats['status'] == 'COMPLETED'
        ]
        total_successes = sum(stats['successes'] for stats in completed)
        total_trials = sum(stats['total_episodes'] for stats in completed)
        total_time = sum(float(stats['duration'] or 0) for stats in completed)
        completed_tasks = len(completed)
        return {
            'total_successes':
            total_successes,
            'total_trials':
            total_trials,
            'completed_tasks':
            completed_tasks,
            'success_rate':
            (total_successes / total_trials * 100 if total_trials else None),
            'total_time':
            total_time,
            'average_task_time':
            (total_time / completed_tasks if completed_tasks else 0.0),
            'max_time':
            max((float(stats['duration'] or 0) for stats in completed),
                default=0.0),
        }

    def _write_summary_json(self, task_results: Dict[str, Dict]) -> Dict:
        """Replace ``summary.json`` atomically and return its payload.

        Single-process runs update the summary after every task. Distributed
        runs persist per-task progress and only rank zero writes the shared
        summary, at startup and after gathering the final results.
        """
        cfg_filename = getattr(self.cfg, 'filename', None)
        summary = {
            'run_id': Path(self.run_dir).name,
            'ckpt': self.ckpt_path,
            'config': Path(cfg_filename).stem if cfg_filename else '',
            'task_suite_name': self.task_suite_name,
            'instruction_type': self.instruction_type,
            'seed': self.seed,
            'eval_chunk_size': self.eval_chunk_size,
            'max_episode_steps': self.max_episode_steps,
            'trials_per_task': self.num_trials_per_task,
            'task_list': self.task_list,
        }
        task_results = {
            task: dict(stats)
            for task, stats in task_results.items()
        }
        for task, stats in task_results.items():
            completed = stats['status'] == 'COMPLETED'
            successes = int(stats['successes']) if completed else 0
            episodes = int(stats['total_episodes']) if completed else 0
            if completed and (episodes <= 0 or not 0 <= successes <= episodes):
                raise ValueError(f'Invalid RoboTwin episode counts for {task}')
            stats.update(
                successes=successes,
                total_episodes=episodes,
                success_rate=successes / episodes * 100 if episodes else None)

        totals = self._summarize_completed_tasks(task_results)
        group = {
            'clean': 'Easy',
            'random': 'Hard',
        }.get(self.task_suite_name)
        group_stats = {
            group: {
                'total_successes': totals['total_successes'],
                'total_trials': totals['total_trials'],
            }
        } if group else {}
        summary.update({
            'duration_scope': 'task_wall_time_including_expert_validation',
            'group_stats': group_stats,
            'task_results': task_results,
            'overall': {
                'success_rate': totals['success_rate'],
                'total_tasks': len(self.task_list),
                'completed_tasks': totals['completed_tasks'],
                'total_time': totals['total_time'],
                'average_task_time': totals['average_task_time'],
            },
        })
        summary_dir = Path(self.run_dir)
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_json = self.summary_path
        tmp_path = f'{summary_json}.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4)
            f.write('\n')
        os.replace(tmp_path, summary_json)
        return summary

    def _write_robotwin_summary_artifacts(self, task_results: Dict) -> str:
        """Write per-task and overall reports for this evaluation condition."""
        summary = self._write_summary_json(task_results)
        summary_dir = Path(self.run_dir)
        overall = summary['overall']
        overall_rate = overall['success_rate']
        completed_tasks = overall['completed_tasks']
        avg_time = overall['average_task_time']
        totals = self._summarize_completed_tasks(summary['task_results'])
        total_trials = totals['total_trials']
        total_successes = totals['total_successes']
        max_time = totals['max_time']
        metrics = [
            ('Success Rate (%)',
             '' if overall_rate is None else f'{overall_rate:.2f}'),
            ('Average Time (s)', f'{avg_time:.2f}'),
            ('Max Time (s)', f'{max_time:.2f}'),
            ('Episodes', total_trials),
            ('Successes', total_successes),
            ('Tasks Completed', completed_tasks),
            ('Tasks Expected', overall['total_tasks']),
        ]
        summary_csv = summary_dir / 'summary.csv'
        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([Path(summary['ckpt']).name])
            writer.writerow(['', 'Overall'])
            writer.writerows(metrics)

        task_csv = summary_dir / 'task_success_rates.csv'
        with open(task_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['Task', 'Successes', 'Episodes', 'Success Rate (%)'])
            for task, stats in summary['task_results'].items():
                rate = stats['success_rate']
                writer.writerow([
                    task,
                    stats['successes'],
                    stats['total_episodes'],
                    '' if rate is None else f'{rate:.2f}',
                ])

        display_rate = ('N/A'
                        if overall_rate is None else f'{overall_rate:.2f}%')
        lines = [
            '=== Evaluation Results Summary ===',
            '',
            'Overall statistics:',
            f'- Tasks completed: {completed_tasks}',
            f"- Tasks expected: {overall['total_tasks']}",
            f'- Total attempts: {total_trials}',
            f'- Successful attempts: {total_successes}',
            f'- Success rate: {display_rate}',
            f"- Total time: {self._format_duration(overall['total_time'])}",
            f'- Average time per task: {self._format_duration(avg_time)}',
            f'- Longest task time: {self._format_duration(max_time)}',
        ]
        failed = [
            task for task, stats in summary['task_results'].items()
            if stats['status'] != 'COMPLETED'
        ]
        if failed:
            lines.append(f"Failed/missing tasks: {', '.join(failed)}")
        summary_txt = summary_dir / 'summary.txt'
        with open(summary_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        return str(self.summary_path)

    def _run_local_tasks(self, task_results: Dict) -> None:
        """Evaluate complete tasks assigned to this rank."""
        if not self.local_task_list:
            return
        module = self._load_robotwin_eval_module()
        num_tasks = len(self.local_task_list)

        self._log_rank(f'Task suite: {self.task_suite_name}')
        self._log_rank(f'Running evaluation on {num_tasks} tasks '
                       f'with {self.num_trials_per_task} trials each.')
        self._log_rank(f'Using model family: {self.model_family}')
        self._log_rank(f'Using evaluation chunk size: {self.eval_chunk_size}')
        self._log_rank(f'Instruction type: {self.instruction_type}')
        self._log_rank(f'Seed: {self.seed}')

        for index, task_name in enumerate(self.local_task_list, start=1):
            task_dir = Path(self.run_dir) / 'tasks' / task_name
            result_path = task_dir / '_result.txt'
            old = task_results.get(task_name)
            if (old and old['status'] == 'COMPLETED'
                    and old['total_episodes'] == self.num_trials_per_task
                    and result_path.is_file()):
                self._log_rank(
                    f'Task {index}/{num_tasks} ({task_name}): resumed')
                continue

            task_dir.mkdir(parents=True, exist_ok=True)
            # Task randomness must not depend on its rank or the tasks that
            # previously ran in this process. Keep the configured seed;
            # upstream still selects its own valid episode seeds.
            set_seed_everywhere(self.seed)
            started = time.time()
            self._log_rank(
                f'Evaluating Task {index}/{num_tasks} ({task_name})')
            try:
                # Run upstream ``main`` for one task; the trial hook records
                # the success count and the result file is collected after.
                self._last_success_count = None
                self._last_episode_count = None
                with self._effective_task_config_file():
                    with self._robotwin_context():
                        module.main(self._task_args(task_name))
                    self._collect_upstream_result(task_name, task_dir, started)
                if (self._last_success_count is None
                        or self._last_episode_count is None):
                    raise RuntimeError(
                        'RoboTwin eval_policy hook did not run for '
                        f'{task_name}')
                successes = self._last_success_count
                episodes = self._last_episode_count
                rate = successes / episodes * 100
                stats = {
                    'status': 'COMPLETED',
                    'successes': successes,
                    'total_episodes': episodes,
                    'success_rate': rate,
                    'duration': time.time() - started,
                    'gpu_id': self.device_id,
                }
            except Exception as exc:
                self._log_rank(f'Task {task_name} failed: {exc!r}')
                # Record the failure so a resumed run retries this task, then
                # surface the error to the caller.
                stats = {
                    'status': 'FAILED',
                    'successes': 0,
                    'total_episodes': 0,
                    'success_rate': None,
                    'duration': time.time() - started,
                    'gpu_id': self.device_id,
                    'error': repr(exc),
                }
                task_results[task_name] = stats
                self._write_task_progress(task_name, stats)
                if self.world_size == 1:
                    self._write_summary_json(task_results)
                raise
            task_results[task_name] = stats
            self._write_task_progress(task_name, stats)
            if self.world_size == 1:
                self._write_summary_json(task_results)
            self._log_rank(
                f'Task {task_name} completed: {successes}/{episodes} successes'
            )
            self._log_rank(f"Time taken: {stats['duration']:.2f} seconds")

    def _wait_for_task_completion(self) -> None:
        """Wait without a deadline until every rank has finished its tasks."""
        self._completion_store.set(str(self.rank), 'done')
        keys = [str(rank) for rank in range(self.world_size)]
        # Poll readiness instead of starting a collective whose communication
        # timeout would also limit how long other ranks can keep evaluating.
        while not self._completion_store.check(keys):
            time.sleep(1.0)

    def run(self) -> str:
        """Run local tasks and write one merged summary on rank zero."""
        previous = self._load_task_progress()
        task_results = {task: previous[task] for task in self.local_task_list}
        error = None
        try:
            self._run_local_tasks(task_results)
        except Exception as exc:
            if self.world_size == 1:
                raise
            error = f'rank {self.rank}: {type(exc).__name__}: {exc}'

        errors = []
        if self.world_size > 1:
            self._wait_for_task_completion()
            results = [None] * self.world_size
            dist.all_gather_object(results, (task_results, error))
            merged = {}
            for rank, (stats, rank_error) in enumerate(results):
                expected = self.task_list[rank::self.world_size]
                if set(stats) != set(expected):
                    raise RuntimeError(
                        f'Unexpected task results from rank {rank}')
                merged.update(stats)
                if rank_error:
                    errors.append(rank_error)
            task_results = {task: merged[task] for task in self.task_list}

        summary_json = str(self.summary_path)
        if self.rank == 0:
            summary_json = self._write_robotwin_summary_artifacts(task_results)
        if errors:
            raise RuntimeError('RoboTwin evaluation failed: ' +
                               '; '.join(errors))
        if self.rank == 0:
            totals = self._summarize_completed_tasks(task_results)
            rate = totals['success_rate']
            display_rate = 'N/A' if rate is None else f'{rate:.1f}%'
            self._log_rank(f"# episodes completed: {totals['total_trials']}")
            self._log_rank(
                f"# successes: {totals['total_successes']} ({display_rate})")
            self._log_rank(
                f'[*] Wrote RoboTwin summary artifacts to {self.run_dir}')
        return summary_json

    def cleanup(self) -> None:
        """Release the cached model before the next evaluation."""
        try:
            if self._policy_module is not None:
                self._policy_module.close_model()
        finally:
            self._policy_module = None
            self._robotwin_eval_module = None
            if self._warp_cache is not None:
                import warp
                warp.config.kernel_cache_dir = self._previous_warp_config
                if self._previous_warp_cache is None:
                    os.environ.pop('WARP_CACHE_PATH', None)
                else:
                    os.environ['WARP_CACHE_PATH'] = self._previous_warp_cache
                self._warp_cache.cleanup()
                self._warp_cache = None
