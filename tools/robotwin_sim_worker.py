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
"""Line-delimited JSON bridge to a RoboTwin 2.0 task.

This worker imports the RoboTwin simulator (``sapien``/``mplib``) and owns the
physics process, so a separately installed policy environment can drive one
task without importing the simulator itself. Run it with the interpreter of a
RoboTwin-capable environment:

    python tools/robotwin_sim_worker.py --root /path/to/RoboTwin

Protocol (one JSON request per line on stdin, one JSON response per line on
stdout; every other byte is forwarded to stderr):

* ``{"op": "reset", "task": "adjust_bottle", "config": "demo_clean",
   "seed": 100000, "episode": 0, "instruction": "...", "expert_check": true}``
* ``{"op": "step", "action": [14 floats]}``
* ``{"op": "close"}``

Responses carry the camera rasters as zlib-compressed base64 ``uint8`` arrays
plus the raw 14-D joint vector, the selected instruction, the task step counter
and its limit, and the task success flag. The worker never fits statistics or
chooses actions; it only executes the direct task API (``setup_demo``,
``get_obs``, ``take_action``, ``check_success``, ``close_env``).

Instruction selection is ``seeded`` by default (a deterministic
``np.random.default_rng(seed)`` pick over a single generated variant, the native
smoke behavior) or ``official`` (the released entrypoint's global
``np.random.choice`` over an ``instruction_pool_size`` variant pool).
"""

import argparse
import base64
import contextlib
import importlib
import json
import os
import sys
import traceback
import zlib

import numpy as np
import yaml


class RetryableSetupError(RuntimeError):
    """The requested seed cannot be used; the caller should try the next one."""


def _read_mapping(path):
    with open(path, 'r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)


@contextlib.contextmanager
def _stdout_to_stderr():
    """Keep the simulator's own prints off the protocol channel."""
    saved = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = saved


@contextlib.contextmanager
def _ray_tracing_denoiser_override(name):
    """Override RoboTwin's hardcoded OIDN choice in this process only.

    Keep the ray-tracing shader, sample count, and path depth unchanged.
    The default uses OIDN; ``none`` remains an explicit fallback for installs
    whose bundled OIDN cannot create a CUDA device.
    """
    if name not in ('none', 'oidn', 'optix'):
        raise ValueError(f'Unsupported ray-tracing denoiser: {name!r}')
    import sapien.render as render

    original = render.set_ray_tracing_denoiser

    def set_denoiser(_requested):
        original(name)

    render.set_ray_tracing_denoiser = set_denoiser
    try:
        yield
    finally:
        render.set_ray_tracing_denoiser = original


def _encode_raster(image):
    array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f'Expected an HWC uint8 camera raster, got {array.shape}')
    payload = zlib.compress(array.tobytes(), level=1)
    return {
        'shape': list(array.shape),
        'data': base64.b64encode(payload).decode('ascii'),
    }


class RoboTwinTask:
    """Owns one RoboTwin task instance and the direct task API."""

    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.task_name = None
        self.task = None
        self.arguments = None
        self.config_name = None
        self.task_config_dir = os.path.join(self.root, 'task_config')
        self.episode_seed = None
        self.instruction = None
        self.instructions_used = []
        self._closed_scenes = 0

    # ------------------------------------------------------------------ setup
    def _runtime_arguments(self, config_name, overrides):
        config_path = os.path.join(self.task_config_dir, config_name + '.yml')
        if not os.path.isfile(config_path):
            raise FileNotFoundError(config_path)
        arguments = _read_mapping(config_path)
        arguments.update(overrides or {})
        frequency = int(arguments.get('clear_cache_freq', 5))
        if frequency < 1:
            raise ValueError('clear_cache_freq must be positive')
        arguments['clear_cache_freq'] = frequency
        conflict = [
            name for name in ('seed', 'now_ep_num', 'is_test')
            if name in arguments
        ]
        if conflict:
            raise ValueError(
                'Episode seed/index/test mode are owned by the evaluation '
                f'runner, not the task configuration: {conflict}')
        arguments.update(
            task_name=self.task_name,
            task_config=config_name,
            eval_mode=True,
            eval_video_save_dir=None,
            collect_data=False,
        )
        camera_path = os.path.join(self.task_config_dir, '_camera_config.yml')
        cameras = _read_mapping(camera_path)
        head = cameras[arguments['camera']['head_camera_type']]
        arguments['head_camera_h'] = int(head['h'])
        arguments['head_camera_w'] = int(head['w'])
        embodiment_path = os.path.join(self.task_config_dir,
                                       '_embodiment_config.yml')
        embodiments = _read_mapping(embodiment_path)
        selected = arguments.get('embodiment')
        if not isinstance(selected, list) or len(selected) not in (1, 3):
            raise ValueError(
                'RoboTwin embodiment must name one robot or left/right robots '
                'plus a separation')
        names = [
            selected[0], selected[0] if len(selected) == 1 else selected[1]
        ]
        for side, name, joint_group in zip(('left', 'right'), names, (0, 1)):
            location = embodiments[name].get('file_path')
            if not location:
                raise ValueError('The selected embodiment needs a local file_path')
            directory = (location if os.path.isabs(location) else
                         os.path.join(self.root, location))
            configuration = _read_mapping(os.path.join(directory, 'config.yml'))
            if len(configuration['arm_joints_name'][joint_group]) != 6:
                raise ValueError(
                    'This bridge requires six arm joints and one gripper per side')
            arguments[side + '_robot_file'] = directory
            arguments[side + '_embodiment_config'] = configuration
        arguments['dual_arm_embodied'] = len(selected) == 1
        if len(selected) == 3:
            arguments['embodiment_dis'] = selected[2]
        data_type = arguments.get('data_type', {})
        if not data_type.get('rgb') or not data_type.get('qpos'):
            raise ValueError(
                'RoboTwin task configuration must enable RGB and qpos observations')
        return arguments, '+'.join(map(str, dict.fromkeys(names)))

    def _instruction(self, request, episode_info):
        explicit = request.get('instruction')
        if explicit:
            return str(explicit)
        selection = str(request.get('instruction_selection', 'seeded'))
        if selection not in ('seeded', 'official'):
            raise ValueError(
                "instruction_selection must be 'seeded' or 'official', got "
                f'{selection!r}')
        pool_size = int(request.get('instruction_pool_size', 1) or 1)
        if pool_size < 1:
            raise ValueError('instruction_pool_size must be positive')
        # The released entrypoint sizes the generator pool with ``test_num``;
        # the native seeded default only needs a single variant.
        count = pool_size if selection == 'official' else 1
        generator = request.get('instruction_provider')
        if generator:
            module_name, _, attribute = generator.partition(':')
            provider = getattr(importlib.import_module(module_name), attribute)
            variants = provider(self.task_name, [episode_info], count)[0]
        else:
            sys.path.append(os.path.join(self.root, 'description', 'utils'))
            from generate_episode_instructions import \
                generate_episode_descriptions
            results = generate_episode_descriptions(
                self.task_name, [episode_info], count)
            variants = results[0][request.get('instruction_type', 'unseen')]
        if not isinstance(variants, (list, tuple)) or not variants:
            raise ValueError('Instruction provider returned no variants')
        if selection == 'official':
            # Released RoboTwin picks with a global ``np.random.choice`` over
            # the generated variants. RoboTwin's scene setup seeds the global
            # numpy RNG per episode; this bridge creates no separate
            # instruction RNG of its own.
            return str(np.random.choice(variants))
        rng = np.random.default_rng(int(request['seed']))
        return str(variants[int(rng.integers(len(variants)))])

    def reset(self, request):
        # A reset replaces the previous rollout; release it first so a failed
        # attempt cannot leave a scene alive behind the new one.
        self._discard_task()
        self.task_name = str(request['task'])
        self.config_name = str(request.get('config', 'demo_clean'))
        self.arguments, embodiment = self._runtime_arguments(
            self.config_name, request.get('setup_overrides'))
        module = importlib.import_module(f'envs.{self.task_name}')
        self.task = getattr(module, self.task_name)()
        seed = int(request['seed'])
        episode_index = int(request.get('episode', 0))
        self.episode_seed = seed
        try:
            self.task.setup_demo(
                now_ep_num=episode_index,
                seed=seed,
                is_test=True,
                **self.arguments)
        except Exception as error:  # noqa: BLE001 - classify unusable seeds
            self._discard_task()
            raise RetryableSetupError(
                f'RoboTwin scene setup failed for seed {seed}: {error!r}')
        episode_info = {}
        if request.get('expert_check', True):
            try:
                episode_record = self.task.play_once()
            except Exception as error:  # noqa: BLE001 - classify unusable seeds
                self._discard_task()
                raise RetryableSetupError(
                    f'Expert rollout raised for RoboTwin seed {seed}: '
                    f'{error!r}')
            if not self.task.plan_success or not self.task.check_success():
                self._discard_task()
                raise RetryableSetupError(
                    f'Expert validity check failed for RoboTwin seed {seed}')
            episode_info = episode_record.get('info', {})
            self._close_task_env()
            try:
                self.task.setup_demo(
                    now_ep_num=episode_index,
                    seed=seed,
                    is_test=True,
                    **self.arguments)
            except Exception as error:  # noqa: BLE001
                self._discard_task()
                raise RetryableSetupError(
                    'RoboTwin scene setup failed after the expert check for '
                    f'seed {seed}: {error!r}')
        else:
            episode_info = getattr(self.task, 'info', {}).get('info', {})
        self.instruction = self._instruction(request, episode_info)
        self.instructions_used.append(self.instruction)
        if hasattr(self.task, 'set_instruction'):
            self.task.set_instruction(instruction=self.instruction)
        return self._observation(embodiment, success=False)

    def _close_task_env(self):
        self._closed_scenes += 1
        frequency = int((self.arguments or {}).get('clear_cache_freq', 5))
        self.task.close_env(clear_cache=(self._closed_scenes % frequency == 0))

    def _discard_task(self):
        if self.task is not None:
            try:
                self._close_task_env()
            except Exception:  # noqa: BLE001 - teardown must not mask the cause
                pass
        self.task = None

    # ------------------------------------------------------------ observation
    def _observation(self, embodiment=None, success=None):
        raw = self.task.get_obs()
        joint = np.asarray(raw['joint_action']['vector'], dtype=np.float32)
        if joint.shape != (14,) or not np.isfinite(joint).all():
            raise ValueError(
                'RoboTwin joint_action.vector must hold fourteen finite '
                'coordinates')
        images = {}
        for name in ('head_camera', 'left_camera', 'right_camera'):
            camera = raw['observation'].get(name)
            if camera is None:
                raise ValueError(f'RoboTwin observation is missing {name}')
            images[name] = _encode_raster(camera['rgb'])
        if success is None:
            success = bool(self.task.check_success())
        step = int(getattr(self.task, 'take_action_cnt', 0))
        limit = int(getattr(self.task, 'step_lim', 0) or 0)
        return {
            'instruction': self.instruction,
            'embodiment': embodiment,
            'seed': self.episode_seed,
            'images': images,
            'state': joint.tolist(),
            'success': bool(success),
            'step': step,
            'step_limit': limit,
        }

    def step(self, request):
        if self.task is None:
            raise RuntimeError('No active RoboTwin episode; call reset first')
        action = np.asarray(request['action'], dtype=np.float32).reshape(-1)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError('RoboTwin actions must hold fourteen finite values')
        self.task.take_action(action, action_type='qpos')
        success = bool(self.task.check_success())
        step = int(getattr(self.task, 'take_action_cnt', 0))
        limit = int(getattr(self.task, 'step_lim', 0) or 0)
        done = success or (limit > 0 and step >= limit)
        return self._observation(success=success) | {'done': bool(done)}

    def close(self):
        self._discard_task()


def serve(stream_in, stream_out, root):
    """Serve the JSON protocol until stdin closes; always release the scene."""
    worker = RoboTwinTask(root)
    try:
        for line in stream_in:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as error:
                print(json.dumps({'ok': False,
                                  'error': f'invalid request: {error}'}),
                      file=stream_out, flush=True)
                continue
            op = request.get('op')
            try:
                with _stdout_to_stderr():
                    if op == 'reset':
                        response = worker.reset(request)
                    elif op == 'step':
                        response = worker.step(request)
                    elif op == 'close':
                        worker.close()
                        response = {'closed': True}
                    else:
                        raise ValueError(f'Unsupported op: {op!r}')
                print(json.dumps({'ok': True, **response}, ensure_ascii=True),
                      file=stream_out, flush=True)
                if op == 'close':
                    break
            except Exception as error:  # noqa: BLE001 - report every failure
                if not isinstance(error, RetryableSetupError):
                    traceback.print_exc(file=sys.stderr)
                print(json.dumps({
                    'ok': False,
                    'error': repr(error),
                    'retryable': isinstance(error, RetryableSetupError),
                }, ensure_ascii=True),
                      file=stream_out, flush=True)
    finally:
        # stdin EOF or an unexpected protocol exit must not leave a scene
        # running in the simulator process.
        worker.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True,
                        help='RoboTwin 2.0 checkout root')
    parser.add_argument(
        '--ray-tracing-denoiser', choices=('none', 'oidn', 'optix'),
        default=os.environ.get('PATCHWAM_RENDER_DENOISER', 'oidn'),
        help='Override RoboTwin denoising; default oidn uses the installed OIDN library.')
    args = parser.parse_args()

    os.chdir(args.root)
    if args.root not in sys.path:
        sys.path.insert(0, args.root)

    print(f'RoboTwin renderer: ray-tracing denoiser={args.ray_tracing_denoiser}',
          file=sys.stderr, flush=True)
    with _ray_tracing_denoiser_override(args.ray_tracing_denoiser):
        serve(sys.stdin, sys.stdout, args.root)


if __name__ == '__main__':
    main()
