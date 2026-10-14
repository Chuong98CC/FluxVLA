# Engine and platform contracts

The [custom-engine tutorial](https://fluxvla.limxdynamics.com/md_source/tutorials/private_engine.html)
distinguishes training execution, inference control, and hardware communication.
Extend the part that changes. Training and robot inference extensions are
independent; an additional model loss does not inherently need either one.

## Shared wiring and configuration

`RUNNERS` and `OPERATORS` are defined in `fluxvla/engines/utils/root.py`.
Implement under `fluxvla/engines/runners/` or `operators/`, register the class,
and add its discovery import to the matching `__init__.py`. Check import-time
optional dependency handling in that package.

| Purpose | Configuration selection | Caller to inspect |
| --- | --- | --- |
| Training | `runner` | `scripts/train.py` |
| Robot inference | `inference`; nested `operator` | `scripts/inference.py`, concrete inference runner |
| Environment evaluation | `eval` or `eval.runner`, depending on checkout | `scripts/eval.py` |

`build_runner_from_cfg` forwards constructor arguments. Entry scripts inject
context such as full configuration and checkpoint paths; inspect those
assignments before writing a standalone construction example. Do not put a
runner configuration in `model` or an inference-only model swap in `inference`.

## Training runner

Read `base_train_runner.py` and the closest concrete runner (`fsdp_train_runner.py`
or `ddp_train_runner.py`). Define setup, wrapping, optimizer/scheduler ownership,
gradient synchronization, checkpoint save/resume, and cleanup against those
actual callers.

The relevant hooks include `run_setup(n_train_examples)`, `save_checkpoint`,
`clip_grad_norm`, `_load_model_state`, and `_load_optimizer_state`. Read their
signatures and return semantics; preserve the base loop when compatible.
`_training_step` consumes `output['loss']` and performs backward/accumulation;
returning only diagnostics or detached loss will not implement a trainable method.

For a different execution backend, define which layer performs backward,
gradient accumulation, clipping, stepping, and scheduling. Prevent double steps
and double scaling. Validate model/optimizer/scheduler state, epoch/step counters,
mixed precision, rank-aware saving, resume behavior, and error/cleanup paths.
Reusing an existing runner is preferable when its execution semantics fit.

## Robot inference runner and operator

Read `base_inference_runner.py`, a similar robot runner, `operators/base_operator.py`,
and the concrete operator being adapted. The inference customization hooks
include `get_ros_observation`, `update_observation_window`, and
`_execute_actions(actions, rate)`. The base supplies model prediction and action
postprocessing; confirm its setup, observation, checkpoint/statistics, history,
and cleanup behavior before overriding loops.

Keep communication and synchronization in the operator, model preprocessing and
action dispatch in the runner. Define actual observation keys and action methods
from the consumers; the names `servoj` or `movegrip` in an example are not a
universal protocol. Reuse `BaseOperator` synchronization/trajectory utilities
when appropriate to the selected transport.

Write a platform schema connecting:

- Camera names and order in observations, `camera_names`, dataset `img_keys`,
  prompt image count, preprocessing statistics, masks, and backbone inputs.
- State/action layout, embodiment IDs, physical and padded dimensions, units,
  frames, normalization statistics, denormalization, and command splitting.
- Sampling/publication rate, action horizon, executed chunk length, reset/history
  behavior, timestamp alignment, and missing-observation handling.

Changing three cameras to two or fourteen controls to seven requires compatible
model/checkpoint behavior and matching dataset/transform metadata. Changing a
single count or slicing an incompatible checkpoint is not a complete adaptation.

Use fake observations and a recording operator for automated verification.
Construction/import checks should not publish robot commands. Treat real
hardware execution as a separate action requiring task authorization.

## Simulation evaluation

Read `base_eval_runner.py`, the closest environment runner, and `scripts/eval.py`.
The current base is a helper mixin; it does not implement every environment's
episode loop. Provide the selected runner's constructor plus `run_setup` and
`run`, registration, import, and compatible `eval` configuration.

Define environment creation, deterministic seeds, resets, observation conversion,
action denormalization/coordinates, horizons, success and termination conditions,
metrics/output paths, and resource cleanup. Reuse compatible checkpoint/statistics
helpers. Test with a small fake environment before a real simulator run, and
report the distinction between harness correctness and measured task success.
