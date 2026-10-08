# PatchWAM RoboTwin 2.0 evaluation

Run the released PatchWAM RoboTwin 2.0 C2R checkpoint
(`QZWang/patchwam-robotwin-c2r-step140000`, step 140000) with the native
FluxVLA model. The policy runs in the FluxVLA environment; the simulator runs
in a separate RoboTwin environment behind `tools/robotwin_sim_worker.py`.

## Checkpoint contract

`step_140000.pt` (7,751,411,897 bytes, SHA256
`22f556d02903a06f23f21a8a28ef91ff92611de0bad4efbf8827c01bd3701fd9`) is a
`torch.save` dict whose top-level keys are `mot`, `proprio_encoder`, `step`,
and `torch_dtype`; `mot` and `proprio_encoder` are nested state dicts:

| Key | Contents |
| --- | --- |
| `mot.mixtures.video.transformer.*` | fine-tuned FLUX.2 Klein base 4B transformer (149 tensors) |
| `mot.mixtures.video.{double,single}_blocks.*` | module aliases of the same blocks (shared storage, no extra parameters) |
| `proprio_encoder.{weight,bias}` | 14 -> 7680 proprio projection |
| `step`, `torch_dtype` | `140000`, `torch.bfloat16` |

The checkpoint carries no autoencoder or text-encoder weights and no
normalization statistics. It is loaded by `PatchWAMVLA.from_pretrained` through
the `action_patch` layout:

* `mot.mixtures.video.transformer.*` -> `vla_head.denoiser.transformer.*`
* `proprio_encoder.*` -> `vla_head.state_projection.*`

`PatchWAMVLA.load_patchwam_state_dict` validates every trainable tensor before
mutating any parameter and rejects unexpected keys, missing keys, shape
mismatches, non-finite values, block aliases that disagree with their
transformer copy, and unexpected tensor entries anywhere in the payload; a
`mot` section without `proprio_encoder` is rejected rather than partially
loaded.

## Local assets

All assets already exist locally; nothing is downloaded.

| Asset | Path |
| --- | --- |
| checkpoint | `/data/Pretrained_models/patchwam-robotwin/step_140000.pt` |
| FLUX.2 transformer | `<ImageWAM>/checkpoints/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors` |
| FLUX.2 autoencoder | `<ImageWAM>/checkpoints/flux2/FLUX.2-dev/ae.safetensors` |
| Qwen3-4B text encoder | `<ImageWAM>/checkpoints/flux2/FLUX.2-klein-base-4B/text_encoder` |
| Qwen3-4B tokenizer | `<ImageWAM>/checkpoints/flux2/FLUX.2-klein-base-4B/tokenizer` |
| FLUX.2 source (pinned `50fe516`) | `<ImageWAM>/third_party/flux2` |
| normalization statistics | `RoboDojo/XPolicyLab/policy/PatchWAM/assets/dataset_stats.json` |
| RoboTwin 2.0 checkout | `<ImageWAM>/third_party/RoboTwin` |
| RoboTwin interpreter | `<ImageWAM>/.venv_rb2/bin/python` |

with `<ImageWAM>` = `/home/chuong/workspace/wam/ImageWAM` by default. Every path
is overridable through the `PATCHWAM_*` environment variables listed in
`scripts/eval_patchwam_robotwin.sh`.

The autoencoder must be the FLUX.2 `ae.safetensors` (the format the pinned
`flux2.autoencoder.AutoEncoder` expects); the Diffusers-style
`FLUX.2-klein-base-4B/vae/diffusion_pytorch_model.safetensors` uses different
parameter names and will not load strictly.

## Normalization and action contract

The released recipe uses z-score normalization from `dataset_stats.json`
(`use_stepwise_action_norm: false`):

```
scale  = 1 / (std + 1e-8)
offset = -mean / (std + 1e-8)
normalize(x)   = clamp(x * scale + offset, -5, 5)
denormalize(y) = (y - offset) / scale
```

The bridge and runner use the `global_mean`/`global_std` entries for both state
and action. Actions are 14-D absolute joint-position commands in
left-arm(6)+left-gripper(1)+right-arm(6)+right-gripper(1) order and are passed
to `RoboTwin.take_action(..., action_type='qpos')` unchanged. Observations use
the compact three-camera mosaic: the head camera resized to 256x192 on top,
the left and right wrist cameras resized to 128x96 side by side below
(288x256 total), scaled to `[-1, 1]`. Instructions are sent through the
training prompt template
`"A video recorded from a robot's point of view executing the following instruction: {task}"`.

## Commands

One-task, one-episode integration smoke (native policy + real simulator):

```bash
cd /home/chuong/workspace/FluxVLA
CUDA_VISIBLE_DEVICES=0 /home/chuong/miniconda3/envs/fluxvla/bin/python \
    scripts/eval.py --config configs/patchwam/patchwam_robotwin_c2r_eval.py
```

or

```bash
PATCHWAM_EVAL_PYTHON=/home/chuong/miniconda3/envs/fluxvla/bin/python \
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_patchwam_robotwin.sh
```

Results are written to
`runs/evaluation/patchwam_robotwin/<task>_<timestamp>/results.json` (checkpoint,
normalization signature, protocol, seed search, per-episode steps/success).

More episodes, a different task, or the randomized condition:

```bash
PATCHWAM_EVAL_PYTHON=/home/chuong/miniconda3/envs/fluxvla/bin/python \
CUDA_VISIBLE_DEVICES=0 PATCHWAM_EPISODES=5 \
bash scripts/eval_patchwam_robotwin.sh \
    --cfg-options eval.task_name=click_alarmclock \
                  eval.task_config=demo_randomized \
                  eval.output_dir=runs/evaluation/patchwam_robotwin_random
```

This is an integration smoke, not the reported benchmark. The published
protocol is 50 tasks x 2 conditions x 100 episodes with the released seen/unseen
instruction splits and seed manifests; shard those runs across GPUs and
aggregate only matching protocols. (The checkpoint distribution's own
`metrics.json` and `results/` directory record that protocol and a 79.14%
average; those are publisher-reported numbers and were not reproduced here.)

## What has been verified

* The released checkpoint loads into `PatchWAMVLA` as `action_patch`
  (151 tensors, step 140000) with strict shape/name validation.
* Real GPU inference: the model builds and predicts `[1, 16, 14]` action chunks
  on an RTX 5090 in about 1 s per chunk, 16.2 GB peak.
* Simulator closed loop: RoboTwin 2.0 `adjust_bottle` / `demo_clean`, one
  episode, expert seed validation, z-score state normalization, 16-step
  replan, 10 denoising steps -> task completed in 112 of 400 steps with 7
  predictions.
* Numerical parity with the released implementation
  (`RoboDojo/XPolicyLab/policy/PatchWAM/PatchWAM/runtime_policy.py`, the
  adapter's bundled upstream checkout): with identical observation, normalized
  state, prompt, and initial noise the native `predict_action` chunk matches the
  reference to max |diff| 8.0e-3 / mean 5.3e-4 (bf16 noise), and the per-step
  denoiser velocities match to at most 4.7e-2.

## Limits

* The RoboTwin simulator environment still lives outside this repository
  (`.venv_rb2`). The runner only reads it; no shared environment is modified.
* Seeds use one monotonic cursor per run, starting at `100000 * (1 + seed)`.
  Every attempt, accepted or rejected, advances the cursor by one, so no two
  episodes can reuse a candidate seed; the used seed and attempt count are
  recorded per episode. The search is bounded by `eval.max_seed_attempts`.
* The simulator bridge waits `eval.response_timeout` seconds (default 1800) for
  each worker response and terminates the worker on timeout instead of
  blocking forever.
* The 79.14% average recorded in the checkpoint distribution is not reproduced
  here; a formal score additionally needs the released task list, instruction
  splits, and episode counts.
