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
| RoboTwin interpreter | `/home/chuong/miniconda3/envs/Robotwin/bin/python` |

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

## Official benchmark launcher

`my_scripts/eval_patchwam.sh` now runs the released protocol by default: the 50
task names in `configs/patchwam/patchwam_robotwin_official_benchmark_tasks.json`
(recovered from the published `results/c2rdr4_140k_100ep_full.json`, names only)
x `demo_clean`/`demo_randomized` x 100 episodes = 10,000 policy episodes, seed
0, action horizon 16, replan 16, 10 denoising steps, unseen instructions and
expert seed validation. Cases run sequentially on one GPU; each writes its own
directory and a `suite_metadata.json` aggregate under
`runs/evaluation/patchwam_robotwin_official/<timestamp>/`.

Plan the suite without loading the model or simulator:

```bash
bash my_scripts/eval_patchwam.sh --list      # 100 case lines
bash my_scripts/eval_patchwam.sh --dry-run   # full plan + per-case commands
```

Subsets and overrides:

```bash
bash my_scripts/eval_patchwam.sh --smoke                 # 1 task, clean, 1 episode
bash my_scripts/eval_patchwam.sh --task adjust_bottle --condition clean
bash my_scripts/eval_patchwam.sh --episodes 4 --task adjust_bottle
bash my_scripts/eval_patchwam.sh --ckpt-path /path/to/step_140000.pt
bash my_scripts/eval_patchwam.sh --cfg-options eval.num_inference_steps=8
```

Anything other than the untouched default is recorded as a custom protocol in
`suite_metadata.json`. `--cfg-options eval.task_name=...`/`eval.task_config=...`
are interpreted as suite filters instead of being forwarded to every case, and
`eval.output_dir`/`eval.run_id` are rejected because the launcher owns them.
`configs/patchwam/patchwam_robotwin_official_benchmark.py` opts into the
released `np.random.choice` instruction selection; the single-task smoke config
`configs/patchwam/patchwam_robotwin_c2r_eval.py` keeps the deterministic seeded
default and one episode.

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

* The RoboTwin simulator runs in the dedicated `Robotwin` Conda environment.
  The previous ImageWAM `.venv_rb2` remains available and was preserved.
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

## Renderer compatibility on this machine

The simulator uses OIDN denoising by default (`PATCHWAM_RENDER_DENOISER=oidn`).
Following [RoboTwin issue #52](https://github.com/RoboTwin-Platform/RoboTwin/issues/52),
the SAPIEN OIDN libraries were upgraded from 2.0.1 to the official RenderKit
OIDN 2.3.3 release and copied into the dedicated simulator environment. The
core, CUDA device and public API libraries now live under
`~/miniconda3/envs/Robotwin/lib/python3.11/site-packages/sapien/oidn_library/`;
`_oidn_tricks.py` now preloads version 2.3.3. Original files and an installation
manifest are saved under `docs/agent-work/patchwam-oidn-upgrade/`.
This is an environment-local fix; reinstalling SAPIEN may replace the loader.

Ray tracing, denoising, sample count and camera geometry use RoboTwin's settings.
`PATCHWAM_RENDER_DENOISER=none` remains an explicit fallback, but disabling
denoising changes rendering and should be recorded when comparing results.
Restart evaluation processes after upgrading the libraries.

The local evaluation config sets `eval.setup_overrides.clear_cache_freq=1`,
following RoboTwin's common-issue recommendation. The bridge applies the setting
on scene teardown (including expert validation and rejected seeds); it does not
modify the shared task YAML. This is separate from OIDN device compatibility.
An unstable initial scene is an expected seed rejection: the runner warns and
tries the next seed. Retryable rejections omit Python tracebacks; other failures
still include them.

## Dedicated evaluation environment

### Portable installer

`scripts/install_robotwin_env.sh` builds the simulator environment from scratch
on another Linux x86_64 NVIDIA machine instead of copying a local virtualenv.
Every component comes from a pinned source:

| Component | Pin |
| --- | --- |
| RoboTwin 2.0 | upstream `RoboTwin-Platform/RoboTwin` revision `bf44be51cf5717a5595ce59447f2cf5263d2aa95`, the revision `third_party/RoboTwin` is vendored from |
| cuRobo | `NVlabs/curobo` tag `v0.7.8`, built from source against the selected Torch |
| Open Image Denoise | official RenderKit `oidn-2.3.3.x86_64.linux.tar.gz` (SHA256 `3c385230d9e6f63527ba72f2229594dbac5051674219d72e0044b5d0b841796f`) |
| pip closure | `scripts/requirements-robotwin-sim.txt` |

```bash
bash scripts/install_robotwin_env.sh                 # default: --env-name Robotwin, --profile auto
bash scripts/install_robotwin_env.sh --dry-run       # no writes, downloads, builds or Conda needed
bash scripts/install_robotwin_env.sh --robotwin-root /data/RoboTwin
bash scripts/install_robotwin_env.sh --prefix /data/envs/robotwin --profile cu128
bash scripts/install_robotwin_env.sh --with-assets   # optional upstream asset downloader
```

`--profile cu128` (the `auto` choice when the selected CUDA toolkit reports
CUDA >= 12.8) installs Python 3.11.15 with torch 2.7.1+cu128 and matches the
verified environment below. `--profile cu121` uses the upstream RoboTwin Torch
pin for pre-Blackwell GPUs, torch 2.4.1+cu121, also on Python 3.11.15: upstream
installs Python 3.10, but the pinned closure (scipy 1.17.1, zarr 3.1.5,
matplotlib 3.11.2, networkx 3.6.1, scikit-image 0.26.0) declares
`Requires-Python >= 3.11`, and torch 2.4.1+cu121 ships cp311 wheels. The
installer verifies each pin's installed version and `Requires-Python` against
the selected interpreter after installation, so a profile that cannot satisfy
the closure fails instead of installing a broken mix.

CUDA extension builds require a CUDA 12.x toolkit; a newer driver alone is
not sufficient, and a CUDA 13.x toolkit is rejected before installation.

The selected CUDA toolkit is authoritative: `<toolkit>/bin/nvcc` decides the
`auto` profile and builds the cuRobo extensions even when a different nvcc is
first on `PATH`, and the toolkit version is checked against the profile before
any environment creation or download. `git`, `curl`, `tar`, a C++ compiler,
`cmake` and `ninja` are preflighted with actionable errors. The installer
creates or reuses the environment under an explicit `-p` prefix, normalizes
every path to absolute before the cwd-sensitive build or exports, drives every
follow-up step with that prefix's interpreter, and never removes an environment
or resets a checkout: a user-supplied `--robotwin-root` is used as-is, a dirty
managed checkout is reported rather than overwritten, and reusing an
environment with the wrong Python version fails before any pip or patch step.

Three environment-local fixes mirror the verified simulator environment:

* `setuptools==69.5.1`, because SAPIEN 3.0.0b1 imports `pkg_resources`;
* RoboTwin's own SAPIEN `urdf_loader.py` and mplib `planner.py` source fixes,
  applied with a one-time `<file>.patchwam-backup` and an atomic replace
  (`scripts/patch_robotwin_runtime.py`);
* Open Image Denoise 2.3.3 copied into `sapien/oidn_library` with the
  `_oidn_tricks.py` preload stub rewritten atomically
  (`scripts/patch_sapien_oidn.py`); the previous 2.0.1 libraries stay in place.

The installer then verifies the interpreter, the pinned package imports, the
cuRobo CUDA extensions, a CUDA Torch tensor, and a fresh OIDN CUDA denoising
device (`commit_error=0`) through the documented Open Image Denoise C API. It
also reports `site-packages` hooks that point outside the environment, so a
copied environment cannot silently reappear.

Finally it exports the two variables the evaluation config reads, both in a
sourceable file and through Conda activation:

```bash
export PATCHWAM_SIM_PYTHON=<prefix>/bin/python
export PATCHWAM_ROBOTWIN_ROOT=<RoboTwin checkout>
```

`<prefix>/etc/conda/activate.d/patchwam-robotwin.sh` sets them on activation and
`<prefix>/patchwam_robotwin_env.sh` can be sourced directly; both files are
written with shell-quoted values, so paths containing spaces, quotes or `$`
source safely. The policy
environment, checkpoint and model assets stay separate: FluxVLA runs in its own
environment (`scripts/install_env.sh`) with `PATCHWAM_CHECKPOINT` and
`PATCHWAM_ASSETS_ROOT`, while only the simulator uses this environment. The
installer never installs drivers, never calls `apt`/`sudo`, and never launches a
benchmark episode; a fresh GPU extension build on another machine is exercised
by that machine's first run, not by this repository.

`--with-assets` runs the checkout's own `scripts/_download_assets.sh` (or
`script/_download_assets.sh`) and then RoboTwin's non-interactive
`update_embodiment_config_path.py` with stdin closed, so the embodiment
templates point at the checkout's asset directory. If the downloader is missing
or the assets and configuration cannot be prepared, the installer exits
non-zero and reports that the simulator environment itself is installed but the
requested assets are incomplete; assets are never silently skipped.

### Local verified environment

`Robotwin` is a Conda environment at `/home/chuong/miniconda3/envs/Robotwin`,
with Python 3.11.15. Its simulator dependency closure was copied from the working
ImageWAM `.venv_rb2` without changing the source environment. Patched SAPIEN,
OIDN 2.3.3 and cuRobo's compiled extensions/configuration are copied into the
new environment; cuRobo is a local package rather than an editable ImageWAM
source hook. Training packages such as ImageWAM, DeepSpeed and Diffusers are
excluded. Existing RoboTwin assets and model weights are reused in place.

The policy continues to use the `fluxvla` interpreter; the simulator uses
`Robotwin` automatically through the evaluation config. Launch the same command:

```bash
cd /home/chuong/workspace/FluxVLA
bash my_scripts/eval_patchwam.sh
```

You can activate the simulator environment for diagnostics with
`conda activate Robotwin`. The old simulator can still be selected explicitly:

```bash
PATCHWAM_SIM_PYTHON=/home/chuong/workspace/wam/ImageWAM/.venv_rb2/bin/python \
bash my_scripts/eval_patchwam.sh
```

Dependency inventory and setup evidence are under
`docs/agent-work/robotwin-conda/`. No full benchmark is run during setup.
