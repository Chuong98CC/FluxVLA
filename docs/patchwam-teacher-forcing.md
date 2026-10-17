# PatchWAM teacher-feature forcing

`TeacherForcingPatchWAMVLA` adds a frozen spatial encoder to PatchWAM without
changing the 128-channel VAE flow states or the pretrained image/action
projection. The method has three independent switches:

| Switch | Effect |
| --- | --- |
| `current_conditioning` | Adds `M(normalize(f_current))` to the current raster hidden states. |
| `future_prediction` | Predicts clean future teacher features from the final future hidden states with `J`. |
| `future_feedback` | Feeds detached predicted future features through `G` between denoiser evaluations. Requires `future_prediction`. |

M and G are separate bias-free linear maps into the denoiser hidden width.
Their default initialization is exactly zero; `scaled_xavier_uniform` is the
opt-in alternative. J defaults to the source three-linear SiLU REPA MLP with
hidden width 2048. `linear` and `mlp2` are available as ablations. Feature
inputs use non-affine LayerNorm at fusion; cosine loss applies L2 normalization
only inside the objective.

## Recipe

The derived RobotWin recipe is
`configs/patchwam/patchwam_teacher_forcing_radio_robotwin.py`. It inherits the
matched data, augmentation, normalization, 288x256 compact mosaic, horizon 16,
and 10-step solver from `configs/patchwam/robotwin/patchwam_robotwin.py`.

Set the local assets:

```bash
export PATCHWAM_PRETRAINED_CHECKPOINT=checkpoints/Pretrained_models/patchwam-robotwin/step_140000.pt
export PATCHWAM_TEACHER_SHA256=<reviewed-model-safetensors-sha256>
```

The default teacher backend is the provisioned PyTorch C-RADIOv4-H export:

```bash
python scripts/train.py \
  --config configs/patchwam/patchwam_teacher_forcing_radio_robotwin.py \
  --work-dir work_dirs/patchwam_teacher_forcing_robotwin
```

Override the model path with `PATCHWAM_TEACHER_MODEL_PATH`. For a TensorRT
runtime compiled for the **exact** 288x256 input, set:

```bash
export PATCHWAM_TEACHER_BACKEND=trt
export PATCHWAM_TEACHER_TRT_ENGINE=/path/to/c_radio_v4_h_288x256.engine
export PATCHWAM_TEACHER_TRT_METADATA=/path/to/c_radio_v4_h_288x256.engine.json
export PATCHWAM_TEACHER_ENGINE_SHA256=<reviewed-engine-sha256>
```

`PATCHWAM_TEACHER_SHA256` is the representation digest. For one source artifact
it is that file's SHA-256; for several artifacts it is the canonical digest
computed by the provider. A TensorRT recipe requires the engine, ONNX source,
and ONNX external-data files to be local. The existing 256x256 and 512x512
engines are not compatible with the 288x256 RobotWin recipe.

The default PyTorch provenance pins the provisioned
`C-RADIOv4-H/model.safetensors` SHA-256
`851f283468118830a08e70173f1ab9a30f613c704958fc0b3c8e638aef738f81`;
override both the path and hash when swapping checkpoints.

## Runtime provenance and batching

Each encoder has an explicit `teacher_provenance` envelope:

```python
dict(
    representation_id='c-radio_v4-h-spatial_features-native_18x16',
    base_dir='checkpoints/Pretrained_models/C-RADIOv4-H',
    artifacts={'model.safetensors': '<sha256>'},
    representation_artifacts=['model.safetensors'],
)
```

The provider verifies every local artifact against its declared SHA-256 once,
caches the result, and binds the configured representation identity to the
actual source artifact set. For TensorRT, include the engine bytes in
`artifacts` and the ONNX source files in `representation_artifacts`.
`test_contract='explicit-unit-fixture'` is reserved for injected test fakes and
bypasses file hashes.

Extractors with a batch profile are called in balanced chunks within their
minimum/maximum batch bounds. For example, five valid rasters with bounds
`[2,4]` are encoded as `3+2`; one raster below `min_batch=2` fails clearly.
Chunk outputs are cloned before the next call.

## Flexible input sizes with PyTorch RADIO

The default recipe uses `RadioViTBackbone` from
`fluxvla/models/backbones/visions/radio_vit.py`. It runs the 288x256 RobotWin
mosaic directly; a matching TensorRT engine is not required. The same PyTorch
weights can accept other resolutions supported by the loaded RADIO model.

For a different resolution, configure the data/VAE raster size, the teacher
specification's `input_size` and `native_grid`, and the encoder's
`default_image_resolution` consistently in both training and inference.
For the tested patch-16 model, 256x256 produces a 16x16 grid and 288x256
produces an 18x16 grid. Verify the loaded encoder and VAE agree at the chosen
size. The provider validates exact geometry for each configuration; it does
not silently resize features or accept arbitrary sizes within a batch.

TensorRT remains an optional backend and requires an engine for the selected
geometry. The missing 288x256 engine affects only that optional backend.

## Geometry and contexts

The RobotWin mosaic is `288x256`. The FLUX VAE and patch-16 RADIO encoder both
produce an `18x16` grid, or 288 tokens per raster. The feature contract records:

- encoder width and representation identity;
- preprocessing and native spatial stream;
- input size and native grid;
- `independent_rasters` view context and view-major row-major ordering;
- `special_tokens='excluded'` for the dense spatial stream;
- `native_exact` alignment.

Current and future frames use separate provider calls. Current descriptors are
extracted once per observation context and reused across the Euler solver.
Training extracts future endpoints only for the feature-supervision target.
Inference never extracts future RGB.

Pre-encoded descriptors must include explicit `current_raster_layout` and, for
training targets, `future_raster_layout`. The layout must match every encoder's
input size, native grid, view count, and view-major row-major order. Supplied
`reference_ids`/`future_ids` are checked against the same raster order.
`teacher_current_tokens` and `teacher_future_tokens` are detached before M, G,
or the feature loss sees them.

## Training

For each update with feedback enabled, the head:

1. prepares the noisy image/action states, sigma, condition-drop mask, and
   current prefix once;
2. runs one no-grad zero-feedback look-ahead in local eval mode;
3. takes raw detached J predictions;
4. substitutes detached true future descriptors independently per sample with
   `true_feature_prob=0.05`;
5. applies LayerNorm, then per-sample `self_cond_drop_prob=0.1` dropout;
6. runs the trainable pass on the same noisy states and computes the unchanged
   PatchWAM flow losses plus masked feature cosine loss.

Every training update executes both passes when feedback is enabled. There is
no synchronized microbatch branch selector and no sample that skips the
look-ahead. Feature loss is a valid-token mean per sample and then a batch mean;
an all-invalid sample contributes zero. Current conditioning remains active
when future feedback is dropped.

Validation uses predicted feedback without true substitution or dropout.
Targets can affect conditioning only through the explicit substitution branch;
with substitution disabled, changing future targets changes only the feature
loss.

## Inference and guidance

Current descriptors and the clean prefix are prepared once. Each Euler step
replaces the feedback buffer with raw detached J predictions. Guided sampling
maintains separate conditional and unconditional feedback buffers and shares
the same current descriptors and M residual. Feedback is local to one
`sample_actions` call, so every new action chunk starts from zeros.

The combined model needs current features from a frozen encoder or validated
pre-encoded descriptors. A future-only feedback recipe with
`current_conditioning=False` can run inference without a teacher encoder.
Teacher future-target keys are ignored at inference; incomplete or differently
sized future targets cannot affect fixed-noise sampling.

## Checkpoints

`teacher_checkpoint_mode` is explicit:

- `base_init` may omit exactly the enabled M/G/J tensors and the contract
  digest, while every inherited PatchWAM tensor remains required;
- `resume` requires every teacher branch and a matching
  `teacher_contract_digest` buffer.

The same preflight is applied to direct head loads, direct VLA loads,
`load_patchwam_state_dict`, and loading through an outer module. A
`strict=False` caller cannot omit teacher tensors or bypass the digest. A
teacher checkpoint that contains teacher tensors is always treated as a
restore, even in `base_init`; mismatches are rejected before inherited weights
are changed.

The digest covers encoder representation identity, native geometry, switches,
predictor architecture, and feature normalization. Runtime paths, device,
loss weight, stochastic probabilities, and projector initialization are
provenance rather than architecture. Changing a representation or predictor
requires a new `base_init`, not a resume.

Frozen encoders and engine buffers are held by `TeacherFeatureProvider`, which
is outside the registered trainable tree. They do not enter the policy state
dict or optimizer. `close_teacher_extractors()` releases owned runtimes and
allows a later reload.

## Scope

This integration supports the ordinary VAE PatchWAM path with full head
finetuning. It rejects Self-Flow, the fused RADIO velocity transform, staged
visual adaptation, PEFT/LoRA combinations, EMA `representation_only` /
`teacher_sampling`, and per-token `token_sigma`. The method trains additional
branches and may change action behavior, but no manipulation improvement is
claimed without matched evaluation runs.

Focused CPU tests live under `test/test_models/patchwam/test_teacher_*.py` and
`test/test_configs/test_patchwam_teacher_forcing_config.py`. The real C-RADIO
provider test is marked `checkpoint` and can be run with
`--run-checkpoint-tests`.

For encoder verification at a compatible 256x256 geometry, the provider can
compare `RadioViTBackbone` with the local
`c_radio_v4_h_256_fp16_ln_fp32.engine` under a matching 16x16 spec. This checks
the generic provider boundary and extraction agreement; it does not make that
engine compatible with the 288x256 RobotWin recipe.

The provisioned comparison measured relative L2 `0.015577` and mean cosine
`0.999880` against PyTorch bfloat16.
