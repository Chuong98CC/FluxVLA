# PatchWAM Teacher Feature Forcing Implementation Plan

> **For agentic workers:** Implementation is not authorized by this planning request. When implementation is requested, follow the supplied AGENTS.md policy: load `astra-flash-orchestrator`, keep Astra responsible for architecture and acceptance, and dispatch one coherent phase bundle to the native `astra_flash_builder`. This overrides the generic writing-plans execution handoff. Do not auto-commit, push, deploy, or run paid setup smoke tests.

**Goal:** Extend the existing VAE-based PatchWAM policy with configurable frozen future-feature teachers, direct linear feature prediction, and prediction feedback between solver evaluations, using the existing C-RADIOv4 TensorRT backbone as the first teacher.

**Architecture:** Keep the original 128-channel image/action token space, pretrained shared FLUX input/output projections, Action-as-Patch codec, attention mask, and joint image/action flow. Add one bias-free, zero-initialized input projection and one randomly initialized linear prediction head per teacher. Extract frozen teacher targets through a separate training service; carry normalized predictions during deployment without constructing or running a teacher.

**Tech stack:** Existing FluxVLA registries and runners, PyTorch, the local official FLUX.2 adapter, `FrozenVisionFeatures` / `SpatialFeatureOutput`, and the existing optional TensorRT 10.x C-RADIO runtime. No dependency on the DINO-Forcing package or its Lightning/Hydra stack.

**Spec:** [PatchWAM_DINO_Forcing_Design_Proposal.md](../../../papers/PatchWAM_DINO_Forcing_Design_Proposal.md), revised 2026-10-13. Read it together with this plan. This document makes implementation choices where the proposal leaves an interface unspecified; it does not claim the proposed extension has been experimentally validated.

**Status:** Plan only. No model, training, configuration, or test implementation has been changed or executed for this plan. Repository inspection included existing uncommitted RADIO changes; preserve them during future implementation.

## Global constraints

- Preserve the VAE future-image flow and action flow, both with width 128. A teacher supervises an additional representation; it does not replace the observation frontend.
- Retain `shift=5.0`, `video_weight=0.5`, `action_weight=1.0`, the configured flow-weight normalization, and the benchmark's action/state normalization and state-token packing.
- Retain action horizon 16 and 10 Euler steps for the initial matched comparison. Keep horizon and solver steps configurable through the existing interfaces.
- Preserve all action-step and action-dimension masks, including supervision of the codec's fill coordinates.
- Add teacher conditioning only to future-image hidden tokens. Add no tokens, MLPs, gates, teacher feature noise, feature velocity, feature ODE, or learned teacher-side target projection.
- Predict features from the output of the last transformer block, **before** the existing final normalization/adaLN velocity readout. Use the same location for the official and tiny denoisers.
- Normalize feedback per patch in float32 with L2 epsilon `1e-6`, then cast to the receiving model dtype. Zero remains zero. Use the same helper in training and sampling.
- Default feedback training: 90% detached look-ahead plus trainable second evaluation; 10% one trainable zero-conditioned evaluation. Sample this choice once per microbatch, synchronized across distributed ranks.
- Frozen teacher targets are supervision only. Do not add the DINO implementation's ground-truth-feature conditioning branch.
- Default teacher cosine-loss weight is `0.1`; evaluate `0.05`, `0.1`, and `0.3`. Keep teacher loss outside the flow weight `w(sigma)`.
- Training teachers stay frozen/eval and run under `torch.no_grad()`. Deployment must work with no engine, sidecar, teacher checkpoint, TensorRT, or teacher runtime installed.
- Use local assets and existing registries. No automatic download, model substitution, engine build, installation, or changes to `/home/chuong/workspace/dino_forcing`.
- Initial support is the ordinary VAE PatchWAM pathway. Reject combinations with visual-feature replacement, staged visual adaptation, LIT, fused RADIO velocity transforms, or EMA/Self-Flow rather than assigning them new semantics in this change.

## Review focus

1. **Wrong spatial correspondence:** rectangular grids, mosaics, separate views, and pooled history must preserve future raster order and field of view. Tasks 2 and 5 own these tests.
2. **Target leakage or broken detachment:** changing teacher targets must change supervision, not velocities or feature conditions. Task 4 owns these tests.
3. **Checkpoint identity collision:** equal feature widths do not make two teachers interchangeable; stale caches and wrong extraction streams must fail before inherited weights are copied. Tasks 1 and 6 own these tests.
4. **Distributed branch divergence:** look-ahead must not produce different collective sequences across ranks; all new trainable weights must exist before optimizer/DDP/FSDP setup. Tasks 4, 6, and 8 own these tests.
5. **Invalid inputs and deployment dependencies:** padded NaNs, fully invalid endpoints, engine batch limits, and unavailable teacher files must have explicit behavior. Tasks 2, 5, and 8 own these tests.

## 1. What the inspected implementations establish

### 1.1 PatchWAM integration points

| File / symbol | Current responsibility | Consequence for this change |
|---|---|---|
| [`PatchWAMVLA`](../../../fluxvla/models/vlas/patchwam_vla.py) | Filters encoder/head inputs, adapts validity masks, calls loss and sampling, loads several checkpoint layouts | Add teacher-target passthrough, training extractor ownership, and explicit initialization/resume rules here |
| [`PatchWAMBackbone.encode`](../../../fluxvla/models/backbones/vlms/patchwam_backbone.py) | Owns frozen VAE and instruction encoder; delegates to `encode_observations` | Keep both encoders; pass a generic target provider into encoding when training |
| [`encode_observations`](../../../fluxvla/models/third_party_models/patchwam/encoding.py) | Selects the actual current/future world rasters, sanitizes invalid frames, flattens views, encodes latents, builds IDs and validity | Extract teacher targets from the exact sanitized future RGB already selected for the VAE |
| [`PatchWAMHead`](../../../fluxvla/models/heads/patchwam_head.py) | Config-aware `PatchFlowPolicy` construction and width validation | Parse teacher configuration and reject unsupported combinations |
| [`PatchFlowPolicy._velocity` / `forward` / `sample_actions`](../../../fluxvla/models/third_party_models/patchwam/policy.py) | Builds prefix/masks, calls denoiser, computes flow loss, performs descending Euler integration | Own trainable teacher projections/predictors; implement two-pass training and sampling feedback here |
| [`OfficialFluxDenoiser.forward`](../../../fluxvla/models/third_party_models/patchwam/flux.py) | Projects `[reference, future, action]` together, runs masked FLUX blocks, applies the shared velocity head | Add future-only hidden residual after `img_in`; use existing last-layer representation capture |
| [`SmallJointTransformer.forward`](../../../fluxvla/models/third_party_models/patchwam/tiny.py) | CPU test counterpart with the same token layout and representation capture | Implement the identical optional residual contract for meaningful offline tests |
| [`ShiftedFlow`](../../../fluxvla/models/third_party_models/patchwam/flow.py) | Noise interpolation, sigma warp, weighting, decreasing schedule | Reuse unchanged |
| [`FrozenVisionFeatures` / `SpatialFeatureOutput`](../../../fluxvla/models/backbones/visions/frozen_vision.py) | Frozen spatial tokens `[N, h*w, D]` and explicit grid | Reuse as the first teacher adapter boundary |
| [`RadioViTBackboneTensorRT`](../../../fluxvla/models/backbones/visions/radio_trt.py) | Frozen C-RADIOv4 native spatial tokens from a fixed-geometry engine | First concrete training teacher, built through `VISION_BACKBONES` |

The encoder already has a RADIO-specific auxiliary hook, `radio_encoder`, and cached key `radio_future_tokens`. Keep their existing behavior. Add a generic teacher-target path rather than routing a configurable teacher through RADIO-named keys. The existing `visual_backbone` option **replaces** VAE latents and is therefore unsuitable for this proposal.

The actual future raster depends on the data recipe. `camera_video` takes precedence over `video`/`images`. With separate views, the encoder processes `B*V` images and flattens them in **view-major, then row-major** order. With a mosaic, it processes the composed image as one raster (`V=1`), including the camera arrangement inside that raster. Do not extract independent original-camera targets and attach them to mosaic VAE tokens.

History extends `reference_tokens` only. `sampling_future_length` comes from `future_ids`, so feedback buffers must not be sized from the full historical reference length.

Two statements in the architecture summary should not be copied into the implementation: its notation table reverses sigma's endpoints, and one input table describes unwarped uniform sigma. The code and proposal are authoritative: `sigma=0` is clean, `sigma=1` is noise, sigma is a warped uniform draw, target velocity is **noise minus clean**, and sampling descends from 1 to 0.

### 1.2 DINO-Forcing: ideas to carry over and details to adapt

The local source was inspected, including [training_module.py](/home/chuong/workspace/dino_forcing/src/training_module.py), [sit_class_cond_network.py](/home/chuong/workspace/dino_forcing/src/sit_class_cond_network.py), [jit_class_cond_network.py](/home/chuong/workspace/dino_forcing/src/jit_class_cond_network.py), [sampler.py](/home/chuong/workspace/dino_forcing/src/sampler.py), [data.py](/home/chuong/workspace/dino_forcing/src/data.py), and [processing.py](/home/chuong/workspace/dino_forcing/src/processing.py).

| Source behavior | Planned PatchWAM behavior |
|---|---|
| `training_step` evaluates the model without gradients, detaches `self_condition`, then evaluates again | Same prediction/stop-gradient principle, inside one native PatchWAM head forward using already sampled noisy inputs |
| SiT concatenates features with patch channels; JiT has a separate fusion projector | Preserve pretrained FLUX `img_in`; add a zero-initialized linear hidden residual on future tokens |
| REPA-style MLP predicts features from a selected block, defaulting to the final block | A single linear readout from final-block future states, as specified by the proposal |
| Alignment normalizes prediction/target for a negative-cosine objective | Masked `1-cosine`, equivalent gradients when token selection is identical; separately weighted from flow loss |
| Example SiT config uses `self_cond_drop_prob=0.1` per sample; training still performs look-ahead | Implement the proposal's explicit 10% whole-microbatch single-pass branch, not merely per-sample feature dropout |
| Optional `true_dino_features_prob=0.05` substitutes ground-truth conditions per sample | Omit; future target features never become model conditioning |
| Optional input LayerNorm; Euler carries raw predicted features separately for conditional/null branches | Shared L2-normalized feedback helper; separate branch buffers when CFG is requested |
| Flow time increases from noise to data; SiT target is clean minus noise | Preserve PatchWAM's opposite sigma convention and velocity sign |
| Latent targets are loaded from `dino_features.npy`; pixel preprocessing extracts frozen DINO patches | Generic online extraction from the actual future world raster; a validated pre-encoded input contract is also supported |

The SiT sampler's CFG/control-flow details and its use of a mutable batch dictionary are not templates to copy. Preserve the existing PatchWAM sampler's guidance behavior and keep input conditions distinct from output predictions.

### 1.3 C-RADIOv4 facts relevant to the plan

- The registered class is **`RadioViTBackboneTensorRT`**, not a new teacher model class.
- `encode_features(images)` accepts `[N,3,H,W]` in `[-1,1]`; it converts to `[0,1]` internally. Do not add ImageNet normalization or convert to unit range again.
- The output is `SpatialFeatureOutput(tokens=[N,h_T*w_T,d_T], grid=(h_T,w_T))`. `normalize_spatial_tokens` standardizes **layout**, not L2 feature magnitude.
- C-RADIOv4-H has `d_T=1280`; SO400M has `d_T=1152`. Read/validate metadata, and never assume the earlier DINO example's width 768.
- Geometry is fixed by the engine sidecar; batch is bounded by `min_batch`, `opt_batch`, and `max_batch`. Neither the backbone nor this integration may silently resize/pad into a different field of view.
- The runtime loads lazily, has no model parameters for `.to()` to move, and uses the engine's actual input/output dtypes. Builder precision is not a substitute for those dtypes.
- The existing metadata does not identify the original teacher checkpoint revision completely. Record that provenance explicitly in the teacher representation specification; engine/ONNX hashes alone do not replace it.

## 2. Selected structure and rejected alternatives

Use an **opt-in extension of the existing VLA/head**, with a small generic target provider and a focused numerical helper module. Leave the default `teacher_forcing=None` path unchanged, including state-dict keys and random draw order.

Changing `img_in` to accept concatenated teacher channels would alter pretrained parameter shapes and checkpoint mapping. Replacing the VAE with a feature backbone would change the modeled flow state. Neither is needed: additive hidden fusion supplies the proposal's architecture while preserving the baseline asset contract.

Own all new trainable weights under `vla_head`, inside the existing `PatchWAMHead` FSDP unit. Own teacher runtime resources in an explicit, training-only provider outside the model's registered parameter tree. That service must manage freezing, device placement, and closure itself; it must not rely on `model.to()` or `model.train()` to configure a teacher. This avoids saving frozen teacher weights and avoids initializing a teacher at deployment.

Use a teacher list and `ModuleDict` from the start. One teacher is the default. This supports different widths and the optional multiple-teacher comparison with the same model evaluations; it does not require implementing DINOv2 or VGGT adapters now.

## 3. Configuration and interfaces

### 3.1 Policy configuration, separate from extraction configuration

Add optional `teacher_forcing` to `PatchWAMHead` / `PatchFlowPolicy`. It contains `mode`, the ordered teacher specifications, normalization settings, and training probabilities. Add optional `teacher_extractors` to `PatchWAMVLA`; it maps teacher names to training runtime definitions. The VLA validates extractor definitions against `vla_head.teacher_specs`.

| Setting | Initial value / rule |
|---|---|
| `teacher_forcing` | `None` disables the entire extension |
| `mode` | `feedback`; also support `prediction_only` for the supervision-only ablation |
| `zero_condition_probability` | `0.1`, finite in `[0,1]`; relevant only to feedback training |
| `feedback_normalization` | `l2`, epsilon `1e-6` |
| Teacher `name` | `radio`; unique stable module key, restricted to letters/digits/underscores, starting with a letter |
| Teacher `feature_dim` | `1280` for the initial C-RADIOv4-H example; positive and validated against extraction output |
| Teacher `loss_weight` | `0.1`, finite and nonnegative |
| Teacher `representation_id` | Immutable identifier for the exact checkpoint revision and selected dense stream |
| Teacher `checkpoint_revision` / `checkpoint_sha256` | Required concrete provenance recorded before a real training run |
| Teacher `feature_stream` / `extraction_point` | `native_spatial` / `export_native_spatial` for this engine; unsupported alternatives fail |
| Teacher `input_size` | `(288,256)` for the RobotWin mosaic example; must match the engine sidecar and world raster |
| Teacher `input_preprocessing` | Native `[-1,1]` endpoint input; RADIO-internal unit-range conversion |
| Teacher `geometry_policy` | `exact`; no implicit resizing, padding, cropping, or flip |
| Teacher `view_context` | `independent_rasters`; a mosaic is one raster |
| Teacher `special_token_policy` | Spatial stream only; no summary/CLS/register tokens |
| Teacher `alignment` | Bilinear grid resampling, `align_corners=False`; identity when grids already match |
| Teacher `loss_reduction` | Valid-token mean per sample, then batch mean; fully invalid samples contribute zero |
| VLA `teacher_checkpoint_mode` | Explicit `base_init` or `resume` in enabled recipes; baseline remains legacy |

The initial `teacher_extractors.radio.backbone` definition uses `type='RadioViTBackboneTensorRT'`, `vision_backbone_id='c-radio_v4-h'`, and local `engine_path` / `metadata_path`. The extractor definition also records `artifact_sha256`, binding the reviewed engine bytes to the declared source-checkpoint identity; validate it during training setup. Source provenance must come from the export record, not inference from a variant name. Use `device=None` so first extraction follows the current rank's CUDA device; reject a conflicting explicitly configured device. Read metadata to validate the contract before training, but load TensorRT only when targets are actually missing.

Inference configuration retains `teacher_forcing` and representation identity but sets `teacher_extractors=None` and checkpoint mode `resume`. It must not read engine files even at model construction. The trained linear layers need `d_T` and the representation signature, not a teacher object.

Changing to another independent spatial teacher means registering an adapter exposing `feature_dim`, `feature_grid`, and `encode_features`, then changing configuration and training new teacher branches. It must not require changes to the policy's flow, loss, or sampler. A joint multi-view geometry teacher requires a provider adapter that consumes the ordered `[B,V,...]` endpoint set jointly; the initial provider rejects that context rather than silently flattening it into independent images.

### 3.2 New helper contracts

These are planned signatures and names, not implementation code.

| Interface | Location / contract |
|---|---|
| `TeacherFeatureSpec` | `teacher_forcing.py`; immutable validated representation specification with name, width, provenance, extraction, geometry, normalization/reduction, and loss weight |
| `TeacherForcingConfig` | Same file; mode, ordered `TeacherFeatureSpec` entries, shared epsilon, and zero-condition probability |
| `teacher_representation_fingerprint(spec, *, feedback_eps, readout_point) -> bytes` | Same file; SHA-256 of canonical representation fields, including width, stream, preprocessing, view context, alignment, feedback normalization, and readout location |
| `normalize_teacher_feedback(tokens: Tensor, valid: Tensor, *, eps: float) -> Tensor` | Same file; `[B,S,d_T]`, FP32 norm, invalid rows sanitized before normalization, output in input dtype |
| `masked_teacher_cosine_loss(prediction: Tensor, target: Tensor, valid: Tensor, *, eps: float) -> Tensor` | Same file; finite FP32 scalar, per-sample masked reduction |
| `TeacherTargetProvider(specs, extractor_configs)` | `backbones/visions/teacher_targets.py`; plain resource-owning service, not a registered model submodule |
| `TeacherTargetProvider.validate_static_contracts() -> None` | Read available configured metadata/provenance; no TensorRT runtime construction |
| `TeacherTargetProvider.encode_future(frames, *, target_grid, target_valid, future_ids) -> mapping` | Frames `[B,V,3,H,W]`; returns aligned targets, validity, and fingerprints described below |
| `TeacherTargetProvider.close() -> None` | Releases only owned runtime resources; supports context-manager use for extraction jobs/tests |
| `PatchFlowPolicy.teacher_new_parameter_names() -> list[str]` | Exact new projection/predictor parameter names for initialization allow-list and optimizer checks |
| `PatchFlowPolicy.teacher_checkpoint_contract() -> mapping` | Canonical human-readable contract and aggregate digest for logging/validation |

Build each registered vision backbone using `build_vision_backbone_from_cfg`. Build/freeze the concrete teacher lazily on the local target device. Never put a teacher in a dataloader worker or create its CUDA context before the training process selects its device.

### 3.3 Encoded batch contract

| Key | Shape / meaning |
|---|---|
| `teacher_future_tokens` | Mapping `name -> Tensor[B,S,d_T]`; clean aligned detached target descriptors |
| `teacher_future_valid` | Mapping `name -> BoolTensor[B,S]`; intersect with `future_valid` |
| `teacher_target_fingerprints` | Mapping `name -> UInt8Tensor[B,32]`; exact representation contract for pre-encoded targets |
| `future_raster_layout` | Explicit views, VAE grid, input geometry, and view-major/row-major packing; emitted on raw encode and supplied for cached raster extraction |

Preserve the existing `reference_tokens`, `future_tokens`, `future_ids`, `future_valid`, `sampling_future_length`, text, action, and state fields. Extend the VLA's `_TOKEN_KEYS` / `_ENCODE_KEYS` so these new keys survive both raw and backbone-free pre-encoded calls. Do not reuse `radio_future_tokens` as an alias for arbitrary teachers.

Targets must exactly match configured teacher names, batch size, future length, width, and representation fingerprint. Valid target vectors must be finite and have norm greater than epsilon; a zero target vector has no cosine direction and is an extraction/cache error. Invalid rows may contain padding before validation, but must be sanitized to zero before any arithmetic. Zero **predictions** remain allowed.

For a pre-encoded training batch, require explicit `future_ids` and `future_raster_layout` and validate the IDs against its declared raster packing; new CPU training fixtures also supply these. Use supplied validated targets without executing the teacher. If targets are missing, require both the actual future RGB and that explicit raster layout; otherwise fail with the missing teacher names. Do not reconstruct teacher targets from decoded VAE latents. An offline cache-writing tool is a later optimization; the initial code supports the validated input contract without introducing a new dataset cache format. Sampling does not require teacher targets or their cache metadata and may still use the existing synthetic sequence-ID fallback.

## 4. Tensor placement, alignment, and loss decisions

Let `R` be the reference length including history, `S=V*h*w` the future raster length, `H` the action horizon, `d` the actual denoiser `representation_dim`, and `d_T` the teacher width. Read `d` from the loaded denoiser; 3072 is the 4B example, not a constant in new modules.

The head owns `teacher_conditioners[name]: Linear(d_T,d,bias=False)` and `teacher_predictors[name]: Linear(d,d_T,bias=True)`. Create them on the denoiser parameter device/dtype before optimizer/wrapping. Zero only conditioner weights. Use ordinary `nn.Linear.reset_parameters()` random predictor initialization, including its bias; do not zero the predictor or reinitialize any pretrained module.

For feedback mode, sum the per-teacher hidden residuals:

\[
\Delta h_{\rm future}=\sum_q W_q\,f_{q,\rm cond},\qquad
h_{\rm future}=W_{\rm in}x_\sigma+b_{\rm in}+\Delta h_{\rm future}.
\]

Conditions reaching this operation are already normalized by the shared helper; do not apply an additional LayerNorm. Add `future_hidden_residual: Tensor | None` to both denoiser forward signatures. Its shape is `[B,S,d]`. The official denoiser adds it only at image-stream slice `[R:R+S]` after the original projection. The tiny denoiser adds it at `[context_length+R:context_length+R+S]`. Preserve the original shared velocity head invocation.

Request `representation_layer=denoiser.representation_layers` to retrieve raw final-block generated states. Slice `[:, :S]` for teacher prediction; exclude action states, reference, text, and history. Do not reuse the existing fused `final_hidden` field, which is conditioned by the final velocity-head normalization/adaLN and is a different extraction point.

Extend `_velocity` with optional `teacher_conditions` and `return_teacher_predictions`. Its legacy return remains unchanged. In teacher mode, return `(velocity_future, velocity_action, predictions_by_name)`; reject simultaneous legacy capture/Self-Flow/fused options so the third return cannot be mistaken for another feature tensor.

Teacher loss uses

\[
L_q=\frac1B\sum_b\frac{\sum_i m_{b,i}\,[1-\cos(\hat f_{q,b,i},f^*_{q,b,i})]}{\max(\sum_i m_{b,i},1)},
\quad m=\text{future_valid}\land\text{teacher_future_valid}_q.
\]

Compute the masked numerator, normalization, and reduction in float32. A fully invalid sample has a zero numerator and contributes zero. Teacher validity does not change the baseline image/action masks.

\[
L=L_{\rm PatchWAM}+\sum_q\lambda_qL_q,
\qquad L_{\rm PatchWAM}=0.5L_{\rm video}+1.0L_{\rm action}.
\]

The existing `loss_video` / `loss_action` already include `w(sigma)`. Do not multiply teacher loss by that weight or alter the original modality reductions. Preserve optional language auxiliary loss addition once in `PatchWAMVLA._run_head`.

Return scalar `loss_teacher_<name>` and aggregate `loss_teacher` diagnostics alongside existing scalar losses. Do not put dense feature predictions in the default training output. Extra timing/feedback diagnostics need an explicit supported metric path; the runner's current collector automatically recognizes only scalar loss-like names.

## 5. Training and sampling behavior

### 5.1 Training

1. Encode prefix and clean future VAE tokens once; extract or validate clean future teacher targets once.
2. Resolve action masks, encode actions, and sample the existing condition-drop mask, sigma, and independent future/action noises once. Preserve baseline draw order; add the new branch draw only after the baseline noisy states have been formed.
3. In feedback mode, select one microbatch branch. Broadcast that boolean from rank 0 on the active distributed process group before branch-dependent evaluation. An optional `teacher_zero_condition` test/debug knob forces the branch and is checked consistently across ranks.
4. For the 90% branch, call `_velocity` with zero feature conditions under `torch.no_grad()`. Keep only predicted features, detach, apply the shared normalization and validity mask, then release first-pass velocities/hidden references. Call `_velocity` again with gradients using the **same** noisy states, sigma, prefix tensors, and condition-drop mask.
5. For the 10% branch, call `_velocity` once with gradients and zero conditions. Apply all image/action/teacher losses to this evaluation.
6. In `prediction_only` mode, use one ordinary velocity evaluation plus teacher readout/loss; create predictors but no conditioners, feedback branch draw, or feedback buffers.
7. Apply the unchanged image/action flow losses and the teacher loss to the trainable evaluation only. The language auxiliary loss is still added once by the wrapper.

Do not implement this by calling the entire head `forward` twice: that would resample noise/dropout and duplicate upstream work. Do not globally toggle `self.eval()` / `self.train()` during a training forward, as the local DINO training code does. Keep the model's current mode; no-gradient recording supplies the required detachment, and existing FLUX gradient checkpointing already checks gradient-enabled state.

Validation uses deterministic two-pass feedback (`teacher_zero_condition=False`) and fixed sampled inputs where comparisons need reproducibility. The stochastic 10% branch is a training rule. Tests cover both branches explicitly rather than testing a noisy empirical percentage.

### 5.2 Sampling

Allocate local zero conditions `[B,S,d_T]` at the start of each `sample_actions` call, using actual generated length and model device/dtype. Each of `K` Euler steps:

1. Evaluate the conditional model once for image/action velocities and all teacher predictions.
2. If either existing guidance scale differs from 1, evaluate the unconditional branch once with its **own** feedback conditions. Preserve the existing dropped-language definition of that branch.
3. Apply the existing guidance formula to velocities only. Normalize and replace each branch's feature conditions using that branch's own prediction; do not guide, average, or integrate feature tensors.
4. Integrate image/action states using the existing negative `end-start` sigma increment.

Decode and return normalized actions as before. No teacher extraction or future RGB decode occurs. Conditions are local to the call and reset for every replanned action chunk; do not store them on the module or accept ground-truth target keys as sampling conditions.

At guidance scales 1, assert exactly `K` denoiser evaluations. With guidance enabled, assert exactly `2K`. `prediction_only` mode keeps conditions absent. Pooled history changes `R` but not `S` or feedback size.

## 6. Planned file map

| Action | File | Responsibility |
|---|---|---|
| Create | `fluxvla/models/third_party_models/patchwam/teacher_forcing.py` | Typed contract, fingerprinting, normalization, masked cosine helper, branch-selection helper |
| Create | `fluxvla/models/backbones/visions/teacher_targets.py` | Generic frozen extraction provider, alignment, batch-profile handling, explicit runtime lifecycle |
| Modify | `fluxvla/models/third_party_models/patchwam/encoding.py` | Generic target hook and explicit future raster metadata; preserve RADIO-specific hook |
| Modify | `fluxvla/models/backbones/vlms/patchwam_backbone.py` | Optional provider keyword passed to encoding |
| Modify | `fluxvla/models/third_party_models/patchwam/flux.py` | Optional future-only hidden residual |
| Modify | `fluxvla/models/third_party_models/patchwam/tiny.py` | Same residual and final-block capture semantics for CPU tests |
| Modify | `fluxvla/models/third_party_models/patchwam/policy.py` | Teacher module ownership, loss integration, two-pass evaluation, sampler feedback |
| Modify | `fluxvla/models/heads/patchwam_head.py` | Configuration parsing and compatibility validation |
| Modify | `fluxvla/models/vlas/patchwam_vla.py` | Extractor service construction, batch passthrough, deployment separation, checkpoint rules |
| Create | `configs/patchwam/patchwam_teacher_forcing_radio_robotwin.py` | Matched C-RADIO training and teacher-free inference model definitions |
| Create | `configs/patchwam/patchwam_teacher_prediction_radio_robotwin.py` | Prediction-only ablation based on the same recipe |
| Create | `docs/patchwam-teacher-forcing.md` | Usage, shapes, supported combinations, checkpoint identity, extraction lifecycle |
| Modify | `test/cpu-tests.txt` | Add new CPU tests without disturbing existing entries |
| Create | `test/test_models/patchwam/test_teacher_feature_contract.py` | Configuration, numerical helpers, fingerprint rejection |
| Create | `test/test_models/patchwam/test_teacher_targets.py` | Endpoint selection, geometry, provider lifecycle/profile handling |
| Create | `test/test_models/patchwam/test_teacher_forcing_denoiser.py` | Injection/readout placement, tiny and local official adapter checks |
| Create | `test/test_models/patchwam/test_teacher_forcing_training.py` | Detachment, pass counts, loss and padding behavior |
| Create | `test/test_models/patchwam/test_teacher_forcing_sampling.py` | Recurrence, guidance, history, reset, deployment independence |
| Create | `test/test_models/patchwam/test_teacher_forcing_checkpoint.py` | Exact initialization allow-list, identity, strict resume, no teacher state |
| Create | `test/test_configs/test_patchwam_teacher_forcing_config.py` | Training/inference config consistency and lazy optional imports |
| Create | `test/test_models/patchwam/test_teacher_forcing_distributed.py` | Gated DDP/FSDP branch and gradient tests |

Do not change `codec.py`, `flow.py`, `geometry.py`, teacher engine serialization/runtime, or the DINO-Forcing checkout for this feature. A real integration test exposing an engine defect would justify a separately identified fix, not an assumed refactor. Prefer targeted additions to these existing files over another hierarchy of PatchWAM VLA/head subclasses.

## 7. Dependency-ordered implementation tasks

The task checklists below are for future implementation. None are completed by this planning-only change.

### Task 1: Define the teacher contract and numerical helpers

**Files:** Create `teacher_forcing.py` and `test_teacher_feature_contract.py` at the paths in the file map.

**Consumes:** The proposal's linear architecture and the configuration contract in section 3.

**Produces:** `TeacherFeatureSpec`, `TeacherForcingConfig`, representation fingerprints, normalization/loss helpers, and validation usable without teacher assets or optional inference packages.

- [ ] Write failing tests for configuration parsing: duplicate/invalid names, nonpositive widths, nonfinite/negative weights, probability outside `[0,1]`, invalid epsilon, unsupported mode, and incomplete representation provenance must fail with the offending field named.
- [ ] Write normalization tests: `[3,4] -> [0.6,0.8]`, zero prediction remains zero, masked NaNs become zero, and valid nonfinite predictions fail. Confirm output dtype is preserved and computation remains finite in half precision.
- [ ] Write loss tests: parallel vectors give 0, orthogonal vectors give 1, opposite vectors give 2; one sample with mean loss 1 plus one fully masked sample gives batch loss `0.5`. Masked NaNs must contribute zero. Confirm gradients reach predictions, not detached targets.
- [ ] Write fingerprint tests: checkpoint revision/hash, stream, feature width, view context, geometry, alignment, normalization, and readout changes alter the digest. Runtime file paths/device and scalar loss weights do not change representation identity. Ordered teacher names and policy mode (`feedback` versus `prediction_only`) contribute to the aggregate policy digest; per-teacher target fingerprints remain independent of policy mode.
- [ ] Run `python -m pytest test/test_models/patchwam/test_teacher_feature_contract.py -q`; confirm failure for the missing functionality before implementation.
- [ ] Implement the immutable contracts and helpers. Canonicalize fingerprint input with stable JSON encoding; keep no Python-pickle extra state in the contract.
- [ ] Run the same command and confirm all tests pass.

**Acceptance:** The numerical core imports and runs on CPU without DINO, RADIO assets, ONNX, TensorRT, CUDA, Lightning, or Hydra. Configuration distinguishes a representation identity from a runtime backend.

### Task 2: Extract and align generic future targets

**Files:** Create `teacher_targets.py` and `test_teacher_targets.py`; modify `encoding.py` and `PatchWAMBackbone.encode`.

**Consumes:** Task 1 specifications and the existing `FrozenVisionFeatures` / `SpatialFeatureOutput` contract.

**Produces:** `TeacherTargetProvider`, the encoded target keys, and `future_raster_layout`. Extend backbone encoding with optional `teacher_provider`; the existing default and RADIO auxiliary hook keep their behavior.

- [ ] Write failing tests with a deterministic coordinate-valued frozen teacher and a small fake VAE. Assert extraction receives the endpoint at frame `-1`, after the exact same world augmentation/composition and invalid-frame sanitization as the VAE; it never receives current images, `vl_current`, history, or decoded latents.
- [ ] Test `camera_video` precedence and both layouts: a mosaic is one raster; separate cameras preserve view order with independent alignment. Use rectangular source/target grids and different values in each view so transposition or cross-view mixing cannot pass.
- [ ] Test that matching grids use identity; mismatched grids follow precisely bilinear `align_corners=False` interpolation. Reshape each teacher output to `[B*V,d_T,h_T,w_T]`, resample only height/width in float32, restore `[B,V,h*w,d_T]`, then flatten view-major/row-major. Never interpolate across channels or across cameras.
- [ ] Test wrong input geometry, output token count, channel width, missing spatial stream, wrong fingerprint, and requested joint-view context. Each must produce a clear error; no silent resize or feature-key guess is permitted.
- [ ] Test invalid views/endpoints, including NaN padded RGB. Skip extraction for invalid rasters and scatter zero targets back into their original positions. If every endpoint is invalid, return zero targets/false validity without constructing the runtime.
- [ ] Test exact engine batch-profile behavior with an injected fake runtime. For `N` valid rasters, choose `q=ceil(N/max_batch)` balanced chunks, with first `N mod q` chunks of size `floor(N/q)+1`; require every chunk to lie in `[min_batch,max_batch]`. For example, bounds `[2,4]` with `N=5` yield sizes 3 and 2; `N=1` fails. Do not duplicate or pad examples to satisfy a profile. Use `min_batch=1` in the initial real recipe.
- [ ] Test frozen/eval/no-gradient extraction, explicit device management, ordinary detached output tensors, preservation across sequential engine calls, lazy runtime loading, and closing only owned resources. Use injected engines with nonzero descriptors so cosine direction is meaningful.
- [ ] Run `python -m pytest test/test_models/patchwam/test_teacher_targets.py -q`; confirm the missing generic path fails.
- [ ] Implement the provider and generic encoding hook. Read and validate extractor metadata/provenance when configured for training; lazily construct registered encoders on the correct rank device. Use each backend's preprocessing interface exactly once.
- [ ] Emit the explicit raster layout even when encoding only the current observation. Extra metadata must not change baseline numerical behavior. In teacher-enabled sampling, the VLA derives generated-raster validity from the current raster's view validity before history is appended; it must not require an observed future endpoint.
- [ ] Implement validated pre-encoded target passthrough. If an enabled teacher target is absent, fail unless actual future RGB and explicit raster layout are available for extraction. Preserve existing cached text checks and optional language adapter rules.
- [ ] Run the new tests and the existing `test/test_models/patchwam/test_patchwam_backbone.py` and `test/test_models/patchwam/test_patchwam_visual_backbones.py` tests. Confirm baseline and visual-replacement behavior is preserved.

**Acceptance:** Teacher targets index exactly the future VAE slots and remain detached. C-RADIO implementation knowledge is confined to its backend/config validation; policy and loss code use generic names. Multiple independent providers can use different widths and native grids.

**Cache rule:** The implemented pre-encoded tensor interface validates representation fingerprints, shape, future IDs/layout, and masks. It cannot recover image identity or augmentation history from VAE latents. Its caller owns that correspondence. A persistent dataset cache is deferred until its dataset/transform owner supplies authoritative endpoint identity and augmentation provenance; do not claim the model already validates those missing records. Online extraction is the default with stochastic world augmentation. Offline caching is useful only when the exact augmented raster can be reproduced; do not silently combine unaugmented features with augmented VAE targets.

### Task 3: Add future-only fusion and final-block linear prediction

**Files:** Modify `flux.py`, `tiny.py`, `policy.py`, and `patchwam_head.py`; create `test_teacher_forcing_denoiser.py`.

**Consumes:** Task 1 contracts; existing `representation_dim` / `representation_layers` and final-layer capture.

**Produces:** Optional `future_hidden_residual` in both denoisers; head-owned `teacher_conditioners` / `teacher_predictors`; `_velocity(..., teacher_conditions=None, return_teacher_predictions=False)` with preserved legacy returns.

- [ ] Write failing tests for zero-initialization and exact module structure: conditioner has only a `[d,d_T]` weight; predictor has `[d_T,d]` weight and `[d_T]` bias; both are trainable in feedback mode. Predictor is randomly initialized. No extra gate, norm, MLP, transformer token, or action projection is added.
- [ ] Copy identical inherited weights into baseline and enabled models. With zero conditioner weights, compare image/action velocities on identical fixed tensors. Disabled mode must keep the original state-dict keys and output API; no new random draw is allowed in its forward/sampler.
- [ ] Inspect the hidden inputs to the first block with a nonzero residual. Only future slots change; projected reference/text/state/action inputs and all positions/masks remain unchanged. Do not require final action velocities to stay unchanged: future-to-action attention is how feedback influences control.
- [ ] Capture raw states from the final block and assert the feature predictor consumes exactly `[:, :S]`. Verify the captured state precedes `output_norm` in the tiny model and final adaLN/normalization in the official model.
- [ ] Test residual shape/dtype/device validation, per-teacher widths, summed conditioning, mask sanitization, history offsets, and incompatible legacy capture/fused/Self-Flow options.
- [ ] Run `python -m pytest test/test_models/patchwam/test_teacher_forcing_denoiser.py -q`; confirm failure before changing the denoiser boundary.
- [ ] Allocate teacher modules from the denoiser's declared hidden width/device/dtype. Add the optional residual out of place after the original input projection, padding its prefix/action slices with zeros rather than editing latent inputs.
- [ ] Request final raw representation capture in teacher evaluations and apply teacher predictors only to future states. Keep the original velocity projection call unchanged.
- [ ] Mirror the contract in both denoisers and test backward with gradient checkpointing on/off. Reuse the existing small official FLUX fixture where local source is available; missing source is a reported skip, not evidence of official parity.
- [ ] Run the new tests plus `test/test_models/patchwam/test_patchwam_flux_adapter.py` and `test/test_models/patchwam/test_patchwam_parity.py`.

**Acceptance:** The new branches add `2*d*d_T+d_T` trainable parameters per feedback teacher. For the 4B hidden width 3072 and RADIO-H width 1280, that is **7,865,600** parameters. Disabled execution preserves the baseline model contract. Prediction-only mode creates predictors without conditioners.

### Task 4: Implement the detached training procedure and objective

**Files:** Modify `policy.py` and the branch helper in `teacher_forcing.py`; create `test_teacher_forcing_training.py`. Extend VLA `_HEAD_KNOB_KEYS` in Task 6 with `teacher_zero_condition`.

**Consumes:** Tasks 1–3 targets, teacher modules, and extended velocity evaluation.

**Produces:** Two-pass/zero-pass selection within one head forward, teacher losses, and scalar output diagnostics.

- [ ] Write a spy-denoiser test for the feedback branch: there are exactly two evaluations; gradients are disabled for the first and enabled for the second; future/action noisy tensors, sigma, prefix inputs, and condition-drop masks are identical. First-pass predictions have no graph; second-pass conditions equal normalized detached first predictions.
- [ ] Write the zero-condition branch test: exactly one trainable evaluation, all teacher conditions zero, all three objectives computed. Test probability endpoints 0 and 1 and the explicit branch override. `prediction_only` must also use one evaluation but has no conditioners or feedback selection.
- [ ] Test target leakage directly: hold noise, sigma, prefix, weights, and branch fixed; alter only `teacher_future_tokens`. Model velocities/predictions/feedback must remain identical while teacher loss changes.
- [ ] Test exact scalar composition: `loss_teacher_<name>` is the raw masked cosine loss, `loss_teacher` is the sum of weighted teacher contributions, and total head loss equals baseline weighted flow loss plus `loss_teacher`. At sigma 1, where the baseline flow weight is zero, a valid teacher loss must still train the predictor.
- [ ] Test action-step/dimension validity, fill-coordinate supervision, fully invalid endpoints, masked NaNs, and separate per-modality reductions against the baseline. A teacher-invalid token must not remove otherwise valid action or image supervision.
- [ ] Test gradients: shared DiT, original input/output projections, state projection, and teacher predictor receive gradients; the conditioner receives gradients in a nonzero-feedback fixture; teacher parameters and first-pass tensors do not. In the zero branch, zero conditioner-weight gradient is expected because the condition is zero.
- [ ] Test optional trainable VL adapters: upstream observation/text encoding occurs once, the trainable second pass retains its gradient path, and the wrapper adds the language auxiliary objective once.
- [ ] Run `python -m pytest test/test_models/patchwam/test_teacher_forcing_training.py -q`; confirm the new behavior fails before implementation.
- [ ] Add a narrowly scoped evaluation helper after the existing flow-input preparation so the noisy states and masks are formed once. Keep the disabled flow path and Self-Flow options numerically unchanged; enabled teacher recipes reject those deferred variants.
- [ ] Implement synchronized branch selection on the active distributed group. All ranks consume the same number of model evaluations; expose a deterministic override for tests. Do not collect targets or tensors from other ranks.
- [ ] Compute teacher loss from the trainable prediction in float32, detach targets explicitly, and release look-ahead temporary outputs before the second pass. Keep ordinary model training mode unchanged.
- [ ] Make validation use the deterministic feedback branch, and rerun the new training tests plus existing PatchWAM core/parity and vision-language tests.

**Acceptance:** Flow random draws occur once per enabled update; disabled updates retain their previous draw order. Normal feedback training adds only one no-gradient model evaluation. No loss or gradient is attached to the discarded look-ahead velocities.

### Task 5: Carry predictions through the existing Euler sampler

**Files:** Modify `PatchFlowPolicy.sample_actions`; create `test_teacher_forcing_sampling.py`. VLA sampling validity/passthrough is completed in Task 6.

**Consumes:** Task 3 velocity/feature evaluation and Task 1 normalization helper.

**Produces:** One feature buffer per teacher per sampling branch, local to a single action-chunk generation.

- [ ] Write a deterministic recurrence test with fixed initial latents and known velocities/features. At evaluation 0 the condition is zero; evaluation `k+1` receives normalized prediction `k`. Feature conditions must not depend on sigma step size or be added to their preceding values.
- [ ] Verify both flow states use `x += (end-start)*v_x` / `u += (end-start)*v_u` under `ShiftedFlow.schedule`; the delta is negative. Assert generated action decoding and inverse-normalization ownership remain unchanged.
- [ ] Assert exactly `K` evaluations at both guidance scales 1, and `2K` when either scale differs from 1. Cover scales 0 as well as greater than 1 to preserve the existing API.
- [ ] Test separate conditional/unconditional feature buffers and the unchanged velocity guidance formula. Each branch receives its own previous normalized feature prediction; feature tensors are not guided.
- [ ] Test two consecutive calls, changed batch size, changed valid raster length, separate views, and reference history. Every call starts from zero feedback; buffer length comes from generated `S`, never `R`.
- [ ] Test zero predictions, half-precision normalization, invalid future slots, and padding NaNs. Changing provided teacher **target** tensors must not affect sampling with fixed initial noise.
- [ ] Install a provider/factory spy that raises on any teacher construction/encoding. Sampling through both the head and public `predict_action` must succeed without calling it; no engine path is required.
- [ ] Run `python -m pytest test/test_models/patchwam/test_teacher_forcing_sampling.py -q`; confirm the recurrence is absent before implementation.
- [ ] Add local buffer initialization/replacement to the existing loop, with explicit feature prediction requests. Keep conditions absent in disabled and prediction-only modes. Do not add a second training-style pass per solver step.
- [ ] Run the new tests and existing PatchWAM action-mode/parity tests.

**Acceptance:** Teacher-free sampling returns the existing normalized action API, follows the original flow sign, and resets feedback on replan. No future RGB decoder is introduced.

### Task 6: Integrate VLA configuration, ownership, and checkpoint rules

**Files:** Modify `patchwam_vla.py` and `patchwam_head.py`; create `test_teacher_forcing_checkpoint.py`. Confirm the FSDP runner's `_format_model_state_dict` preserves the signature buffer; change that formatter only if its current selection omits persistent head buffers, with a focused test.

**Consumes:** Tasks 1–5 contracts, head modules, provider service, and existing checkpoint classifiers/resolver.

**Produces:** `teacher_extractors`, `teacher_checkpoint_mode`, target-key passthrough, explicit runtime closure, representation-bound checkpoint validation, and initialization from existing PatchWAM assets.

- [ ] Write constructor tests rejecting visual-frontend replacement (even if it produces width 128), visual stages/LoRA adaptation, LIT heads, fused velocity transforms, and Self-Flow combinations. Validate head teacher widths/specifications before any optimizer or distributed wrapper is constructed.
- [ ] Write batch passthrough tests for raw and backbone-free encoded calls. Extend `_TOKEN_KEYS` / `_ENCODE_KEYS` with target mappings/fingerprints/layout and `_HEAD_KNOB_KEYS` with the scalar branch override. Nested target tensors stay detached and move to the flow device as needed; helper validation also checks runner-prepared dtypes.
- [ ] Write deployment construction tests with `teacher_extractors=None`: constructing, loading, and sampling must not read teacher files or import/initialize TensorRT. Training with complete cached targets must also avoid engine inference.
- [ ] Write initialization tests for released Action-as-Patch, native, compact, and bare baseline layouts. `base_init` permits missing **only** the new teacher projection/predictor keys and the new signature buffer; missing original DiT/state/action tensors, unexpected keys, wrong shapes, and nonfinite inherited tensors fail before weight copies.
- [ ] Write strict enabled-resume tests: all teacher branches and the matching signature must be present; missing one predictor, wrong-width branches, reordered/renamed teachers, or same-width/different-checkpoint teachers must fail. Verify a load failure leaves inherited parameters unchanged on the native initialization path.
- [ ] Test native `.pt` and tensor-only safetensors round-trips, plus runner-style `load_state_dict(strict=False)`: relaxed generic loading must not bypass enabled teacher contract checks. Confirm caches use the corresponding per-teacher signatures.
- [ ] Test that model parameters/state dict/optimizer groups contain the new branches but contain no teacher weights, engine buffers, temporary predictions, or feedback history. `all_module_keys` remains `['vlm_backbone','vla_head']`.
- [ ] Run `python -m pytest test/test_models/patchwam/test_teacher_forcing_checkpoint.py -q`; confirm the new initialization/resume rules fail before implementation.
- [ ] Store an aggregate SHA-256 signature as persistent tensor buffer `vla_head.teacher_contract_digest`, `uint8[32]`, only in enabled models. Hash representation fields and ordered teacher identities plus policy mode and readout/normalization conventions; scalar training knobs remain recorded in the recipe rather than masquerading as feature identity.
- [ ] Extend the existing exact `allow_missing` validation for `base_init` to accept precisely that one new nontrainable buffer in addition to enumerated new trainable parameters. Retain the rejection of any other frozen/inherited missing key. Validate all inherited tensors before `_apply_patchwam_state` mutates them.
- [ ] Add signature/missing-branch preflight checks to both the custom resolver and the registered model load pre-hook, so generic runner loads cannot evade contract validation. Resume requires the full head contract, regardless of current freeze flags. Exercise FSDP full-state restoration to prove hooks and buffer formatting remain effective.
- [ ] Build the provider service only from training extractor definitions, outside the registered module tree. Preserve `get_fsdp_wrapping_policy({PatchWAMHead})`; the new trainable modules live inside that unit. Add explicit `close_teacher_extractors()` for training/extraction lifecycle owners and tests.
- [ ] After backbone encoding, derive enabled sampling `future_valid` from current world raster validity before pooled history. If a pre-encoded caller supplies explicit future validity, validate and preserve it. Disabled sampling retains its previous masking path.
- [ ] Run new checkpoint tests and `test/test_models/patchwam/test_patchwam_checkpoint.py`, `test/test_models/patchwam/test_patchwam_action_patch_checkpoint.py`, and `test/test_models/test_base_vla_checkpoint.py`.

**Acceptance:** A baseline checkpoint can initialize new branches through an exact documented exception; a trained extension cannot silently resume with a different teacher. Teacher runtime resources are excluded from model checkpoint state. Both plain and distributed restoration validate the contract.

**Teacher swapping:** Start new branches against the new representation specification and initialize inherited FLUX/state weights from an appropriate baseline checkpoint. Do not reuse an old teacher's heads simply because widths match. Transferring the learned transformer from an extension while intentionally dropping its old branches needs an explicit transfer mode in later work; the initial strict loader must not guess that intent.

### Task 7: Add matched recipes and explain teacher replacement

**Files:** Create both planned config files, `test_patchwam_teacher_forcing_config.py`, and `docs/patchwam-teacher-forcing.md`; update `test/cpu-tests.txt`.

**Consumes:** All previous interfaces and the existing RobotWin dataset/transforms/runners.

**Produces:** A C-RADIOv4-H feedback recipe, a prediction-only ablation, and a deployment recipe with no extractor definitions.

- [ ] Write failing config tests for exactly matching train/inference teacher specifications, action dimensions/horizon, raster layout, normalization, solver steps, and checkpoint modes. Check that teacher widths/fields are not hidden in policy constants.
- [ ] Base the initial recipe on the existing RobotWin world data pipeline: `horizon=16`, `future_offset=16`, the compact `288x256` mosaic, existing world augmentation, and the baseline's text frontend. Keep the same data transforms/camera order/augmentation for the baseline and forcing experiments.
- [ ] Configure C-RADIO-H native spatial features, width 1280, exact `288x256` input, identity-or-explicit-bilinear grid alignment as defined above, loss weight 0.1, zero-condition probability 0.1, full trainable DiT/state/original projections, and gradient checkpointing. Validate the actual VAE grid instead of assuming spatial compression from the width 128.
- [ ] Require local paths via `PATCHWAM_TEACHER_ENGINE` / `PATCHWAM_TEACHER_METADATA` and concrete teacher provenance through representation ID, checkpoint revision/hash, and extractor artifact checksum. The current engine metadata cannot reconstruct missing checkpoint provenance; bind a reviewed engine artifact checksum to the recorded teacher identity when preparing assets. Do not replace missing fields with a guessed checkpoint name.
- [ ] Configure runtime `device=None` and a profile with `min_batch=1`; let the provider split larger valid-raster batches against the declared maximum. No TensorRT execution is done in dataset transforms/workers.
- [ ] Use the same initial PatchWAM checkpoint for the baseline and forcing experiments. If using the released C2R transformer/state checkpoint, retain its `pack_proprio_after_text=True` and compatible frontend/data conventions in **both** recipes; do not mix it with the full-finetune recipe's differing state packing without a matched control.
- [ ] Set the training VLA checkpoint mode to `base_init` and deployment mode to `resume`. Inference retains learned branches and representation specifications but removes `teacher_extractors` entirely. The prediction-only recipe changes only the feature-feedback architecture/training mode required for that ablation.
- [ ] Use `DDPTrainRunner(static_graph=False)` for the first enabled recipe while branch-dependent execution is being validated. The existing default is true; do not rely on it for the initial stochastic graph. Keep the existing optimizer/LR/precision/accumulation settings for matched comparisons, and log the resulting trainable parameter count.
- [ ] Explain how to select the existing PyTorch `RadioViTBackbone` backend instead, and how to register a different frozen spatial teacher. State that native RADIO and adaptor streams are different representations; the TensorRT example supports the exported native spatial stream only.
- [ ] Document current limits: independent rasters, exact geometry, required provenance, valid pre-encoded targets, no implicit cross-teacher weight reuse, and no automatic claim that more teachers improve control.
- [ ] Add new CPU tests to the existing manifest using actual file paths, and keep gated asset/distributed checks explicit. Run `python -m pytest test/test_configs/test_patchwam_teacher_forcing_config.py -q` and the newly listed CPU tests.

**Acceptance:** The recipe is concrete and reviewable, uses the provided RADIO backend, preserves a matched baseline, and exposes teacher replacement without changing policy code. Constructing inference configuration does not require teacher artifacts.

### Task 8: Verify distributed execution and the optional multiple-teacher path

**Files:** Create `test_teacher_forcing_distributed.py`; extend the earlier teacher tests for two teachers. Update the new documentation with validation results when these checks are actually run.

**Consumes:** The complete implementation and existing DDP/FSDP runner setup.

**Produces:** Evidence that the single-teacher path works in real training infrastructure, and that configurable independent teacher branches can coexist without extra model evaluations.

- [ ] Use two fake spatial teachers with different widths and native grids. Assert their targets align independently, their residuals add, both heads read the same final future states, their losses sum with their own weights, and their feedback buffers remain separate. The 90% branch still makes two total model evaluations, not two per teacher.
- [ ] Test one missing/mismatched teacher target in a multi-teacher batch. Reject it with the specific name; do not silently drop a configured teacher. Reject a partial enabled checkpoint resume.
- [ ] Add a small two-rank distributed test forcing conflicting local branch proposals, then verify broadcast produces the same branch/pass count. Cover alternating zero/two-pass updates and gradient accumulation. Use CPU/Gloo for the selector and small DDP model where possible; use gated GPU tests for real FSDP.
- [ ] With local GPUs, run a short tiny-model DDP training/save/reload sequence using `static_graph=False`; verify every new parameter participates as expected and rank gradients agree. Repeat with `PatchWAMHead` as the FSDP unit, gradient checkpointing, and BF16; confirm no mismatched collectives, unsharded child calls, or missing signature buffers.
- [ ] Exercise both freeze/load setup orders: DDP freezes before initialization; FSDP loads before freezing. Both must leave exactly the specified trainable head parameters open and teachers frozen outside the optimizer.
- [ ] Run a gated one-batch real FLUX/RADIO integration using the selected local engine: verify reported native grid/width, alignment to the VAE raster, finite losses, finite gradients, and one optimizer update. Use existing offline C-RADIO inspection/parity tooling to validate the chosen engine separately; do not rebuild assets as an incidental test step.
- [ ] Repeat deployment loading/sampling with the teacher artifact paths unavailable and a teacher factory that raises if invoked. Success requires the original observation VAE/text assets and learned head weights only.
- [ ] Run the focused regression commands in section 8 once after the final changes. Record exact passing/skipped/blocked checks; do not label unavailable GPU/assets as verified.

**Acceptance:** A real configuration switch changes the teacher backend/representation while preserving policy mechanics. Independent multiple-teacher support is functional but remains an optional experiment. Joint geometry-teacher adapters are deferred and rejected until their spatial/context contracts are implemented and tested.

## 8. Verification commands and release criteria

The commands below are planned verification commands for the project's configured Python environment; they have **not** been run for this planning task. New tests use CPU fixtures and injected engines unless explicitly gated.

| Scope | Command | Required result |
|---|---|---|
| New numerical/provider/training/sampling/checkpoint tests | `python -m pytest test/test_models/patchwam/test_teacher_feature_contract.py test/test_models/patchwam/test_teacher_targets.py test/test_models/patchwam/test_teacher_forcing_denoiser.py test/test_models/patchwam/test_teacher_forcing_training.py test/test_models/patchwam/test_teacher_forcing_sampling.py test/test_models/patchwam/test_teacher_forcing_checkpoint.py -q` | All CPU tests pass; optional local official-source skips are identified |
| Recipe contract | `python -m pytest test/test_configs/test_patchwam_teacher_forcing_config.py -q` | Pass, including teacher-free inference construction |
| Existing PatchWAM behavior | `python -m pytest test/test_models/patchwam test/test_models/test_patchwam.py test/test_configs/test_patchwam_config.py test/test_transforms/test_patchwam_transforms.py test/test_datasets/test_patchwam_dataset.py -q` | No regression in default flow, codec, loader, mask, or data pipeline |
| Existing C-RADIO contract | `python -m pytest test/test_models/test_radio_trt_backbone.py test/test_models/test_radio_trt_engine.py test/test_models/test_radio_trt_offline.py test/test_configs/test_radio_trt_config.py -q` | CPU contracts pass; any real-runtime checks remain explicitly gated |
| Distributed path | `python -m pytest test/test_models/patchwam/test_teacher_forcing_distributed.py -q` | CPU selector/DDP checks pass; enabled GPU/FSDP checks pass on provisioned local hardware |

Do not run expensive external benchmarks, training jobs, or dependency setup as part of ordinary unit verification. Run the gated local integration only when assets/hardware are available and that execution is authorized. Do not announce deployment or improved success based solely on unit tests.

Completion criteria for the implementation:

- Disabled teacher forcing preserves inherited parameter keys, projections, codec, flow weighting/sign, random draws, masking, and public normalized-action output.
- Enabled zero-conditioning weights preserve baseline velocity behavior on identical fixed inputs.
- Teacher target alignment is verified for mosaics, multiple independent views, rectangular grids, masks, and history.
- The training branches, detachment, loss reduction/weighting, gradient destinations, and absence of target leakage are tested.
- Feedback is normalized replacement, with one evaluation per unguided Euler step and complete reset per chunk.
- Pretrained initialization and enabled resume are explicit and representation-bound, including same-width teacher mismatch rejection.
- Teachers are frozen training resources, absent from model checkpoint/optimizer state and deployment construction/execution.
- At least the CPU/mocked contracts pass. Real FLUX, TensorRT, DDP/FSDP, and robot results are reported only to the extent actually verified.

## 9. Experiment sequence and costs

Implementation correctness does not establish a policy improvement. After the integration checks, run these experiments in dependency order, using identical data splits, initial inherited weights, augmentation, action horizon, solver steps, benchmark action-prefix execution, and evaluation seeds:

1. **Matched PatchWAM baseline:** extension disabled, same actual initialization/frontend/state packing and training budget.
2. **C-RADIO prediction only:** linear feature head and loss, no conditioning projection or feedback. This isolates representation supervision.
3. **C-RADIO feedback:** full proposed architecture, epsilon `1e-6`, zero-condition probability `0.1`, initial lambda `0.1`.
4. **Weight sweep:** lambda `0.05`, `0.1`, `0.3`; retain image/action weights `0.5` / `1.0`. Diagnose zero versus feedback performance with held-out fixed-noise evaluations.
5. **Teacher/backend comparisons:** only after a new adapter satisfies the same contract. For equivalent RADIO PyTorch/TensorRT targets, measure extraction parity as well as runtime. Different teacher identities require separately trained new branches.
6. **Optional multiple teachers:** test complementary information with separate loss weights; do not assume adding teachers is beneficial.
7. **Optional reduced image-loss weight:** a later controlled ablation after stabilization. Keep positive supervision for every integrated flow state; do not remove the image loss while continuing to evolve image latents.

Report manipulation success with the benchmark's clean and available perturbation conditions, wall-clock training/extraction time, update count, inference latency at `K=10`, peak training/deployment memory, and masked teacher alignment. If new perturbation suites are needed, treat them as separate evaluation work. Feature similarity alone is not the primary outcome.

For one RADIO-H branch with `d=3072`, the two linear maps add about **7.87 million** trainable parameters. A BF16 feedback buffer at `B=4`, `S=288`, `d_T=1280` occupies about **2.81 MiB** per branch; actual `S` comes from the VAE, and additional targets, predictions, optimizer states, and look-ahead workspace cost more. Unguided sampling adds no transformer evaluation, but the projections/readouts add compute. Standard training uses about 1.9 model evaluations per microbatch under the proposed 90%/10% mixture; its runtime is not 1.9 times ordinary training because the look-ahead has no backward pass and teacher extraction has its own cost.

Measure those costs with the actual engine profile, raster geometry, and precision. Do not claim speedup or reduced memory from the architecture alone.

## 10. Deferred work and planning review

Deferred: new DINOv2/VGGT adapters, joint multi-view geometry extraction, teacher feature noise/ODEs, feature-only world modeling, object-weighted alignment, magnitude-sensitive losses, current-prefix Spatial Forcing, feature-cache creation tooling, cross-teacher checkpoint transfer, solver changes, and staged/LoRA/LIT combinations. The configurable independent spatial interface is implemented first; these extensions need their own contracts and evidence.

Planning review against the proposal:

| Proposal requirement | Owning tasks |
|---|---|
| Preserve shared pretrained image/action flow, codec, attention, positions | 3–5, regression checks in 8 |
| Configurable frozen spatial teacher, C-RADIO example, widths and geometry | 1–2, 6–7 |
| Zero-initialized linear future-only fusion; random linear predictor | 3 |
| Detached look-ahead and single-pass zero branch | 4 |
| Independently weighted patchwise cosine objective and original masks | 1, 4 |
| Normalized next-evaluation feedback; no feature ODE/teacher deployment | 1, 5–6, 8 |
| Optional independent multiple-teacher heads/losses/buffers | 1–5, 8 |
| Strict initialization/resume and teacher identity | 1–2, 6 |
| Matched validation/ablations and cost accounting | 7–9 |

The inspected implementations were used to resolve extraction location, future raster packing, sigma sign, actual RADIO dimensions, wrapper key filtering, FSDP ownership, and strict checkpoint initialization. No model accuracy, runtime parity, or policy-success claim is made by this plan.
