# PatchWAM Current-Feature Conditioning and Future-Feature Forcing Implementation Plan

> **For agentic workers:** The user authorizes implementation. During execution, follow AGENTS.md: load astra-flash-orchestrator, retain architecture/acceptance with the root, and dispatch one coherent phase bundle to the native astra_flash_builder. Preserve unrelated edits. Do not auto-commit, push, deploy, or run paid setup smoke tests.

**Goal:** Add configurable frozen current-image features to PatchWAM observation tokens and supervise predicted future features that feed back between flow evaluations, using the existing C-RADIOv4 PyTorch/TensorRT backbones first.

**Architecture:** Preserve the VAE's 128-channel flow states and shared pretrained image/action projections. Each encoder has a current projection M, future-feedback projection G, and configurable prediction head J. Frozen extraction serves current observations during training/inference and future endpoints during training; G uses predicted future features, with the agreed 5% true-feature substitution during training only.

**Tech stack:** FluxVLA registries/runners, PyTorch, native PatchWAM/FLUX adapters, FrozenVisionFeatures / SpatialFeatureOutput, and the optional existing C-RADIO TensorRT runtime. No dependency on DINO Forcing's Lightning/Hydra stack.

**Spec:** [PatchWAM_DINO_Forcing_Design_Update.md](../../../papers/PatchWAM_DINO_Forcing_Design_Update.md), revision 2026-10-16, plus the user's review clarification: **use the source three-linear-layer MLP as J's default; retain Linear and two-linear-layer MLP options.** This supersedes the proposal's two-layer-only enumeration and its MLP parameter formula. The proposal itself has not been edited.

**Status:** Implemented 2026-10-17 under the inheritance/source-alignment
contract below. The new head/VLA, provider, recipe, tests, docs, real C-RADIO
geometry check, and one-batch real FLUX/RADIO forward-backward-update plus
action sample are recorded in
`docs/agent-work/patchwam-teacher-forcing/REPORT.md`. Existing PatchWAM recipes
remain unchanged, and no manipulation improvement is claimed.

## Implementation contract update: inheritance and source alignment

This section supersedes conflicting earlier task details below. The user now authorizes implementation and requests minimal changes to baseline PatchWAM. Existing fusion agreement remains authoritative. Reconcile the rest of this document during implementation rather than preserving contradictory requirements.

### Architecture and ownership

- Create dedicated `TeacherForcingPatchWAMVLA(PatchWAMVLA)` and registered `TeacherForcingPatchWAMHead(PatchWAMHead)` in new VLA/head modules. Put teacher policy behavior in a dedicated subclass or reusable mixin under `third_party_models/patchwam/teacher_forcing/`; choose a simple MRO that constructs the inherited head exactly once. Do not add teacher-specific branching throughout baseline classes.
- Prefer overriding existing `_prepare_batch`, head execution and loading extension points. Reuse action codec, flow preparation/losses, mask/position construction, language loss ownership, checkpoint machinery and registry factories.
- When a long baseline method has no adequate seam, extract a small general-purpose method/hook and invoke it through polymorphism. Neutral defaults must preserve baseline outputs, state-dict names, RNG ordering and public interfaces. No class-name checks, runtime monkeypatching, temporary module swapping, global mutable forward-hook state, or copied long forward/sampling methods.
- Denoiser input preparation may expose general projected-prefix/residual inputs and representation capture; additions are hidden-width tensors with explicit shapes. This is a reusable API, not a teacher-specific flag. Support tiny and official FLUX paths, preserve shared img_in and final velocity head, and reuse existing capture.
- Frozen teacher extraction lives in a dedicated configurable provider outside the registered trainable tree. Select the backbone by registry `type` and constructor config, not hard-coded RADIO dispatch. Use existing RadioViTBackboneTensorRT as one example; allow RadioViTBackbone and future adapters satisfying the spatial-feature contract. No new DINO adapter is required now.
- New code belongs in new teacher-forcing policy/helper, head, VLA, feature-provider, configuration and test files. Baseline edits are limited to documented general seams, geometry metadata if needed, and registry imports. No runner changes unless a focused integration failure proves one necessary.

### Algorithm and initialization

- M (current) and G (future) are independent bias-free residual projections into the DiT hidden width (example 3072, not VAE width 128), zero initialized by default with independent scaled-Xavier options. Preserve pretrained P/bias; no extra A. Normalize feature inputs with non-affine LayerNorm. J defaults to the source three-linear-layer SiLU MLP, hidden width 2048; Linear/MLP2 remain configurable.
- Frozen current descriptors condition only current raster slots; G conditions only future raster slots. No new tokens; history/action/text positions retain their input semantics. Shared trainable prefix/current projections must receive gradients from the trainable pass.
- Training always runs a detached zero-feedback look-ahead when feedback is enabled, followed by a trainable pass on the same noisy image/action states, sigma, conditioning and masks. Use local mode management if necessary to disable stochastic layers for look-ahead; restore prior states reliably and never toggle the whole VLA/global model. Preserve distributed ownership and checkpointed backward.
- Take raw detached predictions from look-ahead. Independently per sample, substitute detached true future teacher descriptors with probability 0.05 during training, per the user's explicit choice. Then apply non-affine LayerNorm and per-sample 0.1 feature dropout in the trainable pass, matching source ordering. Dropout applies to substituted features too. Always execute two passes; do not use synchronized microbatch branch selection or skip look-ahead for the dropped samples. Invalid target slots stay masked. Current features never use future frames.
- Validation uses deterministic predicted feedback without true substitution/dropout. Inference carries raw detached predictions, starting from zero, with separate guided buffers and reset per action chunk. Normalize only at fusion, and L2-normalize only for cosine loss. No future extraction or true-feature substitution at inference.
- Loss is unchanged PatchWAM flow/action loss plus separately weighted masked feature cosine loss. Feature-loss reduction is valid-token mean per sample then batch mean; all-invalid samples contribute zero. Training target dependence is permitted only through loss and the explicit true-feature substitution branch. Test no target dependence with substitution disabled; test controlled dependence with substitution enabled.
- Native grid, field-of-view, view ordering and feature provenance must match. Reject silent interpolation/resizing and unsupported variants. Current extraction is once per observation context outside the solver; future extraction is training-only in a separate temporal context. Full finetuning/VAE path initially; reject incompatible legacy fused/Self-Flow/staged modes clearly.

### Configuration, checkpoint and verification contracts

- Registry-selected new VLA/head enable the feature; existing PatchWAM recipes remain untouched. Supply a runnable derived RobotWin config with both train/inference class selection, configurable teacher resources, local asset paths, exact geometry and explicit constructor keys. Resolve through normal registry/config discovery.
- Use `true_feature_prob=0.05` and `self_cond_drop_prob=0.1`, validate finite [0,1] values; switches, predictor choice, normalization and teacher identity enter checkpoint compatibility. Initializer settings are provenance, not a reason to reject restoring learned maps.
- Explicit base_init only permits missing new branch tensors/digest; resume requires complete compatible branches and representation identity, including strict=False callers. Never reinitialize restored tensors. Keep frozen encoders, temporary prefixes and solver feedback out of policy state/optimizer.
- Validate meaningful CPU forward/backward, baseline parity with zero maps, nonzero M/G gradients under nonzero features, J supervision, true substitution/dropout order and frequency endpoints, masking/all-invalid cases, geometry rejection, current/future isolation, extraction/call counts, guided sampling/reset, registry recipe construction and checkpoint roundtrip/mismatch failures. Test actual new paths rather than duplicating calculations in mocks.
- Exercise RADIO TensorRT with available local matching assets and compare the PyTorch provider where available. Inspect availability without downloading/exporting/building engines; report unavailable GPU/runtime/geometry as blocked checks, not passes. Run relevant PatchWAM regression tests once after focused tests pass. No long training, installation, network downloads, commits or pushes.
- Feedback training costs two denoiser evaluations per microbatch (one without gradients). Per-sample stochastic decisions need no cross-rank branch synchronization because all ranks execute the same modules/passes.

### Execution bundle

One native Flash writer owns discovery, implementation, focused testing/debugging and reconciliation of this plan within the scope above. Preserve the pre-existing plan edits captured in `/tmp/patchwam-teacher-forcing-baseline/`. Root owns architecture decisions and acceptance. Write a completion/evidence report and resumable checkpoint under `docs/agent-work/patchwam-teacher-forcing/`; record changed baseline seams with reasons, exact checks/results, unavailable assets, and remaining tasks. Source review paths and the detailed requirements below remain useful except where superseded here.

### Implementation disposition (2026-10-17)

- The default teacher runtime is PyTorch `RadioViTBackbone` from `radio_vit.py`, which supports configurable model-compatible input resolutions. The verified 288x256 recipe needs no TensorRT export; other resolutions require matching data/VAE geometry, teacher input size and native grid. The unavailable 288x256 engine limits only the optional TensorRT backend.

- GPT-6.1 Sol performed final review after the user-selected root switch; Flash performed implementation. Final acceptance checks and limitations are recorded in the report.
- Implemented the teacher-forcing package, registered head/VLA subclasses,
  lazy configurable provider, derived RobotWin recipe, docs, and focused tests.
- Verified the real local C-RADIOv4-H PyTorch export at 288x256 produces an
  18x16x1280 grid, and ran one full VLA batch through real Qwen3/VAE/RADIO/FLUX:
  finite losses, nonzero M/G/J gradients, one SGD update, and a 16x14 action
  sample. Peak allocated GPU memory was 23.6 GB.
- Root correction review added strict layout/ID/future-length validation,
  detached teacher descriptors, complete checkpoint preflight for direct,
  resume, parent-recursive, and `strict=False` load paths, exact descendant
  look-ahead mode restoration, and removal of teacher keys from baseline
  `_TOKEN_KEYS`.
- The provider now returns all-invalid zeros without constructing a runtime,
  balances calls within metadata batch profiles, verifies actual local
  provenance hashes once, and binds representation artifacts to the encoder
  identity.
- A compatible local 256x256 PyTorch/TensorRT provider comparison passed:
  relative L2 `0.015577`, mean cosine `0.999880`.
- The local TensorRT engines are 256x256 and 512x512. They are correctly
  rejected for the 288x256 recipe; no 288x256 engine was available to execute.
- Multi-rank DDP/FSDP execution was not run. The implementation uses the same
  two-pass module structure on every rank and has a focused multiple-encoder /
  per-sample-mask test; see the completion report for the remaining hardware
  checks.

## Global constraints

- Preserve the VAE frontend, 128-channel image/action states, fixed codec, pretrained projection sharing, attention/positions, and action normalization ownership.
- Retain shift=5, video_weight=0.5, action_weight=1.0, and the selected recipe's flow-weight normalization. Initial matched runs use horizon 16 and 10 Euler steps.
- Add M only at current-image positions and G only at future-image positions, after the original input projection. Add no tokens or extra fusion gate; action tokens bypass both additions.
- M/G are separate bias-free linear maps, zero-initialized by default. Each supports configurable scaled-Xavier initialization as an alternative. Preserve the existing pretrained shared input projection and its bias; add no image-side A matrix or extra fusion bias. Never reinitialize pretrained modules.
- J defaults to Linear(d,2048) → SiLU → Linear(2048,2048) → SiLU → Linear(2048,d_f), with biases, Xavier-uniform weights, zero biases, and no internal normalization. Linear and two-linear-layer alternatives are explicit options.
- J reads final-block future hidden states before final normalization/adaLN velocity readout. Keep the original velocity head.
- Follow the DINO Forcing example's non-affine LayerNorm on feature inputs to fusion. Carry detached raw predictions between solver steps; use L2 normalization only inside the cosine objective. Zero feature input remains zero after LayerNorm.
- Require matching native grids, field of view, and view/patch order. No extracted-feature interpolation, pooling, or token-count-only alignment.
- Current/future encoder contexts remain separate. Encoders stay frozen/eval/no-grad; M/G/J and inherited trainable modules retain gradients.
- Feedback training always runs a detached, eval-mode look-ahead, followed by a trainable evaluation on the same noisy states. During the trainable pass, drop predicted future-feature conditioning independently for 10% of samples, after input LayerNorm and before fusion. Current conditioning remains active.
- Future teacher outputs supervise J and may replace predicted feedback for 5% of training samples before LayerNorm/dropout. Validation and inference use predicted feedback only.
- Feature loss is separately weighted masked cosine, outside w(sigma), initially weighted 0.1 per encoder.
- Combined deployment requires current features from an encoder or validated external features. It requires no future RGB/extraction.
- Use local assets; no incidental downloads, installation, engine builds, or edits to the DINO Forcing checkout.
- Initially support ordinary VAE PatchWAM with full head finetuning. Reject visual replacement, staged adaptation, LIT, fused RADIO velocity transforms, and EMA/Self-Flow combinations. Keep disabled paths intact.

## Review focus

1. Equal token counts with wrong geometry: test rectangular grids, mosaics, view order, and implicit RADIO resizing in Task 2.
2. Future leakage or detached M: test separate extraction contexts and prefix gradient reuse in Tasks 2 and 4.
3. Missing/stale deployment current features: test required inputs, cache contracts, and representation identity in Tasks 5–6.
4. Wrong checkpoint architecture despite matching widths: test M/G/J completeness, predictor depth, and preflight validation in Task 6.
5. Distributed divergence and invalid data: test consistent two-pass participation, engine profiles, padded NaNs, all-invalid rasters, and checkpointed backward in Tasks 2, 4, and 8.

## 1. Source review and resolved interpretation

### 1.1 PatchWAM

| Finding | Code and implementation decision |
|---|---|
| D_cur/D_fut/D_act share one module in the VAE path | OfficialFluxDenoiser.forward calls core.img_in on concatenated reference/future/action tokens. Preserve this module and checkpoint keys; the diagram's labels do not create three projections. |
| Future/action velocities share the output head | The same forward applies core.final_layer to generated states. J branches from raw final-block states before this call. |
| R is not literally a direct state-to-d map | PatchFlowPolicy.state_projection maps state to text_dim with bias; _context packs it into text, then core.txt_in maps to hidden width. Preserve this composition and pack_proprio_after_text. |
| The logical prefix mask matches | joint_visibility allows prefix→prefix and generated→all groups. Keep padding masks. The initial recipe uses ordinary joint attention, not isolate_actions. |
| Proposed flow direction is correct | ShiftedFlow.perturb returns noise-clean; schedule descends 1→0 and Euler uses negative end-start. Existing modality losses already include w(sigma). |
| AP's physical action is an end-to-end output | PatchWAMVLA.predict_action returns normalized actions. Existing runner/decoder denormalizes once; do not move inverse normalization into the policy. |
| Current raster length equals future length, not necessarily reference length | encoding.py appends pooled history. Define S_current=S_future=S and R=S+S_history. M applies only to first S current raster positions; history retains baseline VAE conditioning. Initial RobotWin recipe disables history. |

Sources: [VLA](../../../fluxvla/models/vlas/patchwam_vla.py), [head](../../../fluxvla/models/heads/patchwam_head.py), [encoding](../../../fluxvla/models/third_party_models/patchwam/encoding.py), [policy](../../../fluxvla/models/third_party_models/patchwam/policy.py), [FLUX](../../../fluxvla/models/third_party_models/patchwam/flux.py), [tiny](../../../fluxvla/models/third_party_models/patchwam/tiny.py), [flow](../../../fluxvla/models/third_party_models/patchwam/flow.py), [geometry](../../../fluxvla/models/third_party_models/patchwam/geometry.py).

### 1.2 DINO Forcing

Reviewed local revision: 1ce45b94abc79fd38716bd1a6e2a38e0536e9d81. Sources: [SiT](../../../../dino_forcing/src/sit_class_cond_network.py), [JiT](../../../../dino_forcing/src/jit_class_cond_network.py), [training](../../../../dino_forcing/src/training_module.py), [sampler](../../../../dino_forcing/src/sampler.py), [SiT example config](../../../../dino_forcing/src/configs/SiT_B_dinoforcing.yaml).

| Source behavior | Plan decision |
|---|---|
| Both RepaProjectionBlock definitions have three linears and two SiLUs | User confirmed this as J's default. Linear/MLP2 remain ablations. |
| Constructor fallback width is 4*embed_dim; example configs explicitly use 2048 | Document 2048 as our initial default following the example configs, configurable. Do not call it the source constructor fallback. |
| Linear initialization is Xavier-uniform weights and zero biases | Apply to J. M/G use the agreed zero default or scaled-Xavier alternative, an intentional initialization adaptation. Never invoke source whole-network initialization on pretrained FLUX. |
| SiT concatenates features before its input linear; JiT uses a concatenation fusion linear | Concatenation plus a linear equals separate projections plus addition with matched biases. Reuse the trainable image projection and add M/G residuals; see Section 3.1 for parameterization and sharing caveats. |
| Configs enable non-affine input LayerNorm; sampler carries raw predictions | Adopt this behavior for future feedback. Use L2 normalization only in the cosine loss. |
| Training always runs look-ahead, then network drops features per sample | Adopt two passes on every update and a per-sample 10% future-condition dropout in the trainable pass. |
| Configs optionally substitute true features with probability 0.05 | Adopt per sample in training only: substitute detached true future descriptors before LayerNorm and feature dropout. Ground-truth futures never enter current conditioning. |
| Alignment is negative cosine | Use masked 1-cosine: matching gradients for the same token selection, different scalar offset. |
| Prediction is from a selected block, default final, before output conditioning | Use raw final future states in both PatchWAM denoisers. |

### 1.3 Native-grid RADIO feasibility

The local official VAE source at /home/chuong/workspace/RoboDojo/XPolicyLab/policy/PatchWAM/PatchWAM/vendor/flux2/src/flux2/autoencoder.py has three spatial downsampling stages and 2×2 packing of 32 channels. Compatible inputs therefore have stride-16, 128-wide tokens: 288×256 gives 18×16=288 tokens. Patch-16 RADIO can match this **if its selected model accepts the exact input geometry**. Verify actual outputs/metadata during implementation; this review did not execute the assets.

RadioViTBackboneTensorRT accepts only its sidecar geometry. RadioViTBackbone may internally resample or pad via _nearest_supported_resolution/_prepare_input. The provider must reject implicit changes for the initial exact-input recipe. A 256-square engine cannot serve a 288×256 mosaic unchanged.

Use native spatial_features, excluding summary/adaptor streams; RADIO-H width is 1280, SO400M 1152. Both wrappers receive world RGB in [-1,1] and convert internally to [0,1]. normalize_spatial_tokens changes layout, not vector norms.

Other encoders are configurable only when declared preprocessing and native output satisfy the spatial contract. Model-name substitution alone cannot fix a different patch stride. Any encoder-specific RGB resolution/preprocessing must be explicit, preserve the field of view, and be validated by its adapter; never resample extracted features. Initial implementation supplies existing RADIO backends, not new DINO/VGGT adapters. Joint-view encoders require a separate context-aware adapter and are rejected by the independent-raster provider.

## 2. Configuration and interfaces

### 2.1 Policy configuration and resource ownership

Add `teacher_forcing` to the registered `TeacherForcingPatchWAMHead`, a
subclass of `PatchWAMHead`. Teacher behavior comes from a policy mixin under
`third_party_models/patchwam/teacher_forcing/`; baseline head/VLA keys,
execution, and random draws remain unchanged. The three independent switches
are:

| Setting | Combined default / rule |
|---|---|
| current_conditioning | True: create M, require current descriptors |
| future_prediction | True: create J, supervise future descriptors |
| future_feedback | True: create G; requires future_prediction |
| predictor.type | repa_mlp; alternatives linear, mlp2 |
| predictor.hidden_dim | 2048 for MLPs; positive; omitted for Linear |
| self_cond_drop_prob | 0.1, finite in [0,1]; independent per sample in the trainable feedback pass |
| current_projector_init / feedback_projector_init | zero default; scaled_xavier_uniform alternative, independently for M/G |
| current_projector_init_scale / feedback_projector_init_scale | Required finite positive multiplier in scaled mode; select experimentally and record |
| feature_input_norm / cosine_eps | non-affine LayerNorm / 1e-6 for cosine computation |
| Ordered encoder specs | Nonempty; unique stable names |
| Spec name | e.g. radio; letters/digits/underscores, starts with a letter |
| feature_dim / loss_weight | RADIO-H 1280 / 0.1; positive width, finite nonnegative weight |
| Representation metadata | Checkpoint revision/hash, stream/layer, preprocessing, input size, native grid, view context/order, special-token policy |
| alignment | native_exact only |
| view_context | independent_rasters; a mosaic is one raster |
| teacher_checkpoint_mode | Explicit base_init or resume when enabled |

MLP2 is Linear(d,m) → SiLU → Linear(m,d_f); Linear is Linear(d,d_f). Both use biases and J's Xavier/zero-bias convention. Predictor type/depth/width enter checkpoint architecture explicitly.

Add teacher_extractors to PatchWAMVLA: name→runtime definition. Combined train **and inference** recipes provide runtime definitions unless validated current features are supplied externally. The TRT example uses RadioViTBackboneTensorRT, exact variant, local engine/sidecar, artifact checksum, and device=None for rank-local placement. PyTorch uses existing RadioViTBackbone with matching local checkpoint and representation.

Keep encoder resources in a plain TeacherFeatureProvider outside the registered model tree. It explicitly manages loading, freezing, device, checksums, and closure. Excluding frozen weights from policy checkpoints does **not** remove the inference dependency. No GPU encoders in dataset workers. Trainable M/G/J live under vla_head within its existing FSDP unit and are created before optimizer/wrapping.

### 2.2 Planned helper contracts

| Interface | Contract |
|---|---|
| TeacherFeatureSpec | Immutable validated representation metadata, width, native geometry, loss weight |
| TeacherForcingConfig | Ordered specs, three switches, predictor, input normalization, per-sample dropout probability |
| teacher_representation_fingerprint(spec) → bytes | SHA-256 of canonical encoder identity; independent of runtime path/device and predictor architecture |
| prepare_teacher_fusion_input(tokens, valid, *, norm) → Tensor | [B,S,d_f]; sanitize invalid rows, non-affine LayerNorm on valid rows, original dtype; zero prediction allowed |
| masked_teacher_cosine_loss(prediction, target, valid, *, eps) → Tensor | FP32 valid-token mean per sample, then batch mean; all-invalid sample contributes zero |
| TeacherFeatureProvider(specs, extractor_configs) | Plain resource service using build_vision_backbone_from_cfg |
| provider.validate_static_contracts() | Validate configured geometry/provenance/profile without loading TRT runtime |
| provider.encode_context(frames, *, role, raster_layout, raster_valid) → mapping | [B,V,3,H,W], role current/future; separate contexts; detached descriptors/validity/fingerprints |
| provider.close() | Release owned resources only |
| PatchFlowPolicy.teacher_new_parameter_names() → list[str] | Exact new M/G/J initialization allow-list |
| PatchFlowPolicy.teacher_checkpoint_contract() → mapping | Representations, switches, predictor architecture, readout/normalization |

### 2.3 Encoded-batch contract

| Key | Meaning |
|---|---|
| teacher_current_tokens / teacher_future_tokens | name→[B,S,d_f], detached observed conditions / future targets |
| teacher_current_valid / teacher_future_valid | name→boolean [B,S], intersect corresponding VAE raster validity |
| teacher_current_fingerprints / teacher_future_fingerprints | name→uint8 [B,32], same encoder identity but separate role containers |
| current_raster_layout / future_raster_layout | Views, RGB size, VAE grid, view-major/row-major order |
| current_raster_length | S before appended history; validate against layout/reference IDs |

Preserve existing token/mask/ID keys and sampling_future_length. Extend _TOKEN_KEYS/_ENCODE_KEYS. Do not reuse radio_future_tokens as a generic alias; keep the existing RADIO auxiliary path unchanged.

Current descriptors are required whenever M is enabled, including pre-encoded inference. Future targets are required only when computing supervised prediction loss. Missing descriptors can be extracted only from their actual RGB plus explicit layout; otherwise fail naming the role/encoder. Never decode VAE latents to fabricate teacher inputs or silently replace missing valid current descriptors with zeros.

Validate names, batch, width, grid, order, IDs, mask, and fingerprint. Valid observed/target vectors must be finite with norm above epsilon; invalid rows are sanitized. Predictions may be zero. Cache callers own image identity and augmentation correspondence: representation fingerprints cannot prove descriptors came from this observation. Online extraction from the actual augmented raster is default; persistent cache-generation tooling is deferred.

## 3. Model and execution details

### 3.1 Fusion and prefix preparation

Each encoder owns the enabled members of teacher_current_projectors (M), teacher_conditioners (G), and teacher_predictors (J). Read d from denoiser.representation_dim; allocate modules on the existing parameter device/dtype.

Current residual is sum_q M_q(normalize(f_current_q)). Add at projected-reference positions [0:S]; extend with zeros over history [S:R]. Future residual is sum_q G_q(f_condition_q), added at image-stream positions [R:R+S]. Tiny sequence indices additionally include context length. Text/state/action projected inputs and positions/masks stay original. Final action outputs may change through attention.

#### Fusion agreement (2026-10-17)

Use `y = P(x) + M(LN(f))` for current tokens and G in place of M for future feedback. P retains the existing pretrained affine input projection and its bias exactly once. D is the DiT hidden width (3072 in the example), not the VAE state width (128). M/G map teacher width d_f to D. LN is non-affine LayerNorm. There is no extra nonlinear activation or learned fusion gate.

`W[x1; x2] + b = W1*x1 + W2*x2 + b`. JiT first projects image tokens, giving `A*P(x) + B*LN(f) + b`. An unconstrained trainable affine P can absorb A and the bias, so the extra D-by-D A is unnecessary for this tokenwise mapping's representational capacity. Optimization dynamics may differ. Absorption does not generally hold across nonlinearities or constrained projections. PatchWAM shares P across current, future and action tokens: preserve this sharing and do not claim whole-network equivalence to independently parameterized stream-specific fusion.

Default M/G initialization is exactly zero, preserving pretrained outputs at initialization. For z=LN(f), `dL/dM = (dL/dy) z^T` can be nonzero even at M=0. Gradients into z through this branch are initially zero; this is appropriate for frozen encoders and detached feedback. J still learns through its direct feature loss. Zero or masked feedback can produce zero projector gradients; gradient tests must use nonzero valid descriptors and a nondegenerate loss.

Optional small-random initialization uses `scale * XavierUniform(shape)` independently for M/G. Require a finite positive scale in this mode; choose it experimentally rather than claiming a universal value. This permits immediate upstream gradients if a trainable encoder is introduced later, but perturbs initial outputs. Log `||residual|| / (||P(x)|| + eps)` on valid current/future tokens with representative nonzero features, and initial velocity deviation. There is no extra trainable scale.

Initialize only new modules during base_init. Resume restores learned maps and never reapplies initialization. Record mode/scale as training provenance; they do not change module shapes or teacher identity and should not invalidate an otherwise compatible resume.

Define PreparedDenoiserPrefix with reference_hidden:[B,R,d] and context_hidden:[B,Lc,d]. Add prepare_prefix(reference, context, *, current_hidden_residual=None) using existing img_in/txt_in (tiny equivalents), with current addition only. Add optional prepared_prefix and future_hidden_residual to denoiser.forward. The enabled prepared path projects noisy groups and reuses clean embeddings; the default retains original joint projection. Timestep modulation, attention and positions remain inside forward. No new base projection modules.

Prepare current residual and clean prefix once per head call under normal gradient recording. Keep them local, with their live graph; never store across updates. A prepared-conditioning record for _velocity also retains resolved context/text validity, so _context/state projection and its condition-drop mask are resolved once. All trainable preparation stays inside the head's distributed ownership.

Request representation_layer=denoiser.representation_layers. Captured generated states contain future then action: J consumes only [:,:S]. Do not use fused final_hidden, which is after velocity-head conditioning. Preserve legacy _velocity return when disabled; enabled prediction returns image/action velocities plus predictions by encoder. Reject simultaneous legacy capture/fused/Self-Flow requests.

### 3.2 Training

1. Encode current/future VAE rasters once. Extract required current descriptors and future targets in separate provider calls/contexts from the corresponding world RGB. Encoders frozen; outputs detached.
2. Prepare original action masks/codec, condition-drop mask, sigma and independent noises once in their original order; add branch randomness afterward.
3. Compute M and clean trainable prefix/state/text projections once with gradients. Do not include this preparation in look-ahead's no-grad scope.
4. With feedback enabled, always evaluate zero feedback under no-grad in local
   denoiser/J eval mode. Discard first-pass velocities and workspace but keep
   raw detached J predictions.
5. During training, substitute detached true future descriptors independently
   per sample with probability 0.05, then apply non-affine LayerNorm and
   per-sample 0.1 feedback dropout. Dropout applies to substituted features too.
   Every sample still takes the second pass; there is no microbatch branch
   selector, rank synchronization, or look-ahead skip.
6. Run the trainable pass on the same noisy tensors, sigma, conditioning and
   masks. G receives the predicted/substituted/dropped buffer; M stays active.
   Validation uses deterministic predicted feedback without substitution or
   dropout.
7. Without feedback run one evaluation. Prediction-only computes J/loss; current-only has M and no J/G/feature loss.
8. Add separately weighted feature loss to unchanged image/action flow losses; the VLA adds optional language auxiliary loss once.

Cosine loss is a valid-token mean per sample, then batch mean with all-invalid samples contributing zero. Intersect future/teacher validity and compute in FP32 with detached targets. Teacher masks do not change flow masks. Preserve action-step/dimension masks and codec fill-coordinate supervision. Return scalar raw loss_teacher_<name> and weighted loss_teacher diagnostics, not dense predictions.

Validation uses deterministic feedback when enabled; tests fix noise/branch/conditions explicitly. Do not copy the source's global eval()/train() toggles inside a forward.

### 3.3 Inference

Obtain current descriptors once per observation/chunk, compute M once, and prepare the clean prefix once. Guided branches share observed current descriptors/residual, with separately prepared existing conditional/language-dropped text contexts. Current vision remains available to both.

Initialize image/action noise and zero future feedback once. Each Euler evaluation jointly predicts velocities/features; update x/u by (end-start)*velocity and replace feedback with raw predictions. Do not integrate/accumulate/guide feature vectors or add feature noise.

Retain separate conditional/unconditional feedback buffers and existing velocity guidance. Both scales one means K calls; guidance means baseline 2K calls. Extraction is outside the solver. Reset feedback on every chunk. Never extract future features or decode future RGB.

Combined pre-encoded inference may omit runtime definitions only with validated current descriptors. Future-only/prediction-only deployment may omit the encoder because M is disabled. The combined ordinary RGB path cannot claim encoder-free deployment.

## 4. File map

Paths are relative to FluxVLA. The implementation status is recorded below.

| Action | File | Responsibility |
|---|---|---|
| Create | fluxvla/models/third_party_models/patchwam/teacher_forcing/ | Contracts, fingerprints, normalization/loss, predictor factory, policy mixin |
| Create | fluxvla/models/backbones/visions/teacher_features.py | Current/future provider, spatial validation, runtime lifecycle |
| Modify | fluxvla/models/third_party_models/patchwam/encoding.py | RGB-boundary hooks, layouts/current length before history |
| Modify | fluxvla/models/third_party_models/patchwam/flux.py | Prepared prefix, current/future additions, existing capture |
| Modify | fluxvla/models/third_party_models/patchwam/tiny.py | Equivalent testable semantics |
| Modify | fluxvla/models/third_party_models/patchwam/policy.py | Generic prepared-prefix inputs and reusable flow preparation/loss helpers |
| Modify | fluxvla/models/vlas/patchwam_vla.py | General key/allow-list passthrough and buffer-aware base_init |
| Create | fluxvla/models/heads/teacher_forcing_patchwam_head.py | Registered teacher head subclass |
| Create | fluxvla/models/vlas/teacher_forcing_patchwam_vla.py | Provider lifecycle, extraction, checkpoint identity |
| Create | configs/patchwam/patchwam_teacher_forcing_radio_robotwin.py | Matched combined train/inference recipe, ablation switches |
| Create | docs/patchwam-teacher-forcing.md | Usage, geometry, lifecycle, replacement, costs |
| Modify | test/cpu-tests.txt | Register new CPU tests |
| Create | test/test_models/patchwam/test_teacher_feature_contract.py | Config, predictor structure, numerical helpers/fingerprints |
| Create | test/test_models/patchwam/test_teacher_features.py | Context separation, geometry, cache/runtime lifecycle |
| Create | test/test_models/patchwam/test_teacher_forcing_denoiser.py | Fusion/readout/sharing, prefix equivalence |
| Create | test/test_models/patchwam/test_teacher_forcing_training.py | Branches, gradients, masks/losses |
| Create | test/test_models/patchwam/test_teacher_forcing_sampling.py | Extraction counts, recurrence/guidance/reset |
| Create | test/test_models/patchwam/test_teacher_forcing_checkpoint.py | Initialization/resume, encoder/head identity |
| Create | test/test_models/patchwam/test_teacher_forcing_distributed.py | Multiple encoders and rank-independent pass structure |
| Create | test/test_configs/test_patchwam_teacher_forcing_config.py | Recipe/ablation contracts |

Do not change codec/flow/attention algebra, TRT serialization, or DINO Forcing. Runner changes require focused evidence of an integration gap, e.g. lost signature buffers or mishandled nested feature dictionaries.

## 5. Dependency-ordered tasks

Checkboxes track future implementation, not this review.

### Task 1: Define configuration, predictor and numerical contracts

**Files:** teacher_forcing.py and test_teacher_feature_contract.py in the file map.
**Consumes:** Sections 1–2 and user-confirmed J default. **Produces:** validated specs/config, predictor factory, fingerprints, normalization/loss helpers.

- [ ] Add failing tests for invalid names/widths/probabilities/weights/provenance, unsupported alignment, and feedback without prediction.
- [ ] Assert REPA has three biased linears/two SiLUs, default m=2048, no norm, Xavier weights/zero biases; assert Linear/MLP2 alternatives and parameter counts.
- [ ] Test normalization of [3,4], zero predictions, half precision, masked NaNs, and rejection of valid nonfinite/zero observed descriptors.
- [ ] Test cosine 0/1/2 for parallel/orthogonal/opposite; one loss-1 sample and one all-invalid sample give 0.5. Verify prediction gradients/target detachment.
- [ ] Test representation changes alter target fingerprints; predictor/switch changes alter only the aggregate policy digest. Paths/device/loss weights do not redefine representation.
- [ ] Implement after confirming missing functionality fails; imports must work without CUDA/TRT/RADIO assets/DINO.
- [ ] Run python -m pytest test/test_models/patchwam/test_teacher_feature_contract.py -q; require CPU cases to pass.

**Acceptance:** Representation, policy architecture, and runtime backend are distinct; J defaults to the inspected source architecture with adapted widths.

### Task 2: Extract current/future features with exact geometry

**Files:** teacher_features.py, encoding.py, patchwam_backbone.py, test_teacher_features.py.
**Consumes:** Task 1 and frozen spatial-backbone contract. **Produces:** provider, role-specific descriptors, raster metadata/current length.

- [ ] Use coordinate-valued fake encoders/VAE to test frame 0 vs -1, same world augmentation/composition, camera_video precedence, masks, rectangular row/view order.
- [ ] Use a context-sensitive fake to prove current features do not depend on future RGB. Assert separate contexts/calls and no future access at inference.
- [ ] Reject equal-count/different-grid shapes, transposed/mixed views, special tokens, wrong widths, geometry-changing preprocessing, and unsupported joint-view encoders. Assert no feature interpolation.
- [ ] Test RADIO PyTorch resolved size and TRT sidecar geometry before encoding; wrong sizes fail and compatible inputs undergo range conversion exactly once.
- [ ] Skip invalid rasters; scatter zeros. Cover all-invalid contexts and padded NaNs without unnecessary runtime loading.
- [ ] Test balanced engine chunks: min/max 2/4 with N=5 gives 3+2; N=1 fails. Do not duplicate/pad examples. Initial recipe requires min batch 1.
- [ ] Test freezing/eval/no-grad, explicit rank device, ordinary detached tensors usable by trainable linears, and output stability across successive engine calls; clone backend outputs if storage is reused.
- [ ] Implement hooks and pre-encoded validation. Current extraction runs at target=False when enabled; future extraction only for required targets. Preserve existing auxiliary and replacement paths.
- [ ] Test supplied complete features bypass extraction; missing current inputs fail at inference, missing future targets fail only for supervised prediction.
- [ ] Run python -m pytest test/test_models/patchwam/test_teacher_features.py test/test_models/patchwam/test_patchwam_backbone.py test/test_models/patchwam/test_patchwam_visual_backbones.py -q.

**Acceptance:** Descriptors correspond directly to VAE slots with isolated temporal contexts and interchangeable matching RADIO runtimes.

### Task 3: Add M/G/J and preserve pretrained projections

**Files:** flux.py, tiny.py, policy.py, patchwam_head.py, test_teacher_forcing_denoiser.py.
**Consumes:** Tasks 1–2 and representation capture. **Produces:** M/G/J, prepared-prefix/residual interfaces.

- [ ] Test separate bias-free M/G, zero-default and scaled-Xavier M/G initialization, source-style J; no pretrained reinitialization or independent D_cur/D_fut/D_act.
- [ ] Test concatenation-linear versus separate-projection addition with matched biases; nonzero M/G gradients at zero initialization for valid nonzero descriptors; no frozen/detached-feature gradients; direct J supervision gradients; scaled initialization; and resume without reinitialization.
- [ ] Inspect first-block inputs: current/future additions affect only their designated slots; history/text/state/action inputs and positions/masks remain original.
- [ ] Assert J reads only raw final future states before output normalization, not action states or velocity output.
- [ ] Compare ordinary/prepared paths with identical weights and residuals explicitly zero. Disabled models preserve keys/API/random draws. Require enabled zero-initialized M/G baseline velocity parity at identical inputs and randomness; small-random mode need not match exactly.
- [ ] Test prefix/residual shape/device/dtype, masking, history offsets, multiple-encoder sums and incompatible legacy capture/fused settings.
- [ ] Implement modules before optimizer setup; prepare clean embeddings once, add residuals out of place, retain base module objects and shared velocity call.
- [ ] Test backward/checkpointing on/off, including prefix graph reuse after no-grad look-ahead, with tiny and available official FLUX fixtures.
- [ ] Run python -m pytest test/test_models/patchwam/test_teacher_forcing_denoiser.py test/test_models/patchwam/test_patchwam_flux_adapter.py test/test_models/patchwam/test_patchwam_parity.py -q.

**Acceptance:** Correct boundary additions/readout and original sharing. Missing official source is a reported skip, not parity evidence.

### Task 4: Implement detached training with current conditioning

**Files:** teacher_forcing/policy.py, generic policy.py flow helpers, test_teacher_forcing_training.py.
**Consumes:** Prepared prefix/M/G/J and detached descriptors. **Produces:** two-pass training, losses, correct gradient paths.

- [ ] Spy on two-pass updates: same noisy tensors/sigma/prefix/masks, no-grad first, trainable second, raw detached predictions with LayerNorm only at fusion; current extraction/M/prefix preparation once.
- [ ] Test substitution/dropout endpoints: M active, configured losses present, G executed but has zero weight gradient when every sample is dropped.
- [ ] Change only future feature targets under fixed inputs: velocities/predictions/feedback unchanged, loss changed. Changing current features may change outputs.
- [ ] Assert gradients reach M and base prefix/state/input modules after look-ahead reuse, J/DiT from feature loss, G with nonzero feedback; encoders/targets/first-pass predictions detached.
- [ ] Assert exact total-loss composition and single language auxiliary addition. Feature loss still trains at sigma endpoints where flow weight is zero.
- [ ] Cover action-step/dimension/fill-coordinate masks, partial cameras, all-invalid endpoints and padded NaNs; teacher masks never redefine flow masks.
- [ ] Implement within one head forward after existing noisy-input preparation; do not call full head twice or toggle global train/eval.
- [ ] Test ablation switches and deterministic validation. Per-sample masks do not change the two-pass module structure on any rank.
- [ ] Run python -m pytest test/test_models/patchwam/test_teacher_forcing_training.py -q plus existing relevant core/parity and vision-language tests.

**Acceptance:** Future targets affect conditioning only through the explicit training substitution branch, and detached look-ahead retains M/base-prefix gradients.

### Task 5: Add current-conditioned Euler inference

**Files:** policy.py, test_teacher_forcing_sampling.py; public wrapper integration in Task 6.
**Consumes:** Current descriptors/prepared prefix and joint velocity/prediction evaluation. **Produces:** per-call feedback and current extraction outside solver.

- [ ] Test zero initial feedback, raw predicted replacement, negative Euler increments, unchanged codec decoding, resets on subsequent calls/different batch sizes.
- [ ] Count one current context extraction per encoder, subject to engine batch chunking, and one M preparation before K solver calls. Future-extraction spy must never execute.
- [ ] Test 2K guided calls, independent feedback buffers, shared current descriptors and preserved language-drop/state semantics.
- [ ] Test current-only, future-only, supervision-only and combined switches. No inference look-ahead. Skip J evaluation when its predictions are unused and feedback disabled.
- [ ] Test pre-encoded current descriptors with no provider and clear errors for missing combined current inputs. Future target keys cannot alter fixed-noise sampling.
- [ ] Test invalid rasters, zero predictions, half precision, history offsets and generated validity derived from current views.
- [ ] Implement local buffers/prefix reuse without persistent graphs/feedback and preserve normalized-action API.
- [ ] Run python -m pytest test/test_models/patchwam/test_teacher_forcing_sampling.py -q plus action-mode/parity tests.

**Acceptance:** Combined inference uses current features, never future RGB, with no extra denoiser calls beyond baseline guidance.

### Task 6: Integrate VLA lifecycle and checkpoint contracts

**Files:** patchwam_vla.py, patchwam_head.py, test_teacher_forcing_checkpoint.py.
**Consumes:** Provider, modules and encoded keys. **Produces:** raw/pre-encoded train/inference integration, resource closure, strict initialization/resume.

- [ ] Extend key/knob passthrough. Test nested tensors/masks/fingerprints through runner preparation without casting fingerprint bytes to floating features.
- [ ] Validate unsupported combinations before wrapping/optimizer; full finetuning includes all enabled M/G/J and baseline trainable modules.
- [ ] Build providers from runtime definitions, validate static metadata, lazily load runtime. Test close_teacher_extractors closes only owned resources and permits explicit reload.
- [ ] base_init permits missing exactly enabled M/G/J and new digest; reject missing inherited tensors, unexpected keys, shape errors or nonfinite inherited weights before mutation.
- [ ] resume requires full branches and matching digest even through load_state_dict(strict=False). Test missing M, wrong J depth/width, equal-width/different encoder, normalization/switch/name changes.
- [ ] Store persistent vla_head.teacher_contract_digest:uint8[32]; test native .pt and tensor-only safetensors. Runtime paths are separate from source provenance.
- [ ] Verify no encoder weights/engine buffers/temporary prefix/feedback enter model checkpoint or optimizer. Preserve all_module_keys. Combined inference still requires assets or external current features.
- [ ] Verify FSDP full-state buffer preservation and load checks; change formatter only if a focused test proves omission.
- [ ] Run python -m pytest test/test_models/patchwam/test_teacher_forcing_checkpoint.py test/test_models/patchwam/test_patchwam_checkpoint.py test/test_models/patchwam/test_patchwam_action_patch_checkpoint.py test/test_models/test_base_vla_checkpoint.py -q.

**Acceptance:** Narrow baseline initialization and representation/architecture-bound resume. New representation requires newly trained M/G/J; cross-encoder transformer transfer needs a later explicit mode.

### Task 7: Add RADIO recipes, ablations and documentation

**Files:** New config/docs/config tests from the file map; test/cpu-tests.txt.
**Consumes:** Full interfaces and existing RobotWin pipeline. **Produces:** matched train/inference recipes and ablation matrix.

- [ ] Base on configs/patchwam/robotwin/patchwam_robotwin.py: horizon/future offset 16, 288×256 compact mosaic, existing augmentation/text frontend/state packing, history disabled, 10 solver steps.
- [ ] Configure RADIO-H native width 1280, exact 288×256 input, expected 18×16 native/VAE grids subject to asset validation. Require matching TRT geometry/min batch 1 or verified PyTorch; no export/resize fallback.
- [ ] Record actual local checkpoint/engine provenance and checksums. Sidecar identity may be insufficient; bind reviewed export bytes to the source representation explicitly.
- [ ] Set M/G zero by default with independent scaled-Xavier options, J repa_mlp/2048, cosine epsilon 1e-6, loss weight 0.1, true-feature probability 0.05 and feedback dropout 0.1 explicitly.
- [ ] Include current encoder definitions in combined deployment; document pre-encoded and future-only deployment exceptions.
- [ ] Expose switches in current/prediction/feedback order: baseline None; supervision-only False/True/False; current-only True/False/False; future forcing False/True/True; combined True/True/True.
- [ ] Use DDP static_graph=False initially; retain matched optimizer/LR/precision/accumulation, log trainable counts.
- [ ] Document geometry, context separation, head options, history, checkpoint replacement, backend swaps and inference extraction cost.
- [ ] Register CPU tests and run python -m pytest test/test_configs/test_patchwam_teacher_forcing_config.py -q.

**Acceptance:** Matched geometry/identity/architecture across train/inference, with no silent teacher-free combined configuration.

### Task 8: Distributed, multiple-encoder and real-asset verification

**Files:** test_teacher_forcing_distributed.py; extend earlier tests/docs.
**Consumes:** Tasks 1–7. **Produces:** focused integration evidence and honest unavailable-check reporting.

- [ ] Use two fake encoders with different widths but matching required native grids. Verify independent M/G/J/buffers, summed residuals/losses and one shared look-ahead.
- [ ] Reject partial encoder-role batches/checkpoints and distinct native grids; never silently omit a configured encoder.
- [ ] Run two-rank DDP tests with different per-sample substitution/dropout masks and gradient accumulation; require consistent two-pass call counts and valid gradients. No synchronized branch selector is needed.
- [ ] With available GPUs, test tiny DDP and head-unit FSDP training/save/reload, BF16/checkpointing, prefix graph reuse and load-before-freeze/freeze-before-load orders.
- [ ] With matching assets, check actual VAE/RADIO grids, finite role descriptors and PyTorch/TRT parity at identical geometry using existing tools.
- [ ] Run one real FLUX/RADIO train batch/update and fixed-noise action sample/reload; record losses/gradients, counts, latency/memory. No long training or incidental asset rebuild.
- [ ] Test combined pre-encoded inference with assets unavailable and supplied current features; combined ordinary RGB must fail clearly without its encoder. Future-only deployment remains encoder-free.
- [ ] Run focused regressions once after final edits; record exact pass/skip/block outcomes. Unavailable hardware/assets are not passing checks.

**Acceptance:** CPU contracts pass; provisioned integration checks establish execution, not manipulation improvement.

## 6. Verification and completion criteria

Commands below are planned; they were not run during this document review.

| Scope | Command |
|---|---|
| New numerical/provider/model tests | python -m pytest test/test_models/patchwam/test_teacher_feature_contract.py test/test_models/patchwam/test_teacher_features.py test/test_models/patchwam/test_teacher_forcing_denoiser.py test/test_models/patchwam/test_teacher_forcing_training.py test/test_models/patchwam/test_teacher_forcing_sampling.py test/test_models/patchwam/test_teacher_forcing_checkpoint.py -q |
| Configuration | python -m pytest test/test_configs/test_patchwam_teacher_forcing_config.py -q |
| PatchWAM regressions | python -m pytest test/test_models/patchwam test/test_models/test_patchwam.py test/test_configs/test_patchwam_config.py test/test_transforms/test_patchwam_transforms.py test/test_datasets/test_patchwam_dataset.py -q |
| RADIO contracts | python -m pytest test/test_models/test_radio_trt_backbone.py test/test_models/test_base_trt_engine.py test/test_scripts/test_c_radio_tensorrt.py test/test_configs/test_radio_trt_config.py -q |
| Distributed | python -m pytest test/test_models/patchwam/test_teacher_forcing_distributed.py -q |

Completion requires preserved disabled behavior; exact native grids; correct fusion/readout/sharing; source-default J; live M/base gradients after look-ahead; no future input outside the explicit training substitution branch; correct losses/masks; current extraction outside solver; feedback reset/guidance; strict architecture/representation resume; and explicit actual test coverage.

Baseline parity is checked with the extension disabled and with enabled default-zero M/G at identical inputs and randomness. Small-random initialization permits initial perturbations; report residual-to-base activation norms and initial velocity deviation.

## 7. Experiments and costs

Compare matched baseline, supervision only, current only, future forcing only, and combined. Then compare J types, feature weights 0.05/0.1/0.3, equivalent PyTorch/TRT extraction, other compatible encoders and optional multiple encoders. Hold data/augmentation, inherited initialization, action horizon, solver steps, evaluation seeds and execution prefix fixed. Report update budgets and elapsed cost.

Combined parameters per encoder are 2*d*d_f + P_J:

| J | P_J | Total: d=3072, d_f=1280, m=2048 |
|---|---|---:|
| Linear | d*d_f+d_f | 11,797,760 |
| MLP2 | d*m+m+m*d_f+d_f | 16,780,544 |
| Source REPA MLP | d*m+m+m*m+m+m*d_f+d_f | 20,976,896 |

A BF16 feedback buffer at B=4,S=288,d_f=1280 is about 2.81 MiB; guided inference needs two. Current descriptors, encoder/runtime workspace, prepared prefix, targets, optimizer state and look-ahead add further memory. Current extraction adds inference latency once per observation context. M/G/J add compute without adding transformer tokens. Feedback training performs exactly two denoiser evaluations per microbatch, one without gradients.

Report manipulation success, feature cosine, available perturbation results, training/extraction time, inference latency, memory and actual geometry. Feature cosine alone does not establish control improvement. Keep image loss while integrating future image latents.

Deferred: new DINO/VGGT adapters, joint-view geometry encoders, feature ODE/noise, current prediction loss, feature-output resampling, persistent cache tools, cross-encoder transfer, solver changes, staged/LoRA/LIT combinations and historical-feature fusion. Future adapters must establish spatial/context contracts.

## 8. Coverage and review disposition

| Requirement | Tasks |
|---|---|
| Current V–M active in training/inference | 2–5, 7–8 |
| Shared frozen weights, separate temporal contexts | 2, 4–5 |
| Preserve actual D/state/text/attention/codec/flow | 3–5, 8 |
| No feature-grid interpolation | 1–2, 7–8 |
| Zero-default/scaled-random M/G and source-family J initialization | 1, 3 |
| User-confirmed source J default and alternatives | 1, 3, 6–7 |
| Detached look-ahead with live M/base gradients | 3–4, 8 |
| Future supervision, explicit training substitution, and raw feedback carry | 1, 4–5 |
| Independent multiple encoders, shared evaluations | 1–5, 8 |
| Real current-encoder deployment dependency | 2, 5–7 |
| Checkpoints, ablations, costs, validation | 6–8; Sections 6–7 |

The J-depth conflict was resolved by the user during review. Remaining source differences are documented adaptations or diagram shorthand with concrete rules above. This document claims no policy implementation, executed asset validation, training result, or measured improvement.
