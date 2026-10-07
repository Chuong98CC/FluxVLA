# PatchWAM integration implementation plan

**Goal:** Implement DESIGN.md through native FluxVLA components and pipelines.

**Architecture:** Native VLA/backbone/head wrappers surround a local PatchWAM core;
dataset and transform extensions preserve temporal and augmentation behavior.

**Execution:** The user-selected `gpt-6.1-sol` root plans/reviews/integrates;
one native `astra_flash_builder`
writer implements each dependency-ordered bundle. This user-selected workflow
supersedes generic multi-reviewer or per-task agent defaults. No recursive agents.

**Tech stack:** Existing FluxVLA Python, Torch, MMEngine registries/configs,
native parquet/video helpers, and locally supplied FLUX.2/Qwen assets.

**Spec:** [DESIGN.md](DESIGN.md).

## Constraints and review focus

- No runtime imports from `/home/chuong/workspace/PatchWAM`, new standalone engine,
  global dependency pin changes, credentials, network downloads, paid setup
  probes, commits, staging, pushing, or deployment.
- Preserve other edits. One shared-workspace writer; it is not alone in the
  codebase and must accommodate others' work without reverting it.
- Coordinator owns DESIGN.md, PLAN.md, CHECKPOINT.md, and BASELINE.json. Workers
  write only their unique phase report and assigned implementation/test/docs paths.
- Mask polarity, all-invalid/NaN padding, endpoint alignment, episode boundaries,
  clean VLM images, LoRA freeze policy, and module ownership are review priorities.
- Small real phase 1 is also the first useful routed task. Inspect host/router
  metadata after it; do not treat static-ready or self-reported model names as
  evidence of an executed Flash request.

## Prerequisite: authorized root and baseline

- [x] User explicitly authorized `gpt-6.1-sol` and overrode the skill's
  Astra-only root requirement. Continue with the current root and effort.
- [x] Reuse recorded static route evidence; doctor should be rerun only if
  configuration changed. The installed role pins `deepseek/deepseek-v4.1-flash`,
  provider `DeepSeek API`, effort `high`.
- [x] Read the actual working tree against BASELINE.json before dispatch. Preserve
  these untracked planning artifacts and any later user edits.

## Phase 1 — Native core and numerical contract

Task ID: `patchwam-core`; prerequisite: correct root and baseline confirmation.

Status: accepted after one mask-broadcast correction; 13 CPU tests passed and
source/copy/wrapper parity had zero numerical delta. Actual host/router Flash
routing was verified during this phase. Remaining raster/asset work is Phase 2.

**Worker owns:**

- `fluxvla/models/third_party_models/patchwam/{__init__,codec,geometry,flow,policy,tiny}.py`
  and source license/provenance files in that directory.
- `fluxvla/models/heads/patchwam_head.py`, `fluxvla/models/vlas/patchwam_vla.py`.
- Their package `__init__.py` registration entries, preserving existing optional
  import behavior.
- `test/test_models/test_patchwam.py`; minimal additions to
  `test/_cpu_model_runtime.py` if needed to register actual CPU modules.
- `docs/agent-work/patchwam/PHASE1_REPORT.md`.

**Contracts:**

- `PatchWAMHead` inherits the copied policy, builds a tiny denoiser from a config,
  and consumes the source encoded-batch contract.
- `PatchWAMVLA` implements the DESIGN.md native forward and prediction signature;
  permit preencoded `reference_tokens`, `future_tokens`, `text_tokens`, ids, and
  masks so phase 1 runs without assets. Build the head with existing builders.
- Mask conversion belongs in one explicit adapter, preserving temporal and
  dimensional invalidity. Returned losses remain native scalar loss dictionaries.
- Do not build a fake backbone or claim actual-asset coverage in this phase.

**Internal steps and verification:**

- [ ] Write failing tests for registry construction, parameter-free action
  round trip, mask polarity/dimensions, all-invalid loss, native prediction shape,
  and fixed-noise source parity. Record the initial failures.
- [ ] Copy the numerical core with relative imports and retained attribution;
  implement the wrappers and explicit native adapter.
- [ ] Compare source/copy/wrapper loss and gradients with identical weights,
  sigma, and Gaussian noise; compare seeded samples with explicit initial latents.
- [ ] Run `CUDA_VISIBLE_DEVICES='' python -m pytest --cpu-model-tests
  test/test_models/test_patchwam.py -q` using the project's compatible Python
  environment. CPU model mode currently requires Transformers 5.3.0.
- [ ] Run syntax and whitespace checks on changed files and report exact commands,
  interpreter, exit statuses, test counts, and limitations.

**Astra acceptance:** inspect actual tracked/untracked diff and test evidence in
one pass for compliance and quality. Check routing metadata from this useful run,
then accept phase 1 before phase 2. Do not rerun the full validation routinely.
Interrupt the finished child to mark it done; reuse it for corrections/phase 2.

## Phase 2 — Assets, dataset, augmentation, and native workflows

Task ID: `patchwam-integration`; prerequisite: phase 1 accepted and present.

Status: accepted. Actual shipped config-derived decoding/data-to-training,
native raw-observation LIBERO inference, checkpoint formats, source camera/
augmentation/history behavior, and focused root acceptance pass. Final evidence
and explicit hardware/feature limits are recorded in ACCEPTANCE.md.

**Worker owns:** phase 1 implementation files as needed for integration, plus:

- `fluxvla/models/third_party_models/patchwam/{flux,vision_language}.py`.
- `fluxvla/models/backbones/vlms/patchwam_backbone.py` and its registration.
- `fluxvla/models/third_party_models/patchwam/data/{__init__,augmentation,appearance,history}.py`
  with MIT/provenance notices where required.
- `fluxvla/datasets/patchwam_dataset.py` and dataset registration; one optional
  protected hook in `fluxvla/datasets/parquet_dataset.py` if inherited extension
  cannot supply history/endpoints before existing transform execution.
- `fluxvla/transforms/transform_patchwam.py` and registration.
- `configs/patchwam/patchwam_libero_full_finetune.py`,
  `patchwam_robotwin_vlm_full_finetune.py`, and
  `patchwam_robodojo_vlm_history_example.py`.
- `test/test_models/test_patchwam_backbone.py`,
  `test/test_datasets/test_patchwam_dataset.py`,
  `test/test_transforms/test_patchwam_transforms.py`,
  `test/test_configs/test_patchwam_config.py`.
- `test/cpu-tests.txt` for explicit inclusion of the new compatible CPU tests.
- `docs/patchwam.md`, a concise README model-list/guide link if appropriate, and
  `docs/agent-work/patchwam/PHASE2_REPORT.md`.
- Existing eval dataset/runner code only for narrow, documented episode-reset,
  observation-cadence, or capability changes justified by history integration.
  Do not change general optimization/resume logic for deferred EMA/Self-Flow.

**Contracts:**

- `PatchWAMBackbone.encode(batch, target: bool) -> dict` uses copied source
  encoding semantics, local strict FLUX.2 asset loading, and copied specialized
  VLM conditioning. Only encoders are registered in this module.
- Head owns the official masked denoiser and trainable proprio projection. Keep
  module state names stable and use native checkpoint methods. FSDP must wrap
  a module whose forward is actually called; the copied FLUX adapter manually
  executes inner blocks and cannot safely auto-wrap those blocks directly.
- Dataset subclass retains native `.stats`, `__len__`, and
  `__getitem__(index, dataset_statistics)`. Do not transplant the source training
  engine or LeRobot reader. Add original-clock causal metadata before transforms.
- Registered transforms return only the fields named by configured DictCollator
  tensor/meta keys. Separate clean VLM tensors before augmenting world tensors.
- Translate source recipes to native config fields. LIBERO has 16 action targets
  and current/future endpoints 16 rows apart. Use inherited normalizers and native
  inverse action decoders; prove parity on fixtures. Document statistics mapping.
- Preserve CFG/history/LoRA/subtask/appearance behavior. Fail unsupported
  Self-Flow/EMA training options explicitly; no misleading advanced recipes.

**Internal steps and verification:**

- [ ] Add failing synthetic fixtures for two episodes/two roots, padded endpoint
  windows, filtered starts on an original clock, variable action masks, missing
  labels, and camera ordering. Test normalizer/inverse decoder against source.
- [ ] Copy the VLM/FLUX.2 adapter and relevant data utilities with notices;
  implement native backbone, narrow dataset extension, and registered transforms.
- [ ] Test exact fixed-seed ClipAugment/appearance parity, no mutation of clean
  VLM images, causal history/reset, and constant future sampling length.
- [ ] Use injected lightweight real modules for the local encoding contract;
  test freeze/train mode, LoRA gradients, cache validation, subtask causality,
  asset errors, and disjoint named-parameter/state_dict ownership.
- [ ] Add native configs/docs showing training, resume, eval/inference overrides,
  asset paths, normalization, denormalization, and current verification limits.
- [ ] Verify one optimizer update through the native loss dictionary and a
  checkpoint round trip with identical seeded predictions. Exercise registry
  construction and normal production import behavior where dependencies permit.
- [ ] Run the new model tests with CPU mode as in phase 1. Run new dataset,
  transform, and config tests plus affected existing
  `test/test_datasets/test_parquet_temporal_window.py`,
  `test/test_datasets/test_transformed_auto_statistics.py`, and
  `test/test_datasets/test_dataset_wrapper.py` in a compatible offline environment.
  Select separate invocations if model import isolation requires it.
- [ ] Run `git diff --check` and source compilation on changed Python paths.
  Run repository formatting checks on native wrapper/test/config files; exclude
  vendored code according to existing third-party style policy.
- [ ] If local assets/GPU are already available, perform a bounded forward,
  backward, save/load, and inference check; otherwise record each unverified
  item without downloading assets or changing global dependency pins.
- [ ] Return ready-for-review with changed paths, verification commands/results,
  risks, source parity evidence, and a resumable checkpoint if blocked.

**Astra acceptance:** review the actual patch and evidence against every criterion
in DESIGN.md with compliance and quality as two lenses in one batch. Return all
precise corrections to the same worker once. Additional cycles require a concrete
material architecture or shared-infrastructure risk. Accept only verified scope,
update CHECKPOINT.md, and report outstanding hardware/asset limits honestly.

## Dispatch brief to attach to either phase

Read DESIGN.md, this phase, BASELINE.json, and relevant repository guidance.
Working directory is `/home/chuong/workspace/FluxVLA`; PatchWAM source is read-only.
You are not alone in the codebase: preserve and accommodate others' edits. Own
discovery, tests, implementation, and debugging within the listed scope. Do not
load the orchestration skill, delegate, commit, stage, push, deploy, download
weights, expose secrets, or alter permissions/router settings. Use the available
native tools and approved interpreter. Ordinary implementation choices are yours;
report material contract conflicts before expanding scope. Keep coordinator files
untouched and write one unique report. Completion means ready for Astra review.
