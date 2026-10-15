---
name: export-trt
description: Export a frozen FluxVLA vision backbone to ONNX and TensorRT using the shared base_trt runtime, FP16 tactics and enforced FP32 LayerNorm. Compare real-image feature accuracy and latency against PyTorch. Use for C-RADIO or another vision backbone; this workflow does not search other precision settings. Always use together with astra-flash-orchestrator.
---

# Export a frozen vision backbone to TensorRT

Build a reusable inference component in the existing FluxVLA structure, then
measure its feature differences and inference latency. Follow repository
instructions for implementation and delegation. This skill defines the export
contract and evaluation policy; it does not change that execution policy.

## Required companion skill

The root agent must always load `$astra-flash-orchestrator` together with this
skill, before planning or execution. Reuse it if already loaded in the current
session. Its installed entrypoint is
`/home/chuong/.agents/skills/astra-flash-orchestrator/SKILL.md`; discover the
installed skill by name if that location changes. If it is unavailable, report
the missing dependency before starting the workflow.

Astra owns planning, architecture, review and integration. Follow the
orchestrator's task classification: substantial implementation goes to one
native `astra_flash_builder`; trivial edits remain with the root. A Flash
worker using these task instructions executes only its assigned bundle and
must not load the orchestration skill or delegate recursively.

## Invoke

Use `$export-trt` or select `export-trt` through Codex's `/skills` selector.
For example: `$export-trt Export C-RADIO at 256x256, batch 1 through 8, and
evaluate assets/frame_000000.jpg.` Treat the rest of the prompt as the model,
checkpoint, geometry, batch and evaluation-image requirements. Do not launch
an export solely because the skill was loaded to edit its instructions.

When installed as a symlink, resolve the skill's source directory to locate
the FluxVLA checkout. Run the tools below from that repository root.

## Fixed precision policy

- Export the reference graph in FP32. Build with FP16 tactics and force every
  LayerNorm layer to FP32 using `OBEY_PRECISION_CONSTRAINTS`.
- The current Python CLI uses `--precision fp16`, repeatable
  `--layer-precision 'PATTERN:fp32'`, and `--precision-constraints obey`.
  These correspond to trtexec's layer precision and constraints flags.
- Discover actual parsed layer names and check coverage of all LayerNorm
  layers. `*/LayerNormalization` works for the current C-RADIO export; do not
  assume it works for another model or another export of the same model.
- An unmatched pattern, incomplete coverage, unsupported precision constraint,
  or TensorRT rejection is a failed build. Do not silently drop the policy or
  replace `obey` with `prefer`. If LayerNorm is decomposed into primitive ops,
  identify the complete normalization subgraph and constrain it, or report that
  enforcement is blocked. If the architecture has no LayerNorm, report that
  fact and resolve the applicable normalization contract before proceeding.
- Do not try BF16, INT8, a full FP32 engine, Softmax/MatMul overrides, or a
  precision sweep. PyTorch FP32 is the accuracy reference, not another engine
  build. If accuracy is insufficient, save the evidence and report the limit;
  changing precision policy requires a new user instruction.
- Internal FP16 tactics do not imply FP16 engine IO. Keep the exported
  boundary dtype, usually FP32, and record the actual engine IO dtypes.

## Establish the model and artifact contract

Inspect the existing PyTorch backbone, callers, local checkpoint and export
tools before writing an adapter. Reuse an existing compatible ONNX graph when
its checkpoint, preprocessing, geometry, output contract and recorded identity
match. Do not re-export a large checkpoint solely to rebuild its engine.

Record the model/checkpoint identity; real evaluation images; RGB/range and
normalization rules; resize/crop/interpolation; image height and width; patch
or stride constraints; feature selection and layout; and batch profile
`min <= opt <= max`. Use 256x256, min/opt/max batch 1/1/8, and
`assets/frame_000000.jpg` as initial defaults when compatible and unspecified.
Preserve user settings and honor model-specific resolution constraints; do
not silently round the requested size. Optimize the batch profile for the
usual deployment workload when it is known.

Use an installed Python environment with the required PyTorch, ONNX and
TensorRT/CUDA dependencies. Record their versions and the target GPU. Do not
depend on `/tmp/cradio-export-venv` existing in a future session. The current
runtime supports TensorRT 10.x; precision APIs vary by release. Check the
installed API before building and report an unsupported environment without
changing the precision policy or upgrading it automatically.

## Reuse this folder structure

| Responsibility | Location and contract |
| --- | --- |
| PyTorch vision component | `fluxvla/models/backbones/visions/<model>_vit.py` or the model's existing high-level file |
| TensorRT vision component | `fluxvla/models/backbones/visions/<model>_trt.py` |
| Shared engine execution | `visions/base_trt/engine.py`: reuse `TensorRTEngine` |
| Shared tensor/profile contracts | `visions/base_trt/metadata.py`: `TensorSpec`, `OptimizationProfile`, `EngineContract` |
| Model-specific engine metadata | `visions/base_trt/<model>_metadata.py`, only if an adapter is needed |
| Offline ONNX/TRT helpers | `tools/trt_export.py`; reuse export, consolidation, identity validation, build and precision matching |
| Model-specific offline adapter/CLI | `tools/<model>_export.py` and/or `tools/<model>_tensorrt.py` |
| Layer-name/type inspection | `tools/trt_layer_report.py` |
| Example config and checks | `configs/visions/`, `test/test_models/`, `test/test_scripts/` |

Keep only high-level backbone classes in the `visions/` root. Put reusable
runtime support under `base_trt`; keep all export/build/evaluation tooling under
`tools/`. Do not copy C-RADIO's metadata or range conversion into an unrelated
backbone. Preserve the existing stream/context lifecycle, named IO validation,
device selection and `close()` behavior of `TensorRTEngine`.

Use `FrozenVisionFeatures` and `SpatialFeatureOutput` for spatial feature
consumers. Keep the encoder frozen and in evaluation mode. Match feature
layout, token ordering, special/register-token removal, grid and feature
dimension to the PyTorch wrapper. Keep preprocessing identical and applied
once. Load optional TensorRT dependencies lazily. When adding a component,
follow [the registry component skill](../fluxvla-add-component/SKILL.md): register
it, re-export it from `visions/__init__.py`, and verify the exact config keys.

## Export and verify the precision policy

1. Export named tensors with a fixed image geometry and a documented batch
   profile. Reuse `trt_export.export_onnx` and the generic contract rather than
   introducing a separate engine executor.
2. For large models, consolidate weights into **one `.onnx.data` file** beside
   the `.onnx` graph. Export in a temporary directory, publish only the graph,
   consolidated data and metadata, and discard only temporary shards created
   by that run. Keep external-data references relative. Repeated exports must
   not append duplicate weights to an existing sidecar. Preserve older
   artifacts and validate graph/data hashes.
3. Parse the graph with `tools/trt_layer_report.py --onnx ... --match ...`.
   Compare matched names/types/counts with the normalization layers expected
   from the model. Use enough explicit patterns to cover all LayerNorms.
4. Build a distinctly named `*_fp16_ln_fp32.engine` with FP16 tactics, FP32
   normalization overrides and `obey`. Enable detailed profiling verbosity
   for inspection and capture build diagnostics. Use existing generic
   `layer_precisions`/`precision_constraints` APIs; avoid model-specific logic
   in the shared builder.
5. Inspect the built engine, confirm the constraints were honored, and
   deserialize it through `TensorRTEngine`. Validate IO names, dtypes, shapes,
   profile bounds and output layout. Record the policy, matched layers,
   checkpoint/ONNX identity, versions, geometry and profile in build metadata.
   Matched names alone do not prove the built precision. If inspection cannot
   establish the policy, report that verification gap.

For verified C-RADIO commands and existing evaluation helpers, read
[the C-RADIO example](references/c-radio.md). Adapt its model adapter, not its
model-specific image range or patch size, for another architecture.

## Measure feature accuracy against PyTorch

Use the same real image tensors for both backends. Prefer representative user
images; the repository sample is a smoke check, not dataset-level validation.
Save image identifiers and preprocessing along with the measurements.

- Run the frozen PyTorch reference in FP32 under inference/no-grad mode, with
  autocast and TF32 disabled. Preserve the model's intended preprocessing and
  attention semantics; use the existing C-RADIO export contexts for C-RADIO.
- Check output shape, grid, token ordering, feature dimension, finiteness and
  boundary dtype before comparing flattened values. Evaluate batch 1 and a
  supported larger batch; cover profile bounds with shape checks when feasible.
- Report cosine similarity, relative L2 `||TRT - reference|| / ||reference||`,
  maximum/mean absolute difference and elementwise `allclose`. Handle a zero
  reference norm explicitly. Record tolerances and reductions.
- Default diagnostic tolerances are `atol=0.01, rtol=0.01`; preserve tighter
  user or consumer requirements. Save the actual pass/fail, including a
  nonzero parity CLI exit on failure. Never loosen thresholds to obtain a pass
  or label high cosine similarity as recovered task accuracy.
- A failed feature tolerance does not authorize another precision experiment.
  Report the observed drop and the need for downstream VLM/task validation.
  Optional PCA plots may supplement the numerical report, not replace it.

## Measure latency fairly

Compare **PyTorch FP32 and the FP16 + FP32 LayerNorm engine only**. Use the
same GPU, resident preprocessed real-image tensors and equivalent wrapper calls
(`encode_features` for spatial backbones). Run backends sequentially, preferably
in separate processes to release model and engine allocations between runs.

Check GPU utilization and compute processes before each backend and save the
snapshot. Never stop another workload. If the GPU is busy, mark timing as
contended or defer it; do not present it as an isolated speedup measurement.

Default to 30 warmup and 100 timed iterations at batch sizes 1, 2, 4 and 8,
restricted to the compiled profile and available memory. Include the requested
deployment batch when different. Synchronize the relevant CUDA device or stream
after warmup and each iteration. Measure wall-clock wrapper latency including
its preprocessing/validation/synchronization; exclude image decode/resize,
host-to-device transfer, model loading and compilation. State any differences
in timing scope. Do not compare CUDA-event time with wall-clock time.

Report median, mean and p90 batch latency; milliseconds per image; and
throughput `batch * 1000 / median_ms`. Save per-backend JSON and the runnable
harness; keep iteration samples when available. Torch allocator statistics do
not represent all TensorRT memory. The existing C-RADIO `benchmark` CLI alone
uses random input at `opt_batch` and times only TRT: it is not the real-image
PyTorch/TRT batch comparison required here. Adapt the existing benchmark helper
or add a small tool rather than pretending the CLI has missing options.

## Finish with reproducible evidence

Run focused checks for registry/config construction, lazy dependencies, frozen
feature contracts, input/profile rejection, normalization match coverage,
constraint enforcement and metadata compatibility. Run checks relevant to code
changed and inspect `git diff --check`; do not rebuild unchanged models just to
author documentation.

Deliver artifact paths (graph, single weight sidecar, engine, metadata), build
logs and matched-layer evidence, real-image accuracy JSON, a latency table,
environment/workload details and the exact rerun command. Keep reports beside
the artifacts or in the repository's existing report convention. State failed
tolerances and skipped hardware checks plainly. Do not claim that results from
C-RADIO, one image, one geometry or one GPU predict another backbone's results.
