# C-RADIO v4 TensorRT backbone

`RadioViTBackboneTensorRT` serves the same frozen C-RADIOv4 spatial features as
`RadioViTBackbone`, but from a serialized TensorRT engine instead of a live
PyTorch `AutoModel`. It is registered with `VISION_BACKBONES`, so a consumer
that already accepts the generic vision interface can swap between the two
without changing its own code: same `[B, N, D]` tokens, same `(h, w)` grid,
same `[-1, 1]` RGB input, same frozen/eval behaviour.

Only `radio_vit.py` and `radio_trt.py` are C-RADIO components at the vision
package root. Reusable `TensorRTEngine`, `EngineContract`, `TensorSpec` and
`OptimizationProfile` live in `visions/base_trt/`; the RADIO metadata adapter
also lives there. Offline helpers live in `tools/trt_export.py` and
`tools/c_radio_export.py`, with `tools/c_radio_tensorrt.py` as the CLI.

The swap is at the *compiled* geometry: the PyTorch encoder accepts any
patch-aligned size, while the engine accepts exactly the one it was built for.
Nothing else moves: the consumer keeps owning the tensor device and the
projection dtype, so a projector that expects a particular `D` or a particular
half precision still has to match what the engine boundary produces (see
`metadata.input_dtype`/`output_dtype` and `half_precision_dtype`).

| Path | Class | Runtime dependency | Geometry |
| --- | --- | --- | --- |
| PyTorch | `RadioViTBackbone` | `transformers` + local export | any patch-aligned size |
| TensorRT | `RadioViTBackboneTensorRT` | `tensorrt` 10.x + CUDA | one fixed patch-aligned size |

## Dependencies

TensorRT and ONNX are optional and lazy: importing the vision package never
imports either, and a config can be built and inspected without them.

| Step | Needs |
| --- | --- |
| `inspect` (metadata only) | nothing beyond the repository |
| `export-onnx` | `onnx` + a complete local official C-RADIO export |
| `build-engine` | `onnx` + TensorRT 10.x + CUDA |
| `validate` | TensorRT 10.x + CUDA |
| `parity` | TensorRT 10.x + CUDA + the local C-RADIO export (via `transformers`) |
| `benchmark` | TensorRT 10.x + CUDA |
| Inference through `RadioViTBackboneTensorRT` | TensorRT 10.x + CUDA |

`build-engine` needs `onnx` because it verifies the graph against the sidecar
before it compiles anything; the other validation and measurement steps only
touch the serialized engine. `sample`, `export-onnx` and `parity` also import
the local export's own model code, which needs `transformers` + `timm` and the
local directory.

`sample` and `export-onnx` temporarily restore upstream `timm` attention:
importing FluxVLA installs a CUDA-only half-precision attention patch (the
Eagle deployment path) that cannot run in FP32 and cannot be traced, so those
commands use C-RADIO's own attention math and put the patch back afterwards.
An FP32 engine build is strict by default: `BuilderFlag.TF32` is cleared and
precision constraints are obeyed, because TF32 tactics alone accounted for a
0.05 absolute feature difference in this repository's verification.
`--allow-tf32` opts back in.

The installed Python bindings must be TensorRT **10.x**. The named-IO and
`execute_async_v3` surface this runtime uses exists there; TensorRT 11 changed
strong-typing and precision semantics, so an 11.x build is rejected with a
clear error instead of being treated as 10.x.

Install the runtime with the pinned requirements file (Linux x86_64, CUDA 12
wheels):

```bash
python -m pip install -r requirements-trt.txt
# or as part of a full environment build:
bash scripts/install_env.sh full --with-tensorrt
```

`requirements-trt.txt` pins `tensorrt-cu12`, `tensorrt-cu12-bindings` and
`tensorrt-cu12-libs` to `10.13.3.9`, the version that built the checked-in
engines. A serialized engine is only deserializable by the TensorRT version
that produced it, so treat those pins as coupled to
`checkpoints/Pretrained_models/C-RADIOv4-H/exported/` rather than as "latest".
Missing bindings surface as a `RuntimeError` naming TensorRT 10.x, not as an
`ImportError`, because the import stays lazy.

## Fixed geometry, dynamic batch

The engine is compiled for exactly one patch-aligned `H x W`; the batch axis is
the only dynamic dimension, bounded by an optimization profile. An input of any
other size is rejected with a message naming the geometry to export, because a
different resolution needs a different engine. This is deliberate: silently
resizing or padding would hide that the request was never compiled.

## Pipeline

Three offline steps produce and validate the artifacts. Nothing here runs at
inference time and nothing is downloaded or substituted.

```bash
# 1. Trace a complete local official export to ONNX at one geometry.
python tools/c_radio_tensorrt.py export-onnx \
    --radio-path /path/to/C-RADIOv4-H \
    --version c-radio_v4-h \
    --image-height 256 --image-width 256 \
    --min-batch 1 --opt-batch 1 --max-batch 8 \
    --out artifacts/c_radio_v4_h_256.onnx

# 2. Compile ONNX to a serialized engine. The graph is verified against the
#    sidecar first. FP32 is the baseline; --precision fp16 sets
#    BuilderFlag.FP16, which selects FP16 tactics and does not retype the
#    graph inputs/outputs. --precision bf16 requests BuilderFlag.BF16
#    instead, with the same boundary. --layer-precision pins individual
#    layers to a precision (see "Layer precision overrides"). There is no
#    INT8 calibration.
python tools/c_radio_tensorrt.py build-engine \
    --onnx artifacts/c_radio_v4_h_256.onnx \
    --out artifacts/c_radio_v4_h_256.engine \
    --precision fp16 --workspace-mib 2048

# 3. Inspect, validate and compare against the PyTorch encoder.
python tools/c_radio_tensorrt.py inspect \
    --metadata artifacts/c_radio_v4_h_256.engine.json
python tools/c_radio_tensorrt.py validate \
    --engine artifacts/c_radio_v4_h_256.engine
python tools/c_radio_tensorrt.py parity \
    --engine artifacts/c_radio_v4_h_256.engine \
    --radio-path /path/to/C-RADIOv4-H
python tools/c_radio_tensorrt.py benchmark \
    --engine artifacts/c_radio_v4_h_256.engine --iterations 50
```

`export-onnx` writes `<out>.onnx` and a provisional `<out>.json` sidecar.
Graphs larger than `--external-data-min-bytes` (default 1 KiB of initializer)
are written with an external data sidecar whose name is recorded in the
metadata together with its sha256; pass `--no-external-data` to keep
everything inline.

`build-engine` verifies the graph (names, dtypes, geometry, token layout and
the recorded hashes) before it parses anything, re-checks the parsed network
against the same sidecar, writes the engine, and then writes
`<engine>.json`. That final record describes the artifact that actually
exists: `tensorrt_version`, builder `precision`, the `input_dtype`/
`output_dtype` read from the network, `workspace_bytes`, `onnx_sha256`,
`external_data_sha256` and `created_utc`. By default the fresh engine is
deserialized and validated (IO, dtypes, geometry and profile) before the
command returns. Re-exporting the ONNX without refreshing the sidecar fails
the hash check instead of silently rebuilding from a different graph.

`benchmark` reports wall-clock latency measured with an explicit
`torch.cuda.synchronize` on every iteration, on that device only. It is a
measurement, not a speedup claim: no baseline is measured by this tool.

## Configuration

Both entries are ordinary `vision_backbone` dictionaries. The TensorRT entry
repeats no geometry or dtype, because the sidecar next to the engine is the
contract. A ready-to-copy example lives in
`configs/visions/radio_vit_tensorrt_example.py`.

```python
import torch

from fluxvla.engines import build_vision_backbone_from_cfg

# PyTorch path: the local official export.
pytorch_backbone = dict(
    type='RadioViTBackbone',
    vision_backbone_id='c-radio_v4-h',
    model_path='/path/to/C-RADIOv4-H',
    patch_size=16,
    dtype=torch.bfloat16,
    default_image_resolution=(3, 256, 256),
)

# TensorRT path: the same tokens from the serialized engine. The sidecar
# supplies the geometry, so no geometry or dtype is repeated here.
tensorrt_backbone = dict(
    type='RadioViTBackboneTensorRT',
    vision_backbone_id='c-radio_v4-h',
    engine_path='artifacts/c_radio_v4_h_256.engine',
    metadata_path='artifacts/c_radio_v4_h_256.engine.json',
    device=0,
)

encoder = build_vision_backbone_from_cfg(pytorch_backbone)
accelerated = build_vision_backbone_from_cfg(tensorrt_backbone)
assert accelerated.feature_grid(256, 256) == encoder.feature_grid(256, 256)
```

In a VLA recipe the same two dictionaries go in the `vision_backbone` slot of
the consumer model config and its `inference_model` counterpart; the swap is
only valid for a consumer whose expected token width and raster match the
engine's, and the consumer still owns the device and projection dtype.

`engine_path` and its sidecar are read at construction, which needs no
TensorRT. The engine is deserialized on the first encode, so a config can be
built, validated and inspected on a machine without the optional stack.

## Input range and downstream projectors

The input contract is unchanged from the PyTorch path: `[B, 3, H, W]` RGB in
`[-1, 1]`. `to_unit_range` converts it to RADIO's `[0, 1]` on the PyTorch side,
so the engine consumes unit-range images directly.

`encode_features` runs under `torch.no_grad()` (the weights are frozen) but not
`torch.inference_mode()`, and the output buffer is an ordinary `torch.empty`
tensor. The returned tokens are therefore normal tensors that a trainable
projector can still save for backward, which is what stage-1/stage-2 projector
training needs. Tokens carry no gradient themselves.

`encode_features` also performs host synchronization on CUDA: the `[-1, 1]`
validation it inherits from `to_unit_range` (finiteness and a min/max range
check) reads tensor values, and that forces a device sync. The executor's
enqueue does not sync; the preprocessing step in front of it does. Timing this
path measures both, and the cost is not hidden.

`half_precision_dtype` reports the *preferred* half precision for a consumer
that wants one: `float16` when the engine was built with FP16 tactics,
`bfloat16` when it was built with BF16 tactics, else the framework
`bfloat16` default. It is not what the engine's boundary necessarily
uses. The boundary dtypes are `metadata.input_dtype` and
`metadata.output_dtype`, and an FP16 or BF16 build of a float32 graph keeps
both at `float32`.

## BF16 builds

`--precision bf16` requests `BuilderFlag.BF16` and otherwise behaves like the
FP16 option: the graph is still traced in FP32, the engine's inputs and outputs
stay float32, and the sidecar records `precision: bf16` under the same schema.
Nothing in the runtime changes, so a BF16 engine is loaded by the same
`RadioViTBackboneTensorRT`.

The flag is a preference, not a promise. TensorRT applies BF16 to the layers
that have BF16 kernels and leaves the rest at the precision the graph declares,
so a BF16 engine is mixed precision. At 256x256 the engine built from this
graph runs 227 of its 367 layers in BF16, 7 kernels mixed BF16/FP32, and the
Constant/Resize layers in FP32. The build record lists the flags that were
actually set (`precision_flags: ["BF16"]`), and the engine inspector can be
read to see the mix, but the ONNX graph itself is never retyped.

BF16 also depends on the build environment. A TensorRT release without
`BuilderFlag.BF16`, a platform reporting `platform_has_fast_bf16=False`, or a
GPU below compute capability 8.0 fails the build with an actionable error
instead of silently selecting FP16 tactics.

`BuilderFlag.BF16` is also marked deprecated from TensorRT 10.12, where strong
typing supersedes the precision flags. It still selected BF16 tactics in the
10.13.3.9 build used here, which was confirmed per layer, but a TensorRT
release that stops honouring it would produce a BF16-labelled engine with FP32
numerics and FP32 timings. Re-check the layer mix when the TensorRT version
changes. A dedicated BF16 build of the same graph is one flag away:

```bash
python tools/c_radio_tensorrt.py build-engine \
    --onnx artifacts/c_radio_v4_h_256.onnx \
    --out artifacts/c_radio_v4_h_256_bf16.engine \
    --precision bf16 --workspace-mib 2048
```

At 256x256 on an RTX 5090 the BF16 engine is faster than FP32 (3.79 ms vs
9.43 ms median at batch 1) but 25-38% slower than FP16 across batches 1-8. The
cause of that gap has not been isolated: no profiling run was made for it, so
the layers that stay FP32 and BF16 reformats are both plausible contributors
without evidence here. FP16 stays the fastest option, and the FP32 engine
stays the most accurate. Treat BF16 as an opt-in for cases where BF16's
dynamic range or numerics are useful, and validate the downstream task
accuracy before relying on it. The engine boundary stays float32, so a BF16
consumer does not get a BF16 interface from it.

## Layer precision overrides

`--layer-precision PATTERN:DTYPE` (repeatable) constrains the precision of the
layers whose *name* matches `PATTERN`, using `*` and `?` wildcards. It is the
API equivalent of `trtexec --layerPrecisions=*/LayerNormalization:fp32`, and
`--precision-constraints obey` is `--precisionConstraints=obey`. Both are
opt-in. A build without `--layer-precision` sets exactly the builder flags it
set before this option existed and its build record gains no keys; its sidecar
does carry the two policy fields, as `null`, because they are optional fields
of the record rather than keys added only when used. A sidecar written before
this option existed still loads, and a sidecar written now is rejected by older
code, which does not know those two keys.

Layer names come from the ONNX nodes, so a pattern is discovered rather than
guessed:

```bash
python tools/trt_layer_report.py --onnx artifacts/c_radio_v4_h_256.onnx \
    --match '*/LayerNormalization' --match 'LayerNormalization'
# graph.pattern_matches: 64 for '*/LayerNormalization', all NORMALIZATION,
# and 0 for 'LayerNormalization'
```

`LayerNormalization` is the ONNX operator name and `NORMALIZATION` is the
TensorRT layer *type*; what a pattern matches is the layer *name*, which for
this graph ends in `/LayerNormalization`. The bare type matches nothing, and
this pipeline fails the build with that count instead of quietly producing an
unconstrained engine, which is what `trtexec` does with such a pattern.
`tools/trt_layer_report.py --engine` reads the per-layer datatypes back out of a
built engine; that needs the engine to have been built with
`--profiling-verbosity detailed`, and it says so when it has not.

What a layer precision constraint is worth is defined by TensorRT rather than
by this pipeline, so the behaviour here follows the TensorRT precision-control
documentation
([Precision Control](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html))
in four places:

- A layer precision is a request for a preferred type, not an instruction.
  TensorRT normally takes that implementation only when it is also the faster
  one, and otherwise keeps its own choice and warns. `PREFER_PRECISION_CONSTRAINTS`
  makes the request carry more weight while still allowing a fallback to the
  fastest implementation, and `OBEY_PRECISION_CONSTRAINTS` is what turns the
  fallback itself into a build failure. Only `obey` refuses to override a
  constraint, which is why it is the mode this pipeline defaults to when an
  override is requested, and why `--precision-constraints none` together with
  an override is rejected instead of being trusted to hold.
- TensorRT ignores both constraint flags when no layer carries a constraint, so
  a constraints mode with no override is rejected here rather than silently
  doing nothing.
- TensorRT's own documentation states that the normalization layer accumulates
  in FP32 by default even in a mixed-precision build, independent of the
  builder flags. If that holds for this graph, it would explain why the
  LayerNorm override below changes the result so little: the reduction is
  already FP32 and the constraint mainly moves the FP16/FP32 conversion points.
  The per-probe measurements below are consistent with that, and the engine
  inspector shows the normalization implementation reading FP32 inputs in the
  plain FP16 build as well.
- `ILayer::setPrecision` is deprecated from TensorRT 10.12 in favour of strong
  typing, and `OBEY_PRECISION_CONSTRAINTS` is removed in TensorRT 11. These
  overrides are a 10.x weak-typing feature; the deprecation warning is
  suppressed at the assignment, and no failure path turns an override into an
  unconstrained build.

### Sensitivity of FP16 accuracy to grouped constraints at 256x256

Measured on `assets/frame_000000.jpg` against the strict FP32 PyTorch encoder at
batch 1, with elementwise `allclose(atol=rtol=1e-2)` as the pass mark. Every
build record, parity report and log is under
`checkpoints/Pretrained_models/C-RADIOv4-H/exported/256x256/benchmarks/fp16_accuracy/`.

These rows are **sensitivity experiments over groups of layers**, not
identifications of a culprit. Forcing a whole group to FP32 says how much that
group matters on this sample; it does not say that the group is where the
difference is produced, and a group that helps may be doing so only because it
sits on the path the error travels through.

| Build | Layers forced to FP32 | relative L2 | max abs | mean abs | allclose |
| --- | --- | --- | --- | --- | --- |
| FP32 | - | 1.66e-06 | 4.3e-05 | 1.0e-06 | yes |
| FP16 | - | 0.00518829 | 0.140724 | 0.00312487 | no |
| FP16 | LayerNorm (64) | 0.00513138 | 0.126075 | 0.00316332 | no |
| FP16 | LayerNorm, Softmax (96) | 0.00553968 | 0.155860 | 0.00321323 | no |
| FP16 | LayerNorm, attention MatMul (192) | 0.00484839 | 0.104591 | 0.00284140 | no |
| FP16 | LayerNorm, attention and MLP MatMul (256) | 0.00393239 | 0.110939 | 0.00229231 | no |

Forcing every `LayerNormalization` to FP32, the recipe this section exists for,
moves relative L2 from 0.00518829 to 0.00513138 and the maximum absolute
difference from 0.140724 to 0.126075, for about 2% more latency at batch 1 and
nothing measurable above batch 4. Adding `Softmax` to that set makes every
metric slightly worse than the plain FP16 build; the measurements do not
establish why, and the plausible explanation that the extra FP32/FP16
conversion points cost more than the FP32 softmax gains has not been tested.
The largest sensitivity in the table is to the MatMuls: attention alone accounts
for about 7% of relative L2, attention plus MLP about 24%, at 38% and 93% more
latency.

None of these builds reaches elementwise `allclose(1e-2)`, and the largest
group tested, all 256 normalization and MatMul layers, is still five hundred
times further from the FP32 reference than the FP32 engine is. That is a bound
on how far grouped FP32 overrides get on this sample, not evidence about where
the remaining difference is produced.

### Where the difference appears inside the network

Grouped constraints cannot say where the difference is produced, so the same
build was instrumented directly: the input to block 0 and the `norm1` output of
blocks 0, 7, 15, 23 and 31 were marked as extra network outputs, and each was
compared against the matching activation of the strict FP32 PyTorch encoder for
the same sample. The saved run records the named tensors and per-probe metrics
in `benchmarks/fp16_accuracy/probe_intermediates.json`. The temporary probe
script and diagnostic engines were removed after these experiments.

`fp32` below is the same instrumented graph built in FP32 with TF32 cleared, the
control that shows the probes themselves are sound.

| Probe (network tensor) | FP32 probe, relative L2 | FP16 probe, relative L2 | FP16 max abs | growth vs previous probe |
| --- | --- | --- | --- | --- |
| `/model/patch_generator/cls_token/Concat_1_output_0`, input to block 0 `norm1` | 6.62e-07 | 0.001807 | 0.0203 | - |
| `blocks.0/norm1/LayerNormalization` output | 8.97e-07 | 0.001860 | 0.0363 | 1.03x |
| `blocks.7/norm1/LayerNormalization` output | 1.76e-06 | 0.005966 | 0.0628 | 3.21x |
| `blocks.15/norm1/LayerNormalization` output | 1.64e-06 | 0.005190 | 0.0370 | 0.87x |
| `blocks.23/norm1/LayerNormalization` output | 2.65e-06 | 0.007238 | 0.0687 | 1.39x |
| `blocks.31/norm1/LayerNormalization` output | 2.83e-06 | 0.008823 | 0.174 | 1.22x |
| engine `tokens` output | 1.66e-06 | 0.005131 | 0.126 | 0.58x |

What this supports, and what it does not:

- The FP16 difference is already three orders of magnitude above the FP32
  control at the first probe, the tensor the transformer blocks start from, and
  it then grows roughly with depth - about 4.9x end to end across the probed
  span, with the largest single step between blocks 0 and 7.
- No probed boundary shows a step change large enough to stand out as the one
  place the difference is produced. At one probe per eight blocks, this
  localizes the growth to "from the start of the transformer stack, increasing
  with depth" and not to any single probed block boundary. Structures inside a
  block, and the 24 unprobed boundaries, are not resolved.
- The final token error is smaller than the last probed boundary, so the head
  reduces the relative difference rather than adding to it.
- Marking intermediate tensors constrains fusion, so these engines are not the
  shipped engines. For FP16 the instrumented engine's own token error matches
  the shipped `c_radio_v4_h_256_fp16_ln_fp32.engine` exactly
  (relative L2 0.00513138, maximum 0.126075), and for FP32 it matches the
  shipped FP32 engine (1.66e-06), so the instrumentation did not move the
  end-to-end numbers here. That is a property of this graph, not a guarantee.
- The build logs contain no precision warnings, which only means TensorRT did
  not report a constraint it could not honour. It is not evidence that no layer
  diverges, and it identifies no layer.

Within these limits the measurements say that an FP16 engine's difference from
the FP32 reference on this sample is broad and small per boundary rather than
concentrated, and the grouped constraints above can recover part of it. The FP32
engine remains the only build that passes elementwise parity.

The engine that ships for that trade is
`c_radio_v4_h_256_fp16_ln_fp32.engine`, built as:

```bash
python tools/c_radio_tensorrt.py build-engine \
    --onnx artifacts/c_radio_v4_h_256.onnx \
    --out artifacts/c_radio_v4_h_256_fp16_ln_fp32.engine \
    --precision fp16 --precision-constraints obey \
    --layer-precision '*/LayerNormalization:fp32' \
    --profiling-verbosity detailed --log-file build_fp16_ln_fp32.log
```

`--profiling-verbosity detailed` is what lets the engine's own inspector report
per-layer datatypes later; it costs nothing measurable, as the latency table
below shows. `--log-file` keeps TensorRT's own diagnostics; for this build they
contained no precision warnings, which means TensorRT accepted every requested
constraint and nothing more.

### Latency

RTX 5090, 256x256, medians in milliseconds, 30 warmup and 100 timed iterations
per batch, one process per backend, the sample resident on the GPU.

| Build | batch 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- |
| PyTorch FP32 (strict) | 19.299 | 22.206 | 38.295 | 62.239 |
| FP32 | 9.561 | 17.407 | 31.395 | 64.319 |
| FP16 | 2.707 | 4.710 | 6.776 | 12.793 |
| FP16, detailed verbosity control | 2.736 | 4.722 | 6.876 | 12.741 |
| FP16, LayerNorm FP32 | 2.768 | 4.736 | 6.744 | 12.724 |
| FP16, LayerNorm and Softmax FP32 | 2.810 | 4.776 | 6.855 | 12.891 |
| FP16, LayerNorm and attention MatMul FP32 | 3.728 | 6.158 | 8.937 | 16.682 |
| FP16, LayerNorm and attention + MLP MatMul FP32 | 5.231 | 8.475 | 12.273 | 23.866 |

The control row is the same graph and precision rebuilt in the same session at
detailed verbosity: it is numerically identical to the shipped FP16 engine
(relative L2 0.00518829, maximum 0.140724) and within about 1% on latency, so
the FP16 row is a fair baseline and detailed verbosity is not a cost. The
LayerNorm override is close to free, +2.2% at batch 1 and inside run-to-run
noise elsewhere; the MatMul overrides cost 38% and 93% for the accuracy gains
in the table above.

## Engine metadata and validation

The sidecar records the C-RADIO variant, feature width, patch size, resolution
step, image geometry, feature grid, builder `precision`, the graph boundary
`input_dtype`/`output_dtype`, input/output names, output layout, batch bounds,
the TensorRT version that built it, and provenance (`source_onnx`,
`onnx_sha256`, `external_data`, `external_data_sha256`, `workspace_bytes`,
`created_utc`). `precision` and the dtypes are independent fields: the former
is a builder choice, the latter describe the tensors. Records are strict:
unknown keys, missing fields and internally inconsistent values are rejected
on read, and the schema version is checked.

At load the runtime validates the sidecar against the real engine: exactly one
input and one output, their names and dtypes, the fixed image geometry, the
output shape implied by the declared layout, and the optimization profile
against the declared batch bounds. A stale or hand-edited sidecar, a graph
that was re-exported after the sidecar was written, or an engine built for a
different geometry all fail with a specific message instead of producing
subtly wrong tokens.

Layouts `NLD` (`[B, N, D]`, the export default) and the raw raster layouts
`NCHW` and `NHWC` are accepted and normalized to `[B, N, D]`.

## Runtime behaviour and limits

The runtime addresses tensors by name, allocates input and output through
PyTorch on the caller's current stream, and keeps one execution context per
CUDA stream so two streams never share mutable context state. It validates
device, dtype, shape and batch bounds before enqueueing, and reports
deserialization, `set_input_shape`/`set_tensor_address` and enqueue failures
with the engine path. One host lock serializes loading, context creation,
teardown and the shape/address mutation window, so concurrent first use or two
callers on the same stream cannot interleave; the enqueue itself stays
asynchronous, so device work on different streams still overlaps. The
returned tensor lives on the same stream as the input and the caller orders
dependent work by using that stream.

Input and output buffers are recorded on the execution stream for the caching
allocator and kept referenced until the next enqueue on that stream, so
neither can be recycled while the stream still reads or writes it. The
TensorRT logger and runtime are retained for as long as the engine lives,
because the engine points into them. `close()` explicitly waits on every
recorded execution stream before destroying
contexts, buffers, the engine and those handles, so nothing is released while
the GPU may still be reading it. An injected runtime belongs to its caller and
is never closed by the backbone.

TensorRT can also synchronize internally when enqueueing on CUDA's default
stream. Use a non-default current CUDA stream for asynchronous execution, and
order input production and output consumption across streams explicitly.

Portability limits, in short:

- A TensorRT engine is built for a specific TensorRT version, CUDA/driver and
  GPU architecture. It is not portable between them. Rebuild per deployment
  target; the metadata records which version produced it.
- FP32 is the baseline. `--precision fp16` and `--precision bf16` are
  build-time opt-ins that leave the graph's `float32` inputs and outputs in
  place. BF16 needs `BuilderFlag.BF16` in the TensorRT release and a device with
  BF16 support; it is a preference, so a BF16 engine keeps FP32 on the layers
  without BF16 kernels, has larger feature error than the FP16 engine on the
  verified sample, and should have downstream task accuracy validated before
  use. `BuilderFlag.BF16` is deprecated from TensorRT 10.12 (strong typing
  supersedes the precision flags) although 10.13.3.9 still honours it, so
  re-check the layer mix after a TensorRT upgrade. INT8 calibration is not
  implemented and is not claimed.
- `--layer-precision` is a TensorRT 10.x weak-typing feature: it goes through
  `ILayer::setPrecision`, which TensorRT deprecates in favour of strong typing
  from 10.12, and `BuilderFlag::kOBEY_PRECISION_CONSTRAINTS` is removed in
  TensorRT 11. It changes the tactics and the conversion points of the named
  layers only, and on the verified sample no set of them came close to the
  elementwise agreement a full FP32 build reaches, so the FP32 engine stays the
  reference for accuracy.
- `metadata.tensorrt_version` must share the runtime's major version; a
  different minor version loads with a warning, because per-minor builds are
  not portable and the kernel selection or numerics may differ.
- `num_patches` reports the token count for the one compiled geometry.
  `default_image_resolution` must equal that geometry and is validated at
  construction, so the property cannot silently disagree with the engine.
- Engine construction still costs a sidecar read; only the engine itself is
  lazy. The tensor device and the consumer's projection dtype stay the
  caller's responsibility.

## Verification

`test/test_models/test_radio_trt_backbone.py` covers the registry/config build,
metadata validation, fixed-geometry rejection, range conversion, layout
normalization, freezing and autograd-safe outputs, and the eager/lazy
dependency boundary on CPU with an injected engine stand-in.
`test/test_models/test_base_trt_engine.py` drives the TensorRT 10.x glue
against a stand-in module: version rejection, named-IO validation,
optimization-profile checks, current-stream enqueue, per-stream contexts,
buffer lifetime, retained logger/runtime references, close-before-release
ordering and its wait for outstanding GPU work, `False` returns from
`set_input_shape`/`set_tensor_address`, allocator `record_stream` calls, and
concurrent same-stream calls.
`test/test_scripts/test_trt_export.py` covers the metadata a build
records, the ONNX boundary and stale-graph checks, a builder stand-in that
keeps FP32 IO while the FP16 or BF16 flag is set, the BF16 failure paths
(missing builder flag, no fast-BF16 platform, pre-Ampere device) and the
guarantee that a failed BF16 build never becomes an FP16 build, the traceable
graph boundary, the
parity/benchmark helpers and their device and argument validation, and holds
the gated end-to-end export -> build -> validate -> parity unit.
`test/test_scripts/test_c_radio_tensorrt.py` covers the CLI parser, `inspect`,
error reporting, the FP32 parity reference and the parity exit code.
`test/test_scripts/test_trt_layer_report.py` covers the layer report's
inspector parsing and its pattern counting, and
`test/test_scripts/test_trt_export.py` holds the layer-override units: the
`PATTERN:DTYPE` syntax, matching only the named layers, a pattern that matches
nothing failing instead of no-opping, the constraints-mode rules, the record
shape with no override, and the descriptor-level `--log-file` capture.
`test/test_configs/test_radio_trt_config.py` checks the example config against
the registered constructor contract.

Real C-RADIOv4-H exports at 256x256 were verified on
`assets/frame_000000.jpg`, using RGB bicubic resize and identical inputs for
PyTorch, ONNX Runtime and TensorRT. Artifacts and reports are in
`checkpoints/Pretrained_models/C-RADIOv4-H/exported/256x256/`; export dependencies
are isolated in `/tmp/cradio-export-venv`. The ONNX graph needs its `.onnx.data`
file alongside it. The engines were built for TensorRT 10.13.3.9 on RTX 5090.

ONNX and the strict FP32 engine pass elementwise parity. FP16 achieves cosine
about 0.999987 and relative L2 about 0.00519, but fails elementwise
`allclose(atol=1e-2, rtol=1e-2)` (maximum absolute difference about 0.141).
Forcing every `LayerNormalization` to FP32 in that FP16 build, which is what
`c_radio_v4_h_256_fp16_ln_fp32.engine` does, moves relative L2 to about
0.00513 and the maximum to about 0.126 and still fails elementwise
`allclose`. "Layer precision overrides" records the grouped sensitivity
experiments, the per-probe measurements that localize the difference to the
whole stack growing with depth, why the LayerNorm constraint is nearly free,
and why no bounded set of per-layer FP32 overrides reaches elementwise parity.
The FP32 engine is provided when tighter numerical preservation is needed.

The BF16 engine (`c_radio_v4_h_256_bf16.engine`, built from the same ONNX at
256x256) achieves cosine about 0.999717 and relative L2 about 0.02381 against
the strict FP32 encoder, with a maximum absolute difference of about 1.06, so
it also fails elementwise `allclose`. Among the three engines it has the
largest feature error: FP16 is closer to the strict FP32 reference (relative
L2 about 0.00519) and FP32 is closest (about 1.7e-06). As a scale reference, a
fully BF16 PyTorch encoder (BF16 weights under autocast) is further from the
FP32 encoder than the mostly-BF16 TensorRT engine is: cosine 0.999510,
relative L2 0.03130, maximum absolute difference 3.94. The engine and the
PyTorch BF16 path differ by cosine 0.999409 and relative L2 0.03439; they are
the same family of rounding but not the same computation, and no exact match
to autocast is claimed.

Median latency for the same image at batches 1/2/4/8 is 9.43/17.20/30.91/63.25
ms for the FP32 engine, 2.75/4.72/6.82/12.86 ms for FP16 and
3.79/6.35/8.51/16.35 ms for BF16, against 19.60/22.22/38.26/62.07 ms for the
strict FP32 PyTorch encoder and 6.44/8.59/14.11/18.37 ms for PyTorch BF16
autocast.

All of this is feature parity and latency on one image, not dataset-level task
accuracy.
