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
#    graph inputs/outputs. There is no INT8 or BF16 calibration.
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
that wants one: `float16` when the engine was built with FP16 tactics, else
the framework `bfloat16` default. It is not what the engine's boundary
necessarily uses. The boundary dtypes are `metadata.input_dtype` and
`metadata.output_dtype`, and an FP16 build of a float32 graph keeps both at
`float32`.

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
- FP32 is the baseline. `--precision fp16` is a build-time opt-in. INT8 and
  BF16 calibration are not implemented and are not claimed. The flag selects
  FP16 tactics and leaves the graph's `float32` inputs and outputs in place.
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
keeps FP32 IO while the FP16 flag is set, the traceable graph boundary, the
parity/benchmark helpers and their device and argument validation, and holds
the gated end-to-end export -> build -> validate -> parity unit.
`test/test_scripts/test_c_radio_tensorrt.py` covers the CLI parser, `inspect`,
error reporting, the FP32 parity reference and the parity exit code.
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
The FP32 engine is provided when tighter numerical preservation is needed.
This is feature parity on one image, not dataset-level task accuracy.
