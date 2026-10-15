# C-RADIO example and reusable helpers

Read this only for a C-RADIO export or when locating the existing implementation
to adapt for another backbone. Run commands from the repository root in a
Python environment containing the export dependencies. They are examples,
not commands to execute merely because this reference was read.

## Existing C-RADIO artifacts

The local checkpoint is `checkpoints/Pretrained_models/C-RADIOv4-H/`.
At 256x256, the reusable graph is
`exported/256x256/c_radio_v4_h_256.onnx` with its `.onnx.data` and `.onnx.json`.
The constrained engine is `c_radio_v4_h_256_fp16_ln_fp32.engine` with its
`.engine.json`. Check that these files exist and match the requested checkpoint,
geometry and profile before reusing them; their presence is not guaranteed.

The current graph's `*/LayerNormalization` pattern matches 64 parsed
`NORMALIZATION` layers. Verify this in the target checkout:

```bash
python tools/trt_layer_report.py \
    --onnx checkpoints/Pretrained_models/C-RADIOv4-H/exported/256x256/c_radio_v4_h_256.onnx \
    --match '*/LayerNormalization' --out layer_matches.json
```

Build from the existing graph without another ONNX export:

```bash
python tools/c_radio_tensorrt.py build-engine \
    --onnx checkpoints/Pretrained_models/C-RADIOv4-H/exported/256x256/c_radio_v4_h_256.onnx \
    --out checkpoints/Pretrained_models/C-RADIOv4-H/exported/256x256/c_radio_v4_h_256_fp16_ln_fp32.engine \
    --precision fp16 --workspace-mib 4096 \
    --layer-precision '*/LayerNormalization:fp32' \
    --precision-constraints obey --profiling-verbosity detailed \
    --log-file build_fp16_ln_fp32.log
```

Choose a new output name when preserving an existing build. Inspect the result
with `tools/trt_layer_report.py --engine ENGINE --out engine_layers.json` and
`tools/c_radio_tensorrt.py validate --engine ENGINE`.

Compare the real image against PyTorch FP32:

```bash
python tools/c_radio_tensorrt.py parity \
    --engine checkpoints/Pretrained_models/C-RADIOv4-H/exported/256x256/c_radio_v4_h_256_fp16_ln_fp32.engine \
    --radio-path checkpoints/Pretrained_models/C-RADIOv4-H \
    --image assets/frame_000000.jpg --out parity.json \
    --atol 0.01 --rtol 0.01
```

The parity command exits 1 when `allclose` fails; retain its JSON and distinguish
a numerical mismatch from a crash. Use `--no-batch2` when the compiled profile
does not support batch 2. Do not hide that exit with an unconditional success.

For a different image geometry, the existing `myscripts/export_cradio.sh`
(if present; it may be git-ignored) exposes `IMAGE_HEIGHT`, `IMAGE_WIDTH`,
`BATCH_SIZE`, `PRECISION`, `LAYER_PRECISION` and `ENGINE_SUFFIX`. Preserve the
user's geometry/batch settings and use `fp16`, `*/LayerNormalization:fp32` and
`_ln_fp32` for the last three. C-RADIO needs dimensions divisible by 16. This
helper re-exports ONNX before building; reuse a compatible graph directly
when no new export is needed.

## Code to reuse

- `tools/trt_export.py`: `export_onnx`, `convert_onnx_to_external_data`,
  `verify_onnx_graph`, `build_engine`, `parse_layer_precision`,
  `apply_layer_precisions` and `capture_std_streams` are shared helpers.
- `tools/c_radio_export.py`: `load_image`, `upstream_attention`, `strict_fp32`,
  `feature_metrics`, `engine_parity` and `benchmark_backbone` are the existing
  C-RADIO evaluation path. Read signatures before using them. The accuracy
  helper uses float64 reductions; the benchmark synchronizes each iteration.
- `radio_vit.py` / `radio_trt.py`: the PyTorch and TRT high-level wrappers.
  C-RADIO wrapper input is `[-1, 1]`; the exported graph receives `[0, 1]`.
  Reuse the existing conversion once. Other models need their own contract.
- `benchmarks/fp16_accuracy/` under the 256x256 artifact folder contains prior
  latency/accuracy JSON when available. It is historical evidence, not a
  substitute for measuring a newly exported model.
- `docs/c-radio-v4-tensorrt.md` documents the current precision override APIs
  and known feature-error limits. Those results must not be advertised as a
  guarantee for another image, backbone or resolution.
