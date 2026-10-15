#!/usr/bin/env bash
set -euo pipefail

# Edit these settings, then run: bash myscripts/export_cradio.sh
IMAGE_HEIGHT=512
IMAGE_WIDTH=512
BATCH_SIZE=8
# fp32 (baseline), fp16 or bf16 (bf16 needs BuilderFlag.BF16 and BF16 hardware).
PRECISION=fp16
# Optional per-layer precision override, matched on layer names with *, or
# empty for a plain build. The suffix keeps it from overwriting the plain
# engine of the same precision, and must be empty for a plain build.
LAYER_PRECISION="*/LayerNormalization:fp32"
ENGINE_SUFFIX="_ln_fp32"

PYTHON="${PYTHON:-/tmp/cradio-export-venv/bin/python}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$REPO_ROOT/checkpoints/Pretrained_models/C-RADIOv4-H"
TOOL="$REPO_ROOT/tools/c_radio_tensorrt.py"

for value in "$IMAGE_HEIGHT" "$IMAGE_WIDTH" "$BATCH_SIZE"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        printf 'Image dimensions and batch size must be positive integers.\n' >&2
        exit 1
    fi
done
if (( IMAGE_HEIGHT % 16 || IMAGE_WIDTH % 16 )); then
    printf 'C-RADIO image dimensions must be multiples of 16.\n' >&2
    exit 1
fi
if [[ "$PRECISION" != fp32 && "$PRECISION" != fp16 && "$PRECISION" != bf16 ]]; then
    printf 'PRECISION must be fp32, fp16 or bf16.\n' >&2
    exit 1
fi

OUTPUT_DIR="$MODEL_DIR/exported/${IMAGE_HEIGHT}x${IMAGE_WIDTH}"
STEM="$OUTPUT_DIR/c_radio_v4_h_${IMAGE_HEIGHT}x${IMAGE_WIDTH}_b${BATCH_SIZE}"
ONNX="$STEM.onnx"
ENGINE="${STEM}_${PRECISION}${ENGINE_SUFFIX}.engine"

# Keep temporary per-tensor weight files out of the final export directory.
# The exporter consolidates them into one .onnx.data sidecar.
mkdir -p "$OUTPUT_DIR"
EXPORT_TMP="$(mktemp -d "$OUTPUT_DIR/.onnx-export.XXXXXX")"
trap 'rm -rf -- "$EXPORT_TMP"' EXIT
STAGED_ONNX="$EXPORT_TMP/$(basename -- "$ONNX")"

# Dynamic batch range: 1..BATCH_SIZE, optimized for BATCH_SIZE.
"$PYTHON" "$TOOL" export-onnx \
    --radio-path "$MODEL_DIR" --version c-radio_v4-h \
    --image-height "$IMAGE_HEIGHT" --image-width "$IMAGE_WIDTH" \
    --min-batch 1 --opt-batch "$BATCH_SIZE" --max-batch "$BATCH_SIZE" \
    --external-data-min-bytes 0 \
    --device cuda --out "$STAGED_ONNX"

# Publish only the graph, consolidated weights, and export metadata.
for suffix in .data '' .json; do
    mv -- "${STAGED_ONNX}${suffix}" "${ONNX}${suffix}"
done

# An override is only enforced with a constraints mode, so requesting one
# implies obey; a pattern that matches no layer fails the build.
LAYER_ARGS=()
if [[ -n "$LAYER_PRECISION" ]]; then
    LAYER_ARGS=(--layer-precision "$LAYER_PRECISION" --precision-constraints obey)
fi

"$PYTHON" "$TOOL" build-engine \
    --onnx "$ONNX" --out "$ENGINE" \
    --precision "$PRECISION" --workspace-mib 4096 "${LAYER_ARGS[@]}"

printf 'Exported engine: %s\n' "$ENGINE"
