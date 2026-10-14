# Spatial-feature PCA visualization

`tools/pca_feature_visualize.py` shows what a frozen vision backbone encodes
for one image: three principal components of its spatial tokens, next to the
image that was actually encoded.

```bash
CKPT=checkpoints/Pretrained_models/C-RADIOv4-H
ENG=$CKPT/exported/256x256/c_radio_v4_h_256.engine

# C-RADIO through the frozen PyTorch wrapper (reference run).
python tools/pca_feature_visualize.py --backbone cradio \
    --model-name nvidia/C-RADIOv4-H --model-path $CKPT \
    --image assets/frame_000000.jpg --image-size 256 \
    --device cuda --output runs/visualization/c_radio_pca_256/pytorch_h

# The same image through a serialized engine, projected onto the reference
# PCA basis so the two panels share axes, signs and scaling.
python tools/pca_feature_visualize.py --backbone cradio-trt \
    --engine-path $ENG --image assets/frame_000000.jpg --image-size 256 \
    --device cuda --pca-reference runs/visualization/c_radio_pca_256/pytorch_h.pt \
    --output runs/visualization/c_radio_pca_256/fp16_trt

# A local DINOv2/DINOv3 ViT.
python tools/pca_feature_visualize.py --backbone dino \
    --model-path /path/to/dinov2-vit --image assets/frame_000000.jpg \
    --image-size 224 --output runs/visualization/dino
```

Each run writes `<output>.png` (input and PCA panels), `<output>.pt` (tokens,
grid, PCA basis, diagnostics) and `<output>.json` (the same diagnostics as
text). `--pca-reference <bundle.pt>` reuses another run's basis, sign and
scaling, and additionally reports numerical feature differences against that
bundle, so independently fitted axes cannot look like an inference
difference.

Notes:

- `--model-name` selects the C-RADIO variant (`c-radio_v4-h`,
  `c-radio_v4-so400m`, `nvidia/C-RADIOv4-H`, `nvidia/C-RADIOv4-SO400M`), with
  `--version` as an alias that must agree; omitting both keeps
  `c-radio_v4-h`. For DINO it is the Hugging Face model id.
- `--image-size` takes one square size or an explicit `H W`. The requested size
  must be one the encoder accepts exactly: a C-RADIO PyTorch size that would be
  rounded, and a TensorRT size that is not the compiled geometry, both fail
  with the supported size named.
- `--dtype fp16|bf16` uses autocast over FP32 weights, because the C-RADIO
  input conditioner emits FP32; fp16 is rejected on CPU, and `cradio-trt`
  rejects `--dtype` entirely because the engine's tactic precision is fixed at
  build time.
- Reference bundles are tensor-only and reloaded with `weights_only=True`;
  bundles from an older release must be regenerated.
- The tool is offline by default. A model name for DINO needs `--allow-remote`;
  the tool never sets proxies or endpoints.
- `--show` keeps the configured interactive matplotlib backend (a headless
  display reports a clear error); without it the figure is written headlessly.
- `--no-figure` skips matplotlib and still writes the feature bundle.
