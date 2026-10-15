CKPT=checkpoints/Pretrained_models/C-RADIOv4-H
RES=512
ENG=$CKPT/exported/${RES}x${RES}/c_radio_v4_h_${RES}x${RES}_b8_fp16.engine

# python tools/pca_feature_visualize.py --backbone cradio \
#     --model-path $CKPT --image assets/frame_000000.jpg --image-size $RES \
#     --output runs/visualization/c_radio_pca_$RES/pytorch_h

python tools/pca_feature_visualize.py --backbone cradio-trt \
    --engine-path $ENG --image assets/frame_000000.jpg --image-size $RES \
    --pca-reference runs/visualization/c_radio_pca_$RES/pytorch_h.pt \
    --output runs/visualization/c_radio_pca_$RES/fp16_trt