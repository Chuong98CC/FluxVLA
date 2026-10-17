"""PatchWAM RobotWin with frozen C-RADIO current features and future forcing.

The data, augmentation, normalization and three-camera 288x256 mosaic are
inherited from the released PatchWAM RobotWin recipe. The teacher encoder is a
separate frozen runtime: set `PATCHWAM_TEACHER_MODEL_PATH` for PyTorch RADIO, or
`PATCHWAM_TEACHER_BACKEND=trt` plus `PATCHWAM_TEACHER_TRT_ENGINE` for a matching
TensorRT engine. Both paths require native 18x16 spatial features; no feature
resampling is performed.

The combined model uses:

* current conditioning: frozen current descriptors -> M -> current raster;
* future prediction: J predicts detached future teacher features;
* future feedback: predicted features -> G between Euler evaluations.

Set `PATCHWAM_TEACHER_SHA256` to the reviewed teacher checkpoint digest. The
recipe records that identity in checkpoints; changing it requires base_init.
"""

import copy
import json
import os
from pathlib import Path

_base_ = './robotwin/patchwam_robotwin.py'

_teacher_model_path = (
    os.environ.get('PATCHWAM_TEACHER_MODEL_PATH')
    or 'checkpoints/Pretrained_models/C-RADIOv4-H')
_teacher_trt_engine = (
    os.environ.get('PATCHWAM_TEACHER_TRT_ENGINE')
    or 'checkpoints/Pretrained_models/C-RADIOv4-H/exported/288x256/'
       'c_radio_v4_h_288x256.engine')
_teacher_metadata = (
    os.environ.get('PATCHWAM_TEACHER_TRT_METADATA')
    or _teacher_trt_engine + '.json')
_teacher_backend = os.environ.get(
    'PATCHWAM_TEACHER_BACKEND', 'pytorch').lower()
_teacher_sha256 = os.environ.get('PATCHWAM_TEACHER_SHA256') or (
    None if _teacher_backend == 'trt'
    else '851f283468118830a08e70173f1ab9a30f613c704958fc0b3c8e638aef738f81')
_teacher_engine_sha256 = os.environ.get(
    'PATCHWAM_TEACHER_ENGINE_SHA256')
_teacher_representation_id = (
    os.environ.get('PATCHWAM_TEACHER_REPRESENTATION_ID')
    or 'c-radio_v4-h-spatial_features-native_18x16')
_teacher_input_size = [288, 256]
_teacher_grid = [18, 16]


def _teacher_runtime():
    if _teacher_backend == 'trt':
        return dict(
            type='RadioViTBackboneTensorRT',
            vision_backbone_id='c-radio_v4-h',
            engine_path=_teacher_trt_engine,
            metadata_path=_teacher_metadata,
            device=None,
            default_image_resolution=(3, *_teacher_input_size),
        )
    if _teacher_backend != 'pytorch':
        raise ValueError(
            'PATCHWAM_TEACHER_BACKEND must be pytorch or trt')
    return dict(
        type='RadioViTBackbone',
        version='c-radio_v4-h',
        model_path=_teacher_model_path,
        feature_dim=1280,
        patch_size=16,
        minimum_resolution=16,
        default_image_resolution=(3, *_teacher_input_size),
        dtype='bfloat16',
        device='cuda',
    )


def _teacher_provenance():
    if _teacher_backend == 'pytorch':
        return {
            'radio': dict(
                representation_id=_teacher_representation_id,
                base_dir=_teacher_model_path,
                artifacts={'model.safetensors': _teacher_sha256},
                representation_artifacts=['model.safetensors'],
            )
        }
    if not _teacher_engine_sha256:
        raise ValueError(
            'PATCHWAM_TEACHER_ENGINE_SHA256 is required for the TRT backend')
    if not _teacher_sha256:
        raise ValueError(
            'PATCHWAM_TEACHER_SHA256 is required for the TRT backend')
    metadata = json.loads(Path(_teacher_metadata).read_text())
    base_dir = Path(_teacher_trt_engine).expanduser().resolve().parent
    artifacts = {
        Path(_teacher_trt_engine).name: _teacher_engine_sha256}
    representation_artifacts = []
    for field, hash_field in (
            ('source_onnx', 'onnx_sha256'),
            ('external_data', 'external_data_sha256')):
        name = metadata.get(field)
        expected = metadata.get(hash_field)
        if (isinstance(name, str) and isinstance(expected, str)
                and (base_dir / name).is_file()):
            artifacts[name] = expected
            representation_artifacts.append(name)
    if not representation_artifacts:
        raise ValueError(
            'TRT provenance requires the local ONNX source files recorded in '
            'the engine sidecar')
    return {
        'radio': dict(
            representation_id=_teacher_representation_id,
            base_dir=str(base_dir),
            artifacts=artifacts,
            representation_artifacts=representation_artifacts,
        )
    }


_teacher_forcing = dict(
    encoders=[
        dict(
            name='radio',
            feature_dim=1280,
            representation_id=(
                _teacher_representation_id),
            checkpoint_sha256=_teacher_sha256,
            stream='spatial_features',
            preprocessing='world_rgb[-1,1]->unit_range_once',
            input_size=_teacher_input_size,
            native_grid=_teacher_grid,
            view_context='independent_rasters',
            view_order='view_major_row_major',
            special_tokens='excluded',
            alignment='native_exact',
            loss_weight=0.1,
        )
    ],
    current_conditioning=True,
    future_prediction=True,
    future_feedback=True,
    predictor=dict(type='repa_mlp', hidden_dim=2048),
    current_projector_init='zero',
    feedback_projector_init='zero',
    self_cond_drop_prob=0.1,
    true_feature_prob=0.05,
    feature_input_norm='layer_norm',
    norm_eps=1.0e-5,
    cosine_eps=1.0e-6,
)

model = dict(
    type='TeacherForcingPatchWAMVLA',
    teacher_checkpoint_mode='base_init',
    teacher_extractors=dict(radio=_teacher_runtime()),
    teacher_provenance=_teacher_provenance(),
    vla_head=dict(
        type='TeacherForcingPatchWAMHead',
        teacher_forcing=_teacher_forcing,
    ),
)

inference_model = copy.deepcopy(model)
