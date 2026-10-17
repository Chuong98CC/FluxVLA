"""Registry and recipe contracts for PatchWAM teacher feature forcing."""

from pathlib import Path

from mmengine.config import Config

import fluxvla.models  # noqa: F401
from fluxvla.models.backbones.vlms import patchwam_backbone  # noqa: F401
from fluxvla.models.heads import teacher_forcing_patchwam_head  # noqa: F401
from fluxvla.models.vlas import teacher_forcing_patchwam_vla  # noqa: F401
from fluxvla.engines import HEADS, VLAS


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / 'configs' / 'patchwam' / (
    'patchwam_teacher_forcing_radio_robotwin.py')


def test_teacher_registry_entries_resolve():
    assert VLAS.get('TeacherForcingPatchWAMVLA') is not None
    assert HEADS.get('TeacherForcingPatchWAMHead') is not None


def test_robotwin_recipe_selects_teacher_model_for_train_and_inference():
    cfg = Config.fromfile(str(CONFIG))
    assert cfg.model.type == 'TeacherForcingPatchWAMVLA'
    assert cfg.inference_model.type == 'TeacherForcingPatchWAMVLA'
    assert cfg.model.vla_head.type == 'TeacherForcingPatchWAMHead'
    assert cfg.inference_model.vla_head.type == (
        'TeacherForcingPatchWAMHead')
    assert cfg.model.teacher_checkpoint_mode == 'base_init'
    assert cfg.inference_model.teacher_checkpoint_mode == 'base_init'
    assert VLAS.get(cfg.model.type) is not None
    assert HEADS.get(cfg.model.vla_head.type) is not None
    assert cfg.inference_model.vla_head.denoiser.gradient_checkpointing is False


def test_recipe_geometry_switches_and_predictor_are_explicit():
    cfg = Config.fromfile(str(CONFIG))
    teacher = cfg.model.vla_head.teacher_forcing
    encoder = teacher.encoders[0]
    assert encoder.name == 'radio'
    assert encoder.feature_dim == 1280
    assert list(encoder.input_size) == [288, 256]
    assert list(encoder.native_grid) == [18, 16]
    assert encoder.alignment == 'native_exact'
    assert encoder.view_context == 'independent_rasters'
    assert encoder.loss_weight == 0.1
    assert teacher.current_conditioning is True
    assert teacher.future_prediction is True
    assert teacher.future_feedback is True
    assert teacher.predictor.type == 'repa_mlp'
    assert teacher.predictor.hidden_dim == 2048
    assert teacher.current_projector_init == 'zero'
    assert teacher.feedback_projector_init == 'zero'
    assert teacher.self_cond_drop_prob == 0.1
    assert teacher.true_feature_prob == 0.05
    provenance = cfg.model.teacher_provenance.radio
    assert provenance.representation_id == encoder.representation_id
    assert provenance.artifacts['model.safetensors'] == (
        encoder.checkpoint_sha256)
    assert list(provenance.representation_artifacts) == [
        'model.safetensors']


def test_recipe_runtime_examples_use_existing_vision_backbones():
    cfg = Config.fromfile(str(CONFIG))
    runtime = cfg.model.teacher_extractors.radio
    assert runtime.type in {'RadioViTBackbone',
                            'RadioViTBackboneTensorRT'}
    assert tuple(runtime.default_image_resolution) == (3, 288, 256)
    if runtime.type == 'RadioViTBackbone':
        assert runtime.feature_dim == 1280
        assert runtime.patch_size == 16
    else:
        assert runtime.vision_backbone_id == 'c-radio_v4-h'
        assert runtime.engine_path.endswith('.engine')


def test_recipe_keeps_existing_robotwin_data_and_solver_contract():
    cfg = Config.fromfile(str(CONFIG))
    assert cfg.model.action_horizon == 16
    assert cfg.model.num_inference_steps == 10
    assert cfg.train_dataloader.dataset.datasets.future_offset == 16
    assert cfg.train_dataloader.dataset.datasets.history.enabled is False
