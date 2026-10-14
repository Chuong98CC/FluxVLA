# Copyright 2026 Limx Dynamics
"""The C-RADIO example config matches the registered component contract."""

import inspect
from pathlib import Path

import pytest
from mmengine.config import Config

from fluxvla.engines import VISION_BACKBONES, build_vision_backbone_from_cfg
from fluxvla.models.backbones.visions.radio_trt import \
    RadioViTBackboneTensorRT
from fluxvla.models.backbones.visions.base_trt.radio_metadata import \
    EngineMetadata
from fluxvla.models.backbones.visions.radio_vit import RadioViTBackbone

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT / 'configs' / 'visions' / 'radio_vit_tensorrt_example.py')


def _accepted_parameters(cls):
    return set(inspect.signature(cls.__init__).parameters) - {'self'}


def test_example_config_only_uses_registered_constructor_arguments():
    cfg = Config.fromfile(str(CONFIG_PATH))
    for key, cls in (('vision_backbone', RadioViTBackbone),
                     ('inference_vision_backbone', RadioViTBackboneTensorRT)):
        entry = dict(getattr(cfg, key))
        assert VISION_BACKBONES.get(entry['type']) is cls
        # The tutorial's checklist: the ``type`` names a registered class and
        # every other key is a real ``__init__`` parameter.
        assert set(entry) - {'type'} <= _accepted_parameters(cls)


def test_tensorrt_example_builds_from_the_sidecar_without_tensorrt(tmp_path):
    cfg = Config.fromfile(str(CONFIG_PATH))
    entry = dict(cfg.inference_vision_backbone)
    engine_path = tmp_path / 'c_radio_v4_h_256.engine'
    engine_path.write_bytes(b'placeholder-engine')
    EngineMetadata(
        variant='c-radio_v4-h',
        feature_dim=1280,
        patch_size=16,
        resolution_step=16,
        input_height=256,
        input_width=256,
        grid_height=16,
        grid_width=16,
        tensorrt_version='10.7.0',
        min_batch=1,
        opt_batch=1,
        max_batch=8).save(f'{engine_path}.json')
    entry['engine_path'] = str(engine_path)
    entry['metadata_path'] = f'{engine_path}.json'
    entry.pop('device')
    backbone = build_vision_backbone_from_cfg(entry)
    assert isinstance(backbone, RadioViTBackboneTensorRT)
    assert backbone.identifier == 'c-radio_v4-h'
    assert backbone.embed_dim == 1280
    assert backbone.feature_grid(256, 256) == (16, 16)
    assert backbone.num_patches == 256
    # Only the documented geometry is accepted by this engine.
    with pytest.raises(ValueError, match='fixed 256x256 input'):
        backbone.feature_grid(224, 224)
