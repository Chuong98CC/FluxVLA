# Copyright 2026 Limx Dynamics
"""C-RADIO TensorRT backbone: registry, geometry, freeze and metadata.

These units run on CPU with an injected engine stand-in: they cover the
contract a consumer relies on (tokens, grid, freezing, autograd-safe output)
and every metadata/geometry failure the runtime would otherwise raise on a
GPU. Running a real TensorRT engine is covered by the gated GPU units in
``test_base_trt_engine.py``.
"""

import copy
import json

import pytest
import torch

from fluxvla.engines import VISION_BACKBONES, build_vision_backbone_from_cfg
from fluxvla.models.backbones.visions.frozen_vision import \
    FrozenVisionFeatures
from fluxvla.models.backbones.visions.radio_trt import \
    RadioViTBackboneTensorRT
from fluxvla.models.backbones.visions.base_trt.metadata import (  # noqa: F401
    DYNAMIC_DIM, EngineContract, TensorSpec)
from fluxvla.models.backbones.visions.base_trt.radio_metadata import (
    EngineMetadata, default_metadata_path)


def _metadata(**overrides):
    values = dict(
        variant='c-radio_v4-h',
        feature_dim=1280,
        patch_size=16,
        resolution_step=16,
        input_height=32,
        input_width=32,
        grid_height=2,
        grid_width=2,
        tensorrt_version='10.7.0',
        min_batch=1,
        opt_batch=2,
        max_batch=8)
    values.update(overrides)
    return EngineMetadata(**values)


class FakeEngineRuntime:
    """Minimal runtime with the backbone's duck-typed engine contract."""

    def __init__(self, metadata, layout=None):
        self.metadata = metadata
        self.layout = layout or metadata.output_layout
        self.calls = []
        self.closed = False

    def run(self, images):
        self.calls.append(images.clone())
        batch = int(images.shape[0])
        grid_h, grid_w = self.metadata.grid
        channels = int(self.metadata.feature_dim)
        if self.layout == 'NLD':
            return torch.zeros(
                batch, grid_h * grid_w, channels, dtype=images.dtype)
        if self.layout == 'NCHW':
            return torch.zeros(
                batch, channels, grid_h, grid_w, dtype=images.dtype)
        return torch.zeros(batch, grid_h, grid_w, channels, dtype=images.dtype)

    def close(self):
        self.closed = True


def _backbone(**overrides):
    values = overrides.pop('metadata', {})
    layout = overrides.pop('layout', None)
    if layout is not None:
        values['output_layout'] = layout
    metadata = _metadata(**values)
    runtime = FakeEngineRuntime(metadata)
    backbone = RadioViTBackboneTensorRT(
        engine=runtime, metadata=metadata, **overrides)
    return backbone, runtime, metadata


def _engine_file(tmp_path, metadata):
    engine_path = tmp_path / 'radio.engine'
    engine_path.write_bytes(b'not-a-real-engine')
    metadata.save(default_metadata_path(engine_path))
    return engine_path


# -- metadata -------------------------------------------------------------


def test_metadata_round_trips_through_json(tmp_path):
    metadata = _metadata(source_onnx='radio.onnx', onnx_sha256='abc')
    path = metadata.save(tmp_path / 'nested' / 'radio.engine.json')
    loaded = EngineMetadata.load(path)
    assert loaded == metadata
    assert loaded.to_dict() == metadata.to_dict()
    assert loaded.input_dtype == 'float32'
    assert loaded.output_dtype == 'float32'
    assert loaded.grid == (2, 2) and loaded.image_size == (32, 32)
    assert loaded.input_shape(3) == (3, 3, 32, 32)
    assert loaded.output_shape(3) == (3, 4, 1280)


def test_build_precision_is_separate_from_tensor_dtypes():
    # BuilderFlag.FP16 picks internal tactics; it does not retype the graph,
    # so an fp16 build keeps the FP32 boundary unless the graph says otherwise.
    fp16_build = _metadata(precision='fp16')
    assert fp16_build.input_dtype == 'float32'
    assert fp16_build.output_dtype == 'float32'
    half_graph = _metadata(input_dtype='float16', output_dtype='float16')
    assert half_graph.input_dtype == 'float16'
    assert half_graph.output_dtype == 'float16'
    assert half_graph.precision == 'fp32'
    with pytest.raises(ValueError, match='input_dtype must be one of'):
        _metadata(input_dtype='float64')


@pytest.mark.parametrize('overrides,match', [
    ({
        'grid_height': 3
    }, 'grid does not match'),
    ({
        'input_height': 40
    }, 'aligned to the resolution step'),
    ({
        'resolution_step': 24
    }, 'patch-size multiple'),
    ({
        'precision': 'int8'
    }, 'precision must be one of'),
    ({
        'output_layout': 'NHWC2'
    }, 'output_layout must be one of'),
    ({
        'min_batch': 4,
        'opt_batch': 2
    }, 'min <= opt <= max'),
    ({
        'feature_dim': 0
    }, 'positive integer'),
    ({
        'input_name': 'tokens',
        'output_name': 'tokens'
    }, 'must differ'),
    ({
        'schema_version': 99
    }, 'schema_version'),
    ({
        'tensorrt_version': 'v10'
    }, 'major'),
])
def test_metadata_rejects_inconsistent_records(overrides, match):
    with pytest.raises(ValueError, match=match):
        _metadata(**overrides)


def test_metadata_from_dict_is_strict_about_shape_and_keys():
    payload = _metadata().to_dict()
    payload['extra'] = 1
    with pytest.raises(ValueError, match='unknown field'):
        EngineMetadata.from_dict(payload)
    del payload['extra']
    del payload['feature_dim']
    with pytest.raises(ValueError, match='missing required field'):
        EngineMetadata.from_dict(payload)
    with pytest.raises(ValueError, match='must be a mapping'):
        EngineMetadata.from_dict(['nope'])
    with pytest.raises(FileNotFoundError, match='sidecar not found'):
        EngineMetadata.load('/nonexistent/radio.engine.json')


def test_metadata_load_rejects_invalid_json(tmp_path):
    path = tmp_path / 'broken.json'
    path.write_text('{not json')
    with pytest.raises(ValueError, match='not valid JSON'):
        EngineMetadata.load(path)


# -- backbone contract ----------------------------------------------------


def test_backbone_is_registered_and_matches_the_generic_contract():
    assert (VISION_BACKBONES.get('RadioViTBackboneTensorRT')
            is RadioViTBackboneTensorRT)
    assert issubclass(RadioViTBackboneTensorRT, FrozenVisionFeatures)
    backbone, _, metadata = _backbone()
    assert backbone.identifier == 'c-radio_v4-h'
    assert backbone.embed_dim == 1280
    assert backbone.num_patches == 4
    assert backbone.feature_grid(32, 32) == (2, 2)
    assert backbone.half_precision_dtype == torch.bfloat16
    assert backbone.precision == 'fp32'
    assert callable(backbone.get_fsdp_wrapping_policy())
    built = build_vision_backbone_from_cfg(
        dict(
            type='RadioViTBackboneTensorRT',
            engine=FakeEngineRuntime(metadata),
            metadata=metadata))
    assert isinstance(built, RadioViTBackboneTensorRT)
    assert built.metadata == metadata


@pytest.mark.parametrize('variant,dim', [('c-radio_v4-h', 1280),
                                         ('c-radio_v4-so400m', 1152)])
def test_backbone_accepts_both_documented_variants(variant, dim):
    metadata = _metadata(variant=variant, feature_dim=dim)
    backbone = RadioViTBackboneTensorRT(
        engine=FakeEngineRuntime(metadata), metadata=metadata)
    assert backbone.feature_dim == dim
    assert backbone.identifier == variant
    output = backbone.encode_features(torch.rand(1, 3, 32, 32) * 2 - 1)
    assert output.tokens.shape == (1, 4, dim)


def test_backbone_converts_range_and_returns_tokens_and_grid():
    backbone, runtime, _ = _backbone()
    images = torch.full((2, 3, 32, 32), -1.0)
    images[:, 0] = 1.0
    output = backbone.encode_features(images)
    seen = runtime.calls[-1]
    assert seen.shape == (2, 3, 32, 32)
    assert seen.dtype == torch.float32 and seen.is_contiguous()
    # [-1, 1] is converted exactly once, on the PyTorch side.
    assert torch.allclose(seen[:, 0], torch.ones_like(seen[:, 0]))
    assert not seen[:, 1].any() and not seen[:, 2].any()
    assert output.grid == (2, 2) and output.feature_dim == 1280
    assert torch.allclose(backbone(images), output.tokens)


def test_backbone_casts_to_the_engine_input_dtype():
    # An fp16 *build* keeps a float32 boundary, so the input stays float32
    # while the preferred half precision becomes fp16.
    backbone, runtime, _ = _backbone(metadata={'precision': 'fp16'})
    assert backbone.half_precision_dtype == torch.float16
    backbone.encode_features(torch.rand(1, 3, 32, 32) * 2 - 1)
    assert runtime.calls[-1].dtype == torch.float32
    # An fp16 graph casts the input to fp16 and returns fp16 tokens.
    half, half_runtime, _ = _backbone(metadata={
        'input_dtype': 'float16',
        'output_dtype': 'float16'
    })
    output = half.encode_features(torch.rand(1, 3, 32, 32) * 2 - 1)
    assert half_runtime.calls[-1].dtype == torch.float16
    assert output.tokens.dtype == torch.float16


@pytest.mark.parametrize('layout', ['NLD', 'NCHW', 'NHWC'])
def test_backbone_normalizes_every_engine_layout(layout):
    backbone, _, _ = _backbone(layout=layout)
    output = backbone.encode_features(torch.rand(1, 3, 32, 32) * 2 - 1)
    assert output.tokens.shape == (1, 4, 1280)
    assert output.grid == (2, 2)


def test_backbone_rejects_geometry_the_engine_was_not_built_for():
    backbone, _, _ = _backbone()
    with pytest.raises(ValueError, match='fixed 32x32 input'):
        backbone.encode_features(torch.rand(1, 3, 48, 48) * 2 - 1)
    with pytest.raises(ValueError, match='fixed 32x32 input'):
        backbone.feature_grid(64, 64)


def test_backbone_rejects_bad_images():
    backbone, _, _ = _backbone()
    with pytest.raises(ValueError, match='three RGB channels'):
        backbone.encode_features(torch.rand(1, 4, 32, 32))
    with pytest.raises(ValueError, match='in .-1,1.'):
        backbone.encode_features(torch.rand(1, 3, 32, 32) * 4)
    with pytest.raises(ValueError, match='finite'):
        backbone.encode_features(torch.full((1, 3, 32, 32), float('nan')))
    with pytest.raises(ValueError, match='must be'):
        backbone.encode_features(torch.rand(1, 3, 32))


def test_backbone_stays_frozen_and_outputs_stay_autograd_compatible():
    backbone, _, _ = _backbone()
    backbone.train()
    assert not backbone.training
    assert all(not module.training for module in backbone.modules())
    assert not any(parameter.requires_grad
                   for parameter in backbone.parameters())
    tokens = backbone.encode_features(
        (torch.rand(2, 3, 32, 32) * 2 - 1).requires_grad_()).tokens
    assert not tokens.requires_grad and tokens.grad_fn is None
    assert not tokens.is_inference()
    # A downstream trainable projector must still be able to train on them.
    projector = torch.nn.Linear(backbone.embed_dim, 4)
    projector(tokens).square().mean().backward()
    assert projector.weight.grad is not None
    assert torch.isfinite(projector.weight.grad).all()


def test_backbone_reports_tokens_that_disagree_with_the_metadata():
    metadata = _metadata()

    class WrongWidth(FakeEngineRuntime):

        def run(self, images):
            return torch.zeros(1, 4, 7, dtype=images.dtype)

    backbone = RadioViTBackboneTensorRT(
        engine=WrongWidth(metadata), metadata=metadata)
    with pytest.raises(ValueError, match='was configured'):
        backbone.encode_features(torch.rand(1, 3, 32, 32) * 2 - 1)


# -- construction, laziness and argument validation ----------------------


def test_construction_never_imports_tensorrt(tmp_path):
    metadata = _metadata()
    engine_path = _engine_file(tmp_path, metadata)
    # Building the backbone reads only the sidecar; a missing TensorRT
    # install must not be observable until the engine is first used.
    backbone = RadioViTBackboneTensorRT.from_engine(str(engine_path))
    assert backbone.metadata == metadata
    assert backbone.engine_path == str(engine_path)
    with pytest.raises(RuntimeError, match='TensorRT'):
        backbone.encode_features(torch.rand(1, 3, 32, 32) * 2 - 1)


def test_missing_sidecar_and_missing_engine_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match='sidecar not found'):
        RadioViTBackboneTensorRT(engine_path=str(tmp_path / 'absent.engine'))
    backbone = RadioViTBackboneTensorRT(
        engine_path=str(tmp_path / 'absent.engine'), metadata=_metadata())
    with pytest.raises(FileNotFoundError, match='engine not found'):
        backbone.encode_features(torch.rand(1, 3, 32, 32) * 2 - 1)


def test_close_releases_only_an_owned_engine():
    backbone, runtime, metadata = _backbone()
    backbone.close()
    # An injected runtime belongs to its caller, so close() must not touch it.
    assert not runtime.closed
    assert backbone.engine is runtime
    owned = RadioViTBackboneTensorRT(
        engine_path='/tmp/never-loaded.engine', metadata=metadata)
    owned.close()


def test_backbone_requires_exactly_one_engine_source():
    with pytest.raises(
            ValueError, match='exactly one of engine or engine_path'):
        RadioViTBackboneTensorRT(vision_backbone_id='c-radio_v4-h')
    metadata = _metadata()
    with pytest.raises(
            ValueError, match='exactly one of engine or engine_path'):
        RadioViTBackboneTensorRT(
            engine=FakeEngineRuntime(metadata), engine_path='/tmp/x.engine')


def test_backbone_rejects_two_metadata_sources():
    metadata = _metadata()
    with pytest.raises(ValueError, match='at most one of metadata'):
        RadioViTBackboneTensorRT(
            engine=FakeEngineRuntime(metadata),
            metadata=metadata,
            metadata_path='/tmp/x.json')


def test_backbone_rejects_an_unknown_variant_and_a_variant_mismatch():
    metadata = _metadata()
    with pytest.raises(ValueError, match='Unsupported C-RADIO version'):
        RadioViTBackboneTensorRT(
            engine=FakeEngineRuntime(metadata),
            metadata=metadata,
            vision_backbone_id='c-radio_v5-h')
    with pytest.raises(ValueError, match='does not match the engine metadata'):
        RadioViTBackboneTensorRT(
            engine=FakeEngineRuntime(metadata),
            metadata=metadata,
            vision_backbone_id='c-radio_v4-so400m')


def test_backbone_rejects_a_width_that_contradicts_the_variant():
    metadata = _metadata(feature_dim=1152)
    with pytest.raises(ValueError, match='rebuild the engine'):
        RadioViTBackboneTensorRT(
            engine=FakeEngineRuntime(metadata), metadata=metadata)


def test_backbone_rejects_an_injected_contract_that_disagrees():
    metadata = _metadata()
    other = copy.deepcopy(metadata.to_dict())
    other['max_batch'] = 16
    mismatched = EngineMetadata(**other)
    with pytest.raises(
            ValueError, match='does not match the C-RADIO metadata'):
        stale = FakeEngineRuntime(mismatched)
        stale.contract = mismatched.contract()
        RadioViTBackboneTensorRT(engine=stale, metadata=metadata)


def test_injected_engine_needs_metadata_beside_it():

    class Bare:
        pass

    with pytest.raises(ValueError, match='carries no C-RADIO metadata'):
        RadioViTBackboneTensorRT(engine=Bare())


def test_build_from_cfg_rejects_an_unknown_key():
    with pytest.raises(TypeError):
        build_vision_backbone_from_cfg(
            dict(
                type='RadioViTBackboneTensorRT',
                engine=FakeEngineRuntime(_metadata()),
                metadata=_metadata(),
                not_a_parameter=1))


def test_default_image_resolution_must_match_the_engine_geometry():
    metadata = _metadata()
    # num_patches is derived from this, so a mismatch must fail at
    # construction rather than when the property is first read.
    with pytest.raises(ValueError, match='does not match the engine geometry'):
        RadioViTBackboneTensorRT(
            engine=FakeEngineRuntime(metadata),
            metadata=metadata,
            default_image_resolution=(3, 256, 256))
    ok = RadioViTBackboneTensorRT(
        engine=FakeEngineRuntime(metadata),
        metadata=metadata,
        default_image_resolution=(3, 32, 32))
    assert ok.num_patches == 4


def test_sidecar_helper_and_saved_record(tmp_path):
    assert str(default_metadata_path('/a/b/radio.engine')) == (
        '/a/b/radio.engine.json')
    path = _metadata().save(tmp_path / 'radio.engine.json')
    assert json.loads(path.read_text())['variant'] == 'c-radio_v4-h'
