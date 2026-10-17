"""CPU tests for the frozen teacher-feature provider."""

import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from fluxvla.engines import VISION_BACKBONES
from fluxvla.models.backbones.visions.frozen_vision import SpatialFeatureOutput
from fluxvla.models.backbones.visions import radio_trt  # noqa: F401
from fluxvla.models.backbones.visions import radio_vit  # noqa: F401
import fluxvla.models.backbones.visions.teacher_features as teacher_features
from fluxvla.models.backbones.visions.teacher_features import (
    TeacherFeatureProvider,
)
from fluxvla.models.third_party_models.patchwam.teacher_forcing import (
    RasterLayout,
    TeacherEncoderSpec,
    TeacherForcingConfig,
    teacher_representation_fingerprint,
)

RADIO_ROOT = Path(
    '/home/chuong/workspace/FluxVLA/checkpoints/Pretrained_models/'
    'C-RADIOv4-H')


class CoordinateEncoder(nn.Module):
    def __init__(self, feature_dim=4, grid=(2, 3), input_size=(8, 12)):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid = tuple(grid)
        self.input_size = tuple(input_size)
        self.calls = []
        self.closed = 0

    def feature_grid(self, height, width):
        if (height, width) != self.input_size:
            raise ValueError(
                f'fixed input {self.input_size}; got {height}x{width}')
        return self.grid

    def encode_features(self, images):
        self.calls.append(images.detach().clone())
        batch = images.shape[0]
        rows, columns = self.grid
        values = torch.arange(rows * columns, device=images.device,
                              dtype=torch.float32)
        # Frame values enter a second channel so role inputs stay identifiable.
        frame = images.mean(dim=(1, 2, 3))
        tokens = torch.stack(
            (values.expand(batch, -1), frame[:, None].expand(-1, values.numel())),
            -1)
        if self.feature_dim > 2:
            tokens = torch.cat(
                (tokens, torch.zeros(batch, values.numel(),
                                     self.feature_dim - 2,
                                     device=images.device)), -1)
        return SpatialFeatureOutput(tokens=tokens, grid=self.grid)

    def close(self):
        self.closed += 1


class RegistryEncoder(CoordinateEncoder):
    instances = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        type(self).instances.append(self)


class ChunkedEncoder(nn.Module):
    feature_dim = 4
    encoder_batch_bounds = (2, 4)

    def __init__(self):
        super().__init__()
        self.calls = []
        self._buffer = None

    def feature_grid(self, height, width):
        if (height, width) != (8, 12):
            raise ValueError('fixed input 8x12')
        return 1, 1

    def encode_features(self, images):
        count = images.shape[0]
        self.calls.append(count)
        if self._buffer is None or self._buffer.shape[0] < count:
            self._buffer = torch.empty(
                4, 1, 4, device=images.device, dtype=images.dtype)
        # Reuse one backing buffer across calls to prove the provider copies
        # chunk outputs before the next enqueue can overwrite them.
        self._buffer.fill_(-7)
        self._buffer[:count, 0, 0] = images.mean(dim=(1, 2, 3))
        return SpatialFeatureOutput(
            tokens=self._buffer[:count], grid=(1, 1))


VISION_BACKBONES.register_module(
    name='TeacherProviderRegistryFake', module=RegistryEncoder)


def spec(**overrides):
    values = dict(
        name='radio',
        feature_dim=4,
        representation_id='fake-radio-v1',
        checkpoint_sha256='c' * 64,
        stream='spatial_features',
        preprocessing='unit_range_once',
        input_size=(8, 12),
        native_grid=(2, 3),
    )
    values.update(overrides)
    return TeacherEncoderSpec(**values)


def config(encoder=None):
    return TeacherForcingConfig.from_mapping(
        dict(encoders=[encoder or spec()], predictor='linear'))


def layout(batch=2, views=2, video_size=(8, 12), grid=(2, 3)):
    return RasterLayout(views=views, video_size=video_size, grid=grid)


def frames(batch=2, views=2, video_size=(8, 12), value=0.0):
    return torch.full((batch, views, 3, *video_size), float(value))


def provenance(representation_id='fake-radio-v1'):
    return {
        'radio': dict(
            representation_id=representation_id,
            test_contract='explicit-unit-fixture',
        )
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def test_provider_encodes_roles_in_separate_calls_and_fingerprints():
    encoder = CoordinateEncoder()
    provider = TeacherFeatureProvider(
        config(), {'radio': encoder}, provenance())
    current = frames(value=-0.5)
    future = frames(value=0.5)
    valid = torch.ones(2, 12, dtype=torch.bool)
    current_output = provider.encode_context(
        current, role='current', layout=layout(), raster_valid=valid)
    future_output = provider.encode_context(
        future, role='future', layout=layout(), raster_valid=valid)

    assert len(encoder.calls) == 2
    assert encoder.calls[0].shape == (4, 3, 8, 12)
    assert encoder.calls[0].mean() == pytest.approx(-0.5)
    assert encoder.calls[1].mean() == pytest.approx(0.5)
    for output in (current_output, future_output):
        assert output['tokens']['radio'].shape == (2, 12, 4)
        assert output['valid']['radio'].all()
        expected = torch.tensor(
            list(teacher_representation_fingerprint(spec())),
            dtype=torch.uint8)
        torch.testing.assert_close(
            output['fingerprints']['radio'][0], expected)
        torch.testing.assert_close(
            output['fingerprints']['radio'][1], expected)


def test_provider_rejects_wrong_input_and_feature_geometry_without_calls():
    encoder = CoordinateEncoder()
    provider = TeacherFeatureProvider(
        config(), {'radio': encoder}, provenance())
    valid = torch.ones(1, 12, dtype=torch.bool)
    with pytest.raises(ValueError, match='native grid'):
        provider.encode_context(
            frames(batch=1, views=2, value=0.0),
            role='current',
            layout=layout(batch=1, grid=(3, 2)),
            raster_valid=valid)
    assert encoder.calls == []

    with pytest.raises(ValueError, match='requires input'):
        provider.encode_context(
            frames(batch=1, views=2, video_size=(16, 12), value=0.0),
            role='current',
            layout=layout(batch=1, video_size=(16, 12)),
            raster_valid=valid)
    assert encoder.calls == []


def test_provider_skips_invalid_rasters_and_sanitizes_padded_nans():
    encoder = CoordinateEncoder()
    provider = TeacherFeatureProvider(
        config(), {'radio': encoder}, provenance())
    images = frames(batch=2)
    images[0, 1] = float('nan')
    images[1, 0] = float('nan')
    valid = torch.tensor([
        [True] * 6 + [False] * 6,
        [False] * 6 + [True] * 6,
    ])
    output = provider.encode_context(
        images, role='current', layout=layout(), raster_valid=valid)
    assert len(encoder.calls) == 1
    assert encoder.calls[0].shape == (2, 3, 8, 12)
    assert torch.isfinite(encoder.calls[0]).all()
    tokens = output['tokens']['radio']
    assert torch.count_nonzero(tokens[0, 6:]) == 0
    assert torch.count_nonzero(tokens[1, :6]) == 0
    assert torch.isfinite(tokens).all()
    torch.testing.assert_close(output['valid']['radio'], valid)


def test_provider_all_invalid_context_does_not_load_or_call_encoder():
    encoder = CoordinateEncoder()
    provider = TeacherFeatureProvider(
        config(), {'radio': encoder}, provenance())
    output = provider.encode_context(
        frames(), role='future', layout=layout(),
        raster_valid=torch.zeros(2, 12, dtype=torch.bool))
    assert encoder.calls == []
    assert torch.count_nonzero(output['tokens']['radio']) == 0
    assert not output['valid']['radio'].any()


def test_all_invalid_does_not_construct_registry_runtime():
    RegistryEncoder.instances.clear()
    provider = TeacherFeatureProvider(
        config(), {
            'radio': dict(
                type='TeacherProviderRegistryFake',
                feature_dim=4,
                grid=(2, 3),
                input_size=(8, 12),
            )
        }, provenance())
    output = provider.encode_context(
        frames(), role='current', layout=layout(),
        raster_valid=torch.zeros(2, 12, dtype=torch.bool))
    assert RegistryEncoder.instances == []
    assert not output['tokens']['radio'].requires_grad
    assert not output['fingerprints']['radio'].requires_grad
    assert output['tokens']['radio'].shape == (2, 12, 4)


def test_provider_balances_runtime_chunks_and_copies_reused_storage():
    encoder = ChunkedEncoder()
    chunk_spec = spec(feature_dim=4, input_size=(8, 12), native_grid=(1, 1))
    chunk_config = TeacherForcingConfig.from_mapping(
        dict(encoders=[chunk_spec], predictor='linear'))
    chunk_layout = RasterLayout(
        views=5, video_size=(8, 12), grid=(1, 1))
    provider = TeacherFeatureProvider(
        chunk_config, {'radio': encoder},
        {'radio': dict(
            representation_id=chunk_spec.representation_id,
            test_contract='explicit-unit-fixture')})
    images = torch.arange(5, dtype=torch.float32).reshape(
        1, 5, 1, 1, 1).expand(1, 5, 3, 8, 12).clone()
    output = provider.encode_context(
        images, role='current', layout=chunk_layout,
        raster_valid=torch.ones(1, 5, dtype=torch.bool))
    assert encoder.calls == [3, 2]
    torch.testing.assert_close(
        output['tokens']['radio'][0, :, 0],
        torch.arange(5, dtype=torch.float32))
    torch.testing.assert_close(
        output['tokens']['radio'][0, :, 1],
        torch.full((5,), -7.0))

    tiny = ChunkedEncoder()
    provider = TeacherFeatureProvider(
        chunk_config, {'radio': tiny},
        {'radio': dict(
            representation_id=chunk_spec.representation_id,
            test_contract='explicit-unit-fixture')})
    one_layout = RasterLayout(
        views=1, video_size=(8, 12), grid=(1, 1))
    with pytest.raises(ValueError, match='at least 2'):
        provider.encode_context(
            images[:, :1], role='current', layout=one_layout,
            raster_valid=torch.ones(1, 1, dtype=torch.bool))
    assert tiny.calls == []


def test_provider_registry_construction_and_owned_close_reload():
    RegistryEncoder.instances.clear()
    provider = TeacherFeatureProvider(
        config(), {
            'radio': dict(
                type='TeacherProviderRegistryFake',
                feature_dim=4,
                grid=(2, 3),
                input_size=(8, 12),
            )
        }, provenance())
    first = provider.extractors['radio']
    assert isinstance(first, RegistryEncoder)
    provider.close()
    assert first.closed == 1
    second = provider.extractors['radio']
    assert second is not first
    assert len(RegistryEncoder.instances) == 2


def test_provider_never_closes_an_injected_extractor():
    encoder = CoordinateEncoder()
    provider = TeacherFeatureProvider(
        config(), {'radio': encoder}, provenance())
    assert provider.extractors['radio'] is encoder
    provider.close()
    assert encoder.closed == 0


def test_provider_requires_exact_named_runtime_definitions():
    provider = TeacherFeatureProvider(config(), {}, provenance())
    valid = torch.ones(1, 12, dtype=torch.bool)
    with pytest.raises(ValueError, match='radio'):
        provider.encode_context(
            frames(batch=1, views=2), role='current', layout=layout(batch=1),
            raster_valid=valid)


def test_provider_binds_actual_artifacts_and_hashes_once(tmp_path, monkeypatch):
    weights = tmp_path / 'weights.bin'
    weights.write_bytes(b'first-runtime-bytes')
    digest = file_sha256(weights)
    bound_spec = spec(
        representation_id='provenance-test', checkpoint_sha256=digest)
    bound_config = TeacherForcingConfig.from_mapping(
        dict(encoders=[bound_spec], predictor='linear'))
    calls = []
    original = teacher_features._sha256_file

    def counted(path):
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(teacher_features, '_sha256_file', counted)
    encoder = CoordinateEncoder()
    envelope = {'radio': dict(
        representation_id=bound_spec.representation_id,
        base_dir=str(tmp_path),
        artifacts={'weights.bin': digest},
        representation_artifacts=['weights.bin'])}
    provider = TeacherFeatureProvider(
        bound_config, {'radio': encoder}, envelope)
    valid = torch.ones(2, 12, dtype=torch.bool)
    provider.encode_context(
        frames(), role='current', layout=layout(), raster_valid=valid)
    provider.encode_context(
        frames(), role='future', layout=layout(), raster_valid=valid)
    assert calls == [weights.resolve()]

    swapped = tmp_path / 'swapped.bin'
    swapped.write_bytes(b'other-runtime-bytes')
    with pytest.raises(ValueError, match='Provenance hash mismatch'):
        TeacherFeatureProvider(
            bound_config, {'radio': CoordinateEncoder()}, {
                'radio': dict(
                    representation_id=bound_spec.representation_id,
                    base_dir=str(tmp_path),
                    artifacts={'swapped.bin': digest},
                    representation_artifacts=['swapped.bin'])}
        ).encode_context(
            frames(), role='current', layout=layout(), raster_valid=valid)

    wrong_identity = spec(
        representation_id='provenance-test', checkpoint_sha256='0' * 64)
    with pytest.raises(ValueError, match='Representation artifact set'):
        TeacherFeatureProvider(
            TeacherForcingConfig.from_mapping(
                dict(encoders=[wrong_identity], predictor='linear')),
            {'radio': CoordinateEncoder()}, {
                'radio': dict(
                    representation_id=wrong_identity.representation_id,
                    base_dir=str(tmp_path),
                    artifacts={'weights.bin': digest},
                    representation_artifacts=['weights.bin'])}
        ).encode_context(
            frames(), role='current', layout=layout(), raster_valid=valid)


def test_static_validation_does_not_touch_a_lazy_runtime_engine():
    engine = RADIO_ROOT / 'exported' / '256x256' / (
        'c_radio_v4_h_256_fp16_ln_fp32.engine')
    metadata = Path(str(engine) + '.json')
    if not engine.is_file() or not metadata.is_file():
        pytest.skip('local C-RADIO TensorRT engine is unavailable')
    trt_spec = spec(
        feature_dim=1280, input_size=(256, 256), native_grid=(16, 16),
        representation_id='c-radio_v4-h-trt-static')
    trt_config = TeacherForcingConfig.from_mapping(
        dict(encoders=[trt_spec], predictor='linear'))
    from fluxvla.models.backbones.visions.radio_trt import (
        RadioViTBackboneTensorRT,
    )
    extractor = RadioViTBackboneTensorRT(
        vision_backbone_id='c-radio_v4-h',
        engine_path=str(engine),
        metadata_path=str(metadata),
        default_image_resolution=(3, 256, 256),
        device=None,
    )

    def forbidden():
        raise AssertionError('static validation deserialized the GPU engine')

    extractor._ensure_engine = forbidden
    provider = TeacherFeatureProvider(
        trt_config, {'radio': extractor}, {
            'radio': dict(
                representation_id=trt_spec.representation_id,
                test_contract='explicit-unit-fixture')})
    report = provider.validate_static_contracts()
    assert report['radio']['native_grid'] == [16, 16]


def test_tensorrt_256_square_engine_rejects_288x256_recipe():
    engine = RADIO_ROOT / 'exported' / '256x256' / (
        'c_radio_v4_h_256.engine')
    if not engine.is_file():
        pytest.skip('local C-RADIO TensorRT engine is unavailable')
    provider = TeacherFeatureProvider(
        config(), {
            'radio': dict(
                type='RadioViTBackboneTensorRT',
                vision_backbone_id='c-radio_v4-h',
                engine_path=str(engine),
                metadata_path=str(engine) + '.json',
                default_image_resolution=(3, 288, 256),
            )
        }, provenance())
    with pytest.raises(ValueError, match='geometry|input'):
        provider.extractors


@pytest.mark.checkpoint
def test_real_radio_provider_emits_native_288x256_features():
    model_path = RADIO_ROOT
    if not (model_path / 'config.json').is_file():
        pytest.skip('local C-RADIO PyTorch checkpoint is unavailable')
    weights = model_path / 'model.safetensors'
    if not weights.is_file():
        pytest.skip('local C-RADIO safetensors weights are unavailable')
    digest = file_sha256(weights)
    real_spec = spec(
        feature_dim=1280, input_size=(288, 256), native_grid=(18, 16),
        representation_id='c-radio_v4-h-py-288x256',
        checkpoint_sha256=digest)
    real_config = TeacherForcingConfig.from_mapping(
        dict(encoders=[real_spec], predictor='linear'))
    provider = TeacherFeatureProvider(
        real_config, {
            'radio': dict(
                type='RadioViTBackbone',
                version='c-radio_v4-h',
                model_path=str(model_path),
                feature_dim=1280,
                patch_size=16,
                minimum_resolution=16,
                default_image_resolution=(3, 288, 256),
                dtype='bfloat16',
                device='cuda',
            )
        }, {
            'radio': dict(
                representation_id=real_spec.representation_id,
                base_dir=str(model_path),
                artifacts={'model.safetensors': digest},
                representation_artifacts=['model.safetensors'],
            )
        })
    frames_ = torch.zeros(1, 1, 3, 288, 256, device='cuda')
    valid = torch.ones(1, 288, dtype=torch.bool, device='cuda')
    output = provider.encode_context(
        frames_, role='current',
        layout=RasterLayout(views=1, video_size=(288, 256),
                            grid=(18, 16)),
        raster_valid=valid)
    assert output['tokens']['radio'].shape == (1, 288, 1280)
    assert output['tokens']['radio'].dtype == torch.bfloat16
    assert torch.isfinite(output['tokens']['radio']).all()
    assert torch.count_nonzero(output['tokens']['radio']) > 0
    provider.close()


@pytest.mark.checkpoint
def test_real_256_radio_pytorch_and_tensorrt_provider_comparison():
    pytest.importorskip('tensorrt')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    export = RADIO_ROOT / 'exported' / '256x256'
    weights = RADIO_ROOT / 'model.safetensors'
    engine = export / 'c_radio_v4_h_256_fp16_ln_fp32.engine'
    metadata = Path(str(engine) + '.json')
    source_onnx = export / 'c_radio_v4_h_256.onnx'
    source_data = export / 'c_radio_v4_h_256.onnx.data'
    required = (weights, engine, metadata, source_onnx, source_data)
    if not all(path.is_file() for path in required):
        pytest.skip('compatible local RADIO PyTorch/TRT assets are missing')
    weight_hash = file_sha256(weights)
    artifact_hashes = {
        engine.name: file_sha256(engine),
        source_onnx.name: file_sha256(source_onnx),
        source_data.name: file_sha256(source_data),
    }
    representation_hashes = {
        source_onnx.name: artifact_hashes[source_onnx.name],
        source_data.name: artifact_hashes[source_data.name],
    }
    pt_spec = spec(
        name='radio_pt', feature_dim=1280, input_size=(256, 256),
        native_grid=(16, 16), representation_id='c-radio-pt-256',
        checkpoint_sha256=weight_hash)
    trt_spec = spec(
        name='radio_trt', feature_dim=1280, input_size=(256, 256),
        native_grid=(16, 16), representation_id='c-radio-trt-256',
        checkpoint_sha256=teacher_features._representation_digest(
            representation_hashes))
    cfg = TeacherForcingConfig.from_mapping(dict(
        encoders=[pt_spec, trt_spec], predictor='linear'))
    provider = TeacherFeatureProvider(
        cfg, {
            'radio_pt': dict(
                type='RadioViTBackbone',
                version='c-radio_v4-h',
                model_path=str(RADIO_ROOT),
                feature_dim=1280,
                patch_size=16,
                minimum_resolution=16,
                default_image_resolution=(3, 256, 256),
                dtype='bfloat16',
                device='cuda',
            ),
            'radio_trt': dict(
                type='RadioViTBackboneTensorRT',
                vision_backbone_id='c-radio_v4-h',
                engine_path=str(engine),
                metadata_path=str(metadata),
                default_image_resolution=(3, 256, 256),
                device=None,
            ),
        }, {
            'radio_pt': dict(
                representation_id=pt_spec.representation_id,
                base_dir=str(RADIO_ROOT),
                artifacts={'model.safetensors': weight_hash},
                representation_artifacts=['model.safetensors'],
            ),
            'radio_trt': dict(
                representation_id=trt_spec.representation_id,
                base_dir=str(export),
                artifacts=artifact_hashes,
                representation_artifacts=[
                    source_onnx.name, source_data.name],
            ),
        })
    images = torch.rand(1, 1, 3, 256, 256, device='cuda') * 2 - 1
    output = provider.encode_context(
        images, role='current',
        layout=RasterLayout(
            views=1, video_size=(256, 256), grid=(16, 16)),
        raster_valid=torch.ones(1, 256, dtype=torch.bool, device='cuda'))
    pt = output['tokens']['radio_pt'].float()
    trt = output['tokens']['radio_trt'].float()
    assert pt.shape == trt.shape == (1, 256, 1280)
    assert torch.isfinite(pt).all() and torch.isfinite(trt).all()
    relative = float((pt - trt).norm() / pt.norm().clamp_min(1e-6))
    cosine = float(torch.nn.functional.cosine_similarity(
        pt.flatten(0, 1), trt.flatten(0, 1), dim=-1).mean())
    print(
        f'RADIO_256_PROVIDER_METRICS relative={relative:.6f} '
        f'cosine={cosine:.6f}')
    assert relative < 0.2
    assert cosine > 0.95
    provider.close()


def test_impossible_runtime_profile_partition_fails_without_looping():
    # Increasing the number of chunks cannot resolve a below-minimum tail.
    for total, minimum, maximum in ((3, 2, 2), (5, 3, 4), (9, 5, 8)):
        with pytest.raises(ValueError, match='Cannot split'):
            teacher_features._balanced_chunk_sizes(total, minimum, maximum)
    assert teacher_features._balanced_chunk_sizes(5, 2, 4) == [3, 2]
