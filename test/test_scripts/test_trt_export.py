# Copyright 2026 Limx Dynamics
"""Offline export/build helpers and C-RADIO sample orchestration.

The generic helpers are exercised against a stand-in TensorRT builder whose
network IO is fixed by the test, which is what makes "an FP16 build keeps the
FP32 graph boundary" observable without the real stack. The gated end-to-end
unit runs where onnx, TensorRT and a GPU are present.
"""

import importlib.util

import numpy as np
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from fluxvla.models.backbones.visions.base_trt import DYNAMIC_DIM
from fluxvla.models.backbones.visions.base_trt.metadata import (
    EngineContract, OptimizationProfile, TensorSpec)
from fluxvla.models.backbones.visions.base_trt.radio_metadata import (
    EngineMetadata, default_metadata_path)
from fluxvla.models.backbones.visions.radio_trt import \
    RadioViTBackboneTensorRT
from fluxvla.models.backbones.visions.radio_vit import RadioViTBackbone
from tools import c_radio_export, trt_export

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_IMAGE = REPO_ROOT / 'assets' / 'frame_000000.jpg'

HAS_ONNX = importlib.util.find_spec('onnx') is not None
HAS_TENSORRT = importlib.util.find_spec('tensorrt') is not None
NEEDS_ONNX_ABSENT = pytest.mark.skipif(
    HAS_ONNX,
    reason='onnx is installed, so the missing-dependency path is '
    'unreachable')
NEEDS_FULL_STACK = pytest.mark.skipif(
    not (HAS_ONNX and HAS_TENSORRT and torch.cuda.is_available()),
    reason='Requires onnx, TensorRT and CUDA.')
REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason='Requires CUDA.')


class StubRadio(nn.Module):
    """Minimal C-RADIO stand-in with the official call contract."""

    def __init__(self, feature_dim=8, patch_size=16, resolution_step=16):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.resolution_step = resolution_step
        self.stem = nn.Conv2d(3, feature_dim, patch_size, patch_size)

    def get_nearest_supported_resolution(self, height, width):
        step = self.resolution_step
        return (-(-height // step) * step, -(-width // step) * step)

    def forward(self, images, feature_fmt='NLC'):
        spatial = self.stem(images)
        summary = spatial.mean(dim=(2, 3))[:, None]
        if feature_fmt == 'NCHW':
            return summary, spatial
        return summary, spatial.flatten(2).transpose(1, 2)


def _radio_backbone(feature_dim=8):
    torch.manual_seed(0)
    return RadioViTBackbone(
        'c-radio_v4-h',
        model=StubRadio(feature_dim),
        feature_dim=feature_dim,
        patch_size=16,
        resolution_step=16,
        default_image_resolution=(3, 32, 32))


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
        tensorrt_version='10.7.0')
    values.update(overrides)
    return EngineMetadata(**values)


def _images(batch=1):
    torch.manual_seed(1)
    return torch.rand(batch, 3, 32, 32) * 2 - 1


# -- C-RADIO record -------------------------------------------------------


def test_build_metadata_separates_precision_from_tensor_dtypes():
    backbone = _radio_backbone(feature_dim=8)
    metadata = c_radio_export.build_metadata(
        backbone,
        image_height=32,
        image_width=32,
        min_batch=1,
        opt_batch=2,
        max_batch=4,
        tensorrt_version='10.7.0',
        precision='fp16')
    assert metadata.precision == 'fp16'
    assert metadata.input_dtype == 'float32'
    assert metadata.output_dtype == 'float32'
    contract = metadata.contract()
    assert contract.input_names == ('pixel_values', )
    assert contract.inputs[0].shape == (DYNAMIC_DIM, 3, 32, 32)
    assert contract.outputs[0].shape == (DYNAMIC_DIM, 4, 8)
    assert contract.profile_for('pixel_values').max_shape == (4, 3, 32, 32)


def test_build_metadata_rejects_geometry_that_is_not_patch_aligned():
    backbone = _radio_backbone(feature_dim=8)
    with pytest.raises(ValueError, match='aligned to the resolution step'):
        c_radio_export.build_metadata(
            backbone,
            image_height=40,
            image_width=32,
            min_batch=1,
            opt_batch=1,
            max_batch=1,
            tensorrt_version='10.7.0')


def test_sha256_file_is_stable(tmp_path):
    path = tmp_path / 'blob.bin'
    path.write_bytes(b'c-radio')
    first = trt_export.sha256_file(path)
    assert first == trt_export.sha256_file(path)
    path.write_bytes(b'c-radio!')
    assert trt_export.sha256_file(path) != first


def test_verify_onnx_identity_detects_a_stale_graph(tmp_path):
    source = tmp_path / 'graph.onnx'
    source.write_bytes(b'graph-v1')
    recorded = trt_export.sha256_file(source)
    trt_export.verify_onnx_identity(source, onnx_sha256=recorded)
    source.write_bytes(b'graph-v2')
    with pytest.raises(ValueError, match='does not match the recorded sha256'):
        trt_export.verify_onnx_identity(source, onnx_sha256=recorded)
    with pytest.raises(FileNotFoundError, match='ONNX model not found'):
        trt_export.verify_onnx_identity(tmp_path / 'absent.onnx')


def test_verify_onnx_identity_checks_external_data(tmp_path):
    source = tmp_path / 'graph.onnx'
    source.write_bytes(b'graph')
    data = tmp_path / 'graph.onnx.data'
    data.write_bytes(b'weights')
    recorded = trt_export.sha256_file(data)
    trt_export.verify_onnx_identity(
        source, external_data=data.name, external_data_sha256=recorded)
    data.write_bytes(b'other')
    with pytest.raises(ValueError, match='external data'):
        trt_export.verify_onnx_identity(
            source, external_data=data.name, external_data_sha256=recorded)
    data.unlink()
    with pytest.raises(FileNotFoundError, match='does not exist'):
        trt_export.verify_onnx_identity(source, external_data=data.name)
    with pytest.raises(ValueError, match='without external_data'):
        trt_export.verify_onnx_identity(
            source, external_data_sha256='deadbeef')


# -- export boundary ------------------------------------------------------


def test_export_module_returns_tokens_for_unit_range_input():
    backbone = _radio_backbone(feature_dim=8)
    module = c_radio_export.RadioTokenExportModule(backbone).eval()
    unit = torch.rand(2, 3, 32, 32)
    tokens = module(unit)
    assert tokens.shape == (2, 4, 8)
    assert module.input_dtype is torch.bfloat16
    with torch.no_grad():
        expected = backbone.spatial_tokens(unit.to(dtype=module.input_dtype))
    assert torch.allclose(tokens, expected)


def test_dummy_inputs_follow_the_contract_optimum_shape():
    contract = _metadata(min_batch=1, opt_batch=3, max_batch=8).contract()
    dummy = trt_export.dummy_inputs(contract)
    assert len(dummy) == 1
    assert tuple(dummy[0].shape) == (3, 3, 32, 32)
    assert dummy[0].dtype == torch.float32


@NEEDS_ONNX_ABSENT
def test_export_onnx_reports_the_missing_onnx_dependency(tmp_path):
    backbone = _radio_backbone(feature_dim=8)
    with pytest.raises(RuntimeError, match='onnx'):
        c_radio_export.export_onnx(
            backbone, tmp_path / 'radio.onnx', image_height=32, image_width=32)
    assert not (tmp_path / 'radio.onnx').exists()


@NEEDS_ONNX_ABSENT
def test_graph_verification_reports_the_missing_onnx_dependency(tmp_path):
    blob = tmp_path / 'radio.onnx'
    blob.write_bytes(b'not-onnx')
    contract = _metadata().contract()
    with pytest.raises(RuntimeError, match='onnx'):
        trt_export.verify_onnx_graph(blob, contract)
    with pytest.raises(RuntimeError, match='onnx'):
        trt_export.convert_onnx_to_external_data(blob)


# -- builder stand-in: precision is tactics, the graph owns the dtypes -----


class _Dims:

    def __init__(self, shape):
        self.values = tuple(int(value) for value in shape)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


class _FakeTensor:

    def __init__(self, name, dtype, shape):
        self.name = name
        self.dtype = dtype
        self.shape = _Dims(shape)


class _FakeNetwork:

    def __init__(self, inputs, outputs):
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.num_inputs = len(self.inputs)
        self.num_outputs = len(self.outputs)

    def get_input(self, index):
        return self.inputs[index]

    def get_output(self, index):
        return self.outputs[index]


class _FakeBuilderConfig:

    def __init__(self):
        self.flags = []
        self.memory_pool_limits = []
        self.profiles = []

    def set_flag(self, flag):
        self.flags.append(flag)

    def set_memory_pool_limit(self, pool, size):
        self.memory_pool_limits.append((pool, size))

    def add_optimization_profile(self, profile):
        self.profiles.append(profile)


class _FakeProfile:

    def __init__(self):
        self.shapes = {}

    def set_shape(self, name, min_shape, opt_shape, max_shape):
        self.shapes[name] = (tuple(min_shape), tuple(opt_shape),
                             tuple(max_shape))


class _FakeEngine:
    """Engine IO mirroring the network, as TensorRT reports it."""

    def __init__(self, owner, records):
        self.owner = owner
        self.records = records
        self.num_io_tensors = len(records)

    def _record(self, name):
        return next(record for record in self.records
                    if record['name'] == name)

    def get_tensor_name(self, index):
        return self.records[index]['name']

    def get_tensor_dtype(self, name):
        contract_name = self._record(name)['dtype']
        return self.owner.dtypes[self.owner.dtype_names[contract_name]]

    def get_tensor_shape(self, name):
        return list(self._record(name)['shape'])

    def get_tensor_mode(self, name):
        return self._record(name)['mode']

    def get_tensor_profile_shape(self, name, index):
        return [list(shape) for shape in self.owner.profile.shapes[name]]

    def get_tensor_format(self, name):
        return 'LINEAR'

    def get_tensor_location(self, name):
        return 'DEVICE'

    def is_shape_inference_io(self, name):
        return False

    def create_execution_context(self):
        return types.SimpleNamespace()


class FakeBuilderTRT:
    """Stand-in builder and runtime for ``trt_export.build_engine``."""

    _TRT_NAMES = {
        'float32': 'FLOAT',
        'float16': 'HALF',
        'bfloat16': 'BF16',
        'int32': 'INT32',
        'int8': 'INT8',
        'bool': 'BOOL',
    }

    def __init__(self, contract, *, version='10.7.0', dtype_override=None):
        self.version = version
        self.contract = contract
        self.serialized = b'built-engine-bytes'
        self.dtypes = {
            trt_name: types.SimpleNamespace(name=trt_name)
            for trt_name in self._TRT_NAMES.values()
        }
        self.dtype_names = dict(self._TRT_NAMES)
        dtype_override = dtype_override or {}
        self.network = _FakeNetwork([
            _FakeTensor(spec.name, self.dtypes[self.dtype_names[spec.dtype]],
                        spec.shape) for spec in contract.inputs
        ], [
            _FakeTensor(
                spec.name, self.dtypes[self.dtype_names[dtype_override.get(
                    spec.name, spec.dtype)]], spec.shape)
            for spec in contract.outputs
        ])
        self.profile = _FakeProfile()
        self.config = _FakeBuilderConfig()
        self.parsed_bytes = None
        self.deserialized = []
        self.module = None

    def _dtype_name(self, dtype):
        for name, value in self.dtypes.items():
            if value is dtype:
                for contract_name, trt_name in self._TRT_NAMES.items():
                    if trt_name == name:
                        return contract_name
        raise AssertionError(dtype)

    def engine(self):
        records = []
        for mode, tensors in (('INPUT', self.network.inputs),
                              ('OUTPUT', self.network.outputs)):
            for tensor in tensors:
                records.append({
                    'name': tensor.name,
                    'dtype': self._dtype_name(tensor.dtype),
                    'shape': list(tensor.shape.values),
                    'mode': mode,
                })
        return _FakeEngine(self, records)

    def install(self, monkeypatch):
        owner = self
        module = types.ModuleType('tensorrt')
        module.__version__ = self.version
        module.DataType = types.SimpleNamespace(**self.dtypes)
        module.TensorIOMode = types.SimpleNamespace(
            INPUT='INPUT', OUTPUT='OUTPUT')
        module.TensorFormat = types.SimpleNamespace(LINEAR='LINEAR')
        module.TensorLocation = types.SimpleNamespace(DEVICE='DEVICE')
        module.NetworkDefinitionCreationFlag = types.SimpleNamespace(
            EXPLICIT_BATCH=1)
        module.BuilderFlag = types.SimpleNamespace(FP16='FP16')
        module.MemoryPoolType = types.SimpleNamespace(WORKSPACE='WORKSPACE')

        class Logger:
            VERBOSE = 0
            INFO = 1
            WARNING = 2
            ERROR = 3
            INTERNAL_ERROR = 4

            def __init__(self, severity=WARNING):
                self.severity = severity

        module.Logger = Logger

        class OnnxParser:

            def __init__(self, network, logger):
                self.network = network

            def parse(self, data):
                owner.parsed_bytes = bytes(data)
                return True

            num_errors = 0

            def get_error(self, index):
                return 'no error'

        module.OnnxParser = OnnxParser

        class Builder:

            def __init__(self, logger):
                self.logger = logger

            def create_network(self, flags=0):
                owner.network_flags = flags
                return owner.network

            def create_builder_config(self):
                return owner.config

            def create_optimization_profile(self):
                return owner.profile

            def build_serialized_network(self, network, config):
                return owner.serialized

        module.Builder = Builder

        class Runtime:

            def __init__(self, logger):
                self.logger = logger

            def deserialize_cuda_engine(self, data):
                owner.deserialized.append(bytes(data))
                return owner.engine()

        module.Runtime = Runtime
        monkeypatch.setitem(sys.modules, 'tensorrt', module)
        self.module = module
        return module


def _build_inputs(tmp_path):
    source = tmp_path / 'c_radio.onnx'
    source.write_bytes(b'fake-onnx-bytes')
    return source, tmp_path / 'c_radio.engine'


def _skip_verification(monkeypatch):
    monkeypatch.setattr(trt_export, 'verify_onnx_graph',
                        lambda path, contract, **kw: None)


@pytest.mark.parametrize('precision', ['fp32', 'fp16'])
def test_build_engine_keeps_the_graph_dtypes(monkeypatch, tmp_path, precision):
    metadata = _metadata(min_batch=1, opt_batch=2, max_batch=4)
    source, engine_path = _build_inputs(tmp_path)
    fake = FakeBuilderTRT(metadata.contract())
    fake.install(monkeypatch)
    _skip_verification(monkeypatch)
    record = trt_export.build_engine(
        source,
        engine_path,
        metadata.contract(),
        precision=precision,
        workspace_bytes=1 << 21,
        verify=False)
    assert fake.parsed_bytes == source.read_bytes()
    # FP16 is a builder flag; the boundary stays float32.
    assert record['precision'] == precision
    assert record['input_dtypes'] == ['float32']
    assert record['output_dtypes'] == ['float32']
    assert record['tensorrt_version'] == '10.7.0'
    assert record['engine_bytes'] == len(fake.serialized)
    assert engine_path.read_bytes() == fake.serialized
    assert fake.profile.shapes['pixel_values'] == ((1, 3, 32, 32),
                                                   (2, 3, 32, 32), (4, 3, 32,
                                                                    32))
    assert fake.config.memory_pool_limits == [('WORKSPACE', 1 << 21)]
    assert (fake.config.flags == ['FP16']) == (precision == 'fp16')


def test_build_engine_records_the_network_boundary(monkeypatch, tmp_path):
    metadata = _metadata(input_dtype='float16', output_dtype='float16')
    source, engine_path = _build_inputs(tmp_path)
    FakeBuilderTRT(metadata.contract()).install(monkeypatch)
    _skip_verification(monkeypatch)
    record = trt_export.build_engine(
        source, engine_path, metadata.contract(), verify=False)
    assert record['input_dtypes'] == ['float16']
    assert record['output_dtypes'] == ['float16']


def test_build_engine_rejects_a_network_that_contradicts_the_contract(
        monkeypatch, tmp_path):
    metadata = _metadata()
    source, engine_path = _build_inputs(tmp_path)
    FakeBuilderTRT(
        metadata.contract(), dtype_override={
            'tokens': 'float16'
        }).install(monkeypatch)
    _skip_verification(monkeypatch)
    with pytest.raises(ValueError, match='dtype'):
        trt_export.build_engine(
            source, engine_path, metadata.contract(), verify=False)


def test_build_engine_rejects_an_unsupported_precision(tmp_path):
    with pytest.raises(ValueError, match='Unsupported build precision'):
        trt_export.build_engine(
            tmp_path / 'a.onnx',
            tmp_path / 'a.engine',
            _metadata().contract(),
            precision='int8')


def test_build_engine_verifies_the_graph_before_building(
        monkeypatch, tmp_path):
    metadata = _metadata()
    source, engine_path = _build_inputs(tmp_path)
    FakeBuilderTRT(metadata.contract()).install(monkeypatch)
    calls = []
    monkeypatch.setattr(trt_export, 'verify_onnx_graph',
                        lambda path, contract, **kw: calls.append(path))
    trt_export.build_engine(
        source, engine_path, metadata.contract(), verify=False)
    assert calls == [source]


@REQUIRES_CUDA
@pytest.mark.parametrize('precision', ['fp32', 'fp16'])
def test_build_engine_verifies_the_fresh_engine(monkeypatch, tmp_path,
                                                precision):
    metadata = _metadata()
    source, engine_path = _build_inputs(tmp_path)
    fake = FakeBuilderTRT(metadata.contract())
    fake.install(monkeypatch)
    _skip_verification(monkeypatch)
    record = trt_export.build_engine(
        source,
        engine_path,
        metadata.contract(),
        precision=precision,
        verify=True)
    assert fake.deserialized == [fake.serialized]
    assert record['validated']['loaded']
    assert record['precision'] == precision


# -- sample preprocessing, parity and benchmark ---------------------------


class FakeRuntime:
    """Injected execution handle standing in for a real engine."""

    def __init__(self, metadata, fill=0.0, device=None):
        self.contract = metadata.contract()
        self.metadata = metadata
        self.fill = fill
        self.device = torch.device(device) if device is not None else None
        self.seen = []

    def run(self, tensor):
        self.seen.append(tensor)
        batch = int(tensor.shape[0])
        grid_h, grid_w = self.metadata.grid
        return torch.full(
            (batch, grid_h * grid_w, self.metadata.feature_dim),
            float(self.fill),
            dtype=tensor.dtype,
            device=self.device if self.device is not None else tensor.device)

    def close(self):
        pass


def _trt_backbone(fill=0.0, metadata=None, device=None):
    metadata = metadata or _metadata()
    return RadioViTBackboneTensorRT(
        engine=FakeRuntime(metadata, fill, device),
        metadata=metadata,
        device=device)


def test_load_image_preprocesses_the_sample_directly():
    tensor, report = c_radio_export.load_image(
        SAMPLE_IMAGE, height=256, width=256)
    assert tensor.shape == (1, 3, 256, 256)
    assert tensor.dtype == torch.float32
    assert report['resample'] == 'PIL bicubic' and report['crop'] is None
    assert report['resize_wh'] == [256, 256]
    assert report['original_size_wh'] and report['original_mode']
    assert -1.0 <= report['tensor_min'] <= report['tensor_max'] <= 1.0
    unit = c_radio_export.unit_range(tensor)
    assert unit.min() >= 0.0 and unit.max() <= 1.0
    with pytest.raises(FileNotFoundError, match='Image not found'):
        c_radio_export.load_image(SAMPLE_IMAGE.parent / 'absent.jpg')


def test_parity_check_compares_on_the_requested_device():
    images = _images(2)
    report = c_radio_export.parity_check(
        _trt_backbone(0.5), _trt_backbone(0.5), images, device='cpu')
    assert report['allclose'] and report['cosine'] > 0.999
    assert report['relative_l2'] == 0.0
    mismatch = c_radio_export.parity_check(
        _trt_backbone(0.5), _trt_backbone(0.9), images, device='cpu')
    assert not mismatch['allclose'] and mismatch['max_abs_diff'] > 0.2
    with pytest.raises(ValueError, match='same feature raster'):
        other = _metadata(
            patch_size=8, resolution_step=8, grid_height=4, grid_width=4)
        c_radio_export.parity_check(
            _trt_backbone(0.0), _trt_backbone(0.0, other), images)


@REQUIRES_CUDA
def test_parity_check_handles_a_gpu_engine_and_a_cpu_reference():
    metadata = _metadata()
    device = torch.device('cuda', torch.cuda.current_device())
    engine_backbone = _trt_backbone(0.5, metadata, device=device)
    reference = _trt_backbone(0.5, metadata)
    images = _images(2)
    assert engine_backbone.encode_features(images).tokens.is_cuda
    on_cpu = c_radio_export.parity_check(
        engine_backbone, reference, images, device='cpu')
    assert on_cpu['allclose']
    default = c_radio_export.parity_check(engine_backbone, reference, images)
    assert default['allclose'] and default['cosine'] > 0.999


def test_benchmark_validates_counts_before_touching_cuda():
    backbone = _trt_backbone()
    with pytest.raises(ValueError, match='iterations must be at least 1'):
        c_radio_export.benchmark_backbone(backbone, _images(1), iterations=0)
    with pytest.raises(ValueError, match='warmup must be non-negative'):
        c_radio_export.benchmark_backbone(backbone, _images(1), warmup=-1)


@REQUIRES_CUDA
def test_benchmark_synchronizes_the_selected_device():
    metadata = _metadata()
    device = torch.device('cuda', torch.cuda.current_device())
    backbone = _trt_backbone(0.0, metadata, device=device)
    images = _images(1).to(device)
    report = c_radio_export.benchmark_backbone(
        backbone, images, warmup=1, iterations=3, device=device)
    assert report['device'] == str(device) and report['iterations'] == 3
    with pytest.raises(ValueError, match='requires a CUDA device'):
        c_radio_export.benchmark_backbone(
            backbone, images, warmup=0, iterations=1, device='cpu')


@NEEDS_FULL_STACK
@pytest.mark.parametrize('precision', ['fp32', 'fp16'])
def test_export_build_and_parity_end_to_end(tmp_path, precision):
    backbone = _radio_backbone(feature_dim=1280)
    device = torch.device('cuda', torch.cuda.current_device())
    onnx_path, metadata = c_radio_export.export_onnx(
        backbone,
        tmp_path / f'c_radio_{precision}.onnx',
        image_height=32,
        image_width=32,
        min_batch=1,
        opt_batch=1,
        max_batch=2,
        external_data=True,
        tensorrt_version='10')
    assert metadata.precision == 'fp32' and metadata.input_dtype == 'float32'
    final, record = c_radio_export.build_engine(
        onnx_path,
        tmp_path / f'c_radio_{precision}.engine',
        metadata,
        precision=precision,
        verify=True)
    assert final.precision == precision
    assert final.input_dtype == 'float32' and final.output_dtype == 'float32'
    engine_backbone = RadioViTBackboneTensorRT.from_engine(
        str(tmp_path / f'c_radio_{precision}.engine'), device=device.index)
    report = c_radio_export.parity_check(
        engine_backbone,
        backbone,
        _images(2),
        atol=1e-2,
        rtol=1e-2,
        device=device)
    assert report['allclose'], report
    assert record['engine_bytes'] > 0
    engine_backbone.close()


# -- real ONNX external-data conversion (needs onnx) ----------------------

NEEDS_ONNX = pytest.mark.skipif(
    not HAS_ONNX, reason='Requires the onnx package.')


class _SmallModel(nn.Module):
    """A graph with one initializer big enough to externalize."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, images):
        return self.linear(images)


def _small_onnx(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / 'small.onnx'
    torch.onnx.export(
        _SmallModel().eval(), (torch.rand(1, 64), ),
        str(path),
        input_names=['pixel_values'],
        output_names=['tokens'],
        external_data=False,
        dynamo=False)
    return path


@NEEDS_ONNX
def test_repeat_external_data_conversion_does_not_grow_the_files(tmp_path):
    import onnx
    import onnxruntime
    path = _small_onnx(tmp_path)
    feeds = {
        'pixel_values': np.random.default_rng(0).random((1, 64),
                                                        dtype='float32')
    }

    def run():
        session = onnxruntime.InferenceSession(
            str(path), providers=['CPUExecutionProvider'])
        return session.run(None, feeds)[0]

    before = run()
    name = trt_export.convert_onnx_to_external_data(path)
    data = path.parent / name
    first_size = data.stat().st_size
    graph = onnx.load(str(path), load_external_data=False)
    offsets = [
        int(entry.value) for tensor in graph.graph.initializer
        for entry in tensor.external_data if entry.key == 'offset'
    ]
    lengths = [
        int(entry.value) for tensor in graph.graph.initializer
        for entry in tensor.external_data if entry.key == 'length'
    ]
    assert min(offsets) == 0
    assert max(offset + length
               for offset, length in zip(offsets, lengths)) == first_size
    assert first_size == sum(lengths)
    # A second conversion must replace the data, not append to it, and must
    # not change what the graph computes.
    again = trt_export.convert_onnx_to_external_data(path)
    assert again == name
    assert data.stat().st_size == first_size
    after = run()
    assert np.array_equal(before[0] if isinstance(before, list) else before,
                          after[0] if isinstance(after, list) else after)


@NEEDS_ONNX
def test_failed_conversion_preserves_the_original_artifacts(
        tmp_path, monkeypatch):
    import onnx
    path = _small_onnx(tmp_path)
    name = trt_export.convert_onnx_to_external_data(path)
    data = path.parent / name
    graph_before = trt_export.sha256_file(path)
    data_before = trt_export.sha256_file(data)

    def fail(*args, **kwargs):
        raise RuntimeError('staging failure')

    monkeypatch.setattr(onnx, 'save_model', fail)
    with pytest.raises(RuntimeError, match='staging failure'):
        trt_export.convert_onnx_to_external_data(path)
    assert trt_export.sha256_file(path) == graph_before
    assert trt_export.sha256_file(data) == data_before
