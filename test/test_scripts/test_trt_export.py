# Copyright 2026 Limx Dynamics
"""Offline export/build helpers and C-RADIO sample orchestration.

The generic helpers are exercised against a stand-in TensorRT builder whose
network IO is fixed by the test, which is what makes "an FP16 build keeps the
FP32 graph boundary" observable without the real stack. The gated end-to-end
unit runs where onnx, TensorRT and a GPU are present.
"""

import importlib.util

import numpy as np
import os
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from fluxvla.models.backbones.visions.base_trt import (DYNAMIC_DIM,
                                                       require_tensorrt)
from fluxvla.models.backbones.visions.base_trt.metadata import (
    EngineContract, OptimizationProfile, TensorSpec)
from fluxvla.models.backbones.visions.base_trt.radio_metadata import (
    EngineMetadata, default_metadata_path)
from fluxvla.models.backbones.visions.radio_trt import \
    RadioViTBackboneTensorRT
from fluxvla.models.backbones.visions.radio_vit import RadioViTBackbone
from tools import c_radio_export, trt_export, trt_layer_report

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

    def __init__(self, inputs, outputs, layers=()):
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.layers = list(layers)
        self.num_inputs = len(self.inputs)
        self.num_outputs = len(self.outputs)
        self.num_layers = len(self.layers)

    def get_input(self, index):
        return self.inputs[index]

    def get_output(self, index):
        return self.outputs[index]

    def get_layer(self, index):
        return self.layers[index]


class _FakeLayer:
    """A parsed-network layer: a name, a type and a settable precision."""

    def __init__(self, name, layer_type='NORMALIZATION'):
        self.name = name
        self.type = layer_type
        self.precision_is_set = False
        self._precision = 'DEFAULT'
        self._outputs = [types.SimpleNamespace(dtype='FLOAT')]
        self.num_outputs = len(self._outputs)

    def get_output(self, index):
        return self._outputs[index]

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, value):
        self._precision = value
        self.precision_is_set = True


class _PrecisionTrackingLayer(_FakeLayer):
    """Records every precision assignment so the last one can be asserted."""

    def __init__(self, name, layer_type='NORMALIZATION'):
        #: Every precision build_engine assigned, in order.
        self.assignments = []
        super().__init__(name, layer_type)

    @_FakeLayer.precision.setter
    def precision(self, value):
        self.assignments.append(value)
        _FakeLayer.precision.fset(self, value)


class _FakeBuilderConfig:

    def __init__(self):
        self.flags = []
        self.cleared = []
        self.memory_pool_limits = []
        self.profiles = []
        #: TensorRT's own default, which ``build_engine`` only overrides when
        #: profiling_verbosity is requested.
        self.profiling_verbosity = 'LAYER_NAMES_ONLY'

    def set_flag(self, flag):
        self.flags.append(flag)

    def clear_flag(self, flag):
        self.cleared.append(flag)
        if flag in self.flags:
            self.flags.remove(flag)

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

    #: Parsed-network layers the stand-in reports when a test does not supply
    #: its own; the names mirror the real C-RADIO graph's ONNX node names.
    DEFAULT_LAYERS = (
        ('/model/blocks/blocks.0/norm1/LayerNormalization', 'NORMALIZATION'),
        ('/model/blocks/blocks.1/norm1/LayerNormalization', 'NORMALIZATION'),
        ('/model/blocks/blocks.0/attn/Softmax', 'SOFTMAX'),
        ('/model/patch_generator/embedder/MatMul', 'MATRIX_MULTIPLY'),
    )

    _TRT_NAMES = {
        'float32': 'FLOAT',
        'float16': 'HALF',
        'bfloat16': 'BF16',
        'int32': 'INT32',
        'int8': 'INT8',
        'bool': 'BOOL',
    }

    def __init__(self,
                 contract,
                 *,
                 version='10.7.0',
                 dtype_override=None,
                 bf16_flag=True,
                 platform_has_fast_bf16=True,
                 layers=None):
        self.version = version
        self.contract = contract
        self.bf16_flag = bf16_flag
        self.platform_has_fast_bf16 = platform_has_fast_bf16
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
        ], [
            _PrecisionTrackingLayer(name, layer_type)
            for name, layer_type in (layers or FakeBuilderTRT.DEFAULT_LAYERS)
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
        # TF32 and the precision-constraint flags exist in the TensorRT 10.x
        # builds this pipeline targets, so the stand-in has them too; a
        # stand-in without them could not show that an FP32 build disables
        # TF32 and that layer overrides are enforced.
        flags = {
            'FP16': 'FP16',
            'TF32': 'TF32',
            'OBEY_PRECISION_CONSTRAINTS': 'OBEY_PRECISION_CONSTRAINTS',
            'PREFER_PRECISION_CONSTRAINTS': 'PREFER_PRECISION_CONSTRAINTS',
        }
        if self.bf16_flag:
            flags['BF16'] = 'BF16'
        module.BuilderFlag = types.SimpleNamespace(**flags)
        module.MemoryPoolType = types.SimpleNamespace(WORKSPACE='WORKSPACE')
        module.ProfilingVerbosity = types.SimpleNamespace(
            NONE='NONE',
            LAYER_NAMES_ONLY='LAYER_NAMES_ONLY',
            DETAILED='DETAILED')

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
                self.platform_has_fast_bf16 = owner.platform_has_fast_bf16

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


#: Builder flags each precision must ask for, and nothing else. An FP32 build
#: is strict by default, so it also clears TF32 and asks for the declared
#: precisions to be obeyed.
_PRECISION_FLAGS = {
    'fp32': ['NO_TF32', 'OBEY_PRECISION_CONSTRAINTS'],
    'fp16': ['FP16'],
    'bf16': ['BF16'],
}

#: The subset of those that is a flag actually set on the builder config.
#: NO_TF32 is a cleared flag, so it never appears in ``config.flags``.
_PRECISION_CONFIG_FLAGS = {
    'fp32': ['OBEY_PRECISION_CONSTRAINTS'],
    'fp16': ['FP16'],
    'bf16': ['BF16'],
}


@pytest.mark.parametrize('precision', ['fp32', 'fp16', 'bf16'])
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
    # FP16 and BF16 are builder flags; the boundary stays float32.
    assert record['precision'] == precision
    assert record['precision_flags'] == _PRECISION_FLAGS[precision]
    assert record['input_dtypes'] == ['float32']
    assert record['output_dtypes'] == ['float32']
    assert record['tensorrt_version'] == '10.7.0'
    assert record['engine_bytes'] == len(fake.serialized)
    assert engine_path.read_bytes() == fake.serialized
    assert fake.profile.shapes['pixel_values'] == ((1, 3, 32, 32),
                                                   (2, 3, 32, 32), (4, 3, 32,
                                                                    32))
    assert fake.config.memory_pool_limits == [('WORKSPACE', 1 << 21)]
    assert fake.config.flags == _PRECISION_CONFIG_FLAGS[precision]


@pytest.mark.parametrize('kwargs,message', [
    ({
        'bf16_flag': False
    }, 'BuilderFlag.BF16'),
    ({
        'platform_has_fast_bf16': False
    }, 'platform_has_fast_bf16=False'),
])
def test_build_engine_bf16_never_falls_back_to_fp16(monkeypatch, tmp_path,
                                                   kwargs, message):
    """An unsupported BF16 build must fail instead of running FP16 tactics."""
    metadata = _metadata()
    source, engine_path = _build_inputs(tmp_path)
    fake = FakeBuilderTRT(metadata.contract(), **kwargs)
    fake.install(monkeypatch)
    _skip_verification(monkeypatch)
    with pytest.raises(RuntimeError, match=message) as error:
        trt_export.build_engine(
            source,
            engine_path,
            metadata.contract(),
            precision='bf16',
            verify=False)
    assert 'fp32 or fp16 for this device' in str(error.value)
    assert fake.config.flags == []
    assert not engine_path.exists()


def test_build_engine_bf16_probes_the_cuda_capability_without_the_platform_flag(
        monkeypatch, tmp_path):
    """TensorRT 10.13 exposes no platform_has_fast_bf16, so the device is asked.

    A pre-Ampere device cannot run BF16 kernels; the build must stop with a
    reason rather than quietly select FP16 tactics.
    """
    metadata = _metadata()
    source, engine_path = _build_inputs(tmp_path)
    fake = FakeBuilderTRT(metadata.contract(),
                          platform_has_fast_bf16=None)
    fake.install(monkeypatch)
    _skip_verification(monkeypatch)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability',
                        lambda *args, **kwargs: (7, 5))
    monkeypatch.setattr(torch.cuda, 'get_device_name',
                        lambda *args, **kwargs: 'Test Turing GPU')
    with pytest.raises(RuntimeError, match='compute capability') as error:
        trt_export.build_engine(
            source,
            engine_path,
            metadata.contract(),
            precision='bf16',
            verify=False)
    assert '7.5' in str(error.value) and 'Test Turing GPU' in str(error.value)
    assert fake.config.flags == []
    assert not engine_path.exists()


# -- per-layer precision overrides ---------------------------------------


def test_parse_layer_precision_follows_the_trtexec_layer_precision_syntax():
    parsed = trt_export.parse_layer_precision('*/LayerNormalization:fp32')
    assert parsed.pattern == '*/LayerNormalization'
    assert parsed.precision == 'fp32' and parsed.dtype == 'float32'
    assert parsed.spec == '*/LayerNormalization:fp32'
    # An already-parsed constraint passes through unchanged.
    assert trt_export.parse_layer_precision(parsed) is parsed
    # Long and short labels are the same request.
    assert trt_export.parse_layer_precision('/a/b:float16').precision == 'fp16'
    assert trt_export.parse_layer_precision('/a/b:BF16').dtype == 'bfloat16'
    # The dtype is the last colon-separated field, so a pattern may contain
    # colons.
    odd = trt_export.parse_layer_precision('odd:name:fp16')
    assert odd.pattern == 'odd:name' and odd.dtype == 'float16'


@pytest.mark.parametrize('spec,message', [
    ('LayerNormalization', 'PATTERN:DTYPE'),
    (':fp32', 'PATTERN:DTYPE'),
    ('LayerNormalization:int8', 'not supported'),
    ('LayerNormalization:fp8', 'not supported'),
])
def test_parse_layer_precision_rejects_a_malformed_spec(spec, message):
    with pytest.raises(ValueError, match=message):
        trt_export.parse_layer_precision(spec)


def test_build_engine_constrains_only_the_matching_layers(
        monkeypatch, tmp_path):
    metadata = _metadata()
    source, engine_path = _build_inputs(tmp_path)
    fake = FakeBuilderTRT(metadata.contract())
    fake.install(monkeypatch)
    _skip_verification(monkeypatch)
    record = trt_export.build_engine(
        source,
        engine_path,
        metadata.contract(),
        precision='fp16',
        layer_precisions=['*/LayerNormalization:fp32'],
        precision_constraints='obey',
        verify=False)
    assert record['precision'] == 'fp16'
    # FP16 tactics plus the constraints flag, and only those.
    assert record['precision_flags'] == ['FP16', 'OBEY_PRECISION_CONSTRAINTS']
    assert fake.config.flags == ['FP16', 'OBEY_PRECISION_CONSTRAINTS']
    applied = record['layer_precisions']
    assert [entry['pattern'] for entry in applied] == ['*/LayerNormalization']
    entry = applied[0]
    assert entry['precision'] == 'fp32' and entry['dtype'] == 'float32'
    assert entry['matched_layers'] == 2
    assert entry['layer_types'] == ['NORMALIZATION']
    assert entry['truncated'] is False
    assert entry['layer_names'] == [
        '/model/blocks/blocks.0/norm1/LayerNormalization',
        '/model/blocks/blocks.1/norm1/LayerNormalization',
    ]
    assert record['network_layers'] == 4
    # The matched layers were constrained to FP32 and nothing else was, and
    # the graph boundary stayed float32.
    assignments = {
        layer.name: layer.assignments
        for layer in fake.network.layers
    }
    assert [
        value.name for value in assignments['/model/blocks/blocks.0/norm1/'
                                            'LayerNormalization']
    ] == ['FLOAT']
    assert assignments['/model/blocks/blocks.0/attn/Softmax'] == []
    assert assignments['/model/patch_generator/embedder/MatMul'] == []
    assert record['input_dtypes'] == ['float32']
    assert record['output_dtypes'] == ['float32']
    assert engine_path.read_bytes() == fake.serialized


def test_build_engine_fails_when_a_pattern_matches_no_layer(
        monkeypatch, tmp_path):
    """A pattern naming an op type rather than a layer name must not no-op.

    ``LayerNormalization`` is the ONNX operator and the TensorRT layer *type*;
    the layer *names* end in ``/LayerNormalization``. Writing the bare type is
    the natural mistake, and trtexec accepts it silently.
    """
    metadata = _metadata()
    source, engine_path = _build_inputs(tmp_path)
    fake = FakeBuilderTRT(metadata.contract())
    fake.install(monkeypatch)
    _skip_verification(monkeypatch)
    with pytest.raises(
            ValueError, match='matched none of the 4 layers') as error:
        trt_export.build_engine(
            source,
            engine_path,
            metadata.contract(),
            precision='fp16',
            layer_precisions=['LayerNormalization:fp32'],
            precision_constraints='obey',
            verify=False)
    assert 'trt_layer_report' in str(error.value)
    assert not engine_path.exists()
    assert all(not layer.assignments for layer in fake.network.layers)


def test_build_engine_rejects_overrides_without_a_constraints_mode(
        monkeypatch, tmp_path):
    metadata = _metadata()
    source, engine_path = _build_inputs(tmp_path)
    fake = FakeBuilderTRT(metadata.contract())
    fake.install(monkeypatch)
    _skip_verification(monkeypatch)
    with pytest.raises(ValueError, match='silently drop') as error:
        trt_export.build_engine(
            source,
            engine_path,
            metadata.contract(),
            precision='fp16',
            layer_precisions=['*/LayerNormalization:fp32'],
            verify=False)
    assert "precision_constraints='obey'" in str(error.value)
    assert fake.config.flags == []
    assert not engine_path.exists()


def test_build_engine_rejects_a_constraints_mode_without_overrides(
        monkeypatch, tmp_path):
    metadata = _metadata()
    source, engine_path = _build_inputs(tmp_path)
    fake = FakeBuilderTRT(metadata.contract())
    fake.install(monkeypatch)
    _skip_verification(monkeypatch)
    with pytest.raises(ValueError, match='silently do nothing') as error:
        trt_export.build_engine(
            source,
            engine_path,
            metadata.contract(),
            precision='fp16',
            precision_constraints='obey',
            verify=False)
    assert 'layer_precisions' in str(error.value)
    assert fake.config.flags == []


def test_build_engine_rejects_an_unknown_constraints_mode(tmp_path):
    with pytest.raises(ValueError, match='Unsupported precision constraints'):
        trt_export.build_engine(
            tmp_path / 'a.onnx',
            tmp_path / 'a.engine',
            _metadata().contract(),
            precision='fp16',
            precision_constraints='maybe')


def test_layer_precision_policy_describes_a_build_without_an_override():
    """A build with no override states that there is no policy."""
    cleared = {
        'precision_constraints': None,
        'layer_precisions': None,
    }
    assert c_radio_export.layer_precision_policy({}) == cleared
    assert c_radio_export.layer_precision_policy({
        'layer_precisions': [],
        'network_layers': 4,
    }) == cleared
    assert c_radio_export.layer_precision_policy({
        'precision_constraints':
        'obey',
        'network_layers':
        4972,
        'layer_precisions': [{
            'pattern': '*/LayerNormalization',
            'precision': 'fp32',
            'matched_layers': 64,
        }],
    }) == {
        'precision_constraints': 'obey',
        'layer_precisions':
        '*/LayerNormalization:fp32 matched 64 of 4972 layers',
    }


def test_layer_precision_policy_clears_a_stale_policy():
    """Rebuilding an unconstrained engine must not inherit an old policy.

    ``dataclasses.replace`` only overwrites the keys it is given, so a policy
    helper that returned nothing for an unconstrained build would leave the
    metadata object's previous constraints in the new sidecar.
    """
    stale = _metadata(
        precision='fp16',
        precision_constraints='obey',
        layer_precisions='*/LayerNormalization:fp32 matched 64 of 4972 layers')
    record = {'layer_precisions': [], 'network_layers': 4972}
    cleared = replace(stale, **c_radio_export.layer_precision_policy(record))
    assert cleared.precision_constraints is None
    assert cleared.layer_precisions is None
    # And a constrained build still overwrites them.
    record = {
        'precision_constraints': 'obey',
        'network_layers': 4972,
        'layer_precisions': [{
            'pattern': '*/LayerNormalization',
            'precision': 'fp32',
            'matched_layers': 64,
        }],
    }
    applied = replace(stale, **c_radio_export.layer_precision_policy(record))
    assert applied.precision_constraints == 'obey'
    assert applied.layer_precisions == (
        '*/LayerNormalization:fp32 matched 64 of 4972 layers')


def test_build_engine_keeps_the_fp32_baseline_record_without_overrides(
        monkeypatch, tmp_path):
    """The FP32 baseline keeps its strict flags and its pre-existing record."""
    metadata = _metadata()
    source, engine_path = _build_inputs(tmp_path)
    fake = FakeBuilderTRT(metadata.contract())
    fake.install(monkeypatch)
    _skip_verification(monkeypatch)
    record = trt_export.build_engine(
        source, engine_path, metadata.contract(), verify=False)
    assert record['precision'] == 'fp32'
    assert record['precision_flags'] == [
        'NO_TF32', 'OBEY_PRECISION_CONSTRAINTS'
    ]
    # No override means no new record keys: the pre-existing shape is kept.
    for key in ('layer_precisions', 'precision_constraints', 'network_layers',
                'profiling_verbosity'):
        assert key not in record


def test_build_engine_lets_the_last_matching_pattern_win(
        monkeypatch, tmp_path):
    metadata = _metadata()
    source, engine_path = _build_inputs(tmp_path)
    fake = FakeBuilderTRT(metadata.contract())
    fake.install(monkeypatch)
    _skip_verification(monkeypatch)
    record = trt_export.build_engine(
        source,
        engine_path,
        metadata.contract(),
        precision='fp16',
        layer_precisions=[
            '*/LayerNormalization:fp32',
            '*/blocks.0/norm1/LayerNormalization:fp16',
        ],
        precision_constraints='obey',
        verify=False)
    assert [entry['matched_layers']
            for entry in record['layer_precisions']] == [2, 1]
    first = next(layer for layer in fake.network.layers
                 if 'blocks.0/norm1' in layer.name)
    assert [value.name for value in first.assignments] == ['FLOAT', 'HALF']


def test_build_engine_truncates_a_broad_layer_list(monkeypatch, tmp_path):
    metadata = _metadata()
    source, engine_path = _build_inputs(tmp_path)
    layers = [(f'/model/layer{index}/LayerNormalization', 'NORMALIZATION')
              for index in range(trt_export._LAYER_NAME_RECORD_LIMIT + 5)]
    fake = FakeBuilderTRT(metadata.contract(), layers=layers)
    fake.install(monkeypatch)
    _skip_verification(monkeypatch)
    record = trt_export.build_engine(
        source,
        engine_path,
        metadata.contract(),
        precision='fp16',
        layer_precisions=['*/LayerNormalization:fp32'],
        precision_constraints='prefer',
        verify=False)
    entry = record['layer_precisions'][0]
    assert entry['matched_layers'] == trt_export._LAYER_NAME_RECORD_LIMIT + 5
    assert len(entry['layer_names']) == trt_export._LAYER_NAME_RECORD_LIMIT
    assert entry['truncated'] is True
    assert record['precision_constraints'] == 'prefer'
    assert record['precision_flags'] == [
        'FP16', 'PREFER_PRECISION_CONSTRAINTS'
    ]


def test_build_engine_never_sets_contradictory_constraint_flags(
        monkeypatch, tmp_path):
    """A strict FP32 build must not also be told to prefer constraints.

    The FP32 baseline sets OBEY_PRECISION_CONSTRAINTS on its own, so an
    explicit ``prefer`` alongside a layer override has to win instead of
    leaving both flags set.
    """
    metadata = _metadata()
    source, engine_path = _build_inputs(tmp_path)
    fake = FakeBuilderTRT(metadata.contract())
    fake.install(monkeypatch)
    _skip_verification(monkeypatch)
    record = trt_export.build_engine(
        source,
        engine_path,
        metadata.contract(),
        precision='fp32',
        layer_precisions=['*/LayerNormalization:fp16'],
        precision_constraints='prefer',
        verify=False)
    assert record['precision_flags'] == [
        'NO_TF32', 'PREFER_PRECISION_CONSTRAINTS'
    ]
    assert fake.config.flags == ['PREFER_PRECISION_CONSTRAINTS']
    # The default strict FP32 build is unchanged.
    fake.config.flags.clear()
    strict = trt_export.build_engine(
        source,
        engine_path,
        metadata.contract(),
        precision='fp32',
        verify=False)
    assert strict['precision_flags'] == [
        'NO_TF32', 'OBEY_PRECISION_CONSTRAINTS'
    ]
    assert fake.config.flags == ['OBEY_PRECISION_CONSTRAINTS']


def test_build_engine_records_the_requested_profiling_verbosity(
        monkeypatch, tmp_path):
    metadata = _metadata()
    source, engine_path = _build_inputs(tmp_path)
    fake = FakeBuilderTRT(metadata.contract())
    fake.install(monkeypatch)
    _skip_verification(monkeypatch)
    record = trt_export.build_engine(
        source,
        engine_path,
        metadata.contract(),
        precision='fp16',
        profiling_verbosity='detailed',
        verify=False)
    assert fake.config.profiling_verbosity == 'DETAILED'
    assert record['profiling_verbosity'] == 'detailed'
    with pytest.raises(ValueError, match='Unsupported profiling verbosity'):
        trt_export.build_engine(
            source,
            engine_path,
            metadata.contract(),
            profiling_verbosity='verbose')


def test_capture_std_streams_keeps_output_and_restores_the_streams(tmp_path):
    """The file descriptors are redirected, which is what TensorRT writes."""
    log = tmp_path / 'build.log'
    log.write_text('earlier diagnostic\n')
    before = (os.fstat(1).st_ino, os.fstat(2).st_ino)
    with trt_export.capture_std_streams(log):
        inside = (os.fstat(1).st_ino, os.fstat(2).st_ino)
        assert inside != before
        # Write to the descriptors, not to sys.stdout: the builder is C++ and
        # writes to the streams, and a test harness may rebind sys.stdout.
        os.write(1, b'builder stdout\n')
        os.write(2, b'builder stderr\n')
    assert (os.fstat(1).st_ino, os.fstat(2).st_ino) == before
    text = log.read_text()
    assert 'earlier diagnostic' in text
    assert 'builder stdout' in text and 'builder stderr' in text
    # A failure inside the block still restores the streams.
    with pytest.raises(RuntimeError, match='build exploded'):
        with trt_export.capture_std_streams(log):
            raise RuntimeError('build exploded')
    assert (os.fstat(1).st_ino, os.fstat(2).st_ino) == before


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
@pytest.mark.parametrize('precision', ['fp32', 'fp16', 'bf16'])
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
@pytest.mark.parametrize('precision', ['fp32', 'fp16', 'bf16'])
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


@NEEDS_FULL_STACK
def test_layer_precision_override_reaches_the_real_engine(tmp_path):
    """TensorRT accepts the constraint and the constrained layer follows it.

    The comparison is relative to a pure FP32 build of the same graph rather
    than to a datatype spelling, because TensorRT names its per-layer datatypes
    itself. On a one-convolution stand-in graph TensorRT may well choose FP32
    for that convolution in the FP16 build too, so this asserts the constraint
    is honoured, not that it changed the outcome.
    """
    require_tensorrt()
    backbone = _radio_backbone(feature_dim=8)
    onnx_path, metadata = c_radio_export.export_onnx(
        backbone,
        tmp_path / 'tiny.onnx',
        image_height=32,
        image_width=32,
        external_data=False,
        tensorrt_version='10')
    graph = trt_layer_report.describe_graph(onnx_path)
    convolution = next(entry['name'] for entry in graph['layers']
                       if entry['type'] == 'CONVOLUTION')

    def datatypes(engine_name, **kwargs):
        engine_path = tmp_path / engine_name
        _, record = c_radio_export.build_engine(
            onnx_path,
            engine_path,
            metadata,
            verify=False,
            profiling_verbosity='detailed',
            **kwargs)
        report = trt_layer_report.describe_engine(engine_path)
        assert report['datatype_is_available'], report['note']
        layer = next(entry for entry in report['layers']
                     if entry['name'] == convolution)
        return layer['datatypes'], record

    half, _ = datatypes('half.engine', precision='fp16')
    constrained, override_record = datatypes(
        'constrained.engine',
        precision='fp16',
        layer_precisions=[f'{convolution}:fp32'],
        precision_constraints='obey')
    full, _ = datatypes('full.engine', precision='fp32')
    assert constrained == full, (constrained, full, half)
    entry = override_record['layer_precisions'][0]
    assert entry['matched_layers'] == 1
    assert entry['layer_names'] == [convolution]
    assert override_record['precision_flags'] == [
        'FP16', 'OBEY_PRECISION_CONSTRAINTS'
    ]
    # Naming the ONNX operator instead of a layer name matches nothing, and the
    # real parser must say so rather than build an unconstrained engine.
    with pytest.raises(ValueError, match='matched none of'):
        c_radio_export.build_engine(
            onnx_path,
            tmp_path / 'rejected.engine',
            metadata,
            precision='fp16',
            layer_precisions=['LayerNormalization:fp32'],
            precision_constraints='obey',
            verify=False)
    assert not (tmp_path / 'rejected.engine').exists()


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
