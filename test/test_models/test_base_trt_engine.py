# Copyright 2026 Limx Dynamics
"""Generic TensorRT engine and contract: version guard, IO, lifetime.

TensorRT is not a repository dependency, so the engine surface is exercised
against a stand-in module that reproduces the TensorRT 10.x Python API this
code targets. The units that deserialize or enqueue need a CUDA device and are
skipped without one.
"""

import sys
import threading
import types

import pytest
import torch

from fluxvla.models.backbones.visions.base_trt import engine as engine_module
from fluxvla.models.backbones.visions.base_trt.engine import (
    SUPPORTED_TENSORRT_MAJORS, TensorRTEngine, require_tensorrt,
    torch_dtype_for, trt_dtype_name)
from fluxvla.models.backbones.visions.base_trt.metadata import (
    DTYPE_NAMES, DYNAMIC_DIM, EngineContract, OptimizationProfile, TensorSpec,
    validate_engine_contract)

REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason='Requires CUDA.')


class _DType:

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'trt.DataType.{self.name}'


class _TensorIOMode:
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'


_DTYPES = {
    'float32': _DType('FLOAT'),
    'float16': _DType('HALF'),
    'bfloat16': _DType('BF16'),
    'int32': _DType('INT32'),
    'int8': _DType('INT8'),
    'bool': _DType('BOOL'),
}


class _Context:

    def __init__(self, owner):
        self.owner = owner
        self.input_shapes = {}
        self.addresses = {}
        self.enqueues = []

    def set_input_shape(self, name, shape):
        if self.owner.set_input_shape_error:
            raise RuntimeError(self.owner.set_input_shape_error)
        self.input_shapes[name] = tuple(int(dim) for dim in shape)
        return self.owner.set_input_shape_result

    def get_tensor_shape(self, name):
        if self.owner.output_shape_override is not None:
            return list(self.owner.output_shape_override)
        declared = list(self.owner.declared_shape(name))
        if declared and declared[0] == DYNAMIC_DIM:
            batch = next(iter(self.input_shapes.values()), (1, ))[0]
            declared[0] = int(batch)
        return declared

    def set_tensor_address(self, name, pointer):
        self.addresses[name] = int(pointer)
        return self.owner.address_result

    def execute_async_v3(self, stream_handle):
        self.enqueues.append(int(stream_handle))
        if self.owner.enqueue_error:
            raise RuntimeError(self.owner.enqueue_error)
        return self.owner.enqueue_result


class _Engine:

    def __init__(self, owner):
        self.owner = owner

    @property
    def num_io_tensors(self):
        return len(self.owner.records)

    def _record(self, name):
        for record in self.owner.records:
            if record['name'] == name:
                return record
        raise KeyError(name)

    def get_tensor_name(self, index):
        return self.owner.records[index]['name']

    def get_tensor_dtype(self, name):
        return _DTYPES[self._record(name)['dtype']]

    def get_tensor_shape(self, name):
        return list(self.owner.declared_shape(name))

    def get_tensor_mode(self, name):
        return self._record(name)['mode']

    def get_tensor_profile_shape(self, name, index):
        if self.owner.omit_profile:
            return None
        return [list(shape) for shape in self.owner.profile[name]]

    def get_tensor_format(self, name):
        return self.owner.formats.get(name, 'LINEAR')

    def get_tensor_location(self, name):
        return self.owner.locations.get(name, 'DEVICE')

    def is_shape_inference_io(self, name):
        return name in self.owner.shape_inference

    def create_execution_context(self):
        if self.owner.context_is_none:
            return None
        self.owner.context_creation_devices.append(
            int(torch.cuda.current_device()) if torch.cuda.is_available(
            ) else -1)
        context = getattr(self.owner, 'context_class', _Context)(self.owner)
        self.owner.contexts.append(context)
        return context


class FakeTensorRT:
    """Stand-in TensorRT module with an explicitly declared engine boundary."""

    def __init__(self,
                 *,
                 inputs=None,
                 outputs=None,
                 profile=None,
                 version='10.7.0',
                 **options):
        self.version = version
        self.records = []
        for spec in inputs or ():
            self.records.append(dict(spec, mode=_TensorIOMode.INPUT))
        for spec in outputs or ():
            self.records.append(dict(spec, mode=_TensorIOMode.OUTPUT))
        self.profile = dict(profile or {})
        self.omit_profile = options.pop('omit_profile', False)
        self.formats = dict(options.pop('formats', {}))
        self.locations = dict(options.pop('locations', {}))
        self.shape_inference = set(options.pop('shape_inference', ()))
        self.contexts = []
        self.context_creation_devices = []
        self.deserialized = []
        self.deserialize_ok = options.pop('deserialize_ok', True)
        self.enqueue_result = options.pop('enqueue_result', True)
        self.enqueue_error = options.pop('enqueue_error', None)
        self.set_input_shape_error = options.pop('set_input_shape_error', None)
        self.set_input_shape_result = options.pop('set_input_shape_result',
                                                  True)
        self.address_result = options.pop('address_result', True)
        self.output_shape_override = options.pop('output_shape_override', None)
        self.context_is_none = options.pop('context_is_none', False)
        self.omit_execute_async_v3 = options.pop('omit_execute_async_v3',
                                                 False)
        self.module = None

    def declared_shape(self, name):
        for record in self.records:
            if record['name'] == name:
                return tuple(record['shape'])
        raise KeyError(name)

    def engine(self):
        return _Engine(self) if self.deserialize_ok else None

    def install(self, monkeypatch):
        owner = self
        module = types.ModuleType('tensorrt')
        module.__version__ = self.version
        module.DataType = types.SimpleNamespace(**_DTYPES)
        module.TensorIOMode = _TensorIOMode
        module.TensorFormat = types.SimpleNamespace(
            LINEAR='LINEAR', HWC8='HWC8')
        module.TensorLocation = types.SimpleNamespace(
            DEVICE='DEVICE', HOST='HOST')
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

        class Runtime:

            def __init__(self, logger):
                self.logger = logger

            def deserialize_cuda_engine(self, serialized):
                owner.deserialized.append(bytes(serialized))
                return owner.engine()

        module.Runtime = Runtime
        if owner.omit_execute_async_v3:
            owner.context_class = type('_NoEnqueue', (_Context, ),
                                       {'execute_async_v3': None})
        monkeypatch.setitem(sys.modules, 'tensorrt', module)
        self.module = module
        return module


# -- model-independent contract ------------------------------------------


def _contract(specs, profile=None):
    inputs = tuple(
        TensorSpec(name=name, dtype=dtype, shape=shape)
        for name, dtype, shape in specs['inputs'])
    outputs = tuple(
        TensorSpec(name=name, dtype=dtype, shape=shape)
        for name, dtype, shape in specs['outputs'])
    profiles = ()
    if profile:
        profiles = tuple(
            OptimizationProfile(
                tensor=name,
                min_shape=bounds[0],
                opt_shape=bounds[1],
                max_shape=bounds[2]) for name, bounds in profile.items())
    return EngineContract(inputs=inputs, outputs=outputs, profiles=profiles)


def _image_contract(**overrides):
    specs = {
        'inputs': [('pixel_values', 'float32', (DYNAMIC_DIM, 3, 32, 32))],
        'outputs': [('tokens', 'float32', (DYNAMIC_DIM, 4, 1280))],
    }
    specs.update(overrides)
    return _contract(
        specs,
        {'pixel_values': ((1, 3, 32, 32), (2, 3, 32, 32), (4, 3, 32, 32))})


def _profile_bounds():
    return ((1, 3, 32, 32), (2, 3, 32, 32), (4, 3, 32, 32))


def _records(contract, *, dtype_override=None, shape_override=None):
    """Engine records matching a contract, with optional perturbations."""
    inputs, outputs = [], []
    for kind, specs in (('INPUT', contract.inputs), ('OUTPUT',
                                                     contract.outputs)):
        for spec in specs:
            shape = list(spec.shape)
            if shape_override and (spec.name, kind) in shape_override:
                shape = list(shape_override[(spec.name, kind)])
            dtype = spec.dtype
            if dtype_override and (spec.name, kind) in dtype_override:
                dtype = dtype_override[(spec.name, kind)]
            record = {
                'name': spec.name,
                'dtype': dtype,
                'shape': shape,
            }
            (inputs if kind == 'INPUT' else outputs).append(record)
    return inputs, outputs


def test_contract_accepts_the_declared_boundary():
    contract = _image_contract()
    inputs, outputs = _records(contract)
    validate_engine_contract(contract,
                             [TensorSpec(**record) for record in inputs],
                             [TensorSpec(**record) for record in outputs])


def test_contract_rejects_a_mismatched_name():
    contract = _image_contract()
    inputs, outputs = _records(contract)
    outputs[0] = dict(outputs[0], name='other')
    with pytest.raises(ValueError, match='reports no output named'):
        validate_engine_contract(contract,
                                 [TensorSpec(**record) for record in inputs],
                                 [TensorSpec(**record) for record in outputs])


def test_contract_rejects_dtype_rank_and_static_axis_mismatches():
    contract = _image_contract()
    inputs, outputs = _records(
        contract, dtype_override={('pixel_values', 'INPUT'): 'float16'})
    with pytest.raises(ValueError, match='dtype'):
        validate_engine_contract(contract,
                                 [TensorSpec(**record) for record in inputs],
                                 [TensorSpec(**record) for record in outputs])
    inputs, outputs = _records(
        contract, shape_override={('tokens', 'OUTPUT'): (DYNAMIC_DIM, 4)})
    with pytest.raises(ValueError, match='rank'):
        validate_engine_contract(contract,
                                 [TensorSpec(**record) for record in inputs],
                                 [TensorSpec(**record) for record in outputs])
    inputs, outputs = _records(
        contract,
        shape_override={('tokens', 'OUTPUT'): (DYNAMIC_DIM, 5, 1280)})
    with pytest.raises(ValueError, match='static size'):
        validate_engine_contract(contract,
                                 [TensorSpec(**record) for record in inputs],
                                 [TensorSpec(**record) for record in outputs])
    inputs, outputs = _records(
        contract,
        shape_override={('pixel_values', 'INPUT'): (DYNAMIC_DIM, 1, 32, 32)})
    with pytest.raises(ValueError, match='static size'):
        validate_engine_contract(contract,
                                 [TensorSpec(**record) for record in inputs],
                                 [TensorSpec(**record) for record in outputs])


def test_contract_rejects_a_dynamic_axis_outside_the_profile():
    contract = _image_contract()
    inputs, outputs = _records(contract)
    inputs[0] = dict(inputs[0], shape=(7, 3, 32, 32))
    with pytest.raises(ValueError, match='dynamic'):
        validate_engine_contract(contract,
                                 [TensorSpec(**record) for record in inputs],
                                 [TensorSpec(**record) for record in outputs])


def test_contract_rejects_extra_counts_and_bad_profiles():
    contract = _image_contract()
    with pytest.raises(ValueError, match='has 0 inputs'):
        validate_engine_contract(contract, [], [])
    with pytest.raises(ValueError, match='needs an optimization profile'):
        _contract({
            'inputs': [('x', 'float32', (DYNAMIC_DIM, 3))],
            'outputs': [('y', 'float32', (DYNAMIC_DIM, 2))]
        })
    with pytest.raises(ValueError, match='min <= opt <= max'):
        OptimizationProfile(
            tensor='x', min_shape=(2, ), opt_shape=(1, ), max_shape=(3, ))
    with pytest.raises(ValueError, match='must be unique'):
        _contract({
            'inputs': [('x', 'float32', (1, ))],
            'outputs': [('x', 'float32', (1, ))]
        })
    with pytest.raises(ValueError, match='not an engine input'):
        EngineContract(
            inputs=(TensorSpec('x', 'float32', (1, )), ),
            outputs=(TensorSpec('y', 'float32', (1, )), ),
            profiles=(OptimizationProfile('z', (1, ), (1, ), (1, )), ))


def test_tensor_spec_round_trips_and_resolves():
    spec = TensorSpec(name='x', dtype='float32', shape=(DYNAMIC_DIM, 3))
    assert spec.rank == 2 and spec.dynamic_axes == (0, )
    assert spec.resolve({0: 5}) == (5, 3)
    with pytest.raises(ValueError, match='concrete value'):
        spec.resolve({})
    assert TensorSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(ValueError, match='Unsupported tensor dtype'):
        TensorSpec(name='x', dtype='float64', shape=(1, ))


def test_contract_json_round_trip(tmp_path):
    contract = _image_contract()
    path = contract.save(tmp_path / 'contract.json')
    assert EngineContract.load(path) == contract
    assert EngineContract.load(path).to_dict() == contract.to_dict()
    with pytest.raises(FileNotFoundError, match='contract not found'):
        EngineContract.load(tmp_path / 'absent.json')


# -- version guard and dtype mapping --------------------------------------


def test_only_tensorrt_10_is_supported():
    assert SUPPORTED_TENSORRT_MAJORS == (10, )


@pytest.mark.parametrize('version,match', [
    ('11.0.0', 'not supported'),
    ('9.3.0', 'not supported'),
    ('nightly', 'unparseable'),
    ('', 'no __version__'),
])
def test_require_tensorrt_rejects_other_builds(monkeypatch, version, match):
    FakeTensorRT(version=version).install(monkeypatch)
    with pytest.raises(RuntimeError, match=match):
        require_tensorrt()


def test_require_tensorrt_reports_missing_bindings(monkeypatch):
    monkeypatch.setitem(sys.modules, 'tensorrt', None)
    with pytest.raises(RuntimeError, match='bindings are required'):
        require_tensorrt()


def test_dtype_names_cover_the_contract():
    assert torch_dtype_for('float32') is torch.float32
    assert torch_dtype_for('int32') is torch.int32
    assert trt_dtype_name(_DType('HALF')) == 'float16'
    assert trt_dtype_name(_DType('INT32')) == 'int32'
    assert trt_dtype_name(_DType('UNKNOWN')) is None
    with pytest.raises(ValueError, match='Unsupported tensor dtype'):
        torch_dtype_for('float64')
    assert set(DTYPE_NAMES) >= {'float32', 'float16', 'int32', 'bool'}


# -- engine loading ------------------------------------------------------


def _engine_file(tmp_path):
    path = tmp_path / 'model.engine'
    path.write_bytes(b'serialized-engine-bytes')
    return path


def _loaded(monkeypatch, tmp_path, contract=None, **options):
    contract = contract or _image_contract()
    records_inputs, records_outputs = _records(contract)
    profile = {}
    for name in contract.input_names:
        bounds = contract.profile_for(name)
        if bounds is not None:
            profile[name] = (bounds.min_shape, bounds.opt_shape,
                             bounds.max_shape)
    fake = FakeTensorRT(
        inputs=records_inputs,
        outputs=records_outputs,
        profile=profile,
        **options)
    path = _engine_file(tmp_path)
    fake.install(monkeypatch)
    engine = TensorRTEngine(path, contract)
    engine.load()
    return engine, fake, contract


@REQUIRES_CUDA
def test_load_validates_the_engine_against_the_contract(monkeypatch, tmp_path):
    engine, fake, contract = _loaded(monkeypatch, tmp_path)
    assert engine.loaded and engine.device_index == torch.cuda.current_device()
    assert fake.deserialized == [b'serialized-engine-bytes']
    report = engine.describe()
    assert report['loaded'] and report['tensorrt_handles_retained']
    assert report['contract'] == contract.to_dict()
    assert report['inputs'][0]['name'] == 'pixel_values'
    assert engine._runtime is not None and engine._trt_logger is not None
    engine.close()
    assert engine._runtime is None and engine._trt_logger is None
    assert engine.describe()['tensorrt_handles_retained'] is False


@REQUIRES_CUDA
def test_load_reports_deserialize_failure_and_io_mismatch(
        monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match='failed to deserialize'):
        _loaded(monkeypatch, tmp_path, deserialize_ok=False)
    contract = _image_contract()
    inputs, outputs = _records(
        contract,
        shape_override={('tokens', 'OUTPUT'): (DYNAMIC_DIM, 5, 1280)})
    fake = FakeTensorRT(
        inputs=inputs,
        outputs=outputs,
        profile={'pixel_values': _profile_bounds()})
    path = _engine_file(tmp_path)
    fake.install(monkeypatch)
    engine = TensorRTEngine(path, contract)
    with pytest.raises(ValueError, match='static size'):
        engine.load()


def test_load_fails_before_tensorrt_is_needed(monkeypatch, tmp_path):
    engine = TensorRTEngine(tmp_path / 'absent.engine', _image_contract())
    with pytest.raises(FileNotFoundError, match='engine not found'):
        engine.load()
    monkeypatch.setitem(sys.modules, 'tensorrt', None)
    path = _engine_file(tmp_path)
    engine = TensorRTEngine(path, _image_contract())
    with pytest.raises(RuntimeError, match='bindings are required'):
        engine.load()


def test_engine_argument_validation(monkeypatch, tmp_path):
    contract = _image_contract()
    inputs, outputs = _records(contract)
    FakeTensorRT(inputs=inputs, outputs=outputs).install(monkeypatch)
    path = _engine_file(tmp_path)
    with pytest.raises(ValueError, match='EngineContract'):
        TensorRTEngine(path, contract='nope')
    if torch.cuda.is_available():
        # The device is validated after the CUDA availability check, so this
        # case only exists where a CUDA runtime is present.
        with pytest.raises(ValueError, match='must be None, an int'):
            TensorRTEngine(path, contract, device='cuda').load()


# -- execution -----------------------------------------------------------


def _cuda_images(batch=2):
    return torch.rand(batch, 3, 32, 32, device='cuda') * 2 - 1


@REQUIRES_CUDA
def test_execute_enqueues_named_tensors_on_the_current_stream(
        monkeypatch, tmp_path):
    engine, fake, contract = _loaded(monkeypatch, tmp_path)
    images = _cuda_images()
    outputs = engine.execute({'pixel_values': images})
    context = fake.contexts[0]
    stream = torch.cuda.current_stream(engine.device_index)
    assert context.enqueues == [int(stream.cuda_stream)]
    assert context.addresses['pixel_values'] == images.data_ptr()
    assert context.addresses['tokens'] == outputs['tokens'].data_ptr()
    assert tuple(outputs['tokens'].shape) == (2, 4, 1280)
    assert outputs['tokens'].dtype == torch.float32
    assert engine.run(images).shape == (2, 4, 1280)


@REQUIRES_CUDA
def test_execute_rejects_impossible_inputs(monkeypatch, tmp_path):
    engine, _, contract = _loaded(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match='CUDA tensor'):
        engine.execute({'pixel_values': torch.rand(2, 3, 32, 32)})
    with pytest.raises(ValueError, match='Missing engine input'):
        engine.execute({})
    with pytest.raises(ValueError, match='Unknown engine input'):
        engine.execute({
            'pixel_values': _cuda_images(),
            'extra': _cuda_images()
        })
    with pytest.raises(ValueError, match='the engine expects'):
        engine.execute(
            {'pixel_values': torch.rand(2, 3, 64, 64, device='cuda')})
    with pytest.raises(ValueError, match='dtype'):
        engine.execute({'pixel_values': _cuda_images().half()})
    with pytest.raises(ValueError, match='mapping'):
        engine.execute(_cuda_images())


@REQUIRES_CUDA
def test_execute_rejects_non_contiguous_and_out_of_profile(
        monkeypatch, tmp_path):
    engine, _, contract = _loaded(monkeypatch, tmp_path)
    strided = torch.rand(2, 3, 32, 64, device='cuda')[..., ::2]
    assert not strided.is_contiguous()
    with pytest.raises(ValueError, match='contiguous'):
        engine.execute({'pixel_values': strided})
    with pytest.raises(ValueError, match='the engine expects'):
        engine.execute({'pixel_values': _cuda_images(batch=9)})


@REQUIRES_CUDA
def test_scalar_and_multi_tensor_contracts_that_are_not_images(
        monkeypatch, tmp_path):
    # A contract with two inputs, two outputs, different ranks and a width
    # that has nothing to do with RADIO: the engine must not assume images.
    contract = _contract(
        {
            'inputs': [('signal', 'float32', (DYNAMIC_DIM, 7)),
                       ('mask', 'int32', (DYNAMIC_DIM, ))],
            'outputs': [('embedding', 'float32', (DYNAMIC_DIM, 7, 5)),
                        ('score', 'float32', (DYNAMIC_DIM, ))],
        }, {
            'signal': ((1, 7), (2, 7), (4, 7)),
            'mask': ((1, ), (2, ), (4, ))
        })
    engine, fake, _ = _loaded(monkeypatch, tmp_path, contract=contract)
    signal = torch.rand(3, 7, device='cuda')
    mask = torch.ones(3, dtype=torch.int32, device='cuda')
    outputs = engine.execute({'signal': signal, 'mask': mask})
    assert tuple(outputs['embedding'].shape) == (3, 7, 5)
    assert tuple(outputs['score'].shape) == (3, )
    context = fake.contexts[0]
    assert context.input_shapes == {'signal': (3, 7), 'mask': (3, )}
    assert set(context.addresses) == {'signal', 'mask', 'embedding', 'score'}
    with pytest.raises(ValueError, match='single-input, single-output'):
        engine.run(signal)


@REQUIRES_CUDA
def test_engine_keeps_one_context_per_stream(monkeypatch, tmp_path):
    engine, fake, _ = _loaded(monkeypatch, tmp_path)
    images = _cuda_images()
    stream_a = torch.cuda.Stream(device=engine.device_index)
    stream_b = torch.cuda.Stream(device=engine.device_index)
    with torch.cuda.stream(stream_a):
        engine.execute({'pixel_values': images})
    with torch.cuda.stream(stream_b):
        engine.execute({'pixel_values': images})
    with torch.cuda.stream(stream_a):
        engine.execute({'pixel_values': images})
    assert len(fake.contexts) == 2
    assert [c.enqueues[0] for c in fake.contexts
            ] == [int(stream_a.cuda_stream),
                  int(stream_b.cuda_stream)]


@REQUIRES_CUDA
def test_context_for_creates_the_context_on_the_engine_device(
        monkeypatch, tmp_path):
    engine, fake, _ = _loaded(monkeypatch, tmp_path)
    entered = []
    real_device = torch.cuda.device

    def recording_device(index=None):
        entered.append(index)
        return real_device(index)

    monkeypatch.setattr(torch.cuda, 'device', recording_device)
    engine.context_for(torch.cuda.Stream(device=engine.device_index))
    assert entered[-1] == engine.device_index
    assert fake.context_creation_devices == [engine.device_index]


@REQUIRES_CUDA
@pytest.mark.parametrize('options,match', [
    ({
        'set_input_shape_result': False
    }, 'set_input_shape returned False'),
    ({
        'address_result': False
    }, 'set_tensor_address returned False'),
    ({
        'enqueue_result': False
    }, 'reported failure'),
    ({
        'enqueue_error': 'device lost'
    }, 'execute_async_v3 failed'),
    ({
        'set_input_shape_error': 'no profile'
    }, 'rejected the input shape'),
    ({
        'context_is_none': True
    }, 'failed to create an execution context'),
    ({
        'omit_execute_async_v3': True
    }, 'no execute_async_v3'),
    ({
        'output_shape_override': [-1, 4, 1280]
    }, 'unresolved output shape'),
])
def test_execute_reports_runtime_failures(monkeypatch, tmp_path, options,
                                          match):
    engine, _, _ = _loaded(monkeypatch, tmp_path, **options)
    with pytest.raises(RuntimeError, match=match):
        engine.execute({'pixel_values': _cuda_images()})


@REQUIRES_CUDA
def test_execute_records_buffers_and_keeps_them_in_flight(
        monkeypatch, tmp_path):
    engine, _, _ = _loaded(monkeypatch, tmp_path)
    recorded = []
    monkeypatch.setattr(
        engine_module, 'record_stream', lambda tensor, stream: recorded.append(
            (tensor, int(stream.cuda_stream))))
    images = _cuda_images()
    outputs = engine.execute({'pixel_values': images})
    stream = torch.cuda.current_stream(engine.device_index)
    assert [(tensor, handle) for tensor, handle in recorded
            ] == [(images, int(stream.cuda_stream)),
                  (outputs['tokens'], int(stream.cuda_stream))]
    held_stream, held = engine._in_flight[int(stream.cuda_stream)]
    assert int(held_stream.cuda_stream) == int(stream.cuda_stream)
    assert held['pixel_values'] is images
    assert held['tokens'] is outputs['tokens']


@REQUIRES_CUDA
def test_close_synchronizes_before_it_releases(monkeypatch, tmp_path):
    engine, _, _ = _loaded(monkeypatch, tmp_path)
    engine.execute({'pixel_values': _cuda_images()})
    order = []
    monkeypatch.setattr(engine, '_synchronize_in_flight',
                        lambda: order.append('synchronize'))
    monkeypatch.setattr(engine, '_release', lambda: order.append('release'))
    engine.close()
    assert order == ['synchronize', 'release']


@REQUIRES_CUDA
def test_close_waits_for_outstanding_work_on_recorded_streams(
        monkeypatch, tmp_path):
    engine, _, _ = _loaded(monkeypatch, tmp_path)
    stream = torch.cuda.Stream(device=engine.device_index)
    calls = []
    real_wait = engine_module.wait_for_stream

    def observe_wait(target):
        calls.append(int(target.cuda_stream))
        real_wait(target)

    monkeypatch.setattr(engine_module, 'wait_for_stream', observe_wait)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(5 * 10**7)
        engine.execute({'pixel_values': _cuda_images()})
    engine.close()
    # The engine waited on the recorded stream, and that work is complete.
    assert calls == [int(stream.cuda_stream)]
    assert stream.query()


@REQUIRES_CUDA
def test_close_waits_even_after_an_enqueue_failure(monkeypatch, tmp_path):
    engine, _, _ = _loaded(monkeypatch, tmp_path, enqueue_result=False)
    stream = torch.cuda.Stream(device=engine.device_index)
    calls = []
    real_wait = engine_module.wait_for_stream

    def observe_wait(target):
        calls.append(int(target.cuda_stream))
        real_wait(target)

    monkeypatch.setattr(engine_module, 'wait_for_stream', observe_wait)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(5 * 10**7)
        with pytest.raises(RuntimeError, match='reported failure'):
            engine.execute({'pixel_values': _cuda_images()})
    engine.close()
    assert calls == [int(stream.cuda_stream)]
    assert stream.query()


@REQUIRES_CUDA
def test_close_blocks_further_use(monkeypatch, tmp_path):
    engine, _, _ = _loaded(monkeypatch, tmp_path)
    engine.close()
    assert not engine.loaded
    with pytest.raises(RuntimeError, match='has been closed'):
        engine.load()
    with pytest.raises(RuntimeError, match='has been closed'):
        engine.execute({'pixel_values': _cuda_images()})


@REQUIRES_CUDA
def test_context_manager_loads_and_releases(monkeypatch, tmp_path):
    contract = _image_contract()
    inputs, outputs = _records(contract)
    fake = FakeTensorRT(
        inputs=inputs,
        outputs=outputs,
        profile={'pixel_values': _profile_bounds()})
    path = _engine_file(tmp_path)
    fake.install(monkeypatch)
    with TensorRTEngine(path, contract) as engine:
        assert engine.loaded
    assert not engine.loaded


@REQUIRES_CUDA
def test_concurrent_same_stream_calls_cannot_interleave(monkeypatch, tmp_path):
    engine, fake, _ = _loaded(monkeypatch, tmp_path)
    images = _cuda_images()
    stream = torch.cuda.Stream(device=engine.device_index)
    outputs = []
    errors = []

    def call():
        try:
            with torch.cuda.stream(stream):
                outputs.append(
                    engine.execute({'pixel_values': images})['tokens'])
        except Exception as error:  # noqa: BLE001 - surfaced below
            errors.append(error)

    threads = [threading.Thread(target=call) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(outputs) == 8
    assert len(fake.contexts) == 1
    assert fake.contexts[0].enqueues == [int(stream.cuda_stream)] * 8
    assert fake.contexts[0].input_shapes == {'pixel_values': (2, 3, 32, 32)}
    assert len({output.data_ptr() for output in outputs}) == 8


# -- contract discovery, profiles, IO layout and version checks ------------


def _bare_engine(monkeypatch, tmp_path, contract=None, **options):
    """A fake engine plus path, without a contract passed to the executor."""
    contract = contract or _image_contract()
    inputs, outputs = _records(contract)
    profile = {}
    for name in contract.input_names:
        bounds = contract.profile_for(name)
        if bounds is not None:
            profile[name] = (bounds.min_shape, bounds.opt_shape,
                             bounds.max_shape)
    fake = FakeTensorRT(
        inputs=inputs, outputs=outputs, profile=profile, **options)
    path = _engine_file(tmp_path)
    fake.install(monkeypatch)
    return path, fake, contract


@REQUIRES_CUDA
def test_executor_derives_a_contract_with_profiles_from_a_real_engine(
        monkeypatch, tmp_path):
    path, fake, contract = _bare_engine(monkeypatch, tmp_path)
    engine = TensorRTEngine(path, device=0)
    assert engine.contract is None
    output = engine.run(_cuda_images())
    assert tuple(output.shape) == (2, 4, 1280)
    assert engine.contract is not None
    assert engine.contract.profile_for('pixel_values').max_shape == (4, 3, 32,
                                                                     32)
    assert engine.contract.to_dict() == contract.to_dict()
    engine.close()


def test_contract_rejects_duplicate_profiles():
    bounds = OptimizationProfile('x', (1, ), (2, ), (4, ))
    with pytest.raises(ValueError, match='more than one optimization profile'):
        EngineContract(
            inputs=(TensorSpec('x', 'float32', (DYNAMIC_DIM, )), ),
            outputs=(TensorSpec('y', 'float32', (DYNAMIC_DIM, )), ),
            profiles=(bounds, bounds))


@REQUIRES_CUDA
def test_load_rejects_engine_profiles_that_differ_from_the_contract(
        monkeypatch, tmp_path):
    contract = _image_contract()
    path, fake, _ = _bare_engine(monkeypatch, tmp_path, contract)
    # Same IO, different actual bounds: the declared profile cannot drive it.
    fake.profile['pixel_values'] = ((1, 3, 32, 32), (3, 3, 32, 32), (6, 3, 32,
                                                                     32))
    engine = TensorRTEngine(path, contract)
    with pytest.raises(ValueError, match='does not match the engine'):
        engine.load()


@REQUIRES_CUDA
def test_load_accepts_reordered_named_io(monkeypatch, tmp_path):
    contract = _image_contract()
    path, fake, _ = _bare_engine(monkeypatch, tmp_path, contract)
    fake.records.reverse()
    engine = TensorRTEngine(path, contract)
    engine.load()
    assert [spec.name for spec in engine._inputs] == ['pixel_values']
    engine.close()


@REQUIRES_CUDA
def test_load_requires_a_profile_for_a_dynamic_input(monkeypatch, tmp_path):
    contract = _image_contract()
    path, fake, _ = _bare_engine(
        monkeypatch, tmp_path, contract, omit_profile=True)
    engine = TensorRTEngine(path, contract)
    with pytest.raises(ValueError, match='no optimization profile'):
        engine.load()


@REQUIRES_CUDA
def test_execute_rejects_a_resolved_output_that_contradicts_the_contract(
        monkeypatch, tmp_path):
    engine, _, _ = _loaded(monkeypatch, tmp_path, output_shape_override=[2, 4])
    with pytest.raises(RuntimeError, match='declares rank'):
        engine.execute({'pixel_values': _cuda_images()})
    engine, _, _ = _loaded(
        monkeypatch, tmp_path, output_shape_override=[2, 5, 1280])
    with pytest.raises(RuntimeError, match='declares the static size'):
        engine.execute({'pixel_values': _cuda_images()})


@pytest.mark.parametrize('options,match', [
    ({
        'formats': {
            'tokens': 'HWC8'
        }
    }, 'TensorRT format'),
    ({
        'locations': {
            'tokens': 'HOST'
        }
    }, 'device tensors only'),
    ({
        'shape_inference': ('tokens', )
    }, 'shape-inference tensor'),
])
def test_load_rejects_io_layouts_this_executor_cannot_bind(
        monkeypatch, tmp_path, options, match):
    contract = _image_contract()
    inputs, outputs = _records(contract)
    fake = FakeTensorRT(
        inputs=inputs,
        outputs=outputs,
        profile={'pixel_values': _profile_bounds()},
        **options)
    module = fake.install(monkeypatch)
    with pytest.raises(ValueError, match=match):
        # The check is pure: it reads the engine's reported IO metadata, so it
        # runs without a CUDA device.
        engine_module.tensor_specs(module, fake.engine(), 'OUTPUT')


def test_version_check_rejects_a_major_mismatch_and_warns_on_a_build(caplog):
    engine = TensorRTEngine(
        '/tmp/absent.engine',
        _image_contract(),
        expected_tensorrt_version='11.0.0')
    with pytest.raises(RuntimeError, match='not portable across major'):
        engine._check_expected_version('10.7.0')
    engine = TensorRTEngine(
        '/tmp/absent.engine',
        _image_contract(),
        expected_tensorrt_version='10.6.0')
    with caplog.at_level('WARNING'):
        engine._check_expected_version('10.7.0')
    assert 'not portable across builds' in caplog.text
    engine._check_expected_version('10.6.0')
