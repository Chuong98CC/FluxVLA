# Copyright 2026 Limx Dynamics
"""C-RADIO TensorRT CLI: parser, inspect, and clear failure reporting.

The CLI is loaded by path like the other tool tests, so nothing here needs a
model, ONNX, TensorRT or a GPU. The offline steps themselves are covered in
``test/test_scripts/test_trt_export.py``.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch

from fluxvla.models.backbones.visions.frozen_vision import \
    SpatialFeatureOutput
from fluxvla.models.backbones.visions.base_trt.radio_metadata import \
    PRECISIONS, EngineMetadata
from fluxvla.models.backbones.visions.radio_vit import RadioViTBackbone

REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason='Requires CUDA.')

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / 'tools' / 'c_radio_tensorrt.py'


@pytest.fixture(scope='module')
def cli():
    spec = importlib.util.spec_from_file_location('c_radio_tensorrt',
                                                  MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def sidecar(tmp_path):
    metadata = EngineMetadata(
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
    return metadata.save(tmp_path / 'radio.engine.json')


def test_parser_exposes_every_pipeline_step(cli):
    parser = cli.build_parser()
    actions = [
        action for action in parser._actions if action.dest == 'command'
    ]
    assert actions, 'the CLI must require a subcommand'
    names = set(actions[0].choices)
    assert names == {
        'export-onnx', 'sample', 'ort-parity', 'build-engine', 'inspect',
        'validate', 'parity', 'benchmark'
    }


def test_export_command_defaults_are_the_documented_contract(cli):
    parser = cli.build_parser()
    args = parser.parse_args([
        'export-onnx', '--radio-path', '/radio', '--image-height', '256',
        '--image-width', '272', '--out', '/tmp/radio.onnx'
    ])
    assert args.version == 'c-radio_v4-h'
    assert (args.min_batch, args.opt_batch, args.max_batch) == (1, 1, 1)
    assert args.input_name == 'pixel_values'
    assert args.output_name == 'tokens'
    assert args.no_external_data is False
    assert args.tensorrt_version == '10'
    assert args.device == 'cpu'


def test_build_command_metadata_is_optional(cli):
    parser = cli.build_parser()
    args = parser.parse_args(
        ['build-engine', '--onnx', 'a.onnx', '--out', 'b.engine'])
    assert args.metadata is None and args.precision == 'fp32'
    assert args.workspace_mib == 1024 and args.no_verify is False


def test_build_command_accepts_every_builder_precision(cli):
    parser = cli.build_parser()
    assert PRECISIONS == ('fp32', 'fp16', 'bf16')
    for precision in PRECISIONS:
        args = parser.parse_args([
            'build-engine', '--onnx', 'a.onnx', '--out', 'b.engine',
            '--precision', precision
        ])
        assert args.precision == precision
    # An uncalibrated precision stays a hard parse error.
    with pytest.raises(SystemExit):
        parser.parse_args([
            'build-engine', '--onnx', 'a.onnx', '--out', 'b.engine',
            '--precision', 'int8'
        ])


def test_build_command_parses_repeatable_layer_precisions(cli):
    parser = cli.build_parser()
    args = parser.parse_args([
        'build-engine', '--onnx', 'a.onnx', '--out', 'b.engine', '--precision',
        'fp16', '--layer-precision', '*/LayerNormalization:fp32',
        '--layer-precision', '*/Softmax:fp32'
    ])
    assert [entry.spec for entry in args.layer_precision
            ] == ['*/LayerNormalization:fp32', '*/Softmax:fp32']
    assert [entry.dtype
            for entry in args.layer_precision] == ['float32', 'float32']
    # Every new knob defaults to leaving the build exactly as it was.
    assert args.precision_constraints is None
    assert args.profiling_verbosity is None and args.log_file is None
    plain = parser.parse_args(
        ['build-engine', '--onnx', 'a.onnx', '--out', 'b.engine'])
    assert plain.layer_precision is None


@pytest.mark.parametrize('flag,value', [
    ('--layer-precision', 'LayerNormalization'),
    ('--layer-precision', '*/LayerNormalization:int8'),
    ('--precision-constraints', 'maybe'),
    ('--profiling-verbosity', 'verbose'),
])
def test_build_command_rejects_a_bad_precision_request(cli, flag, value):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            'build-engine', '--onnx', 'a.onnx', '--out', 'b.engine', flag,
            value
        ])


def test_build_command_defaults_the_constraints_mode_to_obey(
        cli, monkeypatch, tmp_path):
    """An override means obey; no override means no constraints flag."""
    from tools import c_radio_export
    calls = []

    def fake_build(onnx_path, engine_path, metadata, **kwargs):
        calls.append(kwargs)
        return metadata, {
            'layer_precisions': [],
            'network_layers': 1,
            'precision': 'fp16',
        }

    monkeypatch.setattr(c_radio_export, 'build_engine', fake_build)
    monkeypatch.setattr(
        cli, '_load_metadata', lambda args, fallback=None: _metadata())
    engine = str(tmp_path / 'b.engine')
    assert cli.main(['build-engine', '--onnx', 'a.onnx', '--out', engine]) == 0
    assert calls[-1]['precision_constraints'] == 'none'
    assert cli.main([
        'build-engine', '--onnx', 'a.onnx', '--out', engine,
        '--layer-precision', '*/LayerNormalization:fp32'
    ]) == 0
    assert calls[-1]['precision_constraints'] == 'obey'
    assert [entry.spec for entry in calls[-1]['layer_precisions']
            ] == ['*/LayerNormalization:fp32']
    assert cli.main([
        'build-engine', '--onnx', 'a.onnx', '--out', engine,
        '--layer-precision', '*/LayerNormalization:fp32',
        '--precision-constraints', 'prefer'
    ]) == 0
    assert calls[-1]['precision_constraints'] == 'prefer'


def test_build_command_can_keep_the_builder_diagnostics(
        cli, monkeypatch, tmp_path, capsys):
    from tools import c_radio_export
    seen = {}

    def fake_build(onnx_path, engine_path, metadata, **kwargs):
        # The builder writes to the process streams; the CLI must have sent
        # them to the log file it was given.
        seen['stdout_fd'] = os.fstat(1).st_ino
        return metadata, {'layer_precisions': [], 'network_layers': 1}

    monkeypatch.setattr(c_radio_export, 'build_engine', fake_build)
    monkeypatch.setattr(
        cli, '_load_metadata', lambda args, fallback=None: _metadata())
    log = tmp_path / 'build.log'
    outside = os.fstat(1).st_ino
    code = cli.main([
        'build-engine', '--onnx', 'a.onnx', '--out',
        str(tmp_path / 'b.engine'), '--log-file',
        str(log)
    ])
    assert code == 0
    assert log.is_file()
    assert seen['stdout_fd'] == log.stat().st_ino
    assert os.fstat(1).st_ino == outside
    assert json.loads(capsys.readouterr().out)['log_path'] == str(log)


def test_inspect_prints_the_sidecar_as_json(cli, sidecar, capsys):
    assert cli.main(['inspect', '--metadata', str(sidecar)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['metadata']['variant'] == 'c-radio_v4-h'
    assert payload['input_shape'] == [2, 3, 32, 32]
    assert payload['output_shape'] == [2, 4, 1280]


def test_inspect_reports_a_missing_sidecar(cli, capsys):
    assert cli.main(['inspect', '--metadata',
                     '/nonexistent/sidecar.json']) == 1
    captured = capsys.readouterr()
    assert 'error:' in captured.err and 'sidecar not found' in captured.err


def test_export_reports_a_missing_local_radio_export(cli, tmp_path, capsys):
    code = cli.main([
        'export-onnx', '--radio-path', '/nonexistent/C-RADIOv4-H',
        '--image-height', '256', '--image-width', '256', '--out',
        str(tmp_path / 'radio.onnx')
    ])
    assert code == 1
    captured = capsys.readouterr()
    assert 'error:' in captured.err
    assert 'nonexistent' in captured.err
    assert not (tmp_path / 'radio.onnx').exists()


def test_validate_reports_a_missing_engine(cli, capsys):
    code = cli.main(['validate', '--engine', '/nonexistent/radio.engine'])
    assert code == 1
    assert 'error:' in capsys.readouterr().err


def test_help_exits_cleanly(cli):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(['--help'])
    assert exit_info.value.code == 0


# -- parity reference and exit code ---------------------------------------


def test_parity_reference_is_built_in_fp32(cli, monkeypatch):
    recorded = {}

    def fake_from_local_export(model_path, **kwargs):
        recorded['model_path'] = model_path
        recorded.update(kwargs)
        return object()

    monkeypatch.setattr(RadioViTBackbone, 'from_local_export',
                        staticmethod(fake_from_local_export))
    cli._build_radio_backbone('/radio', 'c-radio_v4-h')
    assert recorded['model_path'] == '/radio'
    assert recorded['version'] == 'c-radio_v4-h'
    assert recorded['feature_dim'] == 1280
    # The engine boundary is FP32, so the reference must not default to BF16.
    assert recorded['dtype'] is torch.float32


class _FakeBackbone:
    """Minimal encoder stand-in for the CLI parity/benchmark paths."""

    def __init__(self, fill, metadata, device=None):
        self.fill = fill
        self.metadata = metadata
        self.device = device
        self.closed = False

    def to(self, device=None, **kwargs):
        self.device = device
        return self

    def encode_features(self, images):
        grid = self.metadata.grid
        tokens = torch.full(
            (int(images.shape[0]), grid[0] * grid[1],
             int(self.metadata.feature_dim)),
            float(self.fill),
            device=self.device if self.device is not None else images.device)
        return SpatialFeatureOutput(tokens=tokens, grid=grid)

    def close(self):
        self.closed = True


def _metadata():
    return EngineMetadata(
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
        opt_batch=1,
        max_batch=2)


@REQUIRES_CUDA
@pytest.mark.parametrize('engine_fill,reference_fill,expected', [
    (0.5, 0.5, 0),
    (0.5, 0.9, 1),
])
def test_parity_exit_code_reflects_the_numerical_result(
        cli, monkeypatch, engine_fill, reference_fill, expected, capsys):
    metadata = _metadata()
    engine_backbone = _FakeBackbone(engine_fill, metadata)
    monkeypatch.setattr(cli, '_build_engine_backbone',
                        lambda args: engine_backbone)
    monkeypatch.setattr(
        cli,
        '_build_radio_backbone',
        lambda radio_path, version, dtype=None: _FakeBackbone(
            reference_fill, metadata))
    code = cli.main([
        'parity', '--engine', '/tmp/fake.engine', '--radio-path', '/radio',
        '--device', '0'
    ])
    report = json.loads(capsys.readouterr().out)
    assert report['batch1']['allclose'] is (expected == 0)
    assert report['passed'] is (expected == 0)
    # A numerical mismatch must not look like a successful run.
    assert code == expected
    assert engine_backbone.closed


@REQUIRES_CUDA
def test_benchmark_passes_the_selected_device(cli, monkeypatch, capsys):
    metadata = _metadata()
    engine_backbone = _FakeBackbone(0.0, metadata)
    monkeypatch.setattr(cli, '_build_engine_backbone',
                        lambda args: engine_backbone)
    recorded = {}

    def fake_benchmark(backbone, images, **kwargs):
        recorded['images_device'] = images.device
        recorded.update(kwargs)
        return {'device': str(kwargs.get('device')), 'median_ms': 0.0}

    from tools import c_radio_export
    monkeypatch.setattr(c_radio_export, 'benchmark_backbone', fake_benchmark)
    code = cli.main([
        'benchmark', '--engine', '/tmp/fake.engine', '--device', '0',
        '--iterations', '3'
    ])
    assert code == 0
    assert recorded['device'] == torch.device('cuda', 0)
    assert recorded['iterations'] == 3
    assert recorded['images_device'] == torch.device('cuda', 0)
    assert json.loads(capsys.readouterr().out)['device'] == 'cuda:0'
