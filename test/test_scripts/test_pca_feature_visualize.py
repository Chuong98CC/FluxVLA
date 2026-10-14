# Copyright 2026 Limx Dynamics
"""PCA feature visualization CLI: parsing, geometry, PCA and artifacts.

The module is loaded by path like the other tool tests. Backends that need
real weights are exercised through small local substitutes; the real C-RADIO
runs live in the tool's own verification, not in this suite.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from fluxvla.models.backbones.visions.radio_vit import RadioViTBackbone

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / 'tools' / 'pca_feature_visualize.py'


@pytest.fixture(scope='module')
def pca():
    spec = importlib.util.spec_from_file_location('pca_feature_visualize',
                                                  MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StubRadio(nn.Module):
    """Minimal C-RADIO stand-in with the official call contract."""

    def __init__(self, feature_dim=8, patch_size=16):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.resolution_step = patch_size
        self.stem = nn.Conv2d(3, feature_dim, patch_size, patch_size)
        self.seen = []

    def get_nearest_supported_resolution(self, height, width):
        step = self.resolution_step
        return (-(-height // step) * step, -(-width // step) * step)

    def forward(self, images, feature_fmt='NLC'):
        self.seen.append(images.detach().clone())
        spatial = self.stem(images)
        summary = spatial.mean(dim=(2, 3))[:, None]
        if feature_fmt == 'NCHW':
            return summary, spatial
        return summary, spatial.flatten(2).transpose(1, 2)


class ExactRadio(StubRadio):
    """Resolution helper that never rounds, so odd sizes stay unsupported."""

    def get_nearest_supported_resolution(self, height, width):
        return (height, width)


def _backbone(feature_dim=8, patch_size=16, resolution=(32, 32)):
    return RadioViTBackbone(
        'c-radio_v4-h',
        model=StubRadio(feature_dim, patch_size),
        feature_dim=feature_dim,
        patch_size=patch_size,
        resolution_step=patch_size,
        dtype=torch.float32,
        default_image_resolution=(3, *resolution))


def _image(path, size=(64, 48)):
    from PIL import Image
    array = (np.random.default_rng(0).random(
        (size[1], size[0], 3)) * 255).astype('uint8')
    Image.fromarray(array).save(path)
    return str(path)


class StubDinoConfig:

    def __init__(self, patch_size=14, num_register_tokens=4):
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens


class StubProcessor:
    """Records the switches it is called with and normalizes like CLIP."""

    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]

    def __init__(self):
        self.calls = []

    def __call__(self, images=None, return_tensors=None, **kwargs):
        self.calls.append({'size': images.size, **kwargs})
        array = np.asarray(images, dtype='float32') / 255.0
        mean = np.asarray(self.image_mean, dtype='float32')
        std = np.asarray(self.image_std, dtype='float32')
        normalized = (array - mean) / std
        values = torch.from_numpy(normalized).permute(2, 0, 1)[None]
        return {'pixel_values': values}


class SelfResizingProcessor(StubProcessor):
    """A processor that resizes anyway, ignoring the caller's switches."""

    def __call__(self, images=None, return_tensors=None, **kwargs):
        self.calls.append({'size': images.size, **kwargs})
        resized = images.resize((224, 224))
        array = np.asarray(resized, dtype='float32') / 255.0
        values = torch.from_numpy((array - 0.5) / 0.5).permute(2, 0, 1)[None]
        return {'pixel_values': values}


class StubDinoModel(nn.Module):

    def __init__(self, config=None, hidden_tokens=None, width=6):
        super().__init__()
        self.config = config or StubDinoConfig()
        self.hidden_tokens = hidden_tokens
        self.width = width
        self.seen = []

    def forward(self, pixel_values=None):
        self.seen.append(pixel_values)
        batch, _, height, width = pixel_values.shape
        patch = self.config.patch_size
        tokens = (
            self.hidden_tokens if self.hidden_tokens is not None else
            (height // patch) * (width // patch))
        prefix = 1 + self.config.num_register_tokens
        hidden = torch.arange(
            (prefix + tokens) * self.width,
            dtype=torch.float32).reshape(1, prefix + tokens, self.width)
        hidden = hidden.repeat(batch, 1, 1)
        return type('Output', (), {'last_hidden_state': hidden})()


# -- import safety and parsing --------------------------------------------


def test_module_import_is_model_free_and_touches_no_environment(tmp_path):
    import subprocess
    script = (
        'import importlib.util, os, sys\n'
        'spec = importlib.util.spec_from_file_location("pca", r"' +
        f'{MODULE_PATH}")\n'
        'module = importlib.util.module_from_spec(spec)\n'
        'before = dict(os.environ)\n'
        'spec.loader.exec_module(module)\n'
        'assert "torch" not in sys.modules, "torch imported at import time"\n'
        'assert "matplotlib" not in sys.modules\n'
        'assert "sklearn" not in sys.modules\n'
        'for key in ("HF_ENDPOINT", "http_proxy", "https_proxy",'
        ' "HTTP_PROXY"):\n'
        '    assert os.environ.get(key) == before.get(key), key\n'
        'parser = module.build_parser()\n'
        'assert parser.parse_args(["--image", "x.jpg"]).backbone == "cradio"\n'
        'print("ok")\n')
    result = subprocess.run([sys.executable, '-c', script],
                            capture_output=True,
                            text=True,
                            cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr
    assert 'ok' in result.stdout


@pytest.mark.parametrize('value,expected', [
    ('cradio', 'cradio'),
    ('C-RADIO', 'cradio'),
    ('radio-trt', 'cradio-trt'),
    ('trt', 'cradio-trt'),
    ('dinov3', 'dino'),
])
def test_backbone_aliases(pca, value, expected):
    assert pca.normalize_backbone(value) == expected


def test_unknown_backbone_is_rejected(pca):
    with pytest.raises(pca.PcaVisualizeError, match='Unknown backbone'):
        pca.normalize_backbone('siglip')


@pytest.mark.parametrize('values,expected', [
    (None, (256, 256)),
    (['224'], (224, 224)),
    (['256', '192'], (256, 192)),
    ([224], (224, 224)),
])
def test_image_size_parsing(pca, values, expected):
    assert pca.parse_image_size(values) == expected


@pytest.mark.parametrize('values', [['0'], ['-4'], ['abc'], ['1', '2', '3']])
def test_invalid_image_sizes_are_rejected(pca, values):
    with pytest.raises(pca.PcaVisualizeError):
        pca.parse_image_size(values)


# -- PCA -------------------------------------------------------------------


def test_pca_is_deterministic_and_shaped(pca):
    rng = np.random.default_rng(0)
    tokens = rng.random((16, 12)).astype('float32')
    first = pca.fit_spatial_pca(tokens, grid=(4, 4))
    second = pca.fit_spatial_pca(tokens, grid=(4, 4))
    assert first['image'].shape == (4, 4, 3)
    assert np.isfinite(first['image']).all()
    assert first['image'].min() >= 0.0 and first['image'].max() <= 1.0
    assert np.allclose(first['components'], second['components'])
    assert np.allclose(first['image'], second['image'])
    assert first['refit'] is True
    assert len(first['variance_ratio']) == 3


def test_pca_handles_constant_channels_and_constant_tokens(pca):
    rng = np.random.default_rng(1)
    tokens = rng.random((9, 6)).astype('float32')
    tokens[:, 0] = 3.0
    report = pca.fit_spatial_pca(tokens, grid=(3, 3))
    assert np.isfinite(report['image']).all()
    constant = np.full((9, 6), 2.5, dtype='float32')
    report = pca.fit_spatial_pca(constant, grid=(3, 3))
    assert np.isfinite(report['image']).all()
    assert np.allclose(report['image'], 0.0)


def test_pca_rejects_bad_inputs(pca):
    with pytest.raises(pca.PcaVisualizeError, match='non-finite'):
        pca.fit_spatial_pca(np.full((9, 6), np.nan))
    with pytest.raises(pca.PcaVisualizeError, match=r'\[N, D\]'):
        pca.fit_spatial_pca(np.zeros(4))
    with pytest.raises(pca.PcaVisualizeError, match='at least 3'):
        pca.fit_spatial_pca(np.zeros((2, 8)))
    with pytest.raises(pca.PcaVisualizeError, match='at least 3'):
        pca.fit_spatial_pca(np.zeros((16, 2)))
    with pytest.raises(pca.PcaVisualizeError, match='for a 3x3 grid'):
        pca.fit_spatial_pca(np.zeros((8, 5)), grid=(3, 3))


def test_pca_reference_reuses_the_same_basis_and_scaling(pca):
    rng = np.random.default_rng(2)
    tokens = rng.random((16, 10)).astype('float32')
    basis = pca.fit_spatial_pca(tokens, grid=(4, 4))
    other = (tokens + 0.01).astype('float32')
    reference = {
        'mean': basis['mean'],
        'components': basis['components'],
        'scale_min': basis['scale_min'],
        'scale_max': basis['scale_max'],
        'variance': basis['variance'],
        'variance_ratio': basis['variance_ratio'],
    }
    projected = pca.fit_spatial_pca(other, grid=(4, 4), reference=reference)
    assert projected['refit'] is False
    assert np.allclose(projected['components'], basis['components'])
    assert np.allclose(projected['scale_min'], basis['scale_min'])
    with pytest.raises(pca.PcaVisualizeError, match='expects width'):
        pca.fit_spatial_pca(
            rng.random((16, 7)).astype('float32'),
            grid=(4, 4),
            reference=reference)


# -- geometry --------------------------------------------------------------


def test_radio_geometry_is_checked(pca):
    backbone = _backbone()
    assert pca.check_geometry(backbone, 32, 32, 'C-RADIO') == ((2, 2), 16)
    unaligned = RadioViTBackbone(
        'c-radio_v4-h',
        model=ExactRadio(8, 16),
        feature_dim=8,
        patch_size=16,
        resolution_step=16,
        dtype=torch.float32,
        default_image_resolution=(3, 32, 32))
    with pytest.raises(pca.PcaVisualizeError, match='C-RADIO cannot encode'):
        pca.check_geometry(unaligned, 33, 33, 'C-RADIO')
    # The shared stub rounds up to 48x48; that must fail instead of being
    # encoded and plotted as if it were 33x33.
    rounded = _backbone()
    with pytest.raises(pca.PcaVisualizeError, match='encodes 33x33 at 48x48'):
        pca.check_geometry(rounded, 33, 33, 'C-RADIO')
    assert pca.check_geometry(rounded, 48, 48, 'C-RADIO') == ((3, 3), 16)


def test_trt_geometry_mismatch_names_the_compiled_size(pca):
    from fluxvla.models.backbones.visions.base_trt.radio_metadata import (
        EngineMetadata)
    from fluxvla.models.backbones.visions.radio_trt import (
        RadioViTBackboneTensorRT)
    metadata = EngineMetadata(
        variant='c-radio_v4-h',
        feature_dim=1280,
        patch_size=16,
        resolution_step=16,
        input_height=32,
        input_width=32,
        grid_height=2,
        grid_width=2,
        tensorrt_version='10.7.0')

    class FakeEngine:
        contract = metadata.contract()

        def run(self, images):
            return torch.zeros(1, 4, 1280, dtype=images.dtype)

        def close(self):
            pass

    backbone = RadioViTBackboneTensorRT(engine=FakeEngine(), metadata=metadata)
    assert pca.check_geometry(backbone, 32, 32,
                              'C-RADIO TensorRT') == ((2, 2), 16)
    with pytest.raises(pca.PcaVisualizeError, match='fixed 32x32 input'):
        pca.check_geometry(backbone, 64, 64, 'C-RADIO TensorRT')


# -- C-RADIO encoding ------------------------------------------------------


def test_encode_cradio_consumes_unit_range_and_returns_tokens(pca):
    backbone = _backbone()
    wrapper = torch.full((1, 3, 32, 32), -1.0)
    wrapper[:, 0] = 1.0
    tokens, grid, diagnostics = pca.encode_cradio(
        backbone, wrapper, device='cpu', dtype='fp32')
    seen = backbone.model.seen[-1]
    assert torch.allclose(seen[:, 0], torch.ones_like(seen[:, 0]))
    assert not seen[:, 1].any() and not seen[:, 2].any()
    assert tokens.shape == (4, 8) and grid == (2, 2)
    assert diagnostics['precision']['tf32'] is False
    assert diagnostics['precision']['autocast'] is None
    assert diagnostics['tokens'] == 4 and diagnostics['feature_dim'] == 8
    assert diagnostics['finite'] is True


def test_encode_cradio_lower_precision_uses_autocast(pca):
    backbone = _backbone()
    wrapper = torch.rand(1, 3, 32, 32) * 2 - 1
    tokens, grid, diagnostics = pca.encode_cradio(
        backbone, wrapper, device='cpu', dtype='bf16')
    assert tokens.shape == (4, 8) and grid == (2, 2)
    assert diagnostics['precision']['weights'] == 'float32'
    assert 'autocast' in diagnostics['precision']


# -- DINO ------------------------------------------------------------------


def test_dino_geometry_uses_config_patch_and_registers(pca):
    geometry = pca.dino_geometry(StubDinoConfig(), 28, 56)
    assert geometry['patch_size'] == 14
    assert geometry['num_register_tokens'] == 4
    assert geometry['prefix_tokens'] == 5
    assert geometry['grid'] == (2, 4)
    assert geometry['expected_tokens'] == 8
    with pytest.raises(pca.PcaVisualizeError, match='multiples of 14'):
        pca.dino_geometry(StubDinoConfig(), 30, 28)
    with pytest.raises(pca.PcaVisualizeError, match='no usable patch_size'):
        pca.dino_geometry(StubDinoConfig(patch_size=0), 28, 28)


def test_dino_tokens_strip_the_prefix_and_validate_the_count(pca):
    geometry = pca.dino_geometry(StubDinoConfig(), 28, 28)
    hidden = torch.zeros(1, geometry['prefix_tokens'] + 4, 6)
    hidden[0, geometry['prefix_tokens']:] = 1.0
    tokens = pca.dino_tokens(hidden, geometry)
    assert tokens.shape == (4, 6)
    assert np.allclose(tokens, 1.0)
    with pytest.raises(pca.PcaVisualizeError, match='spatial tokens'):
        pca.dino_tokens(torch.zeros(1, 5 + 9, 6), geometry)


def test_dino_pixel_values_disable_implicit_resize(pca, tmp_path):
    from PIL import Image
    processor = StubProcessor()
    image = Image.open(_image(tmp_path / 'dino.png', (64, 48)))
    values = pca.dino_pixel_values(processor, image, 28, 28)
    assert tuple(values.shape) == (1, 3, 28, 28)
    assert processor.calls[0]['do_resize'] is False
    assert processor.calls[0]['do_center_crop'] is False
    # The stub normalizes by its own mean/std, which the tool must not undo.
    resized = image.convert('RGB').resize((28, 28), Image.BICUBIC)
    expected = ((np.asarray(resized, dtype='float32') / 255.0) - 0.5) / 0.5
    assert np.allclose(values[0].permute(1, 2, 0).numpy(), expected, atol=1e-5)
    with pytest.raises(pca.PcaVisualizeError, match='changed the requested'):
        pca.dino_pixel_values(SelfResizingProcessor(), image, 28, 28)


def test_encode_dino_extracts_tokens_and_geometry(pca, tmp_path):
    from PIL import Image
    model = StubDinoModel(StubDinoConfig())
    image = Image.open(_image(tmp_path / 'dino2.png', (56, 28)))
    tokens, grid, diagnostics = pca.encode_dino(
        model,
        StubProcessor(),
        image,
        height=28,
        width=56,
        device='cpu',
        dtype='fp32')
    assert tokens.shape == (8, 6) and grid == (2, 4)
    assert diagnostics['patch_size'] == 14
    assert diagnostics['prefix_tokens'] == 5
    assert diagnostics['finite'] is True


def test_dino_model_without_hidden_state_is_rejected(pca, tmp_path):
    from PIL import Image

    class NoHidden(nn.Module):
        config = StubDinoConfig()

        def forward(self, pixel_values=None):
            return type('Output', (), {})()

    image = Image.open(_image(tmp_path / 'dino3.png', (28, 28)))
    with pytest.raises(pca.PcaVisualizeError, match='no last_hidden_state'):
        pca.encode_dino(
            NoHidden(),
            StubProcessor(),
            image,
            height=28,
            width=28,
            device='cpu',
            dtype='fp32')


def test_dino_loader_requires_local_path_or_opt_in(pca, tmp_path):
    parser = pca.build_parser()
    args = parser.parse_args([
        '--image', 'x.jpg', '--backbone', 'dino', '--model-name',
        'facebook/dinov2-base'
    ])
    with pytest.raises(pca.PcaVisualizeError, match='--allow-remote'):
        pca._load_dino(args, 'cpu')
    args = parser.parse_args([
        '--image', 'x.jpg', '--backbone', 'dino', '--model-path',
        str(tmp_path / 'absent')
    ])
    with pytest.raises(pca.PcaVisualizeError, match='does not exist'):
        pca._load_dino(args, 'cpu')


def test_dino_without_transformers_reports_the_dependency(
        pca, monkeypatch, tmp_path):
    parser = pca.build_parser()
    args = parser.parse_args([
        '--image', 'x.jpg', '--backbone', 'dino', '--model-path',
        str(tmp_path)
    ])
    monkeypatch.setitem(sys.modules, 'transformers', None)
    with pytest.raises(
            pca.PcaVisualizeError, match='transformers is required'):
        pca._load_dino(args, 'cpu')


# -- end to end with a local substitute ------------------------------------


def _cli_args(pca, tmp_path, *extra):
    image = _image(tmp_path / 'frame.png', (64, 48))
    return pca.build_parser().parse_args([
        '--image', image, '--backbone', 'cradio', '--image-size', '32',
        '--device', 'cpu', '--output',
        str(tmp_path / 'run' / 'pca'), *extra
    ])


def test_run_saves_figure_bundle_and_diagnostics(pca, monkeypatch, tmp_path):
    backbone = _backbone()
    monkeypatch.setattr(pca, '_load_cradio_pytorch',
                        lambda args, device: backbone)
    args = _cli_args(pca, tmp_path)
    report = pca.run(args)
    output = Path(args.output)
    for suffix in ('.png', '.pt', '.json'):
        path = Path(f'{output}{suffix}')
        assert path.is_file() and path.stat().st_size > 0, suffix
    saved = json.loads(Path(f'{output}.json').read_text())
    assert saved['backbone'] == 'cradio'
    assert saved['grid'] == [2, 2] and saved['tokens'] == 4
    assert saved['finite'] is True and saved['pca_refit'] is True
    assert saved['image']['resize_wh'] == [32, 32]
    assert saved['backend_diagnostics']['attention']['restored'] is True
    assert 'comparison' not in saved
    bundle = torch.load(Path(f'{output}.pt'), weights_only=False)
    # The input panel is the decoded unit-range RGB, never a CLIP-denormalized
    # version of it.
    assert bundle['input_rgb'].shape == (32, 32, 3)
    assert bundle['input_rgb'].min() >= 0.0
    assert bundle['input_rgb'].max() <= 1.0 + 1e-6
    assert bundle['grid'].tolist() == [2, 2]
    assert report['artifacts'][0].endswith('.pt')


def test_run_without_figure_still_saves_feature_bundle(pca, monkeypatch,
                                                       tmp_path):
    monkeypatch.setattr(pca, '_load_cradio_pytorch',
                        lambda args, device: _backbone())
    args = _cli_args(pca, tmp_path, '--no-figure')
    report = pca.run(args)
    assert Path(f'{args.output}.pt').is_file()
    assert not Path(f'{args.output}.png').exists()
    assert report['artifacts'] == [f'{args.output}.pt', f'{args.output}.json']


def test_pca_reference_run_projects_and_compares(pca, monkeypatch, tmp_path):
    # One shared backbone, so the second run sees exactly the same features.
    backbone = _backbone()
    monkeypatch.setattr(pca, '_load_cradio_pytorch',
                        lambda args, device: backbone)
    args = _cli_args(pca, tmp_path, '--no-figure')
    first = pca.run(args)
    args2 = _cli_args(pca, tmp_path, '--no-figure', '--pca-reference',
                      f'{args.output}.pt', '--output',
                      str(tmp_path / 'run' / 'pca_again'))
    second = pca.run(args2)
    assert first['pca_refit'] is True
    assert second['pca_refit'] is False
    assert second['pca_reference_used'] is True
    assert 'comparison' in second
    assert second['comparison']['allclose'] is True
    base = torch.load(Path(f'{args.output}.pt'), weights_only=False)
    reused = torch.load(Path(f'{args2.output}.pt'), weights_only=False)
    assert np.allclose(base['components'], reused['components'])
    with pytest.raises(pca.PcaVisualizeError, match='does not exist'):
        pca.run(pca.build_parser().parse_args([
            '--image', args.image, '--backbone', 'cradio', '--image-size',
            '32', '--device', 'cpu', '--pca-reference',
            str(tmp_path / 'absent.pt')
        ]))


def test_main_reports_failures_without_a_traceback(pca, tmp_path, capsys):
    args = _cli_args(pca, tmp_path)
    code = pca.main([
        '--image', args.image, '--backbone', 'cradio', '--image-size', '32',
        '--device', 'cpu'
    ])
    assert code == 1
    assert 'error:' in capsys.readouterr().err


def test_main_help_exits_cleanly(pca):
    with pytest.raises(SystemExit) as exit_info:
        pca.main(['--help'])
    assert exit_info.value.code == 0


# -- correction cycle: model selection, geometry, dtype, artifacts --------


@pytest.mark.parametrize('model_name,expected', [
    ('c-radio_v4-h', 'c-radio_v4-h'),
    ('nvidia/C-RADIOv4-H', 'c-radio_v4-h'),
    ('C-RADIOv4-H', 'c-radio_v4-h'),
    ('c-radio_v4-so400m', 'c-radio_v4-so400m'),
    ('nvidia/C-RADIOv4-SO400M', 'c-radio_v4-so400m'),
    ('cradiov4so400m', 'c-radio_v4-so400m'),
])
def test_radio_variant_resolution(pca, model_name, expected):
    assert pca.resolve_radio_variant(model_name, None) == expected
    assert pca.resolve_radio_variant(None, model_name) == expected
    assert pca.resolve_radio_variant(model_name, model_name) == expected


def test_radio_variant_conflicts_and_unknowns_fail(pca):
    assert pca.resolve_radio_variant(None, None) == 'c-radio_v4-h'
    with pytest.raises(pca.PcaVisualizeError, match='Conflicting'):
        pca.resolve_radio_variant('nvidia/C-RADIOv4-SO400M', 'c-radio_v4-h')
    with pytest.raises(pca.PcaVisualizeError, match='not a C-RADIO variant'):
        pca.resolve_radio_variant('facebook/dinov2-base', None)
    with pytest.raises(pca.PcaVisualizeError, match='not a C-RADIO variant'):
        pca.resolve_radio_variant(None, 'c-radio_v5-h')


def test_trt_variant_must_match_the_engine_metadata(pca, monkeypatch,
                                                    tmp_path):
    from fluxvla.models.backbones.visions.base_trt.radio_metadata import (
        EngineMetadata)
    from fluxvla.models.backbones.visions.radio_trt import (
        RadioViTBackboneTensorRT)
    metadata = EngineMetadata(
        variant='c-radio_v4-so400m',
        feature_dim=1152,
        patch_size=16,
        resolution_step=16,
        input_height=32,
        input_width=32,
        grid_height=2,
        grid_width=2,
        tensorrt_version='10.7.0')

    class FakeEngine:
        contract = metadata.contract()

        def run(self, images):
            return torch.zeros(1, 4, 1152, dtype=images.dtype)

        def close(self):
            pass

    monkeypatch.setattr(
        pca, '_load_cradio_trt', lambda args, device: RadioViTBackboneTensorRT(
            engine=FakeEngine(), metadata=metadata))
    args = pca.build_parser().parse_args([
        '--image',
        _image(tmp_path / 'x.png'), '--backbone', 'cradio-trt',
        '--engine-path', '/tmp/x.engine', '--image-size', '32', '--device',
        'cpu'
    ])
    args.version = 'c-radio_v4-h'
    with pytest.raises(pca.PcaVisualizeError, match='was built for'):
        pca.extract(args, (32, 32), 'cpu')


def test_effective_size_uses_the_backend_patch_size(pca, monkeypatch,
                                                    tmp_path):
    backbone = RadioViTBackbone(
        'c-radio_v4-h',
        model=StubRadio(8, 8),
        feature_dim=8,
        patch_size=8,
        resolution_step=8,
        dtype=torch.float32,
        default_image_resolution=(3, 32, 32))
    monkeypatch.setattr(pca, '_load_cradio_pytorch',
                        lambda args, device: backbone)
    args = pca.build_parser().parse_args([
        '--image',
        _image(tmp_path / 'patch8.png'), '--backbone', 'cradio',
        '--image-size', '32', '--device', 'cpu', '--no-figure', '--output',
        str(tmp_path / 'run' / 'patch8')
    ])
    report = pca.run(args)
    assert report['patch_size'] == 8
    assert report['effective_size_hw'] == [32, 32]
    assert report['grid'] == [4, 4]


def test_rounded_request_is_rejected_naming_the_supported_size(
        pca, monkeypatch, tmp_path):
    monkeypatch.setattr(pca, '_load_cradio_pytorch',
                        lambda args, device: _backbone())
    args = pca.build_parser().parse_args([
        '--image',
        _image(tmp_path / 'rounded.png'), '--backbone', 'cradio',
        '--image-size', '33', '--device', 'cpu', '--output',
        str(tmp_path / 'run' / 'rounded')
    ])
    with pytest.raises(pca.PcaVisualizeError, match='at 48x48'):
        pca.run(args)


def test_cpu_fp16_and_trt_dtype_flags_are_rejected(pca, monkeypatch, tmp_path):
    args = pca.build_parser().parse_args([
        '--image',
        _image(tmp_path / 'dtype.png'), '--backbone', 'cradio', '--image-size',
        '32', '--device', 'cpu', '--dtype', 'fp16', '--output',
        str(tmp_path / 'run' / 'dtype')
    ])
    monkeypatch.setattr(pca, '_load_cradio_pytorch',
                        lambda args, device: _backbone())
    with pytest.raises(pca.PcaVisualizeError, match='does not support fp16'):
        pca.run(args)
    args = pca.build_parser().parse_args([
        '--image',
        _image(tmp_path / 'dtype2.png'), '--backbone', 'cradio-trt',
        '--engine-path', '/tmp/x.engine', '--image-size', '32', '--device',
        'cpu', '--dtype', 'fp16', '--output',
        str(tmp_path / 'run' / 'dtype2')
    ])
    with pytest.raises(pca.PcaVisualizeError, match='does not apply'):
        pca.run(args)


def test_precision_diagnostics_report_the_effective_precision(
        pca, monkeypatch, tmp_path):
    monkeypatch.setattr(pca, '_load_cradio_pytorch',
                        lambda args, device: _backbone())
    fp32 = pca.run(pca.build_parser().parse_args([
        '--image',
        _image(tmp_path / 'fp32.png'), '--backbone', 'cradio', '--image-size',
        '32', '--device', 'cpu', '--no-figure', '--output',
        str(tmp_path / 'run' / 'fp32')
    ]))
    assert fp32['backend_diagnostics']['precision']['autocast'] is None
    bf16 = pca.run(pca.build_parser().parse_args([
        '--image',
        _image(tmp_path / 'bf16.png'), '--backbone', 'cradio', '--image-size',
        '32', '--device', 'cpu', '--dtype', 'bf16', '--no-figure', '--output',
        str(tmp_path / 'run' / 'bf16')
    ]))
    # CPU autocast runs bfloat16, and the diagnostics say so.
    assert bf16['backend_diagnostics']['precision'][
        'autocast'] == 'torch.bfloat16'


def test_dino_rejects_the_radio_version_flag(pca, tmp_path):
    args = pca.build_parser().parse_args([
        '--image',
        _image(tmp_path / 'dino.png'), '--backbone', 'dino', '--model-path',
        str(tmp_path), '--version', 'c-radio_v4-h'
    ])
    with pytest.raises(pca.PcaVisualizeError, match='omit it for'):
        pca.run(args)


# -- correction cycle: reference bundles, plotting, manifests -------------


def _run(pca, tmp_path, monkeypatch, backbone, name, *extra):
    monkeypatch.setattr(pca, '_load_cradio_pytorch',
                        lambda args, device: backbone)
    args = pca.build_parser().parse_args([
        '--image',
        _image(tmp_path / f'{name}.png', (64, 48)), '--backbone', 'cradio',
        '--image-size', '32', '--device', 'cpu', '--no-figure', '--output',
        str(tmp_path / 'run' / name), *extra
    ])
    return pca.run(args)


def test_bundle_is_tensor_only_and_reloads_with_weights_only(
        pca, monkeypatch, tmp_path):
    report = _run(pca, tmp_path, monkeypatch, _backbone(), 'bundle')
    bundle = torch.load(
        f"{report['output']}.pt", map_location='cpu', weights_only=True)
    assert bundle['format'] == 'pca-feature-bundle'
    assert bundle['format_version'] == 2
    assert bundle['components'].shape == (3, 8)
    assert bundle['grid'].tolist() == [2, 2]
    reference = pca.load_pca_reference(f"{report['output']}.pt")
    assert np.allclose(reference['components'], bundle['components'].numpy())


def test_saved_manifest_matches_the_returned_artifacts(pca, monkeypatch,
                                                       tmp_path):
    report = _run(pca, tmp_path, monkeypatch, _backbone(), 'manifest')
    saved = json.loads(Path(f"{report['output']}.json").read_text())
    assert saved['artifacts'] == report['artifacts']
    assert saved['artifacts'] == [
        f"{report['output']}.pt", f"{report['output']}.json"
    ]
    bundle = torch.load(f"{report['output']}.pt", weights_only=True)
    assert bundle['diagnostics']['artifacts'] == report['artifacts']
    args = pca.build_parser().parse_args([
        '--image',
        _image(tmp_path / 'figure.png', (64, 48)), '--backbone', 'cradio',
        '--image-size', '32', '--device', 'cpu', '--output',
        str(tmp_path / 'run' / 'figure')
    ])
    with_figure = pca.run(args)
    assert Path(f"{with_figure['output']}.png").is_file()
    assert len(with_figure['artifacts']) == 3
    saved = json.loads(Path(f"{with_figure['output']}.json").read_text())
    assert saved['artifacts'] == with_figure['artifacts']


def test_malicious_and_legacy_reference_bundles_are_rejected(pca, tmp_path):

    class Evil:

        def __reduce__(self):
            return (print, ('pwned', ))

    malicious = tmp_path / 'malicious.pt'
    torch.save(
        {
            'mean': torch.zeros(3),
            'components': torch.zeros(3, 3),
            'scale_min': torch.zeros(3),
            'scale_max': torch.ones(3),
            'evil': Evil(),
        }, malicious)
    with pytest.raises(pca.PcaVisualizeError, match='tensor-only'):
        pca.load_pca_reference(str(malicious))
    legacy = tmp_path / 'legacy.pt'
    torch.save(
        {
            'mean': np.zeros(3),
            'components': np.zeros((3, 3)),
            'scale_min': np.zeros(3),
            'scale_max': np.ones(3),
        }, legacy)
    with pytest.raises(pca.PcaVisualizeError, match='tensor-only'):
        pca.load_pca_reference(str(legacy))


@pytest.mark.parametrize('field,value,match', [
    ('components', 'rank', 'exactly 3 components'),
    ('components', 'width', 'wide but the mean'),
    ('components', 'count', 'exactly 3 components'),
    ('mean', 'rank', 'mean must be'),
    ('scale_min', 'shape', 'scaling must be three'),
    ('scale_max', 'nan', 'non-finite'),
    ('scale_max', 'below', 'scale_max below scale_min'),
])
def test_reference_schema_is_validated(pca, tmp_path, field, value, match):
    bundle = {
        'mean': torch.zeros(4),
        'components': torch.eye(3, 4),
        'scale_min': torch.zeros(3),
        'scale_max': torch.ones(3),
    }
    if value == 'rank':
        bundle[field] = torch.zeros(3, 4, 1)
    elif value == 'width':
        bundle[field] = torch.eye(3, 5)
    elif value == 'count':
        bundle[field] = torch.eye(4, 4)
    elif value == 'shape':
        bundle[field] = torch.zeros(2)
    elif value == 'nan':
        bundle[field] = torch.full((3, ), float('nan'))
    elif value == 'below':
        bundle[field] = torch.full((3, ), -1.0)
    path = tmp_path / 'bad.pt'
    torch.save(bundle, path)
    with pytest.raises(pca.PcaVisualizeError, match=match):
        pca.load_pca_reference(str(path))


def test_reference_with_missing_fields_asks_for_a_regeneration(pca, tmp_path):
    path = tmp_path / 'partial.pt'
    torch.save({'mean': torch.zeros(4)}, path)
    with pytest.raises(pca.PcaVisualizeError, match='missing'):
        pca.load_pca_reference(str(path))
    with pytest.raises(pca.PcaVisualizeError, match='not a PCA bundle'):
        torch.save(['not', 'a', 'mapping'], path)
        pca.load_pca_reference(str(path))


def test_reference_reuse_across_token_counts_skips_comparison(
        pca, monkeypatch, tmp_path):
    backbone = _backbone()
    small = _run(pca, tmp_path, monkeypatch, backbone, 'small')
    bigger = _run(pca, tmp_path, monkeypatch, backbone, 'bigger',
                  '--pca-reference', f"{small['output']}.pt", '--image-size',
                  '64')
    assert bigger['pca_refit'] is False
    assert bigger['tokens'] == 16 and small['tokens'] == 4
    assert 'comparison' not in bigger
    assert 'not comparable' in bigger['comparison_skipped']


def test_show_keeps_the_configured_backend_and_headless_errors(
        pca, monkeypatch, tmp_path):
    import matplotlib
    import matplotlib.pyplot as plt
    calls = []
    real_use = matplotlib.use
    monkeypatch.setattr(
        matplotlib, 'use', lambda name, **kwargs:
        (calls.append(name), real_use(name, **kwargs))[0])
    monkeypatch.setattr(plt, 'show', lambda *a, **k: None)
    result = {
        'input_rgb': np.zeros((4, 4, 3), dtype='float32'),
        'grid': (2, 2),
    }
    pca_report = {'image': np.zeros((2, 2, 3), dtype='float32')}
    (tmp_path / 'run').mkdir(parents=True, exist_ok=True)
    args = pca.build_parser().parse_args([
        '--image',
        _image(tmp_path / 'plot.png'), '--backbone', 'cradio', '--show',
        '--output',
        str(tmp_path / 'run' / 'plot')
    ])
    pca.plot_outputs(args, result, pca_report, tmp_path / 'run' / 'plot')
    assert 'Agg' not in calls
    calls.clear()
    args = pca.build_parser().parse_args([
        '--image',
        _image(tmp_path / 'plot2.png'), '--backbone', 'cradio', '--output',
        str(tmp_path / 'run' / 'plot2')
    ])
    pca.plot_outputs(args, result, pca_report, tmp_path / 'run' / 'plot2')
    assert calls == ['Agg']
    monkeypatch.setattr(
        plt, 'show', lambda *a, **k:
        (_ for _ in ()).throw(RuntimeError('no display')))
    args = pca.build_parser().parse_args([
        '--image',
        _image(tmp_path / 'plot3.png'), '--backbone', 'cradio', '--show',
        '--output',
        str(tmp_path / 'run' / 'plot3')
    ])
    with pytest.raises(pca.PcaVisualizeError, match='interactive'):
        pca.plot_outputs(args, result, pca_report, tmp_path / 'run' / 'plot3')
