# Copyright 2026 Limx Dynamics
"""Local C-RADIO export loading: package isolation and weight checking.

These units are narrow and synthetic: they exercise the direct loader's
namespace, import failure, weight-key checking and asset validation without
needing the full checkpoint.
"""

import json

import pytest
import torch
from safetensors.torch import save_file

from fluxvla.models.third_party_models.patchwam.radio import (
    _import_export_code, _load_export_directly, load_local_radio_teacher)

_HF_MODEL = '''
import torch
from torch import nn


class RADIOConfig:

    version = 'c-radio_v4-h'

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls()


class RADIOModel(nn.Module):

    def __init__(self, config=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4))
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, images):
        return images
'''


def _export_dir(tmp_path, name='export', weights=None, version='c-radio_v4-h'):
    directory = tmp_path / name
    directory.mkdir(parents=True)
    (directory / 'hf_model.py').write_text(_HF_MODEL)
    (directory / 'config.json').write_text(
        json.dumps({
            'version': version,
            'patch_size': 16
        }))
    save_file(
        weights if weights is not None else {
            'weight': torch.ones(4),
            'bias': torch.ones(2)
        }, str(directory / 'model.safetensors'))
    return directory


def test_direct_loader_succeeds_and_builds_the_module(tmp_path):
    directory = _export_dir(tmp_path)
    config, model = _load_export_directly(directory)
    assert type(model).__name__ == 'RADIOModel'
    assert torch.allclose(model.weight, torch.ones(4))
    assert config.version == 'c-radio_v4-h'


def test_direct_loader_rejects_unexpected_and_missing_weights(tmp_path):
    extra = _export_dir(tmp_path, 'extra', {
        'weight': torch.ones(4),
        'bias': torch.ones(2),
        'other': torch.ones(2)
    })
    with pytest.raises(RuntimeError, match='weights this model does not use'):
        _load_export_directly(extra)
    # Only ``bias`` is absent and nothing is unexpected, so this exercises the
    # missing-key branch rather than the unexpected-key one.
    missing = _export_dir(tmp_path, 'missing', {'weight': torch.ones(4)})
    with pytest.raises(RuntimeError, match='missing weights the model needs'):
        _load_export_directly(missing)


def test_similar_paths_get_distinct_packages(tmp_path):
    dashed = _export_dir(tmp_path, 'foo-a')
    underscored = _export_dir(tmp_path, 'foo_a')
    first = _import_export_code(dashed)
    second = _import_export_code(underscored)
    assert first is not second
    assert first.__name__ != second.__name__
    assert first.__name__.endswith('hf_model')


def test_missing_model_code_fails_the_import(tmp_path):
    directory = tmp_path / 'no-code'
    directory.mkdir()
    with pytest.raises(ImportError):
        _import_export_code(directory)


def test_export_without_config_is_rejected(tmp_path):
    directory = tmp_path / 'no-config'
    directory.mkdir()
    with pytest.raises(FileNotFoundError, match='config.json'):
        load_local_radio_teacher(str(directory))


def test_export_with_a_wrong_version_is_rejected(tmp_path):
    directory = _export_dir(
        tmp_path, 'wrong-version', version='c-radio_v4-so400m')
    with pytest.raises(ValueError, match='reports version'):
        load_local_radio_teacher(str(directory), version='c-radio_v4-h')


def test_export_requires_a_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match='local Hugging Face export'):
        load_local_radio_teacher(str(tmp_path / 'absent'))
