"""RoboTwin recipe wiring for the optional PatchWAM FK objective."""

from __future__ import annotations

from pathlib import Path

import pytest
from mmengine import Config

import fluxvla.datasets  # noqa: F401
import fluxvla.transforms  # noqa: F401
from fluxvla.engines import DATASETS, HEADS, TRANSFORMS
from fluxvla.models.heads import patchwam_head  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / 'configs/patchwam/robotwin/patchwam_robotwin.py'


def _load():
    return Config.fromfile(str(CONFIG))


def test_recipe_defaults_keep_the_legacy_action_loss(monkeypatch):
    for key in (
            'PATCHWAM_ACTION_TOKEN_VELOCITY_WEIGHT',
            'PATCHWAM_ACTION_MEAN_CLEAN_WEIGHT',
            'PATCHWAM_FK_WEIGHT',
            'PATCHWAM_FK_STATS'):
        monkeypatch.delenv(key, raising=False)
    cfg = _load()
    assert cfg.model.vla_head.action_loss == {
        'token_velocity_weight': 1.0,
        'mean_clean_weight': 0.0,
    }
    assert cfg.model.vla_head.fk_loss['weight'] == 0.0
    assert cfg.model.vla_head.fk_loss['position_weight'] == 1.0
    assert cfg.model.vla_head.fk_loss['rotation_weight'] == 1.0
    child = cfg.train_dataloader.dataset.datasets
    assert child.fk_stats_path is None
    assert child.require_fk_stats is False
    assert cfg.runner.batch_fp32_keys == ()
    types = [transform['type'] for transform in child.transforms]
    assert 'PatchWAMFkActionMetadata' not in types
    assert 'active_arm' in cfg.runner.collator['keys']
    assert 'fk_action_mean' in cfg.runner.collator['keys']


def test_recipe_exposes_all_weights_and_training_metadata(monkeypatch):
    monkeypatch.setenv('PATCHWAM_ACTION_TOKEN_VELOCITY_WEIGHT', '0.4')
    monkeypatch.setenv('PATCHWAM_ACTION_MEAN_CLEAN_WEIGHT', '0.6')
    monkeypatch.setenv('PATCHWAM_FK_WEIGHT', '0.25')
    monkeypatch.setenv('PATCHWAM_FK_POSITION_WEIGHT', '2.0')
    monkeypatch.setenv('PATCHWAM_FK_ROTATION_WEIGHT', '3.0')
    monkeypatch.setenv('PATCHWAM_FK_STATS', '/tmp/fk-stats.json')
    cfg = _load()
    head = cfg.model.vla_head
    assert head.action_loss == {
        'token_velocity_weight': 0.4,
        'mean_clean_weight': 0.6,
    }
    assert head.fk_loss['weight'] == 0.25
    assert head.fk_loss['position_weight'] == 2.0
    assert head.fk_loss['rotation_weight'] == 3.0
    assert head.fk_loss['stats_path'] == '/tmp/fk-stats.json'
    for split in ('train_dataloader', 'val_dataloader'):
        child = cfg[split].dataset.datasets
        assert child.fk_stats_path == '/tmp/fk-stats.json'
        assert child.require_fk_stats is True
        assert set(cfg.runner.batch_fp32_keys) == {
            'fk_action_mean', 'fk_action_std', 'fk_action_epsilon'}
        assert 'PatchWAMFkActionMetadata' in [
            transform['type'] for transform in child.transforms]
    assert HEADS.get('PatchWAMHead') is not None
    assert DATASETS.get('PatchWAMParquetDataset') is not None
    assert TRANSFORMS.get('PatchWAMFkActionMetadata') is not None


def test_recipe_rejects_invalid_negative_weight(monkeypatch):
    monkeypatch.setenv('PATCHWAM_FK_WEIGHT', '-1')
    with pytest.raises(ValueError, match='nonnegative'):
        _load()


def test_ddp_runner_exposes_the_batch_fp32_key_policy():
    import inspect

    from fluxvla.engines.runners.ddp_train_runner import DDPTrainRunner

    parameters = inspect.signature(DDPTrainRunner.__init__).parameters
    assert 'batch_fp32_keys' in parameters
