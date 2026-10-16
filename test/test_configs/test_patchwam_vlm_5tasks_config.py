"""CPU units for the five-task RoboTwin-Clean PatchWAM recipe."""

import copy
from pathlib import Path

from mmengine.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / 'configs' / 'patchwam' / 'robotwin'
BASE = CONFIG_DIR / 'patchwam_robotwin_vlm_full_finetune.py'
SUBSET = CONFIG_DIR / 'patchwam_robotwin_vlm_5tasks_full_finetune.py'

TASKS = ['turn_switch', 'rotate_qrcode', 'blocks_ranking_size',
         'move_stapler_pad', 'hanging_mug']
STATISTIC = 'robotwin'
SELECTION = {'tasks': list(TASKS)}
ENVIRONMENT = ('PATCHWAM_DATA_ROOT', 'PATCHWAM_NONIDLE_FILTER',
               'PATCHWAM_TRANSFORMER', 'PATCHWAM_AUTOENCODER', 'PATCHWAM_VLM',
               'FLUX2_SRC', 'PATCHWAM_FLUX_ASSETS', 'PRETRAINED_ROOT',
               'PATCHWAM_PRETRAINED_CHECKPOINT')
SPLITS = ('train_dataloader', 'val_dataloader')


def _load(path, monkeypatch=None):
    if monkeypatch is not None:
        for name in ENVIRONMENT:
            monkeypatch.delenv(name, raising=False)
    return Config.fromfile(str(path))


def _child(config, split):
    """The ``PatchWAMParquetDataset`` node for one split."""
    return config[split]['dataset']['datasets']


def test_the_resolved_recipes_differ_only_by_the_task_selection(monkeypatch):
    """The ablation narrows the rows and inherits every other setting."""
    base = _load(BASE, monkeypatch).to_dict()
    subset = _load(SUBSET, monkeypatch).to_dict()
    base_public = {key: value for key, value in base.items()
                   if not key.startswith('_')}
    subset_public = {key: value for key, value in subset.items()
                     if not key.startswith('_')}
    assert sorted(subset_public) == sorted(base_public)
    expected = copy.deepcopy(base_public)
    for split in SPLITS:
        expected[split]['dataset']['datasets']['task_selection'] = \
            copy.deepcopy(SELECTION)
    assert subset_public == expected


def test_both_splits_select_the_five_tasks(monkeypatch):
    config = _load(SUBSET, monkeypatch)
    for split in SPLITS:
        assert dict(_child(config, split)['task_selection']) == SELECTION
        assert list(_child(config, split)['task_selection']['tasks']) == TASKS


def test_train_and_val_selections_are_independent(monkeypatch):
    config = _load(SUBSET, monkeypatch)
    train = _child(config, 'train_dataloader')['task_selection']
    val = _child(config, 'val_dataloader')['task_selection']
    assert train is not val
    train['tasks'].append('/tmp/extra_task')
    assert list(_child(config, 'val_dataloader')
                ['task_selection']['tasks']) == TASKS


def test_the_subset_inherits_roots_assets_statistics_and_the_filter(
        monkeypatch):
    config = _load(SUBSET, monkeypatch)
    base = _load(BASE, monkeypatch)
    for split in SPLITS:
        child = _child(config, split)
        assert child['data_root_path'] == _child(base, split)['data_root_path']
        assert child['statistic_name'] == _child(base, split)['statistic_name']
        assert config[split]['dataset']['statistic_name'] == STATISTIC
        assert child['statistic_name'] == STATISTIC
        # The non-idle filter contract is the source recipe's, unmodified.
        assert child['nonidle_filter_path'] is None
        assert child['require_nonidle_filter'] is True
    for target in ('model', 'inference_model'):
        assert config[target] == base[target]
    # The inherited environment indirection still reaches the child node.
    monkeypatch.setenv('PATCHWAM_NONIDLE_FILTER', '/tmp/nonidle_rows.json')
    moved = _load(SUBSET)
    for split in SPLITS:
        assert _child(moved, split)['nonidle_filter_path'] == \
            '/tmp/nonidle_rows.json'


def test_the_subset_inherits_the_batch_guard_and_auto_scaling(monkeypatch):
    """The ablation keeps the base recipe's LR policy without restating it."""
    config = _load(SUBSET, monkeypatch)
    base = _load(BASE, monkeypatch)
    for cfg in (config, base):
        assert cfg.runner.expected_global_batch_size == 256
        assert cfg.runner.auto_scale_lr is True


def test_the_shipped_subset_passes_the_statistics_guard(monkeypatch):
    from fluxvla.datasets.utils.task_selection import \
        reject_predataset_statistics

    config = _load(SUBSET, monkeypatch)
    # The narrowed recipe configures no pre-dataset statistics, so the dataset
    # keeps aggregating them from the episodes it selected.
    reject_predataset_statistics(config.train_dataloader.dataset)
    for split in SPLITS:
        assert 'dataset_statistics' not in config[split]['dataset']
        assert 'dataset_statistics_path' not in config[split]['dataset']
