"""CPU units for the five-task RobotWin PatchWAM recipe."""

import copy
from pathlib import Path

from mmengine.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / 'configs' / 'patchwam'
BASE = CONFIG_DIR / 'patchwam_robotwin_vlm_full_finetune.py'
SUBSET = CONFIG_DIR / 'patchwam_robotwin_vlm_5tasks_full_finetune.py'

TASKS = ['turn_switch', 'rotate_qrcode', 'blocks_ranking_size',
         'move_stapler_pad', 'hanging_mug']
STATISTIC = 'robotwin_5tasks'
DEFAULT_ROOTS = ['datasets/robotwin_lerobotv2.1']


def _load(path, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.delenv('PATCHWAM_DATA_ROOT', raising=False)
    return Config.fromfile(str(path))


def _split(config, split):
    """The ``DistributedRepeatingDataset`` node for one split."""
    return config[f'{split}_dataloader']['dataset']


def test_train_and_val_select_the_five_tasks_in_order(monkeypatch):
    config = _load(SUBSET, monkeypatch)
    for split in ('train', 'val'):
        assert _split(config, split)['datasets']['task_selection'] == {'tasks': TASKS}


def test_train_and_val_selectors_are_independent(monkeypatch):
    config = _load(SUBSET, monkeypatch)
    config.train_dataloader.dataset.datasets.task_selection.tasks.append('extra_task')
    assert _split(config, 'val')['datasets']['task_selection'] == {'tasks': TASKS}
    assert _split(config, 'train')['datasets']['task_selection']['tasks'] == TASKS + ['extra_task']


def test_statistic_namespace_is_isolated_from_the_full_recipe(monkeypatch):
    config = _load(SUBSET, monkeypatch)
    base = _load(BASE, monkeypatch)
    for split in ('train', 'val'):
        assert _split(config, split)['statistic_name'] == STATISTIC
        assert _split(config, split)['datasets']['statistic_name'] == STATISTIC
        assert _split(base, split)['datasets']['statistic_name'] == 'robotwin'


def test_data_root_honours_the_environment(monkeypatch):
    config = _load(SUBSET, monkeypatch)
    for split in ('train', 'val'):
        assert _split(config, split)['datasets']['data_root_path'] == DEFAULT_ROOTS
    monkeypatch.setenv('PATCHWAM_DATA_ROOT', '/data/robotwin_merged')
    merged = _load(SUBSET)
    for split in ('train', 'val'):
        assert _split(merged, split)['datasets']['data_root_path'] == ['/data/robotwin_merged']


def test_recipe_matches_base_except_deliberate_overrides(monkeypatch):
    base = _load(BASE, monkeypatch).to_dict()
    subset = _load(SUBSET, monkeypatch).to_dict()
    base_public = {key: value for key, value in base.items() if not key.startswith('_')}
    subset_public = {key: value for key, value in subset.items() if not key.startswith('_')}
    assert sorted(subset_public) == sorted(base_public)
    expected = copy.deepcopy(base_public)
    for split in ('train_dataloader', 'val_dataloader'):
        wrapper = expected[split]['dataset']
        child = wrapper['datasets']
        wrapper['statistic_name'] = STATISTIC
        child['statistic_name'] = STATISTIC
        child['data_root_path'] = DEFAULT_ROOTS
        child['task_selection'] = {'tasks': TASKS}
    assert subset_public == expected
