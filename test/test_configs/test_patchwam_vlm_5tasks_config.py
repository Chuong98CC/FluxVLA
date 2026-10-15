"""CPU units for the five-task RoboTwin-Clean PatchWAM recipe."""

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
PARENT = '/data/RoboTwin/RoboTwin-Clean'
DEFAULT_ROOTS = [f'{PARENT}/{task}' for task in TASKS]
ASSETS = {
    'autoencoder_path': ('/home/chuong/workspace/wam/ImageWAM/checkpoints/flux2/'
                         'FLUX.2-dev/ae.safetensors'),
    'text_encoder_path': '/data/Pretrained_models/Qwen3-VL-4B-Instruct',
    'flux_source': '/home/chuong/workspace/wam/ImageWAM/third_party/flux2',
    'transformer_path': ('/home/chuong/workspace/wam/ImageWAM/checkpoints/flux2/'
                         'FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors'),
}
CHECKPOINT = ('checkpoints/Pretrained_models/patchwam-robotwin/'
              'step_140000.pt')


def _load(path, monkeypatch=None):
    if monkeypatch is not None:
        for name in ('PATCHWAM_DATA_ROOT', 'PATCHWAM_NONIDLE_FILTER',
                     'PATCHWAM_TRANSFORMER', 'PATCHWAM_AUTOENCODER',
                     'PATCHWAM_VLM', 'FLUX2_SRC', 'PATCHWAM_FLUX_ASSETS',
                     'PATCHWAM_PRETRAINED_CHECKPOINT'):
            monkeypatch.delenv(name, raising=False)
    return Config.fromfile(str(path))


def _split(config, split):
    """The ``DistributedRepeatingDataset`` node for one split."""
    return config[f'{split}_dataloader']['dataset']


def test_both_splits_list_the_five_task_roots_in_order(monkeypatch):
    config = _load(SUBSET, monkeypatch)
    for split in ('train', 'val'):
        child = _split(config, split)['datasets']
        assert child['data_root_path'] == DEFAULT_ROOTS
        assert 'task_selection' not in child


def test_train_and_val_root_lists_are_independent(monkeypatch):
    config = _load(SUBSET, monkeypatch)
    config.train_dataloader.dataset.datasets.data_root_path.append('/tmp/extra_task')
    assert list(_split(config, 'val')['datasets']['data_root_path']) == DEFAULT_ROOTS
    assert list(_split(config, 'train')['datasets']['data_root_path']) == DEFAULT_ROOTS + ['/tmp/extra_task']


def test_statistic_namespace_is_isolated_from_the_full_recipe(monkeypatch):
    config = _load(SUBSET, monkeypatch)
    base = _load(BASE, monkeypatch)
    for split in ('train', 'val'):
        assert _split(config, split)['statistic_name'] == STATISTIC
        assert _split(config, split)['datasets']['statistic_name'] == STATISTIC
        assert _split(base, split)['datasets']['statistic_name'] == 'robotwin'


def test_release_defaults_and_environment_overrides(monkeypatch):
    config = _load(SUBSET, monkeypatch)
    for split in ('train', 'val'):
        child = _split(config, split)['datasets']
        assert child['data_root_path'] == DEFAULT_ROOTS
        # This release ships no non-idle row filter.
        assert child['nonidle_filter_path'] is None
        assert child['require_nonidle_filter'] is False
    monkeypatch.setenv('PATCHWAM_DATA_ROOT', '/data/RoboTwin/Other-Release')
    monkeypatch.setenv('PATCHWAM_NONIDLE_FILTER', '/data/other/filter.json')
    moved = _load(SUBSET)
    for split in ('train', 'val'):
        child = _split(moved, split)['datasets']
        expected = [f'/data/RoboTwin/Other-Release/{task}' for task in TASKS]
        assert child['data_root_path'] == expected
        assert child['nonidle_filter_path'] == '/data/other/filter.json'
        # An explicit filter still keeps the required-filter contract.
        assert child['require_nonidle_filter'] is True


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
        child['nonidle_filter_path'] = None
        child['require_nonidle_filter'] = False
    for target in ('model', 'inference_model'):
        expected[target]['vlm_backbone']['autoencoder_path'] = ASSETS['autoencoder_path']
        expected[target]['vlm_backbone']['text_encoder_path'] = ASSETS['text_encoder_path']
        expected[target]['vlm_backbone']['flux_source'] = ASSETS['flux_source']
        expected[target]['vla_head']['denoiser']['transformer_path'] = ASSETS['transformer_path']
        expected[target]['vla_head']['denoiser']['flux_source'] = ASSETS['flux_source']
        expected[target]['pretrained_name_or_path'] = CHECKPOINT
    assert subset_public == expected


def test_the_subset_inherits_the_batch_guard_and_auto_scaling(monkeypatch):
    """The ablation keeps the base recipe's LR policy without restating it."""
    config = _load(SUBSET, monkeypatch)
    base = _load(BASE, monkeypatch)
    for cfg in (config, base):
        assert cfg.runner.expected_global_batch_size == 256
        assert cfg.runner.auto_scale_lr is True


def test_initial_weights_are_the_released_policy(monkeypatch):
    config = _load(SUBSET, monkeypatch)
    for target in ('model', 'inference_model'):
        assert config[target]['pretrained_name_or_path'] == CHECKPOINT
    monkeypatch.setenv('PATCHWAM_PRETRAINED_CHECKPOINT', '/tmp/other_policy.pt')
    moved = _load(SUBSET)
    for target in ('model', 'inference_model'):
        assert moved[target]['pretrained_name_or_path'] == '/tmp/other_policy.pt'
