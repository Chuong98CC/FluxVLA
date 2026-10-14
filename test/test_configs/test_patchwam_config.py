# Copyright 2026 Limx Dynamics
"""CPU units validating the native PatchWAM config placement and recipes."""

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from mmengine.config import Config
from torch.optim import AdamW

import fluxvla.datasets  # noqa: F401
import fluxvla.engines.runners  # noqa: F401  (runner registration)
import fluxvla.transforms  # noqa: F401
from fluxvla.collators.dict_collator import DictCollator
from fluxvla.engines import (DATASETS, HEADS, RUNNERS, TRANSFORMS, VLAS,
                             VLM_BACKBONES, build_collator_from_cfg,
                             build_lr_scheduler_from_cfg, build_vla_from_cfg)
from fluxvla.models.backbones.vlms import patchwam_backbone  # noqa: F401
from fluxvla.models.heads import patchwam_head  # noqa: F401
from fluxvla.models.vlas import patchwam_vla  # noqa: F401
from fluxvla.optimizers.lr_scheduler_policies import (
    LinearWarmupCosineDecayLRScheduler)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / 'configs' / 'patchwam'

CONFIGS = {
    'patchwam_libero_full_finetune.py': dict(
        action_dim=7, state_dim=8, norm='min_max', layout='horizontal',
        video_size=[224, 448], camera_size=[224, 224], separate=False,
        variant='klein-base-4b',
        slots=0, subtasks=False, adapters=False),
    'patchwam_robotwin_vlm_full_finetune.py': dict(
        action_dim=14, state_dim=14, norm='mean_std', layout='robotwin',
        video_size=[288, 256], camera_size=[240, 320], separate=False,
        variant='klein-base-4b', slots=0, subtasks=False, adapters=False),
    'patchwam_robodojo_vlm_history_example.py': dict(
        action_dim=14, state_dim=14, norm='mean_std', layout='robotwin',
        video_size=[288, 256], camera_size=[256, 256], separate=True,
        variant='klein-base-4b',
        slots=20, subtasks=True, adapters=True),
}


def _load(name):
    return Config.fromfile(str(CONFIG_DIR / name))


@pytest.mark.parametrize('name,recipe', CONFIGS.items())
def test_runner_collator_statistics_and_recipe(name, recipe):
    cfg = _load(name)
    model = cfg.model
    assert VLAS.get(model.type) is not None
    assert VLM_BACKBONES.get(model.vlm_backbone.type) is not None
    assert HEADS.get(model.vla_head.type) is not None
    assert HEADS.get(model.vla_head.denoiser.type) is not None
    # Klein base 4B consumes a 3x Qwen3 hidden-state readout.
    assert model.vlm_backbone.text_dim == 7680
    assert model.vla_head.text_dim == model.vlm_backbone.text_dim
    assert model.vla_head.action_dim == recipe['action_dim']
    assert model.vla_head.proprio_dim == recipe['state_dim']
    assert model.vla_head.isolate_actions is False
    assert model.vla_head.denoiser.variant == recipe['variant']
    assert model.vlm_backbone.get('vl_trainable_adapters',
                                  False) is recipe['adapters']
    if recipe['subtasks']:
        assert model.vlm_backbone.get('vl_require_subtasks', False) is True

    # The runner owns the collator/optimizer/scheduler/metric.
    runner = cfg.runner
    assert RUNNERS.get(runner.type) is not None
    assert 'collator' in runner and 'optimizer' in runner
    assert 'metric' in runner and 'lr_scheduler' in runner
    assert 'collator' not in cfg.train_dataloader
    collator = build_collator_from_cfg(dict(runner.collator))
    assert isinstance(collator, DictCollator)

    dataset = cfg.train_dataloader.dataset
    child = dataset.datasets
    assert DATASETS.get(child.type) is not None
    assert dataset.statistic_name == child.statistic_name
    assert child.action_key == 'action'
    assert child.future_offset == 16 and child.horizon == 16
    assert child.action_window_size == 16 and child.window_start_idx == 0
    if recipe['slots']:
        assert child.history['enabled'] is True
        assert child.history['slots'] == recipe['slots']
    else:
        assert child.history.get('enabled') is False

    types = [t['type'] for t in child.transforms]
    assert types[0] == 'PatchWAMRawInputs'
    assert 'PatchWAMDecodeFrames' in types
    assert 'NormalizeStatesAndActions' in types
    norm = next(t for t in child.transforms
                if t['type'] == 'NormalizeStatesAndActions')
    assert norm['norm_type'] == recipe['norm']
    # Stats are aggregated under the native ``proprio`` name.
    assert norm['state_key'] == 'proprio'
    assert norm['action_dim'] == recipe['action_dim']
    assert norm['state_dim'] == recipe['state_dim']
    compose = next(t for t in child.transforms
                   if t['type'] == 'PatchWAMComposeWorld')
    assert compose['layout'] == recipe['layout']
    assert list(compose['video_size']) == recipe['video_size']
    assert compose['separate_views'] is recipe['separate']
    resize = next(t for t in child.transforms
                  if t['type'] == 'PatchWAMResizeCameras')
    assert list(resize['camera_size']) == recipe['camera_size']
    for transform in child.transforms:
        assert TRANSFORMS.get(transform['type']) is not None


def test_train_and_validation_configs_are_independent():
    cfg = _load('patchwam_robotwin_vlm_full_finetune.py')
    train_transforms = cfg.train_dataloader.dataset.datasets.transforms
    val_transforms = cfg.val_dataloader.dataset.datasets.transforms
    assert cfg.train_dataloader is not cfg.val_dataloader
    train_augment = next(t for t in train_transforms
                         if t['type'] == 'PatchWAMWorldAugment')
    val_augment = next(t for t in val_transforms
                       if t['type'] == 'PatchWAMWorldAugment')
    assert train_augment['enabled'] is True
    assert val_augment['enabled'] is False
    # Mutating the training config must not mutate validation.
    original = val_augment['enabled']
    train_augment['enabled'] = False
    assert val_augment['enabled'] is original


def test_libero_eval_config_is_present():
    cfg = _load('patchwam_libero_full_finetune.py')
    assert 'eval' in cfg
    assert cfg.eval['type'] == 'LiberoEvalRunner'
    assert 'inference' in cfg
    assert cfg.inference['type'] == 'LiberoEvalRunner'
    # The runner carries the norm stats key (it overwrites the dataset field).
    assert cfg.inference['norm_stats_key'] == 'libero_all_no_noops'
    assert cfg.inference['dataset']['type'] == 'PatchWAMLiberoEvalDataset'
    assert cfg.eval['denormalize_action']['type'] == 'DenormalizeLiberoAction'
    transforms = [t['type'] for t in cfg.eval['dataset']['transforms']]
    assert 'PatchWAMEvalPack' in transforms


def test_shipped_datasets_pass_their_training_flag():
    for name, recipe in CONFIGS.items():
        cfg = _load(name)
        assert cfg.train_dataloader.dataset.datasets['training'] is True
        assert cfg.val_dataloader.dataset.datasets['training'] is False


def test_only_the_source_recipes_declare_an_expected_global_batch():
    """The 256 guard is recipe-owned, never a universal default."""
    declared = {
        'patchwam_robotwin_vlm_full_finetune.py': (256, 4, 2),
        'patchwam_robodojo_vlm_history_example.py': (256, 8, 1),
    }
    for name in CONFIGS:
        cfg = _load(name)
        if name not in declared:
            assert 'expected_global_batch_size' not in cfg.runner
            continue
        expected, per_device, accumulation = declared[name]
        assert cfg.runner.expected_global_batch_size == expected
        # The guard is the source batch of the recipe's own factors, and the
        # optimizer budget stays untouched.
        assert expected == per_device * 32 * accumulation
        assert cfg.train_dataloader.per_device_batch_size == per_device
        assert cfg.runner.grad_accumulation_steps == accumulation
        assert 'max_steps' not in cfg.runner


def test_robotwin_dataset_reads_the_nonidle_filter_from_the_environment(
        monkeypatch):
    monkeypatch.delenv('PATCHWAM_NONIDLE_FILTER', raising=False)
    cfg = _load('patchwam_robotwin_vlm_full_finetune.py')
    for split in ('train_dataloader', 'val_dataloader'):
        child = cfg[split].dataset.datasets
        assert child['nonidle_filter_path'] is None
        assert child['require_nonidle_filter'] is True

    monkeypatch.setenv('PATCHWAM_NONIDLE_FILTER', '/tmp/nonidle_rows.json')
    cfg = _load('patchwam_robotwin_vlm_full_finetune.py')
    for split in ('train_dataloader', 'val_dataloader'):
        child = cfg[split].dataset.datasets
        assert child['nonidle_filter_path'] == '/tmp/nonidle_rows.json'
        assert child['require_nonidle_filter'] is True


def test_other_recipes_keep_the_native_unfiltered_dataset():
    for name in ('patchwam_libero_full_finetune.py',
                 'patchwam_robodojo_vlm_history_example.py'):
        cfg = _load(name)
        child = cfg.train_dataloader.dataset.datasets
        assert 'nonidle_filter_path' not in child
        assert 'require_nonidle_filter' not in child


def test_requiring_the_filter_without_a_path_is_a_clear_dataset_error(
        tmp_path, monkeypatch):
    """Config import must still work with no environment variable set."""
    monkeypatch.delenv('PATCHWAM_NONIDLE_FILTER', raising=False)
    cfg = _load('patchwam_robotwin_vlm_full_finetune.py')
    dataset_cfg = copy.deepcopy(
        cfg.train_dataloader.dataset.datasets.to_dict())
    dataset_cfg['data_root_path'] = [str(tmp_path)]
    from fluxvla.engines import build_dataset_from_cfg

    with pytest.raises(ValueError, match='require_nonidle_filter'):
        build_dataset_from_cfg(dataset_cfg)


def _openpi_reference_multiplier(step, warmup_steps, decay_steps,
                                 min_lr_rate):
    """The shared ``openpi`` rate used by the shipped recipes."""
    if step < warmup_steps:
        init_multiplier = 1.0 / (warmup_steps + 1)
        return init_multiplier + (1.0 - init_multiplier) * step / max(
            1, warmup_steps)
    progress = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_rate + (1.0 - min_lr_rate) * cosine


def _source_patchwam_reference_multiplier(step, warmup_steps, total_steps,
                                          min_lr_rate):
    """Original PatchWAM ``src/patchwam/engine.py`` rate, for comparison."""
    if warmup_steps and step < warmup_steps:
        start = 1.0 / warmup_steps
        return start + (1.0 - start) * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_rate + (1.0 - min_lr_rate) * cosine


def _build_recipe_scheduler(name, total_steps, lr=None):
    cfg = _load(name)
    scheduler_cfg = copy.deepcopy(Config(cfg.runner.lr_scheduler).to_dict())
    policy = build_lr_scheduler_from_cfg(scheduler_cfg)
    peak_lr = float(cfg.runner.optimizer.lr if lr is None else lr)
    optimizer = AdamW([nn.Parameter(torch.ones(()))], lr=peak_lr)
    runner = SimpleNamespace(num_training_steps=total_steps)
    return cfg, policy.build_scheduler(runner, optimizer), optimizer, peak_lr


@pytest.mark.parametrize('name', list(CONFIGS))
def test_recipes_reuse_the_shared_openpi_schedule(name):
    """The recipes reuse one shipped style instead of a PatchWAM branch."""
    cfg = _load(name)
    scheduler_cfg = copy.deepcopy(Config(cfg.runner.lr_scheduler).to_dict())

    assert scheduler_cfg['type'] == 'linear-warmup+cosine-decay'
    assert scheduler_cfg['schedule_style'] == 'openpi'
    assert scheduler_cfg['warmup_ratio'] == 0.05
    assert scheduler_cfg['min_lr_rate'] == 0.01
    # No PatchWAM-only knob, fixed step count, or absolute floor was added.
    assert set(scheduler_cfg) == {
        'type', 'schedule_style', 'warmup_ratio', 'min_lr_rate'}
    # The optimizer budget stays untouched by the schedule selection.
    assert cfg.runner.optimizer.type == 'AdamW'
    assert cfg.runner.optimizer.lr == 1.0e-4
    assert cfg.runner.optimizer.weight_decay == 0.01
    assert cfg.runner.optimizer.betas == (0.9, 0.95)
    assert 'max_epochs' in cfg.runner and 'max_steps' not in cfg.runner
    assert isinstance(build_lr_scheduler_from_cfg(scheduler_cfg),
                      LinearWarmupCosineDecayLRScheduler)


@pytest.mark.parametrize('name', list(CONFIGS))
def test_openpi_schedule_matches_recipe_constants_at_source_scale(name):
    total_steps = 20000
    warmup_steps = int(total_steps * 0.05)
    assert warmup_steps == 1000
    cfg, scheduler, optimizer, peak_lr = _build_recipe_scheduler(
        name, total_steps)

    # The first actual update is nonzero: openpi does not zero the start (the
    # huggingface style zeroes ``param_groups['lr']`` at build time).
    assert optimizer.param_groups[0]['lr'] > 0
    assert optimizer.param_groups[0]['lr'] == pytest.approx(
        peak_lr / (warmup_steps + 1))
    for step in (0, 1, warmup_steps - 1, warmup_steps, total_steps // 2,
                 total_steps, total_steps + 10):
        assert scheduler.lr_lambdas[0](step) == pytest.approx(
            _openpi_reference_multiplier(step, warmup_steps, total_steps,
                                         0.01))
    # The cosine decay clamps at the configured floor rate.
    assert scheduler.lr_lambdas[0](total_steps) == pytest.approx(0.01)
    assert scheduler.lr_lambdas[0](total_steps + 500) == pytest.approx(0.01)


def test_openpi_warmup_deviation_from_the_original_source_is_bounded():
    """Reusing openpi shifts the warmup by a bounded, documented amount."""
    total_steps, warmup_steps, min_lr_rate = 20000, 1000, 0.01
    differences = [
        _source_patchwam_reference_multiplier(step, warmup_steps, total_steps,
                                             min_lr_rate)
        - _openpi_reference_multiplier(step, warmup_steps, total_steps,
                                       min_lr_rate)
        for step in range(warmup_steps)
    ]
    max_abs = max(abs(value) for value in differences)
    assert max_abs == pytest.approx(1.0 / (warmup_steps * (warmup_steps + 1)))
    assert max_abs < 1e-6
    # The largest deviation is the first warmup step, where the source starts
    # at 1/W and openpi starts at 1/(W+1): 1/(W+1) ~ 0.1 % of the source value.
    assert abs(differences[0]) == pytest.approx(max_abs)
    assert abs(differences[0]) == pytest.approx(
        1.0 / (warmup_steps + 1) / warmup_steps)
    assert abs(differences[0]) / (1.0 / warmup_steps) == pytest.approx(
        1.0 / (warmup_steps + 1))
    # Past the warmup boundary the two agreement is exact.
    assert _source_patchwam_reference_multiplier(
        warmup_steps, warmup_steps, total_steps, min_lr_rate
    ) == pytest.approx(_openpi_reference_multiplier(
        warmup_steps, warmup_steps, total_steps, min_lr_rate))


@pytest.mark.parametrize('total_steps', [4, 19, 20])
def test_openpi_small_budgets_start_nonzero(total_steps):
    """Small test runs use the openpi default ratio with no warmup steps."""
    _, scheduler, optimizer, peak_lr = _build_recipe_scheduler(
        'patchwam_libero_full_finetune.py', total_steps)
    warmup_steps = int(total_steps * 0.05)
    # ``int(N * 0.05)`` is 0 below 20 updates, so the first update runs at the
    # full rate instead of a zeroed start.
    if total_steps < 20:
        assert warmup_steps == 0
        assert scheduler.lr_lambdas[0](0) == pytest.approx(1.0)
    assert scheduler.lr_lambdas[0](0) == pytest.approx(
        _openpi_reference_multiplier(0, warmup_steps, total_steps, 0.01))
    assert optimizer.param_groups[0]['lr'] > 0
    assert optimizer.param_groups[0]['lr'] == pytest.approx(
        peak_lr * scheduler.lr_lambdas[0](0))


def test_the_patchwam_specific_schedule_style_is_gone():
    with pytest.raises(ValueError, match='Unsupported schedule_style'):
        LinearWarmupCosineDecayLRScheduler(schedule_style='patchwam')


class TinyAutoencoder(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        self.proj = nn.Conv2d(3, channels, 1)

    def encode(self, video):
        return self.proj(video)


class TinyTextEncoder(nn.Module):
    requires_images = False
    trainable_adapters = False

    def __init__(self, dim: int = 32, length: int = 3):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(length, dim))
        self.feature_dim = dim

    def forward(self, prompts):
        return self.embed.unsqueeze(0).expand(len(prompts), -1, -1)


def test_config_model_builds_with_injected_modules():
    cfg = _load('patchwam_libero_full_finetune.py')
    model_cfg = copy.deepcopy(Config(cfg.model).to_dict())
    model_cfg['vlm_backbone'] = dict(
        type='PatchWAMBackbone',
        autoencoder=TinyAutoencoder(channels=128),
        text_encoder=TinyTextEncoder(dim=32),
        token_dim=128, text_dim=32, history_pool_size=2)
    model_cfg['vla_head'] = dict(
        type='PatchWAMHead',
        denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128, text_dim=32,
                      width=32, heads=4, depth=2),
        action_dim=7, text_dim=32, proprio_dim=8, token_dim=128)
    model_cfg['num_inference_steps'] = 2
    vla = build_vla_from_cfg(model_cfg)
    out = vla(images=torch.rand(2, 3, 2, 32, 64), states=torch.randn(2, 8),
              actions=torch.randn(2, 16, 7),
              task_description=['a', 'b'])
    assert torch.isfinite(out['loss'])
