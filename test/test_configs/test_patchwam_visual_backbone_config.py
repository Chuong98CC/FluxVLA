# Copyright 2026 Limx Dynamics
"""CPU units validating the staged C-RADIO visual-backbone recipes."""

from pathlib import Path

import pytest
from mmengine.config import Config

import fluxvla.datasets  # noqa: F401
import fluxvla.engines.runners  # noqa: F401  (runner registration)
import fluxvla.transforms  # noqa: F401
from fluxvla.engines import (HEADS, VLM_BACKBONES, VLAS,
                             build_vla_from_cfg)
from fluxvla.models.backbones.vlms import patchwam_backbone  # noqa: F401
from fluxvla.models.heads import patchwam_head  # noqa: F401
from fluxvla.models.vlas import patchwam_vla  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / 'configs' / 'patchwam'

BASE = 'patchwam_radio_robotwin_base.py'
STAGE1 = 'patchwam_radio_stage1_robotwin.py'
STAGE2 = 'patchwam_radio_stage2_robotwin.py'
ABLATION_STAGE1 = 'patchwam_radio_ablation_default5_stage1_robotwin.py'
ABLATION_STAGE2 = 'patchwam_radio_ablation_default5_stage2_robotwin.py'


def load(name):
    return Config.fromfile(str(CONFIG_DIR / name))


def test_recipes_are_registered_and_declare_the_visual_backbone():
    for name in (BASE, STAGE1, STAGE2, ABLATION_STAGE1, ABLATION_STAGE2):
        config = load(name)
        assert config.model.type in VLAS.module_dict
        assert config.model.vlm_backbone.type == 'PatchWAMBackbone'
        assert config.model.vlm_backbone.type in VLM_BACKBONES.module_dict
        assert config.model.vla_head.type in HEADS.module_dict
        assert config.model.vlm_backbone.visual_backbone == {
            'kind': 'c-radio_v4-h',
            'model_path':
            config.model.vlm_backbone.visual_backbone['model_path'],
        }
        assert config.model.vla_head.visual_dim == 1280
        assert config.model.vla_head.token_dim == 128
        assert config.model.vla_head.denoiser.visual_dim == 1280


def test_stage_recipes_share_the_subset_and_stage_flags():
    base = load(BASE)
    stage1, stage2 = load(STAGE1), load(STAGE2)
    ablation1, ablation2 = load(ABLATION_STAGE1), load(ABLATION_STAGE2)
    # The base recipe is an ordinary full-task fine-tune.
    assert base.model.get('visual_stage') is None
    assert base.train_dataloader.dataset.datasets.get('task_selection') is None
    # Ordinary staged recipes keep every task and differ only by stage.
    for config in (stage1, stage2):
        assert config.train_dataloader.dataset.datasets.get(
            'task_selection') is None
        assert config.runner.type == 'DDPTrainRunner'
        assert config.runner.optimizer.type == 'AdamW'
    assert stage1.model.visual_stage == 'stage1'
    assert stage1.model.vla_head.stage == 'stage1'
    assert stage1.model.vla_head.lora is None
    assert stage1.model.pretrained_name_or_path
    assert stage2.model.visual_stage == 'stage2'
    assert stage2.model.vla_head.stage == 'stage2'
    assert stage2.model.vla_head.lora['rank'] == 32
    assert stage2.model.get('pretrained_name_or_path') in (None, '')
    # Both ablation stages declare the same deterministic subset.
    subset1 = ablation1.train_dataloader.dataset.datasets.task_selection
    subset2 = ablation2.train_dataloader.dataset.datasets.task_selection
    assert subset1 == subset2 == dict(num_tasks=5, episodes_per_task=20,
                                      seed=42)
    assert ablation1.val_dataloader.dataset.datasets.task_selection == subset1
    assert ablation1.model.visual_stage == 'stage1'
    assert ablation2.model.visual_stage == 'stage2'
    assert ablation1.model.vla_head.lora is None
    assert ablation2.model.vla_head.lora['rank'] == 32
    # The ablations carry a bounded step budget (mutually exclusive with
    # epochs), while the ordinary recipes keep the epoch budget.
    for config in (ablation1, ablation2):
        assert config.runner.get('max_steps') == 2000
        assert config.runner.get('max_epochs') is None
        assert config.runner.save_iter_interval == 500
    assert load(STAGE1).runner.get('max_steps') is None
    assert load(STAGE1).runner.max_epochs == 10


def test_stage2_recipe_supplies_the_checkpoint_through_the_environment(
        monkeypatch):
    # The recipe reads PATCHWAM_STAGE1_CHECKPOINT; without it the model
    # constructor refuses to start (covered by the model tests), so the recipe
    # documents an empty value that the launcher fills in.
    monkeypatch.setenv('PATCHWAM_STAGE1_CHECKPOINT',
                       'work_dirs/stage1/step_1000.pth')
    configured = load(ABLATION_STAGE2)
    assert configured.model.stage1_checkpoint == \
        'work_dirs/stage1/step_1000.pth'
