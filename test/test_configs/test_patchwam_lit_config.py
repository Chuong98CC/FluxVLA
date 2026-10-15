# Copyright 2026 Limx Dynamics
"""CPU units validating the shipped PatchWAM + LIT configs and pipeline."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from mmengine.config import Config

import fluxvla.collators  # noqa: F401
import fluxvla.datasets  # noqa: F401
import fluxvla.engines.runners  # noqa: F401
import fluxvla.transforms  # noqa: F401
from fluxvla.datasets.dataset_wrapper import DistributedRepeatingDataset
from fluxvla.engines import (DATASETS, HEADS, TRANSFORMS, VLAS,
                             build_collator_from_cfg, build_vla_from_cfg)
from fluxvla.engines import RUNNERS
from fluxvla.models.backbones.vlms import patchwam_backbone  # noqa: F401
from fluxvla.models.heads import patchwam_head, patchwam_lit_head  # noqa: F401
from fluxvla.models.vlas import patchwam_lit_vla  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / 'configs' / 'patchwam'

STAGE1 = 'patchwam_lit_stage1_robotwin.py'
STAGE2 = 'patchwam_lit_stage2_robotwin.py'
EVAL = 'patchwam_lit_eval_robotwin.py'
BENCHMARK = 'patchwam_lit_official_benchmark.py'


def _load(name):
    return Config.fromfile(str(CONFIG_DIR / name))


@pytest.mark.parametrize('name', [STAGE1, STAGE2, EVAL, BENCHMARK])
def test_registries_resolve_and_assets_are_explicit(name):
    cfg = _load(name)
    model = cfg.model
    assert VLAS.get(model.type) is not None
    assert HEADS.get(model.lit_head.type) is not None
    assert HEADS.get(model.lit_head.action_expert.type) is not None
    assert HEADS.get(model.vla_head.type) is not None
    assert model.vla_head.denoiser.type == 'PatchWAMFluxDenoiser'
    assert model.lit_head.context_dim == 3072
    assert model.lit_head.prefix_input_dim == 7680
    assert model.lit_head.action_dim == 14
    assert model.lit_head.pose_loss_weight == 0.3
    assert model.lit_head.fm_action_weight == 1.0
    if model.lit_head.get('num_latents') is not None:
        assert model.lit_head.num_latents == 100
        assert model.lit_head.num_pose_tokens == 8
        assert model.lit_head.num_layer_groups == 5
    assert model.lit_head.action_expert.num_layers == 25
    assert model.lit_head.action_expert.hidden_dim == 1024
    assert model.lit_head.action_expert.ffn_dim == 4096
    assert model.lit_head.action_expert.num_heads == 16
    assert model.lit_head.action_expert.attn_head_dim == 64
    assert model.lit_head.action_expert.text_dim == 3072
    assert model.vla_head.denoiser.transformer_path.endswith('.safetensors')
    assert model.vlm_backbone.autoencoder_path
    assert model.vlm_backbone.text_encoder_path
    assert model.vlm_backbone.flux_source


def test_stage_specific_config_placement():
    stage1 = _load(STAGE1).model
    assert stage1.lit_stage == 'stage1'
    assert stage1.training_mode == 'lit'
    assert stage1.active_head == 'lit'
    assert stage1.lit_head.stage == 'stage1'
    assert stage1.vlm_backbone.text_encoder_kind == 'qwen3'
    # Stage 1 initializes the shared FLUX/proprio from the released PatchWAM
    # checkpoint instead of a frozen random projection.
    assert stage1.pretrained_name_or_path.endswith('.pt')
    assert stage1.vla_head.pack_proprio_after_text is True
    stage2 = _load(STAGE2).model
    assert stage2.lit_stage == 'stage2'
    assert stage2.training_mode == 'joint'
    assert stage2.patchwam_loss_weight == 1.0
    assert stage2.lit_loss_weight == 1.0
    assert stage2.pretrained_name_or_path.endswith('.pt')
    assert stage2.vla_head.pack_proprio_after_text is True
    assert stage2.vla_head.video_weight == 0.5
    assert stage2.vla_head.action_weight == 1.0
    assert stage2.lit_head.stage == 'stage2'
    # The bridge is model-owned and wired through from_pretrained; the recipe
    # exposes the env-overridable path and requires it by default.
    assert 'lit_stage1_checkpoint' in stage2
    assert stage2.lit_stage1_required is True
    eval_cfg = _load(EVAL)
    assert eval_cfg.model.active_head == 'lit'
    assert eval_cfg.eval.type == 'RobotwinEvalRunner'
    # The released protocol executes the full trained horizon per prediction.
    assert eval_cfg.eval.eval_chunk_size == 16
    assert eval_cfg.eval.dataset.type == 'PatchWAMRoboTwinEvalDataset'
    assert str(eval_cfg.model.pretrained_name_or_path).endswith(
        'latest-checkpoint.pt')
    # Trained LIT weights normalize with the Stage-2 training statistics.
    assert eval_cfg.eval.norm_stats_path.endswith('dataset_statistics.json')
    assert eval_cfg.eval.unnorm_key == 'robotwin'


def test_stage2_transform_order_and_goal_keys():
    cfg = _load(STAGE2)
    dataset = cfg.train_dataloader.dataset.datasets
    assert DATASETS.get(dataset.type) is not None
    assert dataset.goal_offset == 16
    assert dataset.goal_state_key == 'observation.state'
    types = [transform['type'] for transform in dataset.transforms]
    for name in types:
        assert TRANSFORMS.get(name) is not None
    assert types.index('PatchWAMCleanVL') < types.index('PatchWAMWorldAugment')
    assert (types.index('NormalizeStatesAndActions')
            < types.index('PatchWAMLITGoal'))
    assert (types.index('PatchWAMLITGoal')
            < types.index('PatchWAMWorldAugment'))
    goal = next(transform for transform in dataset.transforms
                if transform['type'] == 'PatchWAMLITGoal')
    assert goal['normalization_epsilon'] == 1e-8
    state_norm = next(transform for transform in dataset.transforms
                      if transform['type'] == 'NormalizeStatesAndActions')
    assert state_norm['normalization_epsilon'] == 1e-8
    keep = next(transform for transform in dataset.transforms
                if transform['type'] == 'PatchWAMKeep')
    assert 'goal_state' in keep['keys'] and 'goal_valid' in keep['keys']
    collator = cfg.runner.collator
    assert 'goal_state' in collator['keys']
    assert 'goal_valid' in collator['keys']
    train_augment = next(transform for transform in dataset.transforms
                         if transform['type'] == 'PatchWAMWorldAugment')
    val_transforms = cfg.val_dataloader.dataset.datasets.transforms
    val_augment = next(transform for transform in val_transforms
                       if transform['type'] == 'PatchWAMWorldAugment')
    assert train_augment['enabled'] is True
    assert val_augment['enabled'] is False
    assert cfg.train_dataloader is not cfg.val_dataloader


def test_stage1_config_is_image_free():
    cfg = _load(STAGE1)
    assert cfg.runner.type == 'DDPTrainRunner'
    assert RUNNERS.get(cfg.runner.type) is not None
    dataset = cfg.train_dataloader.dataset.datasets
    types = [transform['type'] for transform in dataset.transforms]
    assert 'PatchWAMDecodeFrames' not in types
    assert 'PatchWAMComposeWorld' not in types
    assert 'PatchWAMWorldAugment' not in types
    assert 'PatchWAMLITGoal' in types
    assert (types.index('NormalizeStatesAndActions')
            < types.index('PatchWAMLITGoal'))
    goal = next(transform for transform in dataset.transforms
                if transform['type'] == 'PatchWAMLITGoal')
    assert goal['normalization_epsilon'] == 1e-8


def test_lit_eval_uses_the_stage2_training_statistics(monkeypatch):
    monkeypatch.delenv('PATCHWAM_EPISODES', raising=False)
    cfg = _load(EVAL)
    stats_path = str(cfg.eval.norm_stats_path)
    assert stats_path.endswith('dataset_statistics.json')
    assert 'patchwam_lit_stage2_robotwin' in stats_path
    # Single-task integration smoke by default; the benchmark config below is
    # the official protocol.
    assert cfg.eval.num_trials_per_task == 1
    assert cfg.eval.task_suite_name == 'clean'
    benchmark = _load(BENCHMARK)
    assert benchmark.eval.num_trials_per_task == 100


def test_lit_official_benchmark_mirrors_the_released_protocol(monkeypatch):
    monkeypatch.delenv('PATCHWAM_EPISODES', raising=False)
    lit = _load(BENCHMARK)
    reference = _load('patchwam_robotwin_official_benchmark.py')
    for key in ('type', 'seed', 'task_suite_name', 'task_list',
                'num_trials_per_task', 'instruction_type', 'eval_chunk_size'):
        assert lit.eval[key] == reference.eval[key], key
    assert lit.eval.seed == 0
    assert lit.eval.num_trials_per_task == 100
    assert list(lit.eval.task_suite_name) == ['clean', 'random']
    assert len(list(lit.eval.task_list)) == 50
    # LIT-specific difference: trained-weights normalization.
    assert 'patchwam_lit_stage2_robotwin' in str(lit.eval.norm_stats_path)
    assert 'patchwam_lit_robotwin_official' in lit.eval.result_output_dir
    assert lit.model.lit_head.num_latents == 100


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


def _tiny_lit_model(cfg):
    model_cfg = copy.deepcopy(Config(cfg.model).to_dict())
    model_cfg.pop('pretrained_name_or_path', None)
    model_cfg['num_inference_steps'] = 2
    model_cfg['vlm_backbone'] = dict(
        type='PatchWAMBackbone', autoencoder=TinyAutoencoder(),
        text_encoder=TinyTextEncoder(), token_dim=128, text_dim=32,
        history_pool_size=2)
    model_cfg['vla_head'] = dict(
        type='PatchWAMHead',
        denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128, text_dim=32,
                      width=32, heads=4, depth=2),
        action_dim=14, text_dim=32, proprio_dim=14, token_dim=128)
    model_cfg['lit_head'] = dict(
        type='PatchWAMLITFlowMatchingHead', stage=model_cfg['lit_stage'],
        action_dim=14, context_dim=32, prefix_input_dim=32, proprio_dim=14,
        num_latents=6, num_pose_tokens=2, num_layer_groups=2, latent_dim=16,
        inner_dim=16, num_heads=2, pose_loss_weight=0.3, fm_action_weight=1.0,
        action_horizon=16,
        action_expert=dict(
            type='PatchWAMLITActionDiT', hidden_dim=32, action_dim=14,
            ffn_dim=64, text_dim=32, freq_dim=256, eps=1e-6, num_heads=4,
            attn_head_dim=8, num_layers=2))
    return build_vla_from_cfg(model_cfg)


def test_optimizer_visible_trainable_sets_per_stage():
    stage1 = _tiny_lit_model(_load(STAGE1))
    stage1.apply_lit_stage_policy()
    trainable = {name for name, param in stage1.named_parameters()
                 if param.requires_grad}
    assert any(name.startswith('lit_head.goal_pose_encoder.')
               for name in trainable)
    assert any(name.startswith('lit_head.action_expert.') for name in trainable)
    assert not any(name.startswith('vla_head.denoiser.') for name in trainable)
    assert not any(name.startswith('vlm_backbone.') for name in trainable)
    stage2 = _tiny_lit_model(_load(STAGE2))
    stage2.apply_lit_stage_policy()
    trainable = {name for name, param in stage2.named_parameters()
                 if param.requires_grad}
    assert any(name.startswith('vla_head.denoiser.') for name in trainable)
    assert any(name.startswith('vla_head.state_projection.')
               for name in trainable)
    assert any(name.startswith('lit_head.semantic_visual_aggregator.')
               for name in trainable)
    assert any(name.startswith('lit_head.semantic_visual_pose_decoder.')
               for name in trainable)
    assert any(name.startswith('lit_head.action_expert.') for name in trainable)
    assert not any(name.startswith('vlm_backbone.') for name in trainable)
    assert stage1.all_module_keys == ['vlm_backbone', 'vla_head', 'lit_head']


FPS = 2.0
STATE_MEAN, STATE_STD = 10.0, 5.0
ACTION_MEAN, ACTION_STD = 1.0, 0.5


def _build_parquet_fixture(root: Path, rows: int = 24, episodes: int = 2):
    import pandas as pd

    meta = root / 'meta'
    meta.mkdir(parents=True)
    (meta / 'info.json').write_text(json.dumps({
        'codebase_version': 'v2.1', 'fps': FPS, 'chunks_size': 1000,
        'total_episodes': episodes,
        'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_'
                      '{episode_index:06d}.mp4',
    }), encoding='utf-8')
    (meta / 'tasks.jsonl').write_text(
        json.dumps({'task_index': 0, 'task': 'move the cup'}) + '\n',
        encoding='utf-8')
    with (meta / 'episodes.jsonl').open('w', encoding='utf-8') as handle:
        for episode in range(episodes):
            handle.write(json.dumps({'episode_index': episode, 'tasks': [0],
                                     'length': rows}) + '\n')
    state_stats = {'min': [0.0] * 14, 'max': [20.0] * 14,
                   'mean': [STATE_MEAN] * 14, 'std': [STATE_STD] * 14}
    action_stats = {'min': [0.0] * 14, 'max': [2.0] * 14,
                    'mean': [ACTION_MEAN] * 14, 'std': [ACTION_STD] * 14}
    timestamp_stats = {'min': [0.0], 'max': [float(rows - 1)],
                       'mean': [float(rows - 1) / 2], 'std': [1.0]}
    with (meta / 'episodes_stats.jsonl').open('w', encoding='utf-8') as handle:
        for episode in range(episodes):
            handle.write(json.dumps({'episode_index': episode, 'stats': {
                'observation.state': state_stats, 'action': action_stats,
                'timestamp': timestamp_stats}}) + '\n')
    records = {'observation.state': [], 'action': [], 'timestamp': [],
               'frame_index': [], 'episode_index': [], 'task_index': [],
               'subtask': []}
    for episode in range(episodes):
        for frame in range(rows):
            records['observation.state'].append([float(frame)] * 14)
            records['action'].append([0.1 * frame] * 14)
            records['timestamp'].append(float(frame) / FPS)
            records['frame_index'].append(frame)
            records['episode_index'].append(episode)
            records['task_index'].append(0)
            records['subtask'].append('grasp the handle')
    data_dir = root / 'data' / 'chunk-000'
    data_dir.mkdir(parents=True)
    pd.DataFrame(records).to_parquet(data_dir / 'episode_000000.parquet')
    return root


def _stage1_cpu_dataset_config(cfg, root: Path, training: bool):
    split = cfg.train_dataloader if training else cfg.val_dataloader
    dataset = copy.deepcopy(split.to_dict()['dataset'])
    dataset['datasets']['data_root_path'] = [str(root)]
    dataset['datasets']['training'] = training
    return dataset


def test_stage1_parquet_fixture_end_to_end(tmp_path):
    cfg = _load(STAGE1)
    root = _build_parquet_fixture(tmp_path / 'stage1')
    dataset = _stage1_cpu_dataset_config(cfg, root, training=True)
    wrapper = DistributedRepeatingDataset(
        datasets=dataset['datasets'],
        statistic_keys=dataset['statistic_keys'],
        name_mappings=dataset['name_mappings'],
        statistic_name=dataset['statistic_name'], shuffle=False)
    stats = wrapper.dataset_statistics.get(dataset['statistic_name'],
                                           wrapper.dataset_statistics)
    assert 'proprio' in stats and 'action' in stats
    # Distinct state and action statistics make the normalization source
    # observable in the produced goal.
    assert stats['proprio']['mean'][0] != stats['action']['mean'][0]
    collator = build_collator_from_cfg(dict(cfg.runner.collator))
    samples = [next(iter(wrapper)) for _ in range(2)]
    batch = collator(samples)
    assert batch['goal_state'].shape == (2, 14)
    assert batch['goal_valid'].shape == (2,)
    assert bool(batch['goal_valid'].all())
    frame = 16
    np.testing.assert_allclose(
        batch['goal_state'][0].numpy(),
        [(float(frame) - STATE_MEAN) / STATE_STD] * 14, rtol=1e-5)
    assert not np.allclose(
        batch['goal_state'][0].numpy(),
        [(float(frame) - ACTION_MEAN) / ACTION_STD] * 14)
    model = _tiny_lit_model(cfg)
    out = model(**batch)
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    assert model.lit_head.goal_pose_encoder.net[0].weight.grad.abs().sum() > 0


def test_stage2_parquet_fixture_marks_tail_goals_invalid(tmp_path):
    cfg = _load(STAGE2)
    root = _build_parquet_fixture(tmp_path / 'stage2', rows=20)
    dataset = _stage1_cpu_dataset_config(cfg, root, training=True)
    dataset['datasets']['transforms'] = [
        dict(type='PatchWAMRawInputs'),
        dict(type='NormalizeStatesAndActions', state_key='proprio',
             action_key='action', action_dim=14, state_dim=14,
             norm_type='mean_std', statistics_key='stats',
             normalization_epsilon=1e-8),
        dict(type='PatchWAMLITGoal', state_key='proprio', state_dim=14,
             norm_type='mean_std', statistics_key='stats'),
        dict(type='PatchWAMKeep', keys=['actions', 'action_masks', 'states',
                                        'goal_state', 'goal_valid',
                                        'task_description'], require=False),
    ]
    wrapper = DistributedRepeatingDataset(
        datasets=dataset['datasets'],
        statistic_keys=dataset['statistic_keys'],
        name_mappings=dataset['name_mappings'],
        statistic_name=dataset['statistic_name'], shuffle=False)
    collator = build_collator_from_cfg(dict(cfg.runner.collator))
    batch = collator([next(iter(wrapper)) for _ in range(3)])
    assert batch['goal_state'].shape == (3, 14)
    # Every emitted window must be self-consistent about its goal validity.
    assert set(batch['goal_valid'].tolist()) <= {True, False}
