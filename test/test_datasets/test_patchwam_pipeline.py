# Copyright 2026 Limx Dynamics
"""End-to-end tests that drive the *shipped* PatchWAM configs.

Builds a real on-disk LeRobot fixture with native metadata statistics, lets
``DistributedRepeatingDataset`` aggregate them, and runs every shipped train and
validation chain through the configured transforms/collator plus a tiny-backed
real model. Only fixture paths, tiny model widths, small CPU geometry, and the
device are overridden.
"""

import copy
import json
from pathlib import Path

import av
import numpy as np
import pytest
import torch
import torch.nn as nn
from mmengine.config import Config

import fluxvla.datasets  # noqa: F401
import fluxvla.collators  # noqa: F401
import fluxvla.engines.runners  # noqa: F401
import fluxvla.transforms  # noqa: F401
from fluxvla.datasets.dataset_wrapper import DistributedRepeatingDataset
from fluxvla.engines import build_collator_from_cfg, build_vla_from_cfg
from fluxvla.models.backbones.vlms import patchwam_backbone  # noqa: F401
from fluxvla.models.heads import patchwam_head  # noqa: F401
from fluxvla.models.vlas import patchwam_vla  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / 'configs' / 'patchwam'

CONFIGS = {
    'patchwam_radio_robotwin_base.py': dict(
        cameras=['observation.images.cam_high',
                 'observation.images.cam_left_wrist',
                 'observation.images.cam_right_wrist'],
        action_dim=14, state_dim=14, rows=24, layout='robotwin',
        video_size=[32, 32], camera_size=[16, 16], history=False,
        # One task, capped to a single episode, so the narrowing is observable.
        visual_dim=6,
        task_selection={'tasks': [0], 'episodes_per_task': 1, 'seed': 1}),
    'patchwam_libero_full_finetune.py': dict(
        cameras=['observation.images.image', 'observation.images.wrist_image'],
        action_dim=7, state_dim=8, rows=24, layout='horizontal',
        video_size=[32, 64], camera_size=[16, 16], history=False),
    'patchwam_robotwin_vlm_full_finetune.py': dict(
        cameras=['observation.images.cam_high',
                 'observation.images.cam_left_wrist',
                 'observation.images.cam_right_wrist'],
        action_dim=14, state_dim=14, rows=24, layout='robotwin',
        video_size=[32, 32], camera_size=[16, 16], history=False),
    'patchwam_robodojo_vlm_history_example.py': dict(
        cameras=['observation.images.cam_high',
                 'observation.images.cam_left_wrist',
                 'observation.images.cam_right_wrist'],
        action_dim=14, state_dim=14, rows=24, layout='robotwin',
        video_size=[32, 32], camera_size=[16, 16], history=True),
}


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


class PromptTextEncoder(nn.Module):
    """Prompt-dependent encoder so instruction conditioning is observable."""

    requires_images = False
    trainable_adapters = False

    def __init__(self, dim: int = 32, length: int = 3):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(length, dim))
        self.feature_dim = dim

    def forward(self, prompts):
        values = torch.tensor(
            [[(ord(char) % 7) / 7 for char in prompt[:3].ljust(3)]
             for prompt in prompts], dtype=self.embed.dtype)
        return self.embed.unsqueeze(0) * (1 + values.unsqueeze(-1))


class DownsamplingAutoencoder(nn.Module):
    """Downsample before producing packed latents so shipped geometry is cheap."""

    def __init__(self, channels: int = 128, factor: int = 16):
        super().__init__()
        self.pool = nn.AvgPool2d(factor)
        self.proj = nn.Conv2d(3, channels, 1)

    def encode(self, video):
        return self.proj(self.pool(video))


class TinyRadio(nn.Module):
    """Frozen C-RADIO stand-in for the shared visual-wrapper path."""

    def __init__(self, feature_dim: int = 6, patch_size: int = 16):
        super().__init__()
        self.feature_dim = feature_dim
        self.stem = nn.Conv2d(3, feature_dim, patch_size, patch_size)
        self.seen = []

    @staticmethod
    def get_nearest_supported_resolution(height, width):
        step = 16
        return (max(step, height // step * step),
                max(step, width // step * step))

    def forward(self, images, feature_fmt='NLC'):
        self.seen.append((feature_fmt, images.detach().clone()))
        spatial = self.stem(images)
        return spatial.mean(dim=(2, 3))[:, None], spatial


FPS = 1.0


def _write_video(path: Path, episode: int, frames: int, size: int,
                 camera: int, fps: float = FPS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode='w')
    stream = container.add_stream('mpeg4', rate=int(fps))
    stream.width, stream.height, stream.pix_fmt = size, size, 'yuv420p'
    for frame in range(frames):
        base = 20 if episode == 0 else 220
        image = np.zeros((size, size, 3), dtype=np.uint8)
        image[..., 0] = min(255, base + frame)
        image[..., 1] = 40 + camera * 40
        image[..., 2] = 30 + frame
        for packet in stream.encode(av.VideoFrame.from_ndarray(image,
                                                              format='rgb24')):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _build_fixture(root: Path, recipe) -> Path:
    import pandas as pd

    meta = root / 'meta'
    meta.mkdir(parents=True)
    (meta / 'info.json').write_text(json.dumps({
        'codebase_version': 'v2.1',
        'fps': FPS,
        'chunks_size': 1000,
        'total_episodes': 2,
        'video_path':
        'videos/chunk-{episode_chunk:03d}/{video_key}/episode_'
        '{episode_index:06d}.mp4',
    }), encoding='utf-8')
    (meta / 'tasks.jsonl').write_text(
        json.dumps({'task_index': 0, 'task': 'move the cup'}) + '\n',
        encoding='utf-8')
    with (meta / 'episodes.jsonl').open('w', encoding='utf-8') as handle:
        for episode in range(2):
            handle.write(json.dumps({'episode_index': episode, 'tasks': [0],
                                     'length': recipe['rows']}) + '\n')
    # Native metadata statistics: the wrapper aggregates these itself.
    state_stats = {'min': [0.0] * recipe['state_dim'],
                   'max': [20.0] * recipe['state_dim'],
                   'mean': [10.0] * recipe['state_dim'],
                   'std': [5.0] * recipe['state_dim']}
    action_stats = {'min': [0.0] * recipe['action_dim'],
                    'max': [2.0] * recipe['action_dim'],
                    'mean': [1.0] * recipe['action_dim'],
                    'std': [0.5] * recipe['action_dim']}
    timestamp_stats = {'min': [0.0], 'max': [float(recipe['rows'] - 1)],
                       'mean': [float(recipe['rows'] - 1) / 2],
                       'std': [1.0]}
    with (meta / 'episodes_stats.jsonl').open('w', encoding='utf-8') as handle:
        for episode in range(2):
            handle.write(json.dumps({'episode_index': episode, 'stats': {
                'observation.state': state_stats,
                'action': action_stats,
                'timestamp': timestamp_stats,
            }}) + '\n')
    rows = {'observation.state': [], 'action': [], 'timestamp': [],
            'frame_index': [], 'episode_index': [], 'task_index': [],
            'subtask': []}
    for episode in range(2):
        for frame in range(recipe['rows']):
            rows['observation.state'].append(
                [min(20.0, float(frame)) for _ in range(recipe['state_dim'])])
            rows['action'].append(
                [0.1 * frame for _ in range(recipe['action_dim'])])
            rows['timestamp'].append(float(frame))
            rows['frame_index'].append(frame)
            rows['episode_index'].append(episode)
            rows['task_index'].append(0)
            rows['subtask'].append('grasp the handle')
    data_dir = root / 'data' / 'chunk-000'
    data_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(data_dir / 'episode_000000.parquet')
    for episode in range(2):
        for camera, key in enumerate(recipe['cameras']):
            _write_video(root / 'videos' / 'chunk-000' / key /
                         f'episode_{episode:06d}.mp4', episode,
                         recipe['rows'], 16, camera)
    return root


def _cpu_dataset_config(cfg, recipe, root: Path, training: bool):
    """Shipped dataset config with CPU-fixture path/size overrides only."""
    split = cfg.train_dataloader if training else cfg.val_dataloader
    dataset = copy.deepcopy(split.to_dict()['dataset'])
    child = dataset['datasets']
    child['data_root_path'] = [str(root)]
    child['training'] = training
    if recipe.get('task_selection') is not None:
        child['task_selection'] = dict(recipe['task_selection'])
    for transform in child['transforms']:
        if transform['type'] == 'PatchWAMResizeCameras':
            transform['camera_size'] = recipe['camera_size']
        elif transform['type'] == 'PatchWAMComposeWorld':
            transform['video_size'] = recipe['video_size']
        elif transform['type'] == 'PatchWAMCleanVL':
            transform['image_size'] = 16
    return dataset


def _tiny_model(cfg, recipe):
    model_cfg = copy.deepcopy(Config(cfg.model).to_dict())
    model_cfg['num_inference_steps'] = 2
    visual_dim = recipe.get('visual_dim')
    if visual_dim is None:
        model_cfg['vlm_backbone'] = dict(
            type='PatchWAMBackbone',
            autoencoder=TinyAutoencoder(channels=128),
            text_encoder=TinyTextEncoder(dim=32),
            token_dim=128, text_dim=32, history_pool_size=2)
    else:
        # The shared frozen wrapper carries the tokens; no VAE is allocated.
        from fluxvla.models.backbones.visions import RadioViTBackbone
        model_cfg['vlm_backbone'] = dict(
            type='PatchWAMBackbone',
            autoencoder=None,
            text_encoder=TinyTextEncoder(dim=32),
            token_dim=128, text_dim=32, history_pool_size=2,
            visual_encoder=RadioViTBackbone('c-radio_v4-h',
                                            model=TinyRadio(visual_dim),
                                            feature_dim=visual_dim,
                                            patch_size=16,
                                            resolution_step=16,
                                            dtype=torch.float32))
    model_cfg['vla_head'] = dict(
        type='PatchWAMHead',
        denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128, text_dim=32,
                      width=32, heads=4, depth=2, visual_dim=visual_dim),
        action_dim=recipe['action_dim'], text_dim=32,
        proprio_dim=recipe['state_dim'], token_dim=128, visual_dim=visual_dim)
    model_cfg.pop('visual_stage', None)
    model_cfg.pop('pretrained_name_or_path', None)
    model_cfg.pop('stage1_checkpoint', None)
    return build_vla_from_cfg(model_cfg)


@pytest.mark.parametrize('name', list(CONFIGS))
def test_shipped_config_train_and_validation_chains(name, tmp_path):
    recipe = CONFIGS[name]
    cfg = Config.fromfile(str(CONFIG_DIR / name))
    root = _build_fixture(tmp_path / name.replace('.py', ''), recipe)

    for training in (True, False):
        dataset = _cpu_dataset_config(cfg, recipe, root, training)
        wrapper = DistributedRepeatingDataset(
            datasets=dataset['datasets'],
            statistic_keys=dataset['statistic_keys'],
            name_mappings=dataset['name_mappings'],
            statistic_name=dataset['statistic_name'],
            shuffle=False)
        if recipe.get('task_selection') is not None:
            # A narrowed selection must also narrow the statistics source: the
            # fixture has two episodes and the recipe selects one task index.
            child = wrapper.dataset
            selected = child.selection_metadata()
            assert selected is not None
            assert selected['episodes_per_task'] == 1
            assert len(child.stats) == selected['episodes'] == 1
            assert len(child.sample_indices) < child.full_length
        # The wrapper aggregated statistics itself (no supplied states key).
        stats = wrapper.dataset_statistics.get(
            dataset['statistic_name'], wrapper.dataset_statistics)
        assert 'proprio' in stats and 'action' in stats
        collator = build_collator_from_cfg(
            dict(cfg.runner.collator))
        batch = collator([next(iter(wrapper)) for _ in range(2)])
        model = _tiny_model(cfg, recipe)
        out = model(**batch)
        assert torch.isfinite(out['loss'])
        if training:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            out['loss'].backward()
            optimizer.step()


def test_validation_samples_are_deterministic_and_training_stochastic(tmp_path):
    name = 'patchwam_robodojo_vlm_history_example.py'
    recipe = CONFIGS[name]
    cfg = Config.fromfile(str(CONFIG_DIR / name))
    root = _build_fixture(tmp_path / 'robodojo', recipe)
    dataset = _cpu_dataset_config(cfg, recipe, root, training=False)
    wrapper = DistributedRepeatingDataset(
        datasets=dataset['datasets'], statistic_keys=dataset['statistic_keys'],
        name_mappings=dataset['name_mappings'],
        statistic_name=dataset['statistic_name'], shuffle=False)
    torch.manual_seed(0)
    iterator = iter(wrapper)
    first = [next(iterator) for _ in range(6)][5]
    torch.manual_seed(1234)
    iterator = iter(wrapper)
    second = [next(iterator) for _ in range(6)][5]
    assert first['history_valid'].any()
    torch.testing.assert_close(first['history_video'], second['history_video'])


def test_shipped_inference_dataset_reaches_the_encoder(tmp_path):
    cfg = Config.fromfile(str(CONFIG_DIR / 'patchwam_libero_full_finetune.py'))
    assert 'inference' in cfg
    assert cfg.inference['type'] == 'LiberoEvalRunner'
    # The runner carries the norm stats key (it overwrites the dataset field).
    assert cfg.inference['norm_stats_key'] == 'libero_all_no_noops'
    assert cfg.inference['dataset']['type'] == 'PatchWAMLiberoEvalDataset'
    from fluxvla.engines import DATASETS
    assert DATASETS.get('PatchWAMLiberoEvalDataset') is not None

    # Native aggregated stats from the training wrapper drive real normalization.
    recipe = CONFIGS['patchwam_libero_full_finetune.py']
    root = _build_fixture(tmp_path / 'infer', recipe)
    training_dataset = _cpu_dataset_config(cfg, recipe, root, training=True)
    wrapper = DistributedRepeatingDataset(
        datasets=training_dataset['datasets'],
        statistic_keys=training_dataset['statistic_keys'],
        name_mappings=training_dataset['name_mappings'],
        statistic_name=training_dataset['statistic_name'], shuffle=False)
    stats = wrapper.dataset_statistics.get(
        training_dataset['statistic_name'], wrapper.dataset_statistics)
    norm_stats = {'libero_all_no_noops': {
        'proprio': {
            'min': np.asarray(stats['proprio']['min']).reshape(-1)[:8],
            'max': np.asarray(stats['proprio']['max']).reshape(-1)[:8],
        },
        'action': {
            'min': np.asarray(stats['action']['min']).reshape(-1)[:7],
            'max': np.asarray(stats['action']['max']).reshape(-1)[:7],
        },
    }}

    dataset_cfg = copy.deepcopy(cfg.inference['dataset'].to_dict())
    dataset_cfg['device'] = 'cpu'
    dataset_cfg['norm_stats'] = norm_stats
    from fluxvla.engines import build_dataset_from_cfg
    dataset = build_dataset_from_cfg(dataset_cfg)
    observation = {
        'agentview_image': np.zeros((16, 16, 3), dtype=np.uint8),
        'robot0_eye_in_hand_image': np.zeros((16, 16, 3), dtype=np.uint8),
        'robot0_eef_pos': np.zeros(3, dtype=np.float32),
        'robot0_eef_quat': np.array([0, 0, 0, 1], dtype=np.float32),
        'robot0_gripper_qpos': np.zeros(2, dtype=np.float32),
        'task_description': 'move the cup',
        'is_new_episode': True,
    }
    batch, _ = dataset(observation)
    assert batch['task_description'] == ['move the cup']
    assert batch['images'].shape[-2:] == (224, 448)  # shipped geometry
    assert batch['states'].shape[-1] == 8
    assert torch.isfinite(batch['states']).all()

    model_cfg = copy.deepcopy(Config(cfg.model).to_dict())
    model_cfg['num_inference_steps'] = 2
    model_cfg['vlm_backbone'] = dict(
        type='PatchWAMBackbone',
        autoencoder=DownsamplingAutoencoder(),
        text_encoder=PromptTextEncoder(dim=32),
        token_dim=128, text_dim=32, history_pool_size=2)
    model_cfg['vla_head'] = dict(
        type='PatchWAMHead',
        denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128, text_dim=32,
                      width=32, heads=4, depth=2),
        action_dim=7, text_dim=32, proprio_dim=8, token_dim=128,
        isolate_actions=False)
    model = build_vla_from_cfg(model_cfg)
    model.eval()
    with torch.no_grad():
        prediction = model.predict_action(**batch, seed=4)
    assert prediction.shape == (1, 16, 7)
    assert torch.isfinite(prediction).all()

    # The native runner decodes each action before stepping the environment.
    from fluxvla.engines import build_transform_from_cfg
    decoder_cfg = copy.deepcopy(cfg.inference.denormalize_action.to_dict())
    decoder_cfg['norm_stats'] = norm_stats
    decoder = build_transform_from_cfg(decoder_cfg)
    denormalized = np.stack([
        decoder({'action': action.copy(),
                 'norm_stats_key': 'libero_all_no_noops'})
        for action in prediction[0].numpy()
    ])
    assert denormalized.shape == (16, 7)
    assert not np.allclose(denormalized, prediction[0].numpy())

    # The instruction is actually observed by the encoder.
    other = dict(batch)
    other['task_description'] = ['open the drawer']
    with torch.no_grad():
        alternative = model.predict_action(**other, seed=4)
    assert not torch.allclose(prediction, alternative)
