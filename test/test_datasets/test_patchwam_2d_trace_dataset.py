# Copyright 2026 Limx Dynamics
"""CPU units for the offline PatchWAM 2D patch-motion trace dataset."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import fluxvla.datasets  # noqa: F401
from fluxvla.collators.dict_collator import DictCollator
from fluxvla.datasets.patchwam_2d_trace_dataset import (
    PatchWAM2DTraceDataset)
from fluxvla.engines import DATASETS, build_vla_from_cfg
from fluxvla.models.backbones.vlms import patchwam_backbone  # noqa: F401
from fluxvla.models.heads import patchwam_2d_motion_head  # noqa: F401
from fluxvla.models.vlas import patchwam_2d_motion_vla  # noqa: F401

HEIGHT = 64
WIDTH = 64
HORIZON = 3
OFFSETS = (0.25, 0.5, 1.0)
SCHEMA = 'patchwam_2d_trace_v1'
COLLATOR = dict(keys=['images', 'motion_2d', 'motion_valid'],
                meta_keys=['task_description', 'sample_id', 'episode_id'])


def patch_centers(height=HEIGHT, width=WIDTH):
    rows, columns = height // 16, width // 16
    grid_row, grid_column = np.meshgrid(np.arange(rows), np.arange(columns),
                                        indexing='ij')
    centers = np.stack((grid_row * 16 + 7.5, grid_column * 16 + 7.5), -1)
    return centers.astype(np.float32)


def trace_arrays(seed, horizon=HORIZON, height=HEIGHT, width=WIDTH):
    generator = np.random.default_rng(seed)
    centers = patch_centers(height, width)
    steps = generator.normal(scale=4.0, size=(horizon, *centers.shape))
    cumulative = np.cumsum(steps, axis=0)
    traces = np.concatenate((np.zeros((1, *centers.shape), np.float32),
                             cumulative), 0) + centers
    valid = np.ones((horizon + 1, height // 16, width // 16), dtype=bool)
    times = np.asarray([0.0, *OFFSETS], dtype=np.float64)
    return (traces.astype(np.float32), valid, times)


def valid_manifest(**overrides):
    manifest = dict(schema_version=SCHEMA, image_height=HEIGHT,
                    image_width=WIDTH, patch_size=16, motion_block_size=8,
                    horizon=HORIZON, coordinate_order='row_column',
                    timestep_offsets_s=list(OFFSETS))
    manifest.update(overrides)
    return manifest


def make_row(sample_id, episode_id, frame, instruction='move the block'):
    return dict(sample_id=sample_id, episode_id=episode_id,
                instruction=instruction,
                current_rgb=f'rgb/{episode_id}/{frame}.png',
                future_rgb=f'rgb/{episode_id}/{frame + 1}.png',
                labels=f'labels/{sample_id}.npz')


def write_raw(root: Path, manifest, rows, *, labels=None, images=None,
              val_rows=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'dataset.json').write_text(json.dumps(manifest))
    (root / 'train.jsonl').write_text(
        '\n'.join(json.dumps(row) for row in rows) + '\n')
    if val_rows is not None:
        (root / 'val.jsonl').write_text(
            '\n'.join(json.dumps(row) for row in val_rows) + '\n')
    for sample_id, payload in (labels or {}).items():
        path = root / 'labels' / f'{sample_id}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **payload)
    for relative, array in (images or {}).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)
    return root


def valid_dataset(root: Path, *, manifest=None, rows=None, val_rows=None,
                  labels=None, images=None):
    """A minimal well-formed dataset: two train samples plus one val sample."""
    generator = np.random.default_rng(0)
    if rows is None:
        rows = [make_row('train0', 'ep0', 0), make_row('train1', 'ep0', 1)]
    if val_rows is None:
        val_rows = [make_row('val0', 'ep1', 0)]
    if labels is None:
        labels = {}
        for index, row in enumerate(rows + val_rows):
            traces, valid, times = trace_arrays(index)
            labels[row['sample_id']] = dict(traces_2d=traces,
                                            trace_valid=valid,
                                            trace_t_seconds=times)
    if images is None:
        images = {
            f'rgb/{episode}/{frame}.png':
            generator.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
            for episode in ('ep0', 'ep1') for frame in (0, 1, 2)}
    return write_raw(root, valid_manifest() if manifest is None else manifest,
                     rows, labels=labels, images=images, val_rows=val_rows)


def build(root, **kwargs):
    return PatchWAM2DTraceDataset(dataset_root=str(root), **kwargs)


# ----------------------------------------------------------------------
# Manifest, geometry and paths
# ----------------------------------------------------------------------
def test_valid_dataset_loads_with_registered_type_and_geometry(tmp_path):
    root = valid_dataset(tmp_path / 'data')
    dataset = build(root)
    assert isinstance(dataset, PatchWAM2DTraceDataset)
    assert DATASETS.get('PatchWAM2DTraceDataset') is not None
    assert len(dataset) == 2
    assert dataset.sample_ids == ['train0', 'train1']
    assert dataset.horizon == HORIZON
    assert dataset.codec.grid_shape == (4, 4)
    assert dataset.codec.blocks_per_timestep == 1
    assert dataset.timestep_offsets_s == OFFSETS
    # Constructor values must agree with the manifest, never silently override.
    assert build(root, horizon=HORIZON).horizon == HORIZON
    with pytest.raises(ValueError, match='horizon'):
        build(root, horizon=4)
    with pytest.raises(ValueError, match='patch_size|geometry'):
        build(root, patch_size=8)
    with pytest.raises(ValueError, match='coordinate_order'):
        build(root, coordinate_order='column_row')
    assert build(root, split='val').sample_ids == ['val0']


@pytest.mark.parametrize('overrides, match', [
    (dict(schema_version='wrong'), 'schema_version'),
    (dict(patch_size=8), 'patch_size'),
    (dict(motion_block_size=4), 'motion_block_size'),
    (dict(coordinate_order='column_row'), 'coordinate_order'),
    (dict(image_height=250), 'divisible|image_height'),
    (dict(image_width=100), 'divisible|image_width'),
    (dict(horizon=4), 'horizon'),
    (dict(timestep_offsets_s=[0.25, 0.25, 1.0]), 'increasing'),
    (dict(timestep_offsets_s=[0.5, 0.25, 1.0]), 'increasing'),
    (dict(timestep_offsets_s=[0.25, 0.5, 0.0]), 'positive|increasing'),
    (dict(timestep_offsets_s=[0.25, 0.5]), 'horizon'),
])
def test_malformed_manifests_are_rejected(tmp_path, overrides, match):
    root = valid_dataset(tmp_path / 'data',
                         manifest=valid_manifest(**overrides))
    with pytest.raises(ValueError, match=match):
        build(root)


def test_unsafe_and_out_of_root_paths_are_rejected(tmp_path):
    root = valid_dataset(tmp_path / 'data')
    original = (root / 'train.jsonl').read_text().splitlines()
    for field, value, match in (
            ('labels', '/etc/passwd', 'relative|root'),
            ('labels', '../../escape.npz', 'root'),
            ('current_rgb', '../escape.png', 'root'),
            ('future_rgb', '/tmp/escape.png', 'relative|root')):
        broken = json.loads(original[0])
        broken[field] = value
        (root / 'train.jsonl').write_text(
            json.dumps(broken) + '\n' + '\n'.join(original[1:]) + '\n')
        with pytest.raises(ValueError, match=match):
            build(root)
        (root / 'train.jsonl').write_text('\n'.join(original) + '\n')


def test_duplicate_sample_ids_and_split_leakage_are_rejected(tmp_path):
    root = valid_dataset(
        tmp_path / 'data',
        rows=[make_row('train0', 'ep0', 0), make_row('train0', 'ep0', 1)])
    with pytest.raises(ValueError, match='duplicate'):
        build(root)
    leaking = valid_dataset(
        tmp_path / 'leak', rows=[make_row('train0', 'ep0', 0)],
        val_rows=[make_row('val0', 'ep0', 1)])
    with pytest.raises(ValueError, match='episode'):
        build(leaking)
    assert build(leaking, check_split_disjoint=False).sample_ids == ['train0']


def test_missing_label_and_image_files_fail_loudly(tmp_path):
    root = valid_dataset(tmp_path / 'data')
    (root / 'labels/train0.npz').unlink()
    dataset = build(root)
    with pytest.raises(FileNotFoundError, match='train0'):
        dataset[0]
    labels, valid, times = trace_arrays(0)
    np.savez(root / 'labels/train0.npz', traces_2d=labels,
             trace_valid=valid, trace_t_seconds=times)
    (root / 'rgb/ep0/1.png').unlink()
    with pytest.raises(FileNotFoundError, match='future_rgb'):
        dataset[0]


# ----------------------------------------------------------------------
# Sample contents
# ----------------------------------------------------------------------
def test_sample_returns_native_pixel_inputs_and_shared_validity(tmp_path):
    sample = build(valid_dataset(tmp_path / 'data'))[0]
    assert set(sample) == {'images', 'motion_2d', 'motion_valid',
                           'task_description', 'sample_id', 'episode_id'}
    assert sample['images'].shape == (1, 3, 2, HEIGHT, WIDTH)
    assert sample['images'].dtype == np.float32
    assert sample['images'].min() >= -1.0 and sample['images'].max() <= 1.0
    assert sample['motion_2d'].shape == (HORIZON, 4, 4, 2)
    assert sample['motion_2d'].dtype == np.float32
    assert sample['motion_valid'].shape == (HORIZON, 4, 4)
    assert sample['motion_valid'].dtype == np.bool_
    assert bool(sample['motion_valid'].all())
    assert sample['task_description'] == 'move the block'
    assert sample['sample_id'] == 'train0'
    assert sample['episode_id'] == 'ep0'
    # Motion is the consecutive signed pixel displacement of the trace.
    traces, _, _ = trace_arrays(0)
    np.testing.assert_allclose(sample['motion_2d'],
                               traces[1:] - traces[:-1], rtol=1e-5)
    assert not np.allclose(sample['motion_2d'], 0.0)
    # No robot state/action keys are produced or required.
    assert 'actions' not in sample and 'states' not in sample


def test_missing_intermediate_trace_masks_both_adjacent_displacements(tmp_path):
    root = valid_dataset(tmp_path / 'data')
    traces, valid, times = trace_arrays(0)
    valid[1] = False
    traces[1] = np.nan
    np.savez(root / 'labels/train0.npz', traces_2d=traces, trace_valid=valid,
             trace_t_seconds=times)
    sample = build(root)[0]
    assert not sample['motion_valid'][0].any()
    assert not sample['motion_valid'][1].any()
    assert sample['motion_valid'][2].all()
    assert np.isfinite(sample['motion_2d']).all()
    assert np.allclose(sample['motion_2d'][:2], 0.0)


def test_trace_zero_must_match_the_patch_centers(tmp_path):
    root = valid_dataset(tmp_path / 'data')
    traces, valid, times = trace_arrays(0)
    shifted = traces.copy()
    shifted[0, 0, 0, 0] += 0.5
    np.savez(root / 'labels/train0.npz', traces_2d=shifted,
             trace_valid=valid, trace_t_seconds=times)
    with pytest.raises(ValueError, match='centers|anchored'):
        build(root)[0]
    aligned = traces.copy()
    aligned[0, 0, 0, 0] += 0.0005
    np.savez(root / 'labels/train0.npz', traces_2d=aligned,
             trace_valid=valid, trace_t_seconds=times)
    assert build(root)[0]['motion_2d'].shape == (HORIZON, 4, 4, 2)


def test_nonfinite_valid_positions_are_rejected(tmp_path):
    root = valid_dataset(tmp_path / 'data')
    traces, valid, times = trace_arrays(0)
    traces[2, 1, 1, 0] = np.inf
    np.savez(root / 'labels/train0.npz', traces_2d=traces,
             trace_valid=valid, trace_t_seconds=times)
    with pytest.raises(ValueError, match='finite'):
        build(root)[0]


@pytest.mark.parametrize('slot', [0, 2])
def test_nonfinite_timestamps_are_rejected(tmp_path, slot):
    # A NaN at index 0 previously slipped past the zero check (NaN comparisons
    # are false); every timestamp must be finite before zero/schedule checks.
    root = valid_dataset(tmp_path / 'data')
    traces, valid, times = trace_arrays(0)
    broken = times.copy()
    broken[slot] = np.nan
    np.savez(root / 'labels/train0.npz', traces_2d=traces, trace_valid=valid,
             trace_t_seconds=broken)
    with pytest.raises(ValueError, match='finite'):
        build(root)[0]


@pytest.mark.parametrize('key, dtype, match', [
    ('traces_2d', np.float64, 'float32'),
    ('trace_t_seconds', np.float32, 'float64'),
])
def test_declared_dtypes_are_enforced(tmp_path, key, dtype, match):
    root = valid_dataset(tmp_path / 'data')
    traces, valid, times = trace_arrays(0)
    payload = dict(traces_2d=traces, trace_valid=valid,
                   trace_t_seconds=times)
    payload[key] = np.asarray(payload[key], dtype=dtype)
    np.savez(root / 'labels/train0.npz', **payload)
    with pytest.raises(ValueError, match=match):
        build(root)[0]


def test_valid_displacement_overflow_is_rejected(tmp_path):
    # Finite float32 endpoints can still overflow to infinity when subtracted.
    root = valid_dataset(tmp_path / 'data')
    traces, valid, times = trace_arrays(0)
    traces[1, 1, 1, 0] = 3.4e38
    traces[2, 1, 1, 0] = -3.4e38
    np.savez(root / 'labels/train0.npz', traces_2d=traces, trace_valid=valid,
             trace_t_seconds=times)
    with pytest.raises(ValueError, match='finite|overflow'):
        build(root)[0]


def test_sample_ids_must_be_unique_across_splits(tmp_path):
    root = valid_dataset(tmp_path / 'data',
                         rows=[make_row('shared', 'ep0', 0)],
                         val_rows=[make_row('shared', 'ep1', 0)])
    with pytest.raises(ValueError, match='duplicate|unique'):
        build(root)
    # Distinct episodes are not enough: the id collision is always an error.
    with pytest.raises(ValueError, match='duplicate|unique'):
        build(root, check_split_disjoint=False)


@pytest.mark.parametrize('mutation, match', [
    ('missing_key', 'trace_valid'),
    ('bad_trace_shape', r'shape|\[T\+1,h,w,2\]'),
    ('bad_valid_dtype', 'bool'),
    ('shifted_times', 'timestep_offsets_s'),
    ('zero_times', 'timestep_offsets_s'),
    ('bad_coordinate_order', 'coordinate_order'),
])
def test_malformed_npz_payloads_are_rejected(tmp_path, mutation, match):
    root = valid_dataset(tmp_path / 'data')
    traces, valid, times = trace_arrays(0)
    payload = dict(traces_2d=traces, trace_valid=valid,
                   trace_t_seconds=times)
    if mutation == 'missing_key':
        payload.pop('trace_valid')
    elif mutation == 'bad_trace_shape':
        payload['traces_2d'] = traces[:, :3]
    elif mutation == 'bad_valid_dtype':
        payload['trace_valid'] = valid.astype(np.float32)
    elif mutation == 'shifted_times':
        payload['trace_t_seconds'] = times + 0.1
    elif mutation == 'zero_times':
        payload['trace_t_seconds'] = np.zeros(HORIZON + 1)
    else:
        payload['coordinate_order'] = np.asarray('column_row')
    np.savez(root / 'labels/train0.npz', **payload)
    with pytest.raises(ValueError, match=match):
        build(root)[0]


def test_images_must_be_rgb_and_match_the_declared_geometry(tmp_path):
    root = valid_dataset(tmp_path / 'data')
    Image.fromarray(np.zeros((HEIGHT, WIDTH), np.uint8)).save(
        root / 'rgb/ep0/1.png')
    with pytest.raises(ValueError, match='RGB'):
        build(root)[0]
    Image.fromarray(np.zeros((HEIGHT // 2, WIDTH, 3), np.uint8)).save(
        root / 'rgb/ep0/1.png')
    with pytest.raises(ValueError, match='image|geometry'):
        build(root)[0]


def test_statistics_are_geometry_only_not_robot_joint_statistics(tmp_path):
    statistics = build(valid_dataset(tmp_path / 'data')).statistics
    assert statistics['schema_version'] == SCHEMA
    assert statistics['image_height'] == HEIGHT
    assert statistics['horizon'] == HORIZON
    assert statistics['coordinate_order'] == 'row_column'
    for forbidden in ('action', 'actions', 'state', 'states', 'proprio',
                      'mean', 'std'):
        assert forbidden not in statistics


# ----------------------------------------------------------------------
# Collation and a tiny end-to-end training step
# ----------------------------------------------------------------------
def _tiny_vla():
    import torch.nn as nn

    class TinyAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(16)
            self.proj = nn.Conv2d(3, 128, 1)

        def encode(self, video):
            return self.proj(self.pool(video))

    class TinyTextEncoder(nn.Module):
        requires_images = False
        trainable_adapters = False

        def __init__(self):
            super().__init__()
            self.embed = nn.Parameter(torch.randn(3, 32))
            self.feature_dim = 32

        def forward(self, prompts):
            return self.embed.unsqueeze(0).expand(len(prompts), -1, -1)

    return build_vla_from_cfg(dict(
        type='PatchWAM2DMotionVLA', motion_horizon=HORIZON,
        num_inference_steps=2,
        vlm_backbone=dict(type='PatchWAMBackbone',
                          autoencoder=TinyAutoencoder(),
                          text_encoder=TinyTextEncoder(), token_dim=128,
                          text_dim=32, history_pool_size=2),
        vla_head=dict(type='PatchWAM2DMotionHead', text_dim=32,
                      image_height=HEIGHT, image_width=WIDTH,
                      motion_horizon=HORIZON,
                      denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128,
                                    text_dim=32, width=32, heads=4, depth=2))))


def test_real_files_collate_and_run_a_training_step(tmp_path):
    dataset = build(valid_dataset(tmp_path / 'data'))
    collator = DictCollator(**COLLATOR)
    batch = collator([dataset[0], dataset[1]])
    assert batch['images'].shape == (2, 1, 3, 2, HEIGHT, WIDTH)
    assert batch['motion_2d'].shape == (2, HORIZON, 4, 4, 2)
    assert batch['motion_valid'].shape == (2, HORIZON, 4, 4)
    assert batch['task_description'] == ['move the block', 'move the block']
    vla = _tiny_vla()
    out = vla(**batch, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(out['loss'])
    assert torch.isfinite(out['loss_video']) and out['loss_video'] > 0
    assert torch.isfinite(out['loss_motion'])
    predicted = vla.predict_motion(**batch, seed=0, num_inference_steps=2)
    assert predicted['motion_2d'].shape == (2, HORIZON, 4, 4, 2)
    assert predicted['traces_2d'].shape == (2, HORIZON + 1, 4, 4, 2)


# ----------------------------------------------------------------------
# The documented export example
# ----------------------------------------------------------------------
def _write_doc_sample(root, *, sample_id, episode_id, instruction, frames,
                      tracked, valid, current_frame_id, future_frame_id):
    """Mirror of the export snippet in docs/patchwam-2d-motion.md.

    ``frames`` and the tracked/valid arrays are external tracker inputs. Image
    names come from the caller's own frame ids (never a shared index), so two
    samples from one episode cannot overwrite each other.
    """
    tracked = np.asarray(tracked, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    tracked[0] = np.where(valid[0][..., None], patch_centers(), tracked[0])
    episode_root = root / 'rgb' / episode_id
    episode_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, index in ((current_frame_id, 0), (future_frame_id, HORIZON)):
        path = episode_root / f'{name}.png'
        Image.fromarray(np.asarray(frames[index], dtype=np.uint8)).save(path)
        paths.append(str(path.relative_to(root)))
    np.savez(root / 'labels' / f'{sample_id}.npz', traces_2d=tracked,
             trace_valid=valid,
             trace_t_seconds=np.asarray([0.0, *OFFSETS], dtype=np.float64))
    return dict(sample_id=sample_id, episode_id=episode_id,
                instruction=instruction, current_rgb=paths[0],
                future_rgb=paths[1], labels=f'labels/{sample_id}.npz')


def test_documented_export_example_mixes_episodes_without_overwriting(tmp_path):
    root = tmp_path / 'export'
    (root / 'labels').mkdir(parents=True, exist_ok=True)
    (root / 'dataset.json').write_text(json.dumps(valid_manifest()))
    generator = np.random.default_rng(11)
    entries = []
    for index, frame_id in enumerate(('f000', 'f010')):
        traces, valid, _ = trace_arrays(index)
        entries.append(dict(
            sample_id=f'ep0_s{index}', episode_id='ep0', instruction='move',
            frames=generator.integers(0, 256, (HORIZON + 1, HEIGHT, WIDTH, 3),
                                      dtype=np.uint8),
            tracked=traces, valid=valid, current_frame_id=frame_id,
            future_frame_id=f'{frame_id}_t{HORIZON}'))
    traces, valid, _ = trace_arrays(7)
    entries.append(dict(
        sample_id='ep1_s0', episode_id='ep1', instruction='push',
        frames=generator.integers(0, 256, (HORIZON + 1, HEIGHT, WIDTH, 3),
                                  dtype=np.uint8),
        tracked=traces, valid=valid, current_frame_id='f000',
        future_frame_id=f'f{HORIZON:03d}'))
    rows = [_write_doc_sample(root, **entry) for entry in entries]
    (root / 'train.jsonl').write_text(
        '\n'.join(json.dumps(row) for row in rows[:2]) + '\n')
    (root / 'val.jsonl').write_text(json.dumps(rows[2]) + '\n')
    dataset = build(root)
    assert dataset.sample_ids == ['ep0_s0', 'ep0_s1']
    first, second = dataset[0], dataset[1]
    assert first['motion_2d'].shape == (HORIZON, 4, 4, 2)
    # Two samples from one episode kept their own frames and labels.
    assert not np.allclose(first['images'], second['images'])
    assert (root / 'rgb/ep0/f000.png').is_file()
    assert (root / 'rgb/ep0/f000_t3.png').is_file()
    assert (root / 'rgb/ep0/f010.png').is_file()
    assert (root / 'rgb/ep1/f000.png').is_file()
    assert (root / 'labels/ep0_s0.npz').is_file()
    assert (root / 'labels/ep0_s1.npz').is_file()
    val = build(root, split='val')
    assert val.sample_ids == ['ep1_s0']
    assert val[0]['episode_id'] == 'ep1'
