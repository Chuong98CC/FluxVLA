# Copyright 2026 Limx Dynamics
"""CPU units for the PatchWAM 2D patch-motion codec and geometry."""

import pytest
import torch

from fluxvla.models.third_party_models.patchwam.motion2d import (
    PatchMotion2DCodec)


def codec(height=256, width=256, **overrides):
    options = dict(image_height=height, image_width=width)
    options.update(overrides)
    return PatchMotion2DCodec(**options)


def test_geometry_defaults_match_the_sixteen_pixel_patch_grid():
    c = codec()
    assert c.patch_size == 16
    assert c.block_size == 8
    assert c.token_dim == 128
    assert c.coordinate_order == 'row_column'
    assert c.grid_shape == (16, 16)
    assert c.block_grid == (2, 2)
    assert c.blocks_per_timestep == 4
    assert c.patches_per_timestep == 256
    assert c.token_count(3) == 12
    centers = c.patch_centers()
    assert centers.shape == (16, 16, 2)
    torch.testing.assert_close(centers[0, 0], torch.tensor([7.5, 7.5]))
    torch.testing.assert_close(centers[1, 3], torch.tensor([23.5, 55.5]))


def test_complete_region_is_an_identity_in_width_128_token_space():
    # 128x128 -> 8x8 patch grid -> exactly one complete 8x8 block per step.
    c = codec(128, 128)
    assert c.grid_shape == (8, 8)
    assert c.blocks_per_timestep == 1
    horizon = 2
    motion = torch.zeros(1, horizon, 8, 8, 2)
    for t in range(horizon):
        for row in range(8):
            for column in range(8):
                motion[0, t, row, column, 0] = 1000 * t + 10 * row + column + 0.25
                motion[0, t, row, column, 1] = -1000 * t - 10 * row - column - 0.5
    tokens = c.encode(motion)
    assert tokens.shape == (1, horizon, 128)
    normalized = c.normalize(motion)
    for t in range(horizon):
        for row in range(8):
            for column in range(8):
                # local row, local column, then coordinate: coordinate fastest.
                index = (row * 8 + column) * 2
                torch.testing.assert_close(tokens[0, t, index],
                                           normalized[0, t, row, column, 0])
                torch.testing.assert_close(tokens[0, t, index + 1],
                                           normalized[0, t, row, column, 1])
    # No repetition or unused fill: every entry is an actual delta pair.
    assert not torch.equal(tokens[0, 0], tokens[0, 1])


def test_token_block_ordering_is_timestep_then_block_row_then_block_column():
    c = codec()
    horizon = 3
    motion = torch.zeros(2, horizon, 16, 16, 2)
    for t in range(horizon):
        for row in range(16):
            for column in range(16):
                motion[:, t, row, column, 0] = t * 1000 + row * 10 + column
    tokens = c.encode(motion)
    assert tokens.shape == (2, horizon * 4, 128)
    normalized = c.normalize(motion)
    for b in range(2):
        for t in range(horizon):
            for br in range(2):
                for bc in range(2):
                    token_index = t * 4 + br * 2 + bc
                    for lr in range(8):
                        for lc in range(8):
                            row, column = br * 8 + lr, bc * 8 + lc
                            index = (lr * 8 + lc) * 2
                            torch.testing.assert_close(
                                tokens[b, token_index, index],
                                normalized[b, t, row, column, 0])
                            torch.testing.assert_close(
                                tokens[b, token_index, index + 1],
                                normalized[b, t, row, column, 1])


def test_partial_blocks_pad_and_mask_every_coordinate():
    # 160x96 -> 10x6 patch grid -> ceil(10/8)=2 by ceil(6/8)=1 blocks.
    c = codec(160, 96)
    assert c.grid_shape == (10, 6)
    assert c.block_grid == (2, 1)
    assert c.blocks_per_timestep == 2
    valid = torch.ones(1, 2, 10, 6, dtype=torch.bool)
    coordinate_mask = c.coordinate_validity(valid[..., None].expand(1, 2, 10, 6, 2))
    token_mask = c.token_validity(valid)
    assert coordinate_mask.shape == (1, 4, 128)
    assert torch.equal(coordinate_mask, token_mask)
    # Token 0 = block (0, 0): real local columns 0..5, padded 6..7.
    for lr in range(8):
        for lc in range(6):
            index = (lr * 8 + lc) * 2
            assert bool(token_mask[0, 0, index])
            assert bool(token_mask[0, 0, index + 1])
        for lc in range(6, 8):
            index = (lr * 8 + lc) * 2
            assert not bool(token_mask[0, 0, index])
            assert not bool(token_mask[0, 0, index + 1])
    # Token 1 = block (1, 0): real rows 8..9 -> local rows 0..1 only.
    for lr in range(2):
        for lc in range(6):
            index = (lr * 8 + lc) * 2
            assert bool(token_mask[0, 1, index])
    for lr in range(2, 8):
        for lc in range(8):
            index = (lr * 8 + lc) * 2
            assert not bool(token_mask[0, 1, index])
    # Invalid patches propagate through the coordinate mask.
    valid[0, 0, 9, 5] = False
    coordinate_mask = c.coordinate_validity(valid[..., None].expand(1, 2, 10, 6, 2))
    index = (1 * 8 + 5) * 2  # local row 1, local column 5 of token 1
    assert not bool(coordinate_mask[0, 1, index])
    assert not bool(coordinate_mask[0, 1, index + 1])


def test_partial_block_round_trip_drops_padding_fill():
    c = codec(160, 96)
    motion = torch.randn(2, 3, 10, 6, 2) * 20
    tokens = c.encode(motion)
    assert tokens.shape == (2, 3 * 2, 128)
    decoded = c.decode(tokens)
    torch.testing.assert_close(decoded, motion, rtol=1e-5, atol=1e-4)
    valid = c.token_validity(torch.ones(2, 3, 10, 6, dtype=torch.bool))
    assert (tokens[~valid] == 0).all()


def test_round_trip_preserves_signed_out_of_frame_pixels():
    c = codec()
    motion = torch.randn(2, 3, 16, 16, 2) * 400 - 50
    decoded = c.decode(c.encode(motion))
    assert decoded.shape == motion.shape
    torch.testing.assert_close(decoded, motion, rtol=1e-5, atol=1e-3)


def test_normalization_is_by_height_for_rows_and_width_for_columns():
    # A complete 8x8 region has no padding, so every token entry is real.
    c = codec(128, 128)
    motion = torch.zeros(1, 1, 8, 8, 2)
    motion[..., 0] = -128.0
    motion[..., 1] = 64.0
    tokens = c.encode(motion)
    torch.testing.assert_close(tokens[..., 0::2], torch.full((1, 1, 64), -1.0))
    torch.testing.assert_close(tokens[..., 1::2], torch.full((1, 1, 64), 0.5))
    # Decoding multiplies by (H, W) exactly once; no sigmoid or clamping.
    decoded = c.decode(tokens)
    torch.testing.assert_close(decoded, motion)


def test_traces_from_motion_are_source_anchored_cumulative_positions():
    c = codec(128, 128)
    motion = torch.randn(2, 2, 8, 8, 2) * 5
    traces = c.traces_from_motion(motion)
    assert traces.shape == (2, 3, 8, 8, 2)
    centers = c.patch_centers()
    torch.testing.assert_close(traces[..., 0, :, :, :],
                               centers.expand(2, 8, 8, 2))
    torch.testing.assert_close(traces[..., 1, :, :, :], centers + motion[:, 0])
    torch.testing.assert_close(traces[..., 2, :, :, :],
                               centers + motion[:, 0] + motion[:, 1])


def test_missing_intermediate_trace_invalidates_both_adjacent_displacements():
    c = codec(128, 128)
    horizon = 3
    traces = torch.randn(1, horizon + 1, 8, 8, 2) * 3
    valid = torch.ones(1, horizon + 1, 8, 8, dtype=torch.bool)
    valid[:, 1] = False
    motion, motion_valid = c.motion_from_traces(traces, valid)
    assert motion.shape == (1, horizon, 8, 8, 2)
    assert motion_valid.shape == (1, horizon, 8, 8)
    # Both displacements touching the lost trace are invalid; the far one is not.
    assert not motion_valid[:, 0].any()
    assert not motion_valid[:, 1].any()
    assert motion_valid[:, 2].all()
    # Invalid targets are sanitized to zero and never supervised.
    assert (motion[:, :2] == 0).all()
    torch.testing.assert_close(motion[:, 2], traces[:, 3] - traces[:, 2])
    # With a complete trace the plain consecutive difference is returned.
    clean, clean_valid = c.motion_from_traces(traces,
                                              torch.ones_like(valid))
    assert clean_valid.all()
    torch.testing.assert_close(clean, traces[:, 1:] - traces[:, :-1])


def test_valid_nonfinite_positions_are_rejected_and_invalid_ones_sanitized():
    c = codec(128, 128)
    traces = torch.zeros(1, 3, 8, 8, 2)
    traces[..., 0] = 7.5
    traces[..., 1] = 7.5
    valid = torch.ones(1, 3, 8, 8, dtype=torch.bool)
    rejected = traces.clone()
    rejected[0, 1, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        c.motion_from_traces(rejected, valid)
    # A lost track may carry a nonfinite placeholder and is fully sanitized.
    lost_valid = valid.clone()
    lost_valid[0, 1] = False
    lost = traces.clone()
    lost[0, 1] = float('nan')
    motion, motion_valid = c.motion_from_traces(lost, lost_valid)
    assert torch.isfinite(motion).all()
    assert (motion == 0).all()
    assert not motion_valid.any()


def test_motion_token_ids_use_group_20_block_centers_and_zero_based_time():
    c = codec()
    ids = c.token_ids(3)
    assert ids.shape == (12, 4)
    expected = []
    for timestep in range(3):
        for br in range(2):
            for bc in range(2):
                expected.append([20.0, 3.5 + 8 * br, 3.5 + 8 * bc,
                                 float(timestep)])
    torch.testing.assert_close(ids, torch.tensor(expected, dtype=torch.float32))


def test_partial_block_ids_use_the_center_of_the_real_extent():
    # 10x6 patch grid: block row 1 only covers rows 8..9 -> center 8.5.
    c = codec(160, 96)
    ids = c.token_ids(2)
    assert ids.shape == (4, 4)
    torch.testing.assert_close(ids[0], torch.tensor([20.0, 3.5, 2.5, 0.0]))
    torch.testing.assert_close(ids[1], torch.tensor([20.0, 8.5, 2.5, 0.0]))
    torch.testing.assert_close(ids[2], torch.tensor([20.0, 3.5, 2.5, 1.0]))
    torch.testing.assert_close(ids[3], torch.tensor([20.0, 8.5, 2.5, 1.0]))


@pytest.mark.parametrize('overrides, match', [
    (dict(image_height=250), 'divisible'),
    (dict(image_width=100), 'divisible'),
    (dict(patch_size=15), 'divisible'),
    (dict(block_size=4), 'token_dim|block_size'),
    (dict(token_dim=64), 'token_dim|block_size'),
    (dict(coordinate_order='column_row'), 'coordinate_order'),
    (dict(image_height=0), 'positive'),
    (dict(block_size=0), 'token_dim|block_size|positive'),
])
def test_invalid_geometry_is_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        codec(**overrides)


def test_encode_and_decode_reject_wrong_shapes():
    c = codec()
    with pytest.raises(ValueError, match=r'\[B,T,h,w,2\]'):
        c.encode(torch.zeros(1, 3, 256, 256))
    with pytest.raises(ValueError, match=r'\(h,w\)'):
        c.encode(torch.zeros(1, 3, 15, 16, 2))
    with pytest.raises(ValueError, match='128'):
        c.decode(torch.zeros(1, 12, 64))
    with pytest.raises(ValueError, match='timestep|multiple'):
        c.decode(torch.zeros(1, 5, 128))
    with pytest.raises(ValueError, match='horizon|positive'):
        c.token_ids(0)
    with pytest.raises(ValueError, match=r'\[B,T,h,w\]|bool'):
        c.token_validity(torch.zeros(1, 2, 16, 16, 3))


def test_equivalent_masks_across_the_supported_public_layouts():
    # 64x64 -> 4x4 patch grid, h*w = 16 patches per timestep.
    c = codec(64, 64)
    assert c.grid_shape == (4, 4)
    valid = torch.zeros(2, 3, 4, 4, dtype=torch.bool)
    valid[0, 0, 1, 2] = True
    valid[0, 1, 3, 3] = True
    valid[1, 2, 0, 0] = True
    reference = c.token_validity(valid)
    assert reference.shape == (2, 3, 128)
    assert bool(reference[0, 0].sum()) and bool(reference[1, 2].sum())
    # [B, T, h*w] is the documented flattened form of the same mask.
    flattened = c.token_validity(valid.reshape(2, 3, 16))
    assert torch.equal(reference, flattened)
    assert torch.equal(reference, c.token_validity(valid.clone()))
    # A 4x4 grid is one partial 8x8 block: the 16 real local patches are valid
    # and the padded local patches stay masked, in either layout.
    all_valid = c.token_validity(torch.ones(1, 3, 16, dtype=torch.bool))
    assert all_valid.shape == (1, 3, 128)
    assert int(all_valid[0, 0].sum()) == 16 * 2
    assert torch.equal(all_valid[0, 0], all_valid[0, 2])
    assert not bool(all_valid.all())
    # A complete 8x8 region has no padding, so an all-valid mask stays all valid.
    complete = codec(128, 128)
    complete_mask = complete.token_validity(
        torch.ones(1, 3, 64, dtype=torch.bool))
    assert complete_mask.shape == (1, 3, 128)
    assert bool(complete_mask.all())
    # [B, T*h*w] is not a supported layout and must be rejected, not mis-reshaped.
    with pytest.raises(ValueError, match=r'\[B,T,h,w\]|\[B,T,h\*w\]'):
        c.token_validity(valid.reshape(2, 48))
    with pytest.raises(ValueError, match=r'\[B,T,h,w\]|\[B,T,h\*w\]'):
        c.token_validity(valid.reshape(2, 3, 16)[..., :15])


def test_meta_device_encode_decode_and_traces_keep_device_and_float32():
    """Device regression that needs no CUDA: a meta input must stay on meta."""
    c = codec()
    motion = torch.randn(2, 3, 16, 16, 2, device='meta')
    tokens = c.encode(motion)
    assert tokens.device.type == 'meta'
    assert tokens.dtype == torch.float32
    assert tokens.shape == (2, 12, 128)
    decoded = c.decode(tokens)
    assert decoded.device.type == 'meta'
    assert decoded.dtype == torch.float32
    assert decoded.shape == (2, 3, 16, 16, 2)
    traces = c.traces_from_motion(decoded)
    assert traces.device.type == 'meta'
    assert traces.shape == (2, 4, 16, 16, 2)
    mask = c.token_validity(torch.ones(2, 3, 16, 16, dtype=torch.bool,
                                       device='meta'))
    assert mask.device.type == 'meta' and mask.dtype == torch.bool


def test_normalize_and_denormalize_follow_the_input_device_and_float32():
    c = codec(160, 96)
    motion = torch.randn(1, 2, 10, 6, 2, dtype=torch.float64, device='meta')
    normalized = c.normalize(motion)
    assert normalized.device.type == 'meta'
    assert normalized.dtype == torch.float32
    restored = c.denormalize(normalized)
    assert restored.device.type == 'meta'
    assert restored.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA behavioral round trip requires local hardware')
def test_normalization_cuda_round_trip_when_hardware_exists():  # pragma: no cover
    c = codec()
    motion = torch.randn(2, 3, 16, 16, 2, device='cuda')
    tokens = c.encode(motion)
    assert tokens.device.type == 'cuda'
    decoded = c.decode(tokens)
    assert decoded.device == motion.device
    torch.testing.assert_close(decoded, motion, rtol=1e-5, atol=1e-3)


def test_valid_displacement_overflow_is_rejected():
    c = codec(64, 64)
    traces = torch.zeros(1, 3, 4, 4, 2, dtype=torch.float32)
    traces[..., 0] = c.patch_centers()[..., 0]
    traces[..., 1] = c.patch_centers()[..., 1]
    traces[0, 1, 0, 0, 0] = 3.4e38
    traces[0, 2, 0, 0, 0] = -3.4e38
    valid = torch.ones(1, 3, 4, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match='finite|overflow'):
        c.motion_from_traces(traces, valid)
