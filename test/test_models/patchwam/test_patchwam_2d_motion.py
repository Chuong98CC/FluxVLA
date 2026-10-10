# Copyright 2026 Limx Dynamics
"""CPU units for the PatchWAM 2D patch-motion head and native VLA.

The tiny denoiser, autoencoder and instruction encoder are deterministic local
stand-ins; no FLUX.2 assets, checkpoints or GPU are required.
"""

import pytest
import torch
import torch.nn as nn

from fluxvla.engines import (HEADS, VLAS, build_head_from_cfg,
                             build_vla_from_cfg)
from fluxvla.models.backbones.vlms import patchwam_backbone  # noqa: F401
from fluxvla.models.heads import patchwam_2d_motion_head  # noqa: F401
from fluxvla.models.third_party_models.patchwam.geometry import \
    sequence_coordinates
from fluxvla.models.third_party_models.patchwam.motion2d import \
    PatchMotion2DCodec
from fluxvla.models.vlas import patchwam_2d_motion_vla  # noqa: F401

TINY = dict(token_dim=128, text_dim=32, width=32, heads=4, depth=2)


class TinyAutoencoder(nn.Module):
    """Stride-16 stand-in: the official packed AE latent grid is ``H/16``."""

    def __init__(self, channels: int = 128):
        super().__init__()
        self.pool = nn.AvgPool2d(16)
        self.proj = nn.Conv2d(3, channels, 1)

    def encode(self, video):
        return self.proj(self.pool(video))


class TinyTextEncoder(nn.Module):
    requires_images = False
    trainable_adapters = False

    def __init__(self, dim: int = 32, length: int = 3):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(length, dim))
        self.feature_dim = dim

    def forward(self, prompts):
        return self.embed.unsqueeze(0).expand(len(prompts), -1, -1)


class RecordingDenoiser(nn.Module):
    """Tiny denoiser that records every joint-pass input and zeroes velocities."""

    def __init__(self, *, zero_velocities: bool = True, **kwargs):
        super().__init__()
        from fluxvla.models.third_party_models.patchwam.tiny import \
            SmallJointTransformer
        self.inner = SmallJointTransformer(**kwargs)
        self.token_dim = self.inner.token_dim
        self.text_dim = self.inner.text_dim
        self.representation_dim = self.inner.representation_dim
        self.representation_layers = self.inner.representation_layers
        self.zero_velocities = zero_velocities
        self.calls = []

    def forward(self, reference, noisy, sigma, context, **kwargs):
        self.calls.append({
            'reference_shape': tuple(reference.shape),
            'noisy_shape': tuple(noisy.shape),
            'context_shape': tuple(context.shape),
            'reference_ids': kwargs['reference_ids'].detach().clone(),
            'noisy_ids': kwargs['noisy_ids'].detach().clone(),
            'context_ids': kwargs['context_ids'].detach().clone(),
            'visibility': kwargs['visibility'].detach().clone(),
        })
        result = self.inner(reference, noisy, sigma, context, **kwargs)
        return torch.zeros_like(result) if self.zero_velocities else result


def head_cfg(**overrides):
    cfg = dict(type='PatchWAM2DMotionHead',
               denoiser=dict(type='PatchWAMTinyDenoiser', **TINY),
               image_height=64, image_width=64, text_dim=32, motion_horizon=3)
    cfg.update(overrides)
    return cfg


def backbone_cfg(**overrides):
    cfg = dict(type='PatchWAMBackbone', autoencoder=TinyAutoencoder(),
               text_encoder=TinyTextEncoder(), token_dim=128, text_dim=32,
               history_pool_size=2)
    cfg.update(overrides)
    return cfg


def build_vla(**overrides):
    cfg = dict(type='PatchWAM2DMotionVLA', vlm_backbone=backbone_cfg(),
               vla_head=head_cfg(), motion_horizon=3, num_inference_steps=2)
    cfg.update(overrides)
    return build_vla_from_cfg(cfg)


def images(batch=2, height=64, width=64, seed=9):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(batch, 1, 3, 2, height, width,
                      generator=generator) * 2 - 1


def motion_batch(batch=2, horizon=3, rows=4, columns=4, seed=5):
    generator = torch.Generator().manual_seed(seed)
    motion = torch.randn(batch, horizon, rows, columns, 2,
                         generator=generator) * 30
    valid = torch.ones(batch, horizon, rows, columns, dtype=torch.bool)
    return motion, valid


def preencoded_batch(batch=2, reference=16, future=16, text=3):
    generator = torch.Generator().manual_seed(7)
    return {
        'reference_tokens': torch.randn(batch, reference, 128,
                                        generator=generator),
        'future_tokens': torch.randn(batch, future, 128, generator=generator),
        'text_tokens': torch.randn(batch, text, 32, generator=generator),
    }


def _run(vla, motion, valid, seed=4, **kwargs):
    return vla(images=images(), motion_2d=motion, motion_valid=valid,
               task_description=['a', 'b'],
               generator=torch.Generator().manual_seed(seed), **kwargs)


# ----------------------------------------------------------------------
# Registration, defaults and strict knobs
# ----------------------------------------------------------------------
def test_registers_public_names_and_owns_a_real_codec():
    assert HEADS.get('PatchWAM2DMotionHead') is not None
    assert VLAS.get('PatchWAM2DMotionVLA') is not None
    head = build_head_from_cfg(head_cfg())
    assert isinstance(head.codec, PatchMotion2DCodec)
    assert head.codec.token_dim == 128
    assert head.codec.grid_shape == (4, 4)
    assert head.codec.blocks_per_timestep == 1
    assert head.motion_horizon == 3
    # The 2D variant has no robot state projection at all.
    assert head.state_projection is None
    assert not hasattr(head, 'split_modalities')


def test_state_projection_and_self_flow_are_rejected():
    with pytest.raises(ValueError, match='state|proprio'):
        build_head_from_cfg(head_cfg(proprio_dim=14))
    with pytest.raises(ValueError, match='Self-Flow|self_flow'):
        build_head_from_cfg(head_cfg(self_flow_variant=1))
    with pytest.raises(ValueError, match='horizon'):
        build_head_from_cfg(head_cfg(motion_horizon=0))
    with pytest.raises(ValueError, match='divisible'):
        build_head_from_cfg(head_cfg(image_height=120))
    with pytest.raises(TypeError):
        build_head_from_cfg(dict(type='PatchWAM2DMotionHead',
                                 denoiser=dict(type='PatchWAMTinyDenoiser',
                                               **TINY), text_dim=32))


# ----------------------------------------------------------------------
# Training forward
# ----------------------------------------------------------------------
def test_forward_returns_finite_motion_and_video_losses_without_state():
    vla = build_vla()
    motion, valid = motion_batch()
    out = _run(vla, motion, valid)
    assert torch.isfinite(out['loss'])
    assert torch.isfinite(out['loss_video']) and out['loss_video'] > 0
    assert torch.isfinite(out['loss_motion'])
    assert torch.equal(out['loss_motion'], out['loss_action'])
    detached = {key: float(value.detach()) for key, value in out.items()
                if torch.is_tensor(value)}
    assert detached['loss'] == pytest.approx(
        0.5 * detached['loss_video'] + 1.0 * detached['loss_motion'], rel=1e-5)


def test_robot_state_and_action_inputs_are_rejected():
    vla = build_vla()
    motion, valid = motion_batch()
    with pytest.raises(ValueError, match='state|proprio'):
        _run(vla, motion, valid, states=torch.randn(2, 14))
    with pytest.raises(ValueError, match='action chunk|robot action'):
        _run(vla, motion, valid, actions=torch.randn(2, 4, 7))
    head = build_head_from_cfg(head_cfg())
    batch = preencoded_batch()
    batch['proprio'] = torch.randn(2, 14)
    with pytest.raises(ValueError, match='state|proprio'):
        head(batch, motion_2d=torch.zeros(2, 3, 4, 4, 2))
    with pytest.raises(ValueError, match='motion_2d'):
        head(preencoded_batch())


def test_fully_invalid_motion_has_zero_loss_without_suppressing_video_loss():
    vla = build_vla()
    motion, valid = motion_batch()
    valid[:] = False
    out = _run(vla, motion, valid)
    assert float(out['loss_motion'].detach()) == 0.0
    assert float(out['loss_action'].detach()) == 0.0
    assert (torch.isfinite(out['loss_video'])
            and float(out['loss_video'].detach()) > 0)
    assert torch.isfinite(out['loss'])


def test_invalid_nan_motion_cannot_contaminate_the_loss():
    vla = build_vla()
    motion, valid = motion_batch()
    # A missing intermediate trace invalidates both adjacent displacement steps.
    valid[:, 1] = False
    with_nan = motion.clone()
    with_nan[:, 1] = float('nan')
    with_zero = motion.clone()
    with_zero[:, 1] = 0.0
    nan_out = _run(vla, with_nan, valid)
    zero_out = _run(vla, with_zero, valid)
    assert torch.isfinite(nan_out['loss_motion'])
    torch.testing.assert_close(nan_out['loss_motion'], zero_out['loss_motion'],
                               rtol=0, atol=0)
    # The invalidated step is excluded from the loss; still-valid steps are not.
    changed_invalid = motion.clone()
    changed_invalid[:, 1] += 1000.0
    torch.testing.assert_close(_run(vla, changed_invalid, valid)['loss_motion'],
                               zero_out['loss_motion'], rtol=0, atol=0)
    changed_valid = motion.clone()
    changed_valid[:, 2] += 25.0
    assert not torch.allclose(_run(vla, changed_valid, valid)['loss_motion'],
                              zero_out['loss_motion'])


def test_valid_nonfinite_motion_is_rejected_loudly():
    vla = build_vla()
    motion, valid = motion_batch()
    motion[0, 0, 0, 0, 0] = float('inf')
    with pytest.raises(ValueError, match='finite'):
        _run(vla, motion, valid)


def test_absent_motion_valid_means_every_step_is_supervised():
    vla = build_vla()
    motion, valid = motion_batch()
    explicit = _run(vla, motion, valid)
    implicit = vla(images=images(), motion_2d=motion,
                   task_description=['a', 'b'],
                   generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(explicit['loss_motion'], implicit['loss_motion'])


# ----------------------------------------------------------------------
# Joint-pass contract
# ----------------------------------------------------------------------
def test_denoiser_sees_one_motion_token_per_block_per_step_with_spatial_ids():
    denoiser = RecordingDenoiser(**TINY)
    vla = build_vla(vla_head=head_cfg(image_height=256, image_width=256,
                                      denoiser=denoiser))
    codec = vla.vla_head.codec
    assert codec.grid_shape == (16, 16)
    assert codec.blocks_per_timestep == 4
    motion = torch.randn(1, 3, 16, 16, 2)
    valid = torch.ones(1, 3, 16, 16, dtype=torch.bool)
    out = vla(images=images(batch=1, height=256, width=256),
              motion_2d=motion, motion_valid=valid, task_description=['a'])
    assert torch.isfinite(out['loss'])
    call = denoiser.calls[-1]
    latent = 16 * 16
    # 256 current patches + 256 future patches + T*K = 12 motion tokens.
    assert call['reference_shape'] == (1, latent, 128)
    assert call['noisy_shape'] == (1, latent + 12, 128)
    motion_ids = call['noisy_ids'][:, latent:]
    torch.testing.assert_close(motion_ids, codec.token_ids(3).unsqueeze(0))
    assert (motion_ids[..., 0] == 20.0).all()
    timesteps = motion_ids[..., 3].flatten().tolist()
    assert sorted(set(timesteps)) == [0.0, 1.0, 2.0]
    assert [timesteps.count(t) for t in (0.0, 1.0, 2.0)] == [4, 4, 4]
    # Exactly one token per block per step, not one per step.
    assert motion_ids.shape[1] == 12 != 3
    # No state token: the context is only the instruction tokens.
    assert call['context_shape'] == (1, 3, 32)
    assert (call['context_ids'][..., 3] == torch.arange(3)).all()
    # Prefix-masked attention is preserved: the instruction + reference prefix
    # never attends to the future raster or the motion tokens.
    visibility = call['visibility']
    prefix = 3 + latent
    assert visibility.shape == (1, 1, prefix + latent + 12, prefix + latent + 12)
    assert not visibility[..., :prefix, prefix:].any()
    assert visibility[..., :prefix, :prefix].all()
    # Sampling uses the physical horizon and the real grid: T * K tokens.
    sampled = vla.predict_motion(images=images(batch=1, height=256, width=256),
                                 task_description=['a'], seed=0,
                                 num_inference_steps=1)
    assert sampled['motion_2d'].shape == (1, 3, 16, 16, 2)
    assert denoiser.calls[-1]['noisy_shape'] == (1, latent + 12, 128)
    torch.testing.assert_close(denoiser.calls[-1]['noisy_ids'][:, latent:],
                               codec.token_ids(3).unsqueeze(0))


def test_explicit_default_action_ids_reproduce_the_absent_key_behaviour():
    from fluxvla.models.heads.patchwam_head import PatchWAMHead  # noqa: F401
    head = build_head_from_cfg(dict(
        type='PatchWAMHead', denoiser=dict(type='PatchWAMTinyDenoiser', **TINY),
        action_dim=7, text_dim=32, token_dim=128))
    generator = torch.Generator().manual_seed(21)
    batch = preencoded_batch()
    batch['action'] = torch.randn(2, 4, 7, generator=generator)
    base = head(dict(batch), generator=torch.Generator().manual_seed(8))
    extended = dict(batch)
    extended['action_ids'] = sequence_coordinates(2, 4, group=20)
    hooked = head(extended, generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(base['loss'], hooked['loss'], rtol=0, atol=0)
    torch.testing.assert_close(base['loss_action'], hooked['loss_action'],
                               rtol=0, atol=0)
    torch.testing.assert_close(base['loss_video'], hooked['loss_video'],
                               rtol=0, atol=0)
    with pytest.raises(ValueError, match='action_ids'):
        head(dict(extended, action_ids=torch.zeros(2, 3, 4)),
             generator=torch.Generator().manual_seed(8))


# ----------------------------------------------------------------------
# Gradients
# ----------------------------------------------------------------------
def test_backprop_reaches_the_denoiser_and_frozen_frontends_stay_frozen():
    denoiser = RecordingDenoiser(zero_velocities=False, **TINY)
    vla = build_vla(vla_head=head_cfg(denoiser=denoiser))
    motion, valid = motion_batch()
    out = _run(vla, motion, valid)
    out['loss'].backward()
    gradients = [parameter.grad for parameter in denoiser.parameters()
                 if parameter.grad is not None]
    assert gradients
    assert any(float(gradient.abs().sum()) > 0 for gradient in gradients)
    frozen = list(vla.vlm_backbone.parameters())
    assert frozen
    for parameter in frozen:
        assert not parameter.requires_grad
        assert parameter.grad is None


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------
def test_seeded_sampling_is_reproducible_and_source_anchored():
    vla = build_vla()
    codec = vla.vla_head.codec
    first = vla.predict_motion(images=images(), task_description=['a', 'b'],
                               seed=0, num_inference_steps=3)
    second = vla.predict_motion(images=images(), task_description=['a', 'b'],
                                seed=0, num_inference_steps=3)
    assert first['motion_2d'].shape == (2, 3, 4, 4, 2)
    assert first['traces_2d'].shape == (2, 4, 4, 4, 2)
    assert first['future_tokens'].shape == (2, 16, 128)
    torch.testing.assert_close(first['motion_2d'], second['motion_2d'],
                               rtol=0, atol=0)
    centers = codec.patch_centers()
    torch.testing.assert_close(first['traces_2d'][..., 0, :, :, :],
                               centers.expand(2, 4, 4, 2))
    torch.testing.assert_close(first['traces_2d'][..., 1, :, :, :],
                               centers + first['motion_2d'][:, 0])
    torch.testing.assert_close(
        first['traces_2d'][..., 2, :, :, :],
        centers + first['motion_2d'][:, 0] + first['motion_2d'][:, 1])


def test_sampling_returns_signed_pixels_through_one_normalization():
    denoiser = RecordingDenoiser(**TINY)
    vla = build_vla(vla_head=head_cfg(denoiser=denoiser))
    target = torch.randn(2, 3, 4, 4, 2) * 37
    out = vla.predict_motion(images=images(), task_description=['a', 'b'],
                             seed=1, num_inference_steps=1,
                             initial_motion=target)
    # A zero-velocity denoiser leaves the initialized normalized tokens in place,
    # so the decode must return the original signed pixel displacements.
    torch.testing.assert_close(out['motion_2d'], target, rtol=1e-5, atol=1e-4)
    action = vla.predict_action(images=images(), task_description=['a', 'b'],
                                seed=2, num_inference_steps=1,
                                initial_motion=target)
    # The compatibility entry point returns pixel motion, never robot [B,H,A].
    assert action.shape == (2, 3, 4, 4, 2)
    torch.testing.assert_close(action, target, rtol=1e-5, atol=1e-4)


def test_raw_image_callers_derive_the_grid_and_cached_callers_must_declare_it():
    cached = build_vla(vlm_backbone=None)
    batch = preencoded_batch()
    with pytest.raises(ValueError, match='motion_grid_shape'):
        cached.predict_motion(task_description=None, seed=0,
                              num_inference_steps=1, **batch)
    out = cached.predict_motion(task_description=None, seed=0,
                                num_inference_steps=1, motion_grid_shape=(4, 4),
                                **batch)
    assert out['motion_2d'].shape == (2, 3, 4, 4, 2)
    with pytest.raises(ValueError, match='grid'):
        cached.predict_motion(task_description=None, seed=0,
                              num_inference_steps=1, motion_grid_shape=(8, 8),
                              **batch)
    # Raw images supply the grid from the codec's patch size.
    raw = build_vla()
    derived = raw.predict_motion(images=images(), task_description=['a', 'b'],
                                 seed=0, num_inference_steps=1)
    assert derived['motion_2d'].shape == (2, 3, 4, 4, 2)
    with pytest.raises(ValueError, match='state|proprio'):
        raw.predict_motion(images=images(), states=torch.randn(2, 14),
                           task_description=['a', 'b'], seed=0,
                           num_inference_steps=1)


def test_frame_and_view_masks_reach_the_encoder():
    vla = build_vla()
    motion, valid = motion_batch()
    frames = images()
    baseline = _run(vla, motion, valid)
    assert float(baseline['loss_video'].detach()) > 0
    # Every sample's future endpoint is invalid: no video supervision at all.
    future_masked = vla(
        images=frames, motion_2d=motion, motion_valid=valid,
        frame_masks=torch.tensor([[True, False], [True, False]]),
        task_description=['a', 'b'],
        generator=torch.Generator().manual_seed(4))
    assert float(future_masked['loss_video'].detach()) == 0.0
    assert torch.isfinite(future_masked['loss_motion'])
    assert float(future_masked['loss_motion'].detach()) > 0
    # One valid endpoint keeps exactly half of the video supervision.
    partial = vla(
        images=frames, motion_2d=motion, motion_valid=valid,
        frame_masks=torch.tensor([[True, True], [True, False]]),
        task_description=['a', 'b'],
        generator=torch.Generator().manual_seed(4))
    partial_video = float(partial['loss_video'].detach())
    assert 0 < partial_video < float(baseline['loss_video'].detach())
    # A padded single view masks the same rasters through img_masks [B, V].
    view_masked = vla(
        images=frames, motion_2d=motion, motion_valid=valid,
        img_masks=torch.zeros(2, 1, dtype=torch.bool),
        task_description=['a', 'b'],
        generator=torch.Generator().manual_seed(4))
    assert float(view_masked['loss_video'].detach()) == 0.0


def test_cached_token_validity_is_preserved_through_the_vla():
    vla = build_vla(vlm_backbone=None)
    motion, valid = motion_batch()

    def run(future_valid):
        return vla(motion_2d=motion, motion_valid=valid,
                   future_valid=future_valid,
                   generator=torch.Generator().manual_seed(4),
                   **preencoded_batch())

    baseline = run(None)
    assert float(baseline['loss_video'].detach()) > 0
    nothing_valid = run(torch.zeros(2, 16, dtype=torch.bool))
    assert float(nothing_valid['loss_video'].detach()) == 0.0
    partial = run(torch.tensor([[True] * 16, [False] * 16]))
    assert 0 < float(partial['loss_video'].detach()) \
        < float(baseline['loss_video'].detach())


def test_robot_action_inputs_are_rejected_at_every_public_entry_point():
    head = build_head_from_cfg(head_cfg())
    batch = preencoded_batch()
    batch['motion_2d'] = torch.zeros(2, 3, 4, 4, 2)
    for field, value in (('action', torch.randn(2, 3, 2)),
                         ('action_is_pad', torch.zeros(2, 3, dtype=torch.bool)),
                         ('action_dim_is_pad',
                          torch.zeros(2, 3, 2, dtype=torch.bool))):
        with pytest.raises(ValueError, match='robot action'):
            head(dict(batch, **{field: value}))
    raw = build_vla()
    for field, value in (('actions', torch.randn(2, 4, 7)),
                         ('action_masks', torch.ones(2, 4, dtype=torch.bool)),
                         ('action_dim_masks',
                          torch.ones(2, 7, dtype=torch.bool))):
        with pytest.raises(ValueError, match='robot action'):
            raw.predict_motion(images=images(), task_description=['a', 'b'],
                               seed=0, num_inference_steps=1,
                               **{field: value})
    with pytest.raises(ValueError, match='state|proprio'):
        raw.predict_motion(images=images(), task_description=['a', 'b'],
                           seed=0, num_inference_steps=1,
                           states=torch.randn(2, 14))
    with pytest.raises(ValueError, match='action_horizon'):
        raw.predict_action(images=images(), task_description=['a', 'b'],
                           action_horizon=16)
    # This feature's own motion fields stay valid for batch inference.
    cached = build_vla(vlm_backbone=None)
    predicted = cached.predict_motion(
        task_description=None, seed=0, num_inference_steps=1,
        motion_grid_shape=(4, 4), motion_2d=torch.zeros(2, 3, 4, 4, 2),
        motion_valid=torch.ones(2, 3, 4, 4, dtype=torch.bool),
        **preencoded_batch())
    assert predicted['motion_2d'].shape == (2, 3, 4, 4, 2)


# ----------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------
def test_native_checkpoint_round_trip_drops_recognized_state_projection():
    source = build_vla()
    target = build_vla()
    report = target.load_patchwam_state_dict(source.state_dict())
    assert report['format'] == 'native'
    assert report['loaded_tensors'] == len(source.state_dict())
    motion, valid = motion_batch()
    source_out = _run(source, motion, valid, seed=6)
    target_out = _run(target, motion, valid, seed=6)
    torch.testing.assert_close(source_out['loss'], target_out['loss'])
    torch.testing.assert_close(source_out['loss_motion'],
                               target_out['loss_motion'])
    torch.testing.assert_close(source_out['loss_video'],
                               target_out['loss_video'])
    # A robot PatchWAM payload carries a state projection that this variant does
    # not have; it is discarded deliberately and reported, never silently.
    state = source.state_dict()
    state['vla_head.state_projection.weight'] = torch.zeros(32, 14)
    state['vla_head.state_projection.bias'] = torch.zeros(32)
    robot_out = build_vla()
    report = robot_out.load_patchwam_state_dict(state)
    assert set(report['dropped_state_projection']) == {
        'vla_head.state_projection.weight',
        'vla_head.state_projection.bias',
    }
    assert report['loaded_tensors'] == len(source.state_dict())
    # Required DiT weights are never dropped: an unknown tensor still fails.
    state['vla_head.not_a_dit.weight'] = torch.zeros(1)
    with pytest.raises(ValueError, match='unexpected'):
        build_vla().load_patchwam_state_dict(state)
    del state['vla_head.denoiser.token_output.weight']
    with pytest.raises(ValueError, match='missing'):
        build_vla().load_patchwam_state_dict(state)
