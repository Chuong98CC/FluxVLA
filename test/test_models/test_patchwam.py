# Copyright 2026 Limx Dynamics
"""CPU units for the native PatchWAM head/VLA and vendored numerical core."""

import copy

import pytest
import torch

from fluxvla.engines import HEADS, VLAS, build_head_from_cfg, build_vla_from_cfg
from fluxvla.models.third_party_models.patchwam import (PatchFlowPolicy,
                                                       RepeatedActionCodec,
                                                       make_tiny_policy)
from fluxvla.models.third_party_models.patchwam.flow import masked_sample_mse
from fluxvla.models.third_party_models.patchwam.geometry import (
    flatten_image_latents, joint_visibility, restore_image_latents)
from fluxvla.models.vlas.patchwam_vla import (PatchWAMVLA,
                                              resolve_action_padding)

TINY_DENOISER = dict(token_dim=128, text_dim=32, width=32, heads=4, depth=2)


def tiny_head_cfg(**overrides):
    cfg = dict(
        type='PatchWAMHead',
        denoiser=dict(type='PatchWAMTinyDenoiser', **TINY_DENOISER),
        action_dim=14,
        text_dim=32,
        proprio_dim=14,
        token_dim=128,
    )
    cfg.update(overrides)
    return cfg


def preencoded_batch(batch=2, reference=4, future=4, text=3, horizon=16,
                     action_dim=14):
    generator = torch.Generator().manual_seed(1234)
    return {
        'reference_tokens':
        torch.randn(batch, reference, 128, generator=generator),
        'future_tokens':
        torch.randn(batch, future, 128, generator=generator),
        'text_tokens':
        torch.randn(batch, text, 32, generator=generator),
        'action':
        torch.randn(batch, horizon, action_dim, generator=generator),
        'proprio':
        torch.randn(batch, horizon, action_dim, generator=generator),
    }


def test_registries_expose_the_native_components():
    assert HEADS.get('PatchWAMHead') is not None
    assert VLAS.get('PatchWAMVLA') is not None
    head = build_head_from_cfg(tiny_head_cfg())
    assert isinstance(head, PatchFlowPolicy)
    assert head.codec.action_dim == 14
    assert head.denoiser.token_dim == 128
    vla = build_vla_from_cfg(
        dict(
            type='PatchWAMVLA',
            vla_head=tiny_head_cfg(),
            action_horizon=16,
            num_inference_steps=4,
        ))
    assert isinstance(vla, PatchWAMVLA)
    assert vla.all_module_keys == ['vlm_backbone', 'vla_head']
    assert vla.vlm_backbone is None


@pytest.mark.parametrize('action_dim', [7, 14, 80, 128])
def test_codec_is_parameter_free_and_a_left_inverse(action_dim):
    codec = RepeatedActionCodec(action_dim, scale=0.75)
    assert not hasattr(codec, 'parameters')
    actions = torch.randn(2, 16, action_dim, requires_grad=True)
    tokens = codec.encode(actions)
    assert tokens.shape == (2, 16, 128)
    torch.testing.assert_close(codec.decode(tokens), actions)
    assert torch.count_nonzero(tokens[..., codec.occupied:]) == 0
    codec.decode(tokens).sum().backward()
    torch.testing.assert_close(actions.grad, torch.ones_like(actions))


def test_geometry_and_flow_survive_the_port():
    mask = joint_visibility(3, 4, 5, 2)[0, 0]
    assert mask[:7, :7].all() and not mask[:7, 7:].any() and mask[7:].all()
    latent = torch.randn(2, 128, 2, 3)
    torch.testing.assert_close(
        restore_image_latents(flatten_image_latents(latent), 2, 3), latent)
    prediction = torch.tensor([[[2.0, float('nan')]],
                               [[float('nan'), float('nan')]]],
                              requires_grad=True)
    valid = torch.tensor([[[True, False]], [[False, False]]])
    loss = masked_sample_mse(prediction, torch.zeros_like(prediction), valid)
    torch.testing.assert_close(loss, torch.tensor([4.0, 0.0]))


def test_action_padding_preserves_dimension_and_time_separation():
    actions = torch.zeros(2, 3, 4)
    # Native mask polarity: ``True`` means a valid coordinate.
    time_only = torch.ones(2, 3, dtype=torch.bool)
    time_only[1, 2] = False
    step_pad, dim_pad = resolve_action_padding(actions, time_only, None)
    assert step_pad.dtype == torch.bool and dim_pad.dtype == torch.bool
    torch.testing.assert_close(step_pad, ~time_only)
    assert dim_pad.shape == (2, 4) and not dim_pad.any()

    coordinate = torch.ones(2, 3, 4, dtype=torch.bool)
    coordinate[0, 1, 2] = False
    coordinate[1, 2, :] = False
    step_pad, dim_pad = resolve_action_padding(actions, coordinate, None)
    # A fully padded step is time-invalid; per-coordinate padding stays exact.
    torch.testing.assert_close(step_pad,
                               torch.tensor([[False, False, False],
                                             [False, False, True]]))
    torch.testing.assert_close(dim_pad, ~coordinate)

    explicit_dims = torch.ones(2, 4, dtype=torch.bool)
    explicit_dims[:, 3] = False
    step_pad, dim_pad = resolve_action_padding(actions, time_only,
                                               explicit_dims)
    torch.testing.assert_close(step_pad, ~time_only)
    torch.testing.assert_close(
        dim_pad[0], torch.tensor([False, False, False, True]))
    assert dim_pad.shape == (2, 4)


def test_dimension_mask_broadcast_has_no_cross_batch_contamination():
    # B != H: the [B, A] dimension mask must broadcast over the horizon axis.
    actions = torch.zeros(2, 3, 4)
    coordinate = torch.ones(2, 3, 4, dtype=torch.bool)
    coordinate[0, 1, 2] = False
    explicit = torch.ones(2, 4, dtype=torch.bool)
    explicit[1, 3] = False
    step_pad, dim_pad = resolve_action_padding(actions, coordinate, explicit)
    assert dim_pad.shape == (2, 3, 4)
    torch.testing.assert_close(dim_pad, (~coordinate) | (~explicit)[:, None, :])
    torch.testing.assert_close(step_pad, ~coordinate.any(-1))
    assert not dim_pad[0, :, 3].any()  # sample 0 keeps dimension 3 valid
    assert dim_pad[1, :, 3].all()  # sample 1 pads dimension 3 everywhere

    # B == H: source-order right-aligned broadcast would cross-wire batch and
    # horizon. Distinct per-sample dimension masks must stay per-sample.
    actions = torch.zeros(3, 3, 2)
    coordinate = torch.ones(3, 3, 2, dtype=torch.bool)
    coordinate[0, 0, 1] = False
    coordinate[1, 1, 0] = False
    coordinate[2, 2, 1] = False
    explicit = torch.ones(3, 2, dtype=torch.bool)
    explicit[0, 1] = False
    step_pad, dim_pad = resolve_action_padding(actions, coordinate, explicit)
    assert dim_pad.shape == (3, 3, 2)
    torch.testing.assert_close(step_pad, ~coordinate.any(-1))
    torch.testing.assert_close(dim_pad, (~coordinate) | (~explicit)[:, None, :])
    # Only sample 0 pads dimension 1; samples 1 and 2 keep it valid.
    torch.testing.assert_close(dim_pad[:, 0, 1],
                               torch.tensor([True, False, False]))
    assert not dim_pad[1, 0, 1] and not dim_pad[2, 0, 1]

    # Without a coordinate mask the separable [B, A] form is preserved.
    time_only = torch.ones(3, 3, dtype=torch.bool)
    _, dim_pad = resolve_action_padding(actions, time_only, explicit)
    assert dim_pad.shape == (3, 2)
    torch.testing.assert_close(dim_pad, ~explicit)


def test_all_invalid_actions_and_padded_nans_stay_finite():
    core = make_tiny_policy(width=32, heads=4, depth=2)
    sample = preencoded_batch()
    sample['action_is_pad'] = torch.ones(2, 16, dtype=torch.bool)
    sample['action_dim_is_pad'] = torch.ones(2, 14, dtype=torch.bool)
    result = core(sample, generator=torch.Generator().manual_seed(3))
    assert torch.isfinite(result['loss']).all()
    torch.testing.assert_close(result['loss_action'], torch.zeros(()))

    # NaNs in padded action dimensions and text must not reach the loss.
    sample = preencoded_batch()
    sample['action_dim_is_pad'] = torch.zeros(2, 14, dtype=torch.bool)
    sample['action_dim_is_pad'][:, -1] = True
    sample['action'] = sample['action'].clone()
    sample['action'][..., -1] = float('nan')
    sample['text_valid'] = torch.tensor([[True, False, True],
                                         [False, True, True]])
    sample['text_tokens'] = sample['text_tokens'].clone()
    sample['text_tokens'][~sample['text_valid']] = float('nan')
    result = core(sample, generator=torch.Generator().manual_seed(5))
    result['loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in core.parameters()
               if p.grad is not None)


def test_native_forward_and_prediction_shapes():
    vla = build_vla_from_cfg(
        dict(
            type='PatchWAMVLA',
            vla_head=tiny_head_cfg(),
            action_horizon=16,
            num_inference_steps=3,
        ))
    sample = preencoded_batch()
    action_masks = torch.ones(2, 16, dtype=torch.bool)
    action_masks[1, -2:] = False
    losses = vla(
        reference_tokens=sample['reference_tokens'],
        future_tokens=sample['future_tokens'],
        text_tokens=sample['text_tokens'],
        actions=sample['action'],
        states=sample['proprio'],
        action_masks=action_masks,
    )
    for key in ('loss', 'loss_video', 'loss_action'):
        assert key in losses and losses[key].ndim == 0
    assert torch.isfinite(losses['loss'])

    prediction = vla.predict_action(
        reference_tokens=sample['reference_tokens'],
        text_tokens=sample['text_tokens'],
        states=sample['proprio'][:, 0],
        seed=11,
    )
    assert prediction.shape == (2, 16, 14)
    repeat = vla.predict_action(
        reference_tokens=sample['reference_tokens'],
        text_tokens=sample['text_tokens'],
        states=sample['proprio'][:, 0],
        seed=11,
    )
    torch.testing.assert_close(prediction, repeat, rtol=0, atol=0)
    other = vla.predict_action(
        reference_tokens=sample['reference_tokens'],
        text_tokens=sample['text_tokens'],
        states=sample['proprio'][:, 0],
        seed=12,
    )
    assert not torch.allclose(prediction, other)


def test_wrapper_matches_vendored_core_loss_gradients_and_samples():
    torch.manual_seed(0)
    core = make_tiny_policy(width=32, heads=4, depth=2)
    vla = build_vla_from_cfg(
        dict(
            type='PatchWAMVLA',
            vla_head=tiny_head_cfg(),
        ))
    vla.vla_head.load_state_dict(core.state_dict())

    sample = preencoded_batch()
    sample['action_is_pad'] = torch.zeros(2, 16, dtype=torch.bool)
    sample['action_is_pad'][0, -2:] = True
    sample['action_dim_is_pad'] = torch.zeros(2, 14, dtype=torch.bool)
    sample['action_dim_is_pad'][:, -1] = True
    sigma = torch.tensor([0.3, 0.7])
    generator = torch.Generator().manual_seed(21)
    future_noise = torch.randn(2, 4, 128, generator=generator)
    action_noise = torch.randn(2, 16, 128, generator=generator)

    core_out = core(
        sample,
        sigma=sigma,
        future_noise=future_noise,
        action_noise=action_noise,
    )
    wrap_out = vla(
        reference_tokens=sample['reference_tokens'],
        future_tokens=sample['future_tokens'],
        text_tokens=sample['text_tokens'],
        actions=sample['action'],
        states=sample['proprio'],
        action_masks=~sample['action_is_pad'],
        action_dim_masks=~sample['action_dim_is_pad'],
        sigma=sigma,
        future_noise=future_noise,
        action_noise=action_noise,
    )
    for key in ('loss', 'loss_video', 'loss_action', 'sigma_mean'):
        torch.testing.assert_close(wrap_out[key], core_out[key], rtol=0,
                                   atol=0)

    wrap_out['loss'].backward()
    wrapped_grad = {
        name: param.grad.detach().clone()
        for name, param in vla.vla_head.named_parameters()
        if param.grad is not None
    }
    core_out['loss'].backward()
    for name, param in core.named_parameters():
        if param.grad is not None and name in wrapped_grad:
            torch.testing.assert_close(param.grad, wrapped_grad[name], rtol=0,
                                       atol=0)
    assert wrapped_grad['state_projection.weight'].abs().sum() > 0

    initial_future = torch.randn(2, 4, 128,
                                 generator=torch.Generator().manual_seed(31))
    initial_action = torch.randn(2, 16, 128,
                                 generator=torch.Generator().manual_seed(32))
    core_sample = core.sample_actions(sample, steps=3, horizon=16,
                                      initial_future=initial_future,
                                      initial_action=initial_action)
    wrap_sample = vla.predict_action(
        reference_tokens=sample['reference_tokens'],
        text_tokens=sample['text_tokens'],
        states=sample['proprio'][:, 0],
        action_horizon=16,
        num_inference_steps=3,
        initial_future=initial_future,
        initial_action=initial_action,
    )
    torch.testing.assert_close(wrap_sample, core_sample['action'], rtol=0,
                               atol=0)


def test_optimizer_step_and_checkpoint_round_trip():
    vla = build_vla_from_cfg(
        dict(
            type='PatchWAMVLA',
            vla_head=tiny_head_cfg(),
            action_horizon=16,
            num_inference_steps=2,
        ))
    sample = preencoded_batch()
    action_masks = torch.ones(2, 16, dtype=torch.bool)
    action_masks[0, -1] = False
    optimizer = torch.optim.SGD(vla.parameters(), lr=0.01)
    losses = vla(
        reference_tokens=sample['reference_tokens'],
        future_tokens=sample['future_tokens'],
        text_tokens=sample['text_tokens'],
        actions=sample['action'],
        states=sample['proprio'],
        action_masks=action_masks,
    )
    losses['loss'].backward()
    before = vla.vla_head.state_projection.weight.detach().clone()
    optimizer.step()
    assert not torch.allclose(before,
                              vla.vla_head.state_projection.weight.detach())

    vla.eval()
    with torch.no_grad():
        prediction = vla.predict_action(
            reference_tokens=sample['reference_tokens'],
            text_tokens=sample['text_tokens'],
            states=sample['proprio'][:, 0],
            seed=7,
        )
    state = copy.deepcopy(vla.state_dict())
    clone = build_vla_from_cfg(
        dict(
            type='PatchWAMVLA',
            vla_head=tiny_head_cfg(),
            action_horizon=16,
            num_inference_steps=2,
        ))
    clone.load_state_dict(state)
    clone.eval()
    with torch.no_grad():
        restored = clone.predict_action(
            reference_tokens=sample['reference_tokens'],
            text_tokens=sample['text_tokens'],
            states=sample['proprio'][:, 0],
            seed=7,
        )
    torch.testing.assert_close(prediction, restored, rtol=0, atol=0)


def test_self_flow_and_missing_tokens_are_rejected():
    with pytest.raises(ValueError, match='Self-Flow'):
        build_head_from_cfg(tiny_head_cfg(self_flow_variant=1))
    vla = build_vla_from_cfg(
        dict(
            type='PatchWAMVLA',
            vla_head=tiny_head_cfg(),
            action_horizon=16,
            num_inference_steps=2,
        ))
    with pytest.raises(ValueError, match='pre-encoded'):
        vla(actions=torch.zeros(1, 16, 14),
            action_masks=torch.ones(1, 16, dtype=torch.bool))
    with pytest.raises(ValueError, match='pre-encoded'):
        vla.predict_action(seed=0)
