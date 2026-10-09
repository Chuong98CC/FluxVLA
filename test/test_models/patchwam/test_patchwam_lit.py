# Copyright 2026 Limx Dynamics
"""CPU units for the LIT goal prior, aggregator and layer-wise action expert."""

import pytest
import torch

from fluxvla.engines import HEADS, build_head_from_cfg
from fluxvla.models.third_party_models.fastwam.modules.action_dit import ActionDiT
from fluxvla.models.third_party_models.patchwam.lit import (
    LITLayerwiseActionExpert,
    SemanticVisualAggregator,
    compute_pose_reconstruction_loss,
    layer_group_index,
    lit_coupling_layers,
    validate_lit_checkpoint_keys,
)


def _action_dit(num_layers=4):
    return ActionDiT(
        hidden_dim=32, action_dim=14, ffn_dim=64, text_dim=32, freq_dim=256,
        eps=1e-6, num_heads=4, attn_head_dim=8, num_layers=num_layers)


def _expert(num_layers=4, latent_dim=16, context_dim=32):
    return LITLayerwiseActionExpert(
        _action_dit(num_layers), latent_dim=latent_dim,
        context_dim=context_dim)


def test_registry_exposes_a_real_native_action_dit():
    assert HEADS.get('PatchWAMLITActionDiT') is not None
    built = build_head_from_cfg(dict(
        type='PatchWAMLITActionDiT', hidden_dim=32, action_dim=14, ffn_dim=64,
        text_dim=32, freq_dim=256, eps=1e-6, num_heads=4, attn_head_dim=8,
        num_layers=2))
    assert isinstance(built, ActionDiT)
    assert len(built.blocks) == 2


def test_depth_group_sharing_is_exact_and_ordered():
    assert layer_group_index(0, 4, 2) == 0
    assert layer_group_index(1, 4, 2) == 0
    assert layer_group_index(2, 4, 2) == 1
    assert layer_group_index(3, 4, 2) == 1
    with pytest.raises(ValueError, match='divisible'):
        layer_group_index(0, 5, 2)
    with pytest.raises(ValueError, match='layer_idx'):
        layer_group_index(4, 4, 2)


def test_coupling_layers_are_spread_and_strictly_increasing():
    assert lit_coupling_layers(5, 25) == [5, 10, 15, 20, 25]
    assert lit_coupling_layers(1, 8) == [8]
    with pytest.raises(ValueError, match='cannot exceed'):
        lit_coupling_layers(3, 2)


def test_expert_uses_every_layer_context_and_backpropagates_to_latents():
    torch.manual_seed(0)
    expert = _expert(num_layers=4, latent_dim=16, context_dim=32)
    batch, horizon = 2, 5
    latents = [torch.randn(batch, 6, 16, requires_grad=True) for _ in range(4)]
    tokens_in = torch.randn(batch, horizon, 14, requires_grad=True)
    timestep = torch.rand(batch) * 1000
    out = expert.forward_layerwise(tokens_in, timestep, latents)
    assert out.shape == (batch, horizon, 14)
    out.square().mean().backward()
    for index, latent in enumerate(latents):
        assert latent.grad is not None, f'layer {index} context unused'
        assert latent.grad.abs().sum() > 0
    assert tokens_in.grad.abs().sum() > 0


def test_stage2_expert_context_is_latent_only():
    """No text/image prefix path exists for the Stage-2 action expert."""
    expert = _expert(num_layers=2, latent_dim=16, context_dim=32)
    batch, horizon, tokens = 2, 4, 6
    latents = [torch.randn(batch, tokens, 16) for _ in range(2)]
    action = torch.randn(batch, horizon, 14)
    timestep = torch.full((batch,), 300.0)
    base = expert.forward_layerwise(action, timestep, latents)
    # The expert has no prefix argument at all; only the latents and the action
    # window can move the output.
    changed_action = action + 1.0
    assert not torch.allclose(
        base, expert.forward_layerwise(changed_action, timestep, latents))
    assert expert.forward_layerwise(action, timestep, latents).shape == base.shape


def test_masked_latents_cannot_influence_the_expert():
    torch.manual_seed(0)
    expert = _expert(num_layers=1, latent_dim=16, context_dim=32)
    batch, horizon, tokens = 2, 4, 6
    latents = [torch.randn(batch, tokens, 16)]
    action = torch.randn(batch, horizon, 14)
    timestep = torch.full((batch,), 300.0)
    mask = torch.ones(batch, tokens, dtype=torch.bool)
    mask[:, 4] = False
    base = expert.forward_layerwise(action, timestep, latents, [mask])
    poisoned = latents[0].clone()
    poisoned[:, 4] = float('nan')
    masked = expert.forward_layerwise(action, timestep, [poisoned], [mask])
    torch.testing.assert_close(base, masked)
    # A masked latent cannot train either: its gradient stays zero.
    latents[0].requires_grad_(True)
    expert.forward_layerwise(action, timestep, latents, [mask]).sum().backward()
    assert torch.count_nonzero(latents[0].grad[:, 4]) == 0


def test_single_context_path_is_goal_conditioned_and_trainable():
    torch.manual_seed(0)
    expert = _expert(num_layers=2)
    batch, horizon = 2, 4
    prefix = torch.randn(batch, 4, 32)
    tokens_in = torch.randn(batch, horizon, 14)
    timestep = torch.full((batch,), 250.0)
    out = expert.forward_single_context(tokens_in, timestep, prefix)
    assert out.shape == (batch, horizon, 14)
    out.sum().backward()
    grads = [p.grad for p in expert.action_dit.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_aggregator_mask_polarity_blocks_masked_keys():
    torch.manual_seed(0)
    aggregator = SemanticVisualAggregator(
        num_tokens=6, latent_dim=16, context_dim=32, num_layer_groups=2,
        num_pose_tokens=2, num_heads=2, ffn_ratio=2.0)
    queries = aggregator.initial_queries(2)
    semantic = torch.randn(2, 3, 32)
    visual = torch.randn(2, 4, 32)
    keep_all = torch.ones(2, 3, dtype=torch.bool)
    image_all = torch.ones(2, 4, dtype=torch.bool)
    base = aggregator.forward_layer(
        queries, semantic, visual, semantic_mask=keep_all,
        image_mask=image_all, layer_idx=0, num_layers=4)

    blocked = keep_all.clone()
    blocked[:, 1] = False
    masked = aggregator.forward_layer(
        queries, semantic, visual, semantic_mask=blocked,
        image_mask=image_all, layer_idx=0, num_layers=4)
    assert not torch.allclose(base, masked)

    poisoned = semantic.clone()
    poisoned[:, 1] = float('nan')
    poisoned_result = aggregator.forward_layer(
        queries, poisoned, visual, semantic_mask=blocked,
        image_mask=image_all, layer_idx=0, num_layers=4)
    assert torch.isfinite(poisoned_result).all()
    torch.testing.assert_close(poisoned_result, masked)


def test_aggregator_handles_all_invalid_memories_finitely():
    torch.manual_seed(0)
    aggregator = SemanticVisualAggregator(
        num_tokens=6, latent_dim=16, context_dim=32, num_layer_groups=2,
        num_pose_tokens=2, num_heads=2, ffn_ratio=2.0)
    queries = aggregator.initial_queries(2)
    semantic = torch.randn(2, 3, 32)
    visual = torch.randn(2, 4, 32)
    semantic[:, 1] = float('nan')
    out = aggregator.forward_layer(
        queries, semantic, visual,
        semantic_mask=torch.zeros(2, 3, dtype=torch.bool),
        image_mask=torch.zeros(2, 4, dtype=torch.bool),
        layer_idx=0, num_layers=4)
    assert torch.isfinite(out).all()


def test_context_dropout_never_drops_pose_tokens():
    aggregator = SemanticVisualAggregator(
        num_tokens=10, latent_dim=16, context_dim=32, num_layer_groups=2,
        num_pose_tokens=4, num_heads=2, ffn_ratio=2.0,
        context_token_dropout=0.9, context_blackout_prob=0.4)
    aggregator.train()
    keep = aggregator.sample_context_keep_mask(8, torch.device('cpu'))
    assert keep is not None and keep.shape == (8, 10)
    assert keep[:, :4].all()
    aggregator.eval()
    assert aggregator.sample_context_keep_mask(8, torch.device('cpu')) is None


def test_pose_reconstruction_loss_ignores_invalid_goals_and_dims():
    pred = torch.zeros(2, 4)
    target = torch.ones(2, 4)
    loss = compute_pose_reconstruction_loss(
        pred, target, is_pad=torch.tensor([False, True]))
    torch.testing.assert_close(loss, torch.tensor(1.0))
    dims = torch.tensor([[False, False, True, True],
                         [False, False, True, True]])
    torch.testing.assert_close(
        compute_pose_reconstruction_loss(pred, target, dim_is_pad=dims),
        torch.tensor(1.0))
    all_invalid = compute_pose_reconstruction_loss(
        torch.full((2, 4), float('nan')), target,
        is_pad=torch.tensor([True, True]))
    assert torch.isfinite(all_invalid)
    assert float(all_invalid) == 0.0


def test_pose_reconstruction_loss_keeps_masked_gradients_finite():
    torch.manual_seed(0)
    pred = torch.randn(3, 4, requires_grad=True)
    target = torch.randn(3, 4)
    # Sample 0 is valid, sample 1 is a padded goal, sample 2 has padded dims.
    target[1] = float('nan')
    pred_dirty = pred.detach().clone()
    pred_dirty[1] = float('nan')
    pred_dirty[2, 2:] = float('nan')
    pred_dirty.requires_grad_(True)
    dims = torch.zeros(3, 4, dtype=torch.bool)
    dims[2, 2:] = True
    loss = compute_pose_reconstruction_loss(
        pred_dirty, target, is_pad=torch.tensor([False, True, False]),
        dim_is_pad=dims)
    assert torch.isfinite(loss)
    loss.backward()
    assert pred_dirty.grad is not None
    assert torch.isfinite(pred_dirty.grad).all()
    # Invalid coordinates receive exactly zero gradient.
    assert torch.count_nonzero(pred_dirty.grad[1]) == 0
    assert torch.count_nonzero(pred_dirty.grad[2, 2:]) == 0


def test_pose_reconstruction_loss_rejects_nan_in_valid_coordinates():
    pred = torch.zeros(1, 2)
    target = torch.tensor([[0.0, float('nan')]])
    with pytest.raises(ValueError, match='Non-finite goal values'):
        compute_pose_reconstruction_loss(pred, target)
    with pytest.raises(ValueError, match='Non-finite goal values'):
        compute_pose_reconstruction_loss(
            torch.tensor([[float('nan'), 0.0]]), torch.zeros(1, 2),
            dim_is_pad=torch.zeros(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match='dim_is_pad'):
        compute_pose_reconstruction_loss(
            pred, torch.zeros(1, 2), dim_is_pad=torch.zeros(1, 3,
                                                           dtype=torch.bool))


def test_checkpoint_key_validation_is_fail_closed():
    validate_lit_checkpoint_keys(stage='stage1', missing_keys=[],
                                 unexpected_keys=[])
    with pytest.raises(RuntimeError, match='Stage1'):
        validate_lit_checkpoint_keys(stage='stage1', missing_keys=['a'],
                                     unexpected_keys=[])
    validate_lit_checkpoint_keys(
        stage='stage2',
        missing_keys=['lit_head.semantic_visual_aggregator.queries'],
        unexpected_keys=['lit_head.goal_pose_encoder.net.0.weight'],
        bridge_from_stage1=True)
    with pytest.raises(RuntimeError, match='Stage2'):
        validate_lit_checkpoint_keys(
            stage='stage2',
            missing_keys=['lit_head.semantic_visual_aggregator.queries'],
            unexpected_keys=[], bridge_from_stage1=False)
    with pytest.raises(RuntimeError, match='Stage2'):
        validate_lit_checkpoint_keys(
            stage='stage2',
            missing_keys=['lit_head.action_expert.action_encoder.weight'],
            unexpected_keys=[], bridge_from_stage1=True)


def _head_cfg(**overrides):
    cfg = dict(
        type='PatchWAMLITFlowMatchingHead',
        stage='stage2',
        action_dim=14,
        context_dim=32,
        prefix_input_dim=32,
        proprio_dim=14,
        num_latents=8,
        num_pose_tokens=2,
        num_layer_groups=2,
        latent_dim=16,
        inner_dim=16,
        num_heads=2,
        prefix_depth=2,
        action_expert=dict(
            type='PatchWAMLITActionDiT', hidden_dim=32, action_dim=14,
            ffn_dim=64, text_dim=32, freq_dim=256, eps=1e-6, num_heads=4,
            attn_head_dim=8, num_layers=2),
    )
    cfg.update(overrides)
    return cfg


def test_head_registry_and_strict_lit_knobs():
    head = build_head_from_cfg(_head_cfg())
    assert head.coupling_layers == [1, 2]
    assert head.num_action_layers == 2
    assert head.checkpoint_contract()['stage'] == 'stage2'
    with pytest.raises(ValueError, match='latent layout'):
        build_head_from_cfg(_head_cfg(latent_layout='grid'))
    with pytest.raises(ValueError, match='action_sees_ref'):
        build_head_from_cfg(_head_cfg(action_sees_ref=True))
    with pytest.raises(ValueError, match='divisible'):
        build_head_from_cfg(_head_cfg(
            action_expert=dict(
                type='PatchWAMLITActionDiT', hidden_dim=32, action_dim=14,
                ffn_dim=64, text_dim=32, freq_dim=256, eps=1e-6, num_heads=4,
                attn_head_dim=8, num_layers=3)))
