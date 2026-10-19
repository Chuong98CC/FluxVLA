# Copyright 2026 Limx Dynamics
"""CPU integration units for the PatchWAM + LIT VLA and its checkpoints."""

import copy

import pytest
import torch
import torch.nn as nn

from fluxvla.engines import HEADS, VLAS, build_vla_from_cfg
from fluxvla.models.heads import patchwam_lit_head  # noqa: F401
from fluxvla.models.third_party_models.patchwam.tiny import SmallJointTransformer
from fluxvla.models.vlas import patchwam_lit_vla  # noqa: F401
from fluxvla.models.vlas.patchwam_lit_vla import PatchWAMLITVLA

B, TEXT, REFERENCE, HORIZON = 2, 3, 4, 16
ACTION_DIM = 14


def lit_head_cfg(stage='stage2', **overrides):
    cfg = dict(
        type='PatchWAMLITFlowMatchingHead',
        stage=stage,
        action_dim=ACTION_DIM,
        context_dim=32,
        prefix_input_dim=32,
        proprio_dim=ACTION_DIM,
        num_latents=6,
        num_pose_tokens=2,
        num_layer_groups=2,
        latent_dim=16,
        inner_dim=16,
        num_heads=2,
        pose_loss_weight=0.3,
        fm_action_weight=1.0,
        action_horizon=HORIZON,
        action_expert=dict(
            type='PatchWAMLITActionDiT', hidden_dim=32, action_dim=ACTION_DIM,
            ffn_dim=64, text_dim=32, freq_dim=256, eps=1e-6, num_heads=4,
            attn_head_dim=8, num_layers=2),
    )
    cfg.update(overrides)
    return cfg


def vla_cfg(stage='stage2', training_mode=None, **overrides):
    cfg = dict(
        type='PatchWAMLITVLA',
        action_horizon=HORIZON,
        num_inference_steps=2,
        lit_stage=stage,
        lit_head=lit_head_cfg(stage),
        vla_head=dict(
            type='PatchWAMHead',
            denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128,
                          text_dim=32, width=32, heads=4, depth=2),
            action_dim=ACTION_DIM, text_dim=32, proprio_dim=ACTION_DIM,
            token_dim=128),
    )
    if training_mode:
        cfg['training_mode'] = training_mode
    cfg.update(overrides)
    return cfg


def _batch(batch=B, horizon=HORIZON, include_future=True):
    generator = torch.Generator().manual_seed(11)
    data = {
        'reference_tokens': torch.randn(batch, REFERENCE, 128,
                                        generator=generator),
        'reference_valid': torch.ones(batch, REFERENCE, dtype=torch.bool),
        'text_tokens': torch.randn(batch, TEXT, 32, generator=generator),
        'text_valid': torch.ones(batch, TEXT, dtype=torch.bool),
        'proprio': torch.randn(batch, ACTION_DIM, generator=generator),
        'action': torch.randn(batch, horizon, ACTION_DIM, generator=generator),
        'goal_state': torch.randn(batch, ACTION_DIM, generator=generator),
        'goal_valid': torch.ones(batch, dtype=torch.bool),
    }
    if include_future:
        data['future_tokens'] = torch.randn(batch, REFERENCE, 128,
                                            generator=generator)
        data['future_valid'] = torch.ones(batch, REFERENCE, dtype=torch.bool)
    return data


def _forward(model, data, **extra):
    payload = {key: value for key, value in data.items()
               if key not in ('proprio', 'action')}
    payload.update(extra)
    return model(states=data['proprio'], actions=data['action'], **payload)


def _forward_masked(model, data, action_masks=None, dim_masks=None, **extra):
    payload = {key: value for key, value in data.items()
               if key not in ('proprio', 'action')}
    payload.update(extra)
    return model(states=data['proprio'], actions=data['action'],
                 action_masks=action_masks, action_dim_masks=dim_masks,
                 **payload)


def test_registry_and_single_shared_transformer_owner():
    assert VLAS.get('PatchWAMLITVLA') is not None
    assert HEADS.get('PatchWAMLITFlowMatchingHead') is not None
    model = build_vla_from_cfg(vla_cfg())
    assert isinstance(model, PatchWAMLITVLA)
    assert model.all_module_keys == ['vlm_backbone', 'vla_head', 'lit_head']
    owners = [name for name, module in model.named_modules()
              if isinstance(module, SmallJointTransformer)]
    assert owners == ['vla_head.denoiser']
    assert not any(key.startswith('lit_head.denoiser')
                   for key in model.state_dict())
    assert 'vla_head.state_projection.weight' in model.state_dict()
    assert not any('lit_head.state_projection' in key
                   for key in model.state_dict())


def test_lit_rejects_patchwam_optional_objectives():
    cfg = vla_cfg()
    cfg['vla_head'] = dict(
        cfg['vla_head'],
        action_loss=dict(token_velocity_weight=0.0,
                         mean_clean_weight=1.0))
    with pytest.raises(ValueError, match='LIT forward path'):
        build_vla_from_cfg(cfg)


def test_lit_interface_rejects_the_unstaged_adapter_mode():
    """The LIT stage policy owns the shared head; adapters cannot coexist."""
    cfg = vla_cfg()
    cfg['vla_head']['peft'] = dict(peft_type='LORA', r=2, lora_alpha=4,
                                   target_modules=('qkv', 'mix', 'ff.0',
                                                   'ff.2'),
                                   blocks=('blocks',))
    with pytest.raises(ValueError, match='drop `peft`'):
        build_vla_from_cfg(cfg)


def test_stage2_joint_loss_matches_weighted_branches_and_trains_both_heads():
    torch.manual_seed(0)
    model = build_vla_from_cfg(vla_cfg('stage2', 'joint'))
    data = _batch()
    out = _forward(model, data)
    torch.testing.assert_close(
        out['loss'],
        out['loss_patchwam'] + out['loss_lit'])
    assert out['loss_patchwam'].item() > 0
    assert float(out['lit_loss_goal']) > 0
    out['loss'].backward()
    shared = model.vla_head.denoiser.token_input.weight
    assert shared.grad is not None and shared.grad.abs().sum() > 0
    proprio = model.vla_head.state_projection.weight
    assert proprio.grad is not None and proprio.grad.abs().sum() > 0
    aggregator = model.lit_head.semantic_visual_aggregator
    assert aggregator.queries.grad.abs().sum() > 0
    group = aggregator.groups[0]
    assert group.visual_block.cross_attn.out_proj.weight.grad.abs().sum() > 0
    assert model.lit_head.action_expert.context_projection.weight.grad.abs().sum() > 0
    assert model.lit_head.action_expert.action_dit.action_encoder.weight.grad.abs().sum() > 0


def test_stage2_lit_action_path_ignores_goal_and_future_perturbations():
    torch.manual_seed(0)
    model = build_vla_from_cfg(vla_cfg('stage2', 'lit'))
    data = _batch()
    torch.manual_seed(5)
    out = _forward(model, data)
    perturbed = copy.deepcopy(data)
    perturbed['goal_state'] = torch.randn_like(data['goal_state'])
    perturbed['goal_valid'] = torch.zeros(B, dtype=torch.bool)
    perturbed['future_tokens'] = torch.randn_like(data['future_tokens'])
    torch.manual_seed(5)
    other = _forward(model, perturbed)
    torch.testing.assert_close(out['loss_action'], other['loss_action'])
    assert not torch.allclose(out['loss_goal'], other['loss_goal'])


def test_stage2_predict_action_is_seeded_current_only_and_rejects_oracle():
    model = build_vla_from_cfg(vla_cfg('stage2', 'lit'))
    data = _batch()
    kwargs = dict(
        states=data['proprio'], text_tokens=data['text_tokens'],
        text_valid=data['text_valid'],
        reference_tokens=data['reference_tokens'],
        reference_valid=data['reference_valid'])
    first = model.predict_action(seed=3, **kwargs)
    second = model.predict_action(seed=3, **kwargs)
    assert first.shape == (B, HORIZON, ACTION_DIM)
    torch.testing.assert_close(first, second)
    assert not torch.allclose(first, model.predict_action(seed=4, **kwargs))
    with pytest.raises(ValueError, match='oracle'):
        model.predict_action(seed=3, goal_state=data['goal_state'], **kwargs)
    with pytest.raises(ValueError, match='future'):
        model.predict_action(seed=3, future_tokens=data['future_tokens'],
                             **kwargs)
    original = model.predict_action(head='patchwam', seed=3, **kwargs)
    assert original.shape == (B, HORIZON, ACTION_DIM)


def test_stage1_is_image_free_and_goal_conditioned():
    torch.manual_seed(0)
    model = build_vla_from_cfg(vla_cfg('stage1'))
    data = _batch()
    torch.manual_seed(9)
    out = _forward(model, data)
    assert torch.isfinite(out['loss'])
    assert float(out['loss_goal']) == 0.0
    out['loss'].backward()
    assert model.lit_head.goal_pose_encoder.net[0].weight.grad.abs().sum() > 0
    assert model.lit_head.action_expert.action_dit.action_encoder.weight.grad is not None
    for name, param in model.named_parameters():
        if name.startswith(('vla_head.', 'vlm_backbone.')):
            assert not param.requires_grad, name
            assert param.grad is None or param.grad.abs().sum() == 0, name
    perturbed = copy.deepcopy(data)
    perturbed['reference_tokens'] = torch.randn_like(data['reference_tokens'])
    perturbed['future_tokens'] = torch.randn_like(data['future_tokens'])
    torch.manual_seed(9)
    other = _forward(model, perturbed)
    torch.testing.assert_close(out['loss'], other['loss'])


def test_stage1_drops_an_explicit_image_batch():
    model = build_vla_from_cfg(vla_cfg('stage1'))
    data = _batch()
    torch.manual_seed(1)
    out = _forward(model, data)
    images = torch.randn(B, 2, 3, 2, 8, 8)
    torch.manual_seed(1)
    other = model(states=data['proprio'], actions=data['action'], images=images,
                  reference_tokens=data['reference_tokens'],
                  reference_valid=data['reference_valid'],
                  text_tokens=data['text_tokens'],
                  text_valid=data['text_valid'],
                  goal_state=data['goal_state'],
                  goal_valid=data['goal_valid'])
    torch.testing.assert_close(out['loss'], other['loss'])


def test_masking_and_nan_boundaries_stay_finite():
    torch.manual_seed(0)
    model = build_vla_from_cfg(vla_cfg('stage2', 'lit'))
    data = _batch()
    data['goal_valid'] = torch.zeros(B, dtype=torch.bool)
    torch.manual_seed(0)
    out = _forward_masked(
        model, data,
        action_masks=torch.zeros(B, HORIZON, dtype=torch.bool),
        dim_masks=torch.zeros(B, ACTION_DIM, dtype=torch.bool))
    assert torch.isfinite(out['loss'])
    assert float(out['loss_action']) == 0.0
    assert float(out['loss_goal']) == 0.0
    out['loss'].backward()
    data = _batch()
    data['action'][:, 5:, :] = float('nan')
    action_masks = torch.ones(B, HORIZON, dtype=torch.bool)
    action_masks[:, 5:] = False
    data['reference_valid'] = torch.ones(B, REFERENCE, dtype=torch.bool)
    data['reference_valid'][:, 3] = False
    data['reference_tokens'][:, 3] = float('nan')
    torch.manual_seed(0)
    out = _forward_masked(model, data, action_masks=action_masks)
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    grads = [p.grad for p in model.lit_head.action_expert.parameters()
             if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)


def test_all_invalid_text_and_cameras_stay_finite():
    torch.manual_seed(0)
    model = build_vla_from_cfg(vla_cfg('stage2', 'lit'))
    data = _batch()
    data['text_valid'] = torch.zeros(B, TEXT, dtype=torch.bool)
    data['text_tokens'] = torch.full((B, TEXT, 32), float('nan'))
    data['reference_valid'] = torch.zeros(B, REFERENCE, dtype=torch.bool)
    torch.manual_seed(0)
    out = _forward(model, data)
    assert torch.isfinite(out['loss'])
    out['loss'].backward()


def test_released_initialization_leaves_lit_random_and_keeps_originals():
    torch.manual_seed(0)
    model = build_vla_from_cfg(vla_cfg('stage2', 'joint'))
    state = {key: value.clone() for key, value in model.state_dict().items()
             if not key.startswith('lit_head.')}
    other = build_vla_from_cfg(vla_cfg('stage2', 'joint'))
    before = {key: value.clone() for key, value in other.state_dict().items()
              if key.startswith('lit_head.')}
    report = other.load_patchwam_state_dict(state, strict=True)
    assert report['lit_head_initialized'] == 'random'
    for key, value in state.items():
        torch.testing.assert_close(other.state_dict()[key], value)
    for key, value in before.items():
        torch.testing.assert_close(other.state_dict()[key], value)
    with pytest.raises(ValueError, match='load_lit_state_dict'):
        other.load_patchwam_state_dict(
            {'lit_head.semantic_visual_aggregator.queries': torch.zeros(1)},
            strict=True)


def test_stage1_to_stage2_bridge_requires_exact_shared_tensors():
    torch.manual_seed(0)
    stage1 = build_vla_from_cfg(vla_cfg('stage1'))
    stage2 = build_vla_from_cfg(vla_cfg('stage2', 'joint'))
    stage1_state = stage1.state_dict()
    assert any(key.startswith('lit_head.goal_pose_encoder.')
               for key in stage1_state)
    assert any(key.startswith('lit_head.semantic_visual_aggregator.')
               for key in stage2.state_dict())
    report = stage2.load_lit_state_dict(stage1_state, bridge_from_stage1=True)
    assert report['loaded_tensors'] > 0
    assert any(key.startswith('lit_head.semantic_visual_aggregator.')
               for key in report['missing'])
    assert any(key.startswith('lit_head.goal_pose_encoder.')
               for key in report['unexpected'])
    key = 'lit_head.action_expert.action_dit.action_encoder.weight'
    torch.testing.assert_close(stage2.state_dict()[key], stage1_state[key])
    with pytest.raises(RuntimeError, match='Stage2'):
        build_vla_from_cfg(vla_cfg('stage2', 'joint')).load_lit_state_dict(
            stage1_state, bridge_from_stage1=False)


def test_native_dual_head_roundtrip_and_atomic_rejection():
    torch.manual_seed(3)
    source = build_vla_from_cfg(vla_cfg('stage2', 'joint'))
    state = {key: value.clone() for key, value in source.state_dict().items()}
    target = build_vla_from_cfg(vla_cfg('stage2', 'joint'))
    report = target.load_state_dict_payload(state)
    assert report['lit_head']['loaded_tensors'] > 0
    assert 'patchwam' in report
    for key, value in state.items():
        torch.testing.assert_close(target.state_dict()[key], value)
    bad = dict(state)
    bad['lit_head.action_expert.action_dit.action_encoder.weight'] = torch.zeros(
        1, 1)
    keep = {key: value.clone() for key, value in target.state_dict().items()}
    with pytest.raises(ValueError, match='shape differs'):
        target.load_lit_state_dict(bad)
    for key, value in keep.items():
        torch.testing.assert_close(target.state_dict()[key], value)
    missing = {key: value for key, value in state.items()
               if not key.startswith(
                   'lit_head.semantic_visual_pose_decoder.')}
    with pytest.raises(RuntimeError, match='Stage2'):
        target.load_lit_state_dict(missing)
    for key, value in keep.items():
        torch.testing.assert_close(target.state_dict()[key], value)


def test_stage_policy_survives_train_and_freeze_backbones():
    stage1 = build_vla_from_cfg(vla_cfg('stage1'))
    stage1.train()
    assert stage1.lit_head.goal_pose_encoder.net[0].weight.requires_grad
    assert stage1.lit_head.action_expert.context_projection.weight.requires_grad
    assert not stage1.vla_head.denoiser.token_input.weight.requires_grad
    stage1.freeze_backbones()
    assert not stage1.vla_head.denoiser.token_input.weight.requires_grad
    stage2 = build_vla_from_cfg(vla_cfg('stage2', 'joint'))
    stage2.freeze_backbones()
    assert stage2.vla_head.denoiser.token_input.weight.requires_grad
    assert stage2.vla_head.state_projection.weight.requires_grad
    assert stage2.lit_head.semantic_visual_aggregator.queries.requires_grad
    assert stage2.lit_head.action_expert.context_projection.weight.requires_grad
    # Stage 2 has no goal encoder at all (Stage-1-only module).
    assert stage2.lit_head.goal_pose_encoder is None
    trainable = {name for name, param in stage2.named_parameters()
                 if param.requires_grad}
    assert any(name.startswith('lit_head.semantic_visual_aggregator.')
               for name in trainable)
    assert any(name.startswith('vla_head.denoiser.') for name in trainable)
    stage2.train()
    assert stage2.lit_head.goal_pose_encoder is None


def test_invalid_configurations_fail_loudly():
    with pytest.raises(ValueError, match='lit_stage'):
        build_vla_from_cfg(vla_cfg('stage3'))
    with pytest.raises(ValueError, match='training_mode'):
        build_vla_from_cfg(vla_cfg('stage2', 'sometimes'))
    with pytest.raises(ValueError, match='training_mode'):
        build_vla_from_cfg(vla_cfg('stage1', 'joint'))
    with pytest.raises(ValueError, match='training_mode'):
        build_vla_from_cfg(vla_cfg('stage1', 'patchwam'))
    with pytest.raises(ValueError, match='prefix_input_dim'):
        build_vla_from_cfg(vla_cfg('stage2', lit_head=lit_head_cfg(
            'stage2', prefix_input_dim=64)))
    with pytest.raises(ValueError, match='action_dim'):
        build_vla_from_cfg(vla_cfg('stage2', lit_head=lit_head_cfg(
            'stage2', action_dim=7)))
    with pytest.raises(ValueError, match='stage'):
        build_vla_from_cfg(vla_cfg('stage1', lit_head=lit_head_cfg('stage2')))


def test_patchwam_training_mode_dispatch_and_trainability_is_static():
    torch.manual_seed(0)
    model = build_vla_from_cfg(vla_cfg('stage2', 'patchwam'))
    data = _batch()
    out = _forward(model, data)
    assert 'loss_video' in out and 'loss_action' in out
    assert not any(key.startswith('lit_') for key in out)
    before = {name: param.requires_grad
              for name, param in model.named_parameters()}
    model.predict_action(states=data['proprio'],
                         text_tokens=data['text_tokens'],
                         text_valid=data['text_valid'],
                         reference_tokens=data['reference_tokens'],
                         reference_valid=data['reference_valid'], seed=1,
                         head='lit')
    model.predict_action(states=data['proprio'],
                         text_tokens=data['text_tokens'],
                         text_valid=data['text_valid'],
                         reference_tokens=data['reference_tokens'],
                         reference_valid=data['reference_valid'], seed=1,
                         head='patchwam')
    after = {name: param.requires_grad
             for name, param in model.named_parameters()}
    assert before == after


def test_fsdp_wrap_policy_covers_both_heads_without_aliases():
    model = build_vla_from_cfg(vla_cfg('stage2', 'joint'))
    keys = list(model.state_dict())
    # Every trainable tensor is owned by exactly one of the three module keys.
    for name in keys:
        assert name.startswith(('vlm_backbone.', 'vla_head.', 'lit_head.')), name
    for module_key in model.all_module_keys:
        if getattr(model, module_key, None) is None:
            continue
        assert any(name.startswith(module_key + '.') for name in keys), module_key
    try:
        from fluxvla.engines.utils.fsdp_wrapping import require_fsdp_wrap_support
        require_fsdp_wrap_support()
    except RuntimeError as exc:  # pragma: no cover - slim torch builds
        pytest.skip(str(exc))
    policy = model.get_fsdp_wrapping_policy()
    assert callable(policy)
    # Outer-VLA-only policy: no head or FLUX block is wrapped on its own.
    wrapped = policy.keywords['module_classes']
    assert wrapped == {PatchWAMLITVLA}


def test_stage2_action_path_is_isolated_from_the_prefix():
    """Fixed latents must fully determine the Stage-2 action output.

    Root spot check before the correction: changing the prefix while holding
    the latents fixed changed the action prediction. The strict bottleneck must
    route ALL backbone memory through the aggregator, so this must now be
    bit-identical.
    """
    torch.manual_seed(0)
    model = build_vla_from_cfg(vla_cfg('stage2', 'lit'))
    aggregator = model.lit_head.semantic_visual_aggregator
    fixed = torch.randn(B, aggregator.num_tokens, aggregator.latent_dim,
                        generator=torch.Generator().manual_seed(21))
    calls = {'count': 0}

    def fixed_layer(queries, semantic, visual, **kwargs):
        del queries, semantic, visual, kwargs
        calls['count'] += 1
        return fixed.clone()

    aggregator.forward_layer = fixed_layer
    data = _batch()
    perturbed = copy.deepcopy(data)
    perturbed['reference_tokens'] = torch.randn_like(data['reference_tokens'])
    perturbed['text_tokens'] = torch.randn_like(data['text_tokens'])
    perturbed['proprio'] = torch.randn_like(data['proprio'])
    torch.manual_seed(4)
    first = _forward(model, data)
    first_calls = calls['count']
    torch.manual_seed(4)
    second = _forward(model, perturbed)
    torch.testing.assert_close(first['loss_action'], second['loss_action'])
    # The prefix really ran and drove one aggregator step per coupling layer.
    assert first_calls == model.lit_head.num_action_layers
    assert calls['count'] == 2 * model.lit_head.num_action_layers
    kwargs = dict(states=data['proprio'], text_tokens=data['text_tokens'],
                  text_valid=data['text_valid'],
                  reference_tokens=data['reference_tokens'],
                  reference_valid=data['reference_valid'])
    torch.manual_seed(7)
    left = model.predict_action(seed=7, **kwargs)
    right = model.predict_action(
        seed=7,
        states=perturbed['proprio'], text_tokens=perturbed['text_tokens'],
        text_valid=perturbed['text_valid'],
        reference_tokens=perturbed['reference_tokens'],
        reference_valid=perturbed['reference_valid'])
    torch.testing.assert_close(left, right)


def test_stage2_mode_policies_freeze_the_inactive_branch():
    patchwam = build_vla_from_cfg(vla_cfg('stage2', 'patchwam'))
    assert any(param.requires_grad for param in patchwam.vla_head.parameters())
    assert not any(param.requires_grad
                   for param in patchwam.lit_head.parameters())
    assert not patchwam.lit_head.training
    lit = build_vla_from_cfg(vla_cfg('stage2', 'lit'))
    # The LIT path reads the shared prefix, so the FLUX blocks and the proprio
    # projection stay trainable while the original output-only modules (the
    # FLUX `final_layer`, or the tiny `output_norm`/`token_output`) are frozen.
    assert all(param.requires_grad
               for param in lit.vla_head.state_projection.parameters())
    assert all(param.requires_grad
               for param in lit.vla_head.denoiser.token_input.parameters())
    output_modules = lit.original_output_modules()
    assert output_modules
    assert not any(param.requires_grad for module in output_modules
                   for param in module.parameters())
    frozen_output_names = {
        name for name, param in lit.named_parameters()
        if not param.requires_grad}
    assert any('output_norm' in name or 'token_output' in name
               or 'final_layer' in name for name in frozen_output_names)
    assert all(param.requires_grad
               for param in lit.lit_head.action_expert.parameters())
    assert all(param.requires_grad
               for param in lit.lit_head.semantic_visual_aggregator.parameters())
    joint = build_vla_from_cfg(vla_cfg('stage2', 'joint'))
    # Joint and patchwam modes keep every original parameter active.
    assert all(param.requires_grad for param in joint.vla_head.parameters())
    assert all(param.requires_grad
               for param in joint.lit_head.semantic_visual_pose_decoder
               .parameters())
    # A runtime inference head override never changes trainability.
    before = {name: param.requires_grad
              for name, param in patchwam.named_parameters()}
    patchwam.predict_action(states=torch.randn(B, ACTION_DIM),
                            text_tokens=torch.randn(B, TEXT, 32),
                            text_valid=torch.ones(B, TEXT, dtype=torch.bool),
                            reference_tokens=torch.randn(B, REFERENCE, 128),
                            reference_valid=torch.ones(B, REFERENCE,
                                                      dtype=torch.bool),
                            seed=1, head='lit')
    after = {name: param.requires_grad
             for name, param in patchwam.named_parameters()}
    assert before == after
    # Re-freezing after a train() call keeps the LIT interface frozen.
    patchwam.freeze_backbones()
    assert not any(param.requires_grad
                   for param in patchwam.lit_head.parameters())


class _TinyAutoencoder(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        self.proj = nn.Conv2d(3, channels, 1)

    def encode(self, video):
        return self.proj(video)


class _TinyTextEncoder(nn.Module):
    requires_images = False
    trainable_adapters = False

    def __init__(self, dim: int = 32, length: int = 3):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(length, dim))
        self.feature_dim = dim

    def forward(self, prompts):
        return self.embed.unsqueeze(0).expand(len(prompts), -1, -1)


def _backbone_lit_cfg(stage='stage2', training_mode='lit', modules=None,
                      **overrides):
    cfg = vla_cfg(stage, training_mode, **overrides)
    autoencoder, text_encoder = (modules if modules is not None
                                 else (_TinyAutoencoder(), _TinyTextEncoder()))
    cfg['vlm_backbone'] = dict(
        type='PatchWAMBackbone', autoencoder=autoencoder,
        text_encoder=text_encoder, token_dim=128, text_dim=32,
        history_pool_size=2)
    return cfg


def test_stage2_lit_training_encodes_the_current_frame_only():
    torch.manual_seed(0)
    model = build_vla_from_cfg(_backbone_lit_cfg('stage2', 'lit'))
    data = _batch(include_future=False)
    images = torch.rand(B, 1, 3, 2, 8, 8)
    other_future = images.clone()
    other_future[:, :, :, 1] = torch.rand_like(other_future[:, :, :, 1])
    payload = dict(text_tokens=data['text_tokens'],
                   text_valid=data['text_valid'],
                   goal_state=data['goal_state'],
                   goal_valid=data['goal_valid'],
                   states=data['proprio'], actions=data['action'])
    torch.manual_seed(2)
    first = model(images=images, **payload)
    torch.manual_seed(2)
    second = model(images=other_future, **payload)
    assert torch.isfinite(first['loss'])
    torch.testing.assert_close(first['loss'], second['loss'])
    first['loss'].backward()
    # The LIT-only branch trains the shared FLUX prefix and the LIT interface.
    assert model.vla_head.denoiser.token_input.weight.grad.abs().sum() > 0
    assert model.lit_head.action_expert.context_projection.weight.grad is not None


def test_stage1_drops_caller_imagery_history_and_masks():
    model = build_vla_from_cfg(vla_cfg('stage1'))
    data = _batch()
    torch.manual_seed(11)
    clean = _forward(model, data)
    images = torch.rand(B, 2, 3, 4, 8, 8)
    history = torch.rand(B, 2, 3, 8, 8)
    vl_current = torch.rand(B, 3, 8, 8)
    stray_reference = torch.randn(B, 5, 128)
    torch.manual_seed(11)
    polluted = model(
        images=images,
        states=data['proprio'], actions=data['action'],
        history_video=history,
        history_valid=torch.ones(B, 2, dtype=torch.bool),
        vl_current=vl_current,
        frame_masks=torch.ones(B, 4, dtype=torch.bool),
        img_masks=torch.ones(B, 2, dtype=torch.bool),
        reference_tokens=stray_reference,
        reference_valid=torch.ones(B, 5, dtype=torch.bool),
        text_tokens=data['text_tokens'], text_valid=data['text_valid'],
        goal_state=data['goal_state'], goal_valid=data['goal_valid'])
    torch.testing.assert_close(clean['loss'], polluted['loss'])


def _payload(model):
    return {key: value.clone() for key, value in model.state_dict().items()}


def test_dual_head_load_is_atomic_when_the_original_half_is_malformed():
    torch.manual_seed(0)
    source = build_vla_from_cfg(vla_cfg('stage2', 'joint'))
    state = _payload(source)
    state['vla_head.denoiser.token_input.weight'] = torch.zeros(1, 1)
    target = build_vla_from_cfg(vla_cfg('stage2', 'joint'))
    before = _payload(target)
    with pytest.raises(ValueError, match='shape differs'):
        target.load_state_dict_payload(state)
    after = _payload(target)
    changed = [key for key in before
               if not torch.equal(before[key], after[key])]
    assert changed == []


def test_dual_head_load_rejects_missing_original_and_lit_tensors():
    source = build_vla_from_cfg(vla_cfg('stage2', 'joint'))
    state = _payload(source)
    target = build_vla_from_cfg(vla_cfg('stage2', 'joint'))
    before = _payload(target)
    without_original = {key: value for key, value in state.items()
                        if key != 'vla_head.state_projection.weight'}
    with pytest.raises(ValueError, match='missing'):
        target.load_state_dict_payload(without_original)
    without_lit = {key: value for key, value in state.items()
                   if not key.startswith(
                       'lit_head.semantic_visual_aggregator.')}
    with pytest.raises(RuntimeError, match='Stage2'):
        target.load_state_dict_payload(without_lit)
    after = _payload(target)
    assert [key for key in before
            if not torch.equal(before[key], after[key])] == []


def test_stage1_rejects_partial_original_payloads():
    model = build_vla_from_cfg(vla_cfg('stage1'))
    before = _payload(model)
    # Root reproduction: a single frozen projection entry used to satisfy the
    # trainable-key accounting and was accepted.
    partial = {'vla_head.state_projection.bias':
               model.state_dict()['vla_head.state_projection.bias'].clone()}
    with pytest.raises(ValueError, match='missing'):
        model.load_patchwam_state_dict(partial, strict=True)
    after = _payload(model)
    assert [key for key in before
            if not torch.equal(before[key], after[key])] == []
    # A complete released-style original payload still loads.
    full = {key: value for key, value in _payload(model).items()
            if key.startswith('vla_head.') or key.startswith('vlm_backbone.')}
    report = model.load_patchwam_state_dict(full, strict=True)
    assert report['lit_head_initialized'] == 'random'
    assert report['resolved_tensors'] >= len(
        [key for key in full if key.startswith('vla_head.')])


def test_from_pretrained_wires_released_init_and_stage1_bridge(tmp_path):
    torch.manual_seed(0)
    released = build_vla_from_cfg(vla_cfg('stage2', 'joint'))
    released_state = {key: value for key, value in _payload(released).items()
                      if not key.startswith('lit_head.')}
    released_path = tmp_path / 'released.pt'
    torch.save({'model': released_state}, released_path)
    stage1 = build_vla_from_cfg(vla_cfg('stage1'))
    # Real Stage-1 runs start from the same released PatchWAM weights.
    stage1.load_patchwam_state_dict(released_state, strict=True)
    stage1_path = tmp_path / 'stage1.pt'
    torch.save({'model': _payload(stage1)}, stage1_path)
    model = build_vla_from_cfg(vla_cfg(
        'stage2', 'joint', pretrained_name_or_path=str(released_path),
        lit_stage1_checkpoint=str(stage1_path)))
    model.from_pretrained()
    shared = 'vla_head.denoiser.token_input.weight'
    torch.testing.assert_close(model.state_dict()[shared],
                               released_state[shared])
    expert = 'lit_head.action_expert.action_dit.action_encoder.weight'
    torch.testing.assert_close(model.state_dict()[expert],
                               stage1.state_dict()[expert])
    assert torch.isfinite(
        model.state_dict()['lit_head.semantic_visual_aggregator.queries']).all()


def _aligned_stage1(tmp_path, released_state):
    stage1 = build_vla_from_cfg(vla_cfg('stage1'))
    stage1.load_patchwam_state_dict(released_state, strict=True)
    return stage1


def test_lit_only_primary_payload_is_rejected_atomically():
    torch.manual_seed(0)
    model = build_vla_from_cfg(vla_cfg('stage2', 'joint'))
    lit_only = {key: value for key, value in _payload(model).items()
                if key.startswith('lit_head.')}
    target = build_vla_from_cfg(vla_cfg('stage2', 'joint'))
    before = _payload(target)
    with pytest.raises(ValueError, match='complete original/shared half'):
        target.load_state_dict_payload(lit_only)
    after = _payload(target)
    assert [key for key in before
            if not torch.equal(before[key], after[key])] == []


def test_stage1_bridge_requires_goal_encoder_and_matching_shared_weights(
        tmp_path):
    torch.manual_seed(0)
    released = build_vla_from_cfg(vla_cfg('stage2', 'joint'))
    released_state = {key: value for key, value in _payload(released).items()
                      if not key.startswith('lit_head.')}
    released_path = tmp_path / 'released.pt'
    torch.save({'model': released_state}, released_path)
    stage1 = _aligned_stage1(tmp_path, released_state)
    good_path = tmp_path / 'stage1.pt'
    torch.save({'model': _payload(stage1)}, good_path)

    # A Stage-2 checkpoint is not a Stage-1 bridge: no goal encoder.
    stage2_only_path = tmp_path / 'stage2_only.pt'
    torch.save({'model': _payload(released)}, stage2_only_path)
    no_goal = build_vla_from_cfg(vla_cfg(
        'stage2', 'joint', pretrained_name_or_path=str(released_path),
        lit_stage1_checkpoint=str(stage2_only_path)))
    with pytest.raises(ValueError, match='goal_pose_encoder'):
        no_goal.from_pretrained()

    # An edited Stage-1 shared backbone must fail before anything is copied.
    edited = _payload(stage1)
    edited['vla_head.state_projection.weight'] = (
        edited['vla_head.state_projection.weight'] + 1.0)
    edited_path = tmp_path / 'edited.pt'
    torch.save({'model': edited}, edited_path)
    mismatched = build_vla_from_cfg(vla_cfg(
        'stage2', 'joint', pretrained_name_or_path=str(released_path),
        lit_stage1_checkpoint=str(edited_path)))
    before = _payload(mismatched)
    with pytest.raises(ValueError, match='disagree'):
        mismatched.from_pretrained()
    after = _payload(mismatched)
    assert [key for key in before
            if not torch.equal(before[key], after[key])] == []

    # The matching bridge loads and reports the verified shared tensors.
    good = build_vla_from_cfg(vla_cfg(
        'stage2', 'joint', pretrained_name_or_path=str(released_path),
        lit_stage1_checkpoint=str(good_path)))
    good.from_pretrained()
    expert = 'lit_head.action_expert.action_dit.action_encoder.weight'
    torch.testing.assert_close(good.state_dict()[expert],
                               stage1.state_dict()[expert])


def test_stage1_bridge_accepts_normal_bfloat16_storage_rounding(tmp_path):
    torch.manual_seed(0)
    released = build_vla_from_cfg(vla_cfg('stage2', 'joint'))
    released_state = {key: value for key, value in _payload(released).items()
                      if not key.startswith('lit_head.')}
    released_path = tmp_path / 'released.pt'
    torch.save({'model': released_state}, released_path)
    stage1 = _aligned_stage1(tmp_path, released_state)
    bf16_payload = {
        key: (value.to(torch.bfloat16) if value.is_floating_point() else value)
        for key, value in _payload(stage1).items()}
    bf16_path = tmp_path / 'stage1_bf16.pt'
    torch.save({'model': bf16_payload}, bf16_path)
    model = build_vla_from_cfg(vla_cfg(
        'stage2', 'joint', pretrained_name_or_path=str(released_path),
        lit_stage1_checkpoint=str(bf16_path)))
    model.from_pretrained()
    expert = 'lit_head.action_expert.action_dit.action_encoder.weight'
    torch.testing.assert_close(model.state_dict()[expert],
                               bf16_payload[expert].float())


def test_bridge_without_primary_checkpoint_is_rejected(tmp_path):
    torch.manual_seed(0)
    released = build_vla_from_cfg(vla_cfg('stage2', 'joint'))
    released_state = {key: value for key, value in _payload(released).items()
                      if not key.startswith('lit_head.')}
    stage1 = _aligned_stage1(tmp_path, released_state)
    stage1_path = tmp_path / 'stage1.pt'
    torch.save({'model': _payload(stage1)}, stage1_path)
    model = build_vla_from_cfg(vla_cfg(
        'stage2', 'joint', lit_stage1_checkpoint=str(stage1_path)))
    with pytest.raises(ValueError, match='requires the primary'):
        model.from_pretrained()


ASSET_PREFIXES = ('vlm_backbone.autoencoder.', 'vlm_backbone.text_encoder.')


def test_native_full_export_bridge_tolerates_optional_frozen_assets(tmp_path):
    """A real native Stage-1 export carries the frozen AE/Qwen assets.

    The released PatchWAM primary omits them, so the bridge comparison must
    validate those optional tensors against the locally loaded assets instead
    of demanding them from the primary (root reproduction: four optional
    AE/Qwen keys rejected a valid Stage-1 checkpoint).
    """
    torch.manual_seed(0)
    shared_modules = (_TinyAutoencoder(), _TinyTextEncoder())
    released = build_vla_from_cfg(
        _backbone_lit_cfg('stage2', 'joint', modules=shared_modules))
    released_state = {key: value for key, value in _payload(released).items()
                      if not key.startswith('lit_head.')
                      and not key.startswith(ASSET_PREFIXES)}
    released_path = tmp_path / 'released.pt'
    torch.save({'model': released_state}, released_path)
    stage1 = build_vla_from_cfg(
        _backbone_lit_cfg('stage1', modules=shared_modules))
    stage1.load_patchwam_state_dict(released_state, strict=True)
    stage1_state = _payload(stage1)
    optional_keys = [key for key in stage1_state
                     if key.startswith(ASSET_PREFIXES)]
    assert len(optional_keys) >= 3
    assert not any(key in released_state for key in optional_keys)
    stage1_path = tmp_path / 'stage1_native_full.pt'
    torch.save({'model': stage1_state}, stage1_path)

    model = build_vla_from_cfg(
        _backbone_lit_cfg('stage2', 'joint', modules=shared_modules,
                          pretrained_name_or_path=str(released_path),
                          lit_stage1_checkpoint=str(stage1_path)))
    model.from_pretrained()
    expert = 'lit_head.action_expert.action_dit.action_encoder.weight'
    torch.testing.assert_close(model.state_dict()[expert],
                               stage1_state[expert])
    # The bridge plan reports the optional assets it verified.
    plan = model.plan_lit_bridge_load(
        torch.load(stage1_path, map_location='cpu')['model'],
        primary_original_values=model.plan_original_load(
            released_state)['values'])
    assert plan['validated_optional_assets'] == len(optional_keys)
    assert plan['validated_shared_tensors'] > 0


def test_native_full_export_bridge_rejects_changed_assets_and_shared(tmp_path):
    torch.manual_seed(0)
    shared_modules = (_TinyAutoencoder(), _TinyTextEncoder())
    released = build_vla_from_cfg(
        _backbone_lit_cfg('stage2', 'joint', modules=shared_modules))
    released_state = {key: value for key, value in _payload(released).items()
                      if not key.startswith('lit_head.')
                      and not key.startswith(ASSET_PREFIXES)}
    released_path = tmp_path / 'released.pt'
    torch.save({'model': released_state}, released_path)
    stage1 = build_vla_from_cfg(
        _backbone_lit_cfg('stage1', modules=shared_modules))
    stage1.load_patchwam_state_dict(released_state, strict=True)
    stage1_state = _payload(stage1)

    # A different optional asset (for example a different AE) must be refused.
    changed_asset = dict(stage1_state)
    changed_asset['vlm_backbone.autoencoder.proj.weight'] = (
        changed_asset['vlm_backbone.autoencoder.proj.weight'] + 1.0)
    asset_path = tmp_path / 'stage1_changed_asset.pt'
    torch.save({'model': changed_asset}, asset_path)
    model = build_vla_from_cfg(
        _backbone_lit_cfg('stage2', 'joint', modules=shared_modules,
                          pretrained_name_or_path=str(released_path),
                          lit_stage1_checkpoint=str(asset_path)))
    before = _payload(model)
    with pytest.raises(ValueError, match='disagree'):
        model.from_pretrained()
    after = _payload(model)
    assert [key for key in before
            if not torch.equal(before[key], after[key])] == []


    # An edited shared/proprio tensor keeps failing atomically as well.
    changed_shared = dict(stage1_state)
    changed_shared['vla_head.state_projection.weight'] = (
        changed_shared['vla_head.state_projection.weight'] + 1.0)
    shared_path = tmp_path / 'stage1_changed_shared.pt'
    torch.save({'model': changed_shared}, shared_path)
    model = build_vla_from_cfg(
        _backbone_lit_cfg('stage2', 'joint', modules=shared_modules,
                          pretrained_name_or_path=str(released_path),
                          lit_stage1_checkpoint=str(shared_path)))
    before = _payload(model)
    with pytest.raises(ValueError, match='disagree'):
        model.from_pretrained()
    after = _payload(model)
    assert [key for key in before
            if not torch.equal(before[key], after[key])] == []


def test_from_pretrained_requires_the_stage1_bridge_for_stage2(tmp_path):
    released = build_vla_from_cfg(vla_cfg('stage2', 'joint'))
    released_state = {key: value for key, value in _payload(released).items()
                      if not key.startswith('lit_head.')}
    released_path = tmp_path / 'released.pt'
    torch.save({'model': released_state}, released_path)
    strict = build_vla_from_cfg(vla_cfg(
        'stage2', 'joint', pretrained_name_or_path=str(released_path)))
    with pytest.raises(ValueError, match='Stage-1'):
        strict.from_pretrained()
    ablation = build_vla_from_cfg(vla_cfg(
        'stage2', 'joint', pretrained_name_or_path=str(released_path),
        lit_stage1_required=False))
    ablation.from_pretrained()
    assert torch.isfinite(
        ablation.state_dict()['lit_head.action_expert.context_projection'
                              '.weight']).all()
    # A Stage-2 payload that already carries both heads needs no Stage-1 file
    # (this is the evaluation path).
    dual_path = tmp_path / 'dual.pt'
    torch.save({'model': _payload(released)}, dual_path)
    evaluated = build_vla_from_cfg(vla_cfg(
        'stage2', 'joint', pretrained_name_or_path=str(dual_path)))
    evaluated.from_pretrained()
    # Pointing both at LIT-bearing payloads is ambiguous and must not silently
    # overwrite the trained interface.
    ambiguous = build_vla_from_cfg(vla_cfg(
        'stage2', 'joint', pretrained_name_or_path=str(dual_path),
        lit_stage1_checkpoint=str(dual_path)))
    with pytest.raises(ValueError, match='silently overwrite'):
        ambiguous.from_pretrained()


def test_shared_tensor_comparison_keeps_integer_buffers_exact():
    match = PatchWAMLITVLA._shared_tensors_match
    # Float tensors accept normal bfloat16 storage rounding.
    assert match(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))
    assert match(torch.tensor([1.0], dtype=torch.bfloat16),
                 torch.tensor([1.0], dtype=torch.float32))
    assert not match(torch.tensor([1.0]), torch.tensor([1.5]))
    # Integer buffers are compared exactly, never quantized to bfloat16.
    assert match(torch.tensor([7], dtype=torch.int64),
                 torch.tensor([7], dtype=torch.int32))
    assert not match(torch.tensor([7], dtype=torch.int64),
                     torch.tensor([8], dtype=torch.int64))
    big = 2 ** 24 + 1
    assert not match(torch.tensor([big], dtype=torch.int64),
                     torch.tensor([big - 1], dtype=torch.int64))
    assert not match(torch.tensor([True]),
                     torch.tensor([False], dtype=torch.bool))
