# Copyright 2026 Limx Dynamics
"""CPU units for the fused linear RADIO velocity auxiliary."""

import copy
import importlib
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fluxvla.engines import build_vla_from_cfg
from fluxvla.models.backbones.vlms import patchwam_radio_backbone  # noqa: F401
from fluxvla.models.heads import patchwam_radio_head  # noqa: F401
from fluxvla.models.third_party_models.patchwam.flux import \
    OfficialFluxDenoiser
from fluxvla.models.third_party_models.patchwam.geometry import (
    joint_visibility, sequence_coordinates)
from fluxvla.models.third_party_models.patchwam.radio import (
    RadioAuxFlowPolicy, make_radio_noise_projection)
from fluxvla.models.vlas import patchwam_radio_vla  # noqa: F401
from fluxvla.models.vlas.patchwam_radio_vla import PatchWAMRadioAuxVLA

RADIO_DIM = 6
TINY = dict(token_dim=128, text_dim=32, width=32, heads=4, depth=2)
FLUX_SOURCE = Path('/home/chuong/workspace/wam/ImageWAM/third_party/flux2/src')


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


class TinyRadioTeacher(nn.Module):
    """Deterministic stand-in returning the official (summary, spatial) pair."""

    def __init__(self, feature_dim: int = RADIO_DIM, patch_size: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.calls = 0
        self.kernel = nn.Parameter(
            torch.randn(feature_dim, 3, patch_size, patch_size) * 0.1)

    def encode(self, images):
        self.calls += 1
        spatial = F.conv2d(images, self.kernel,
                           stride=self.patch_size).flatten(2).transpose(1, 2)
        return spatial.mean(1), spatial


class ExplodingTeacher(TinyRadioTeacher):
    def encode(self, images):
        raise AssertionError('teacher must not run at inference')


def _aux_head(action_dim=7, **overrides):
    cfg = dict(
        type='PatchWAMRadioAuxHead',
        denoiser=dict(type='PatchWAMTinyDenoiser', **TINY),
        action_dim=action_dim, text_dim=32, proprio_dim=8, token_dim=128,
        radio_dim=RADIO_DIM, radio_weight=1.0, video_weight=0.0,
        action_weight=1.0)
    cfg.update(overrides)
    return cfg


def _backbone(teacher=None, **overrides):
    cfg = dict(type='PatchWAMRadioAuxBackbone', autoencoder=TinyAutoencoder(),
               text_encoder=TinyTextEncoder(), token_dim=128, text_dim=32,
               history_pool_size=2, radio_teacher=teacher, radio_dim=RADIO_DIM,
               radio_patch_size=2, radio_min_resolution=2)
    cfg.update(overrides)
    return cfg


def _aux_vla(teacher=None, **overrides):
    cfg = dict(type='PatchWAMRadioAuxVLA', vlm_backbone=_backbone(teacher),
               vla_head=_aux_head(), action_horizon=4, num_inference_steps=2)
    cfg.update(overrides)
    return build_vla_from_cfg(cfg)


def _plain_vla(action_dim=7):
    return build_vla_from_cfg(dict(
        type='PatchWAMVLA',
        vlm_backbone=dict(type='PatchWAMBackbone',
                          autoencoder=TinyAutoencoder(),
                          text_encoder=TinyTextEncoder(), token_dim=128,
                          text_dim=32, history_pool_size=2),
        vla_head=dict(type='PatchWAMHead',
                      denoiser=dict(type='PatchWAMTinyDenoiser', **TINY),
                      action_dim=action_dim, text_dim=32, proprio_dim=8,
                      token_dim=128),
        action_horizon=4, num_inference_steps=2))


def _video(batch=2, views=2, frames=2, height=4, width=6):
    generator = torch.Generator().manual_seed(11)
    return torch.rand(batch, views, 3, frames, height, width,
                      generator=generator) * 2 - 1


def _cached(batch=2, future=6, horizon=4):
    generator = torch.Generator().manual_seed(7)
    return dict(
        reference_tokens=torch.randn(batch, 4, 128, generator=generator),
        future_tokens=torch.randn(batch, future, 128, generator=generator),
        radio_future_tokens=torch.randn(batch, future, RADIO_DIM,
                                        generator=generator),
        text_tokens=torch.randn(batch, 3, 32, generator=generator))


def _flow(seed=17, batch=2, future=6, horizon=4):
    generator = torch.Generator().manual_seed(seed)
    return dict(sigma=torch.tensor([0.3, 0.7]),
                future_noise=torch.randn(batch, future, 128,
                                         generator=generator),
                action_noise=torch.randn(batch, horizon, 128,
                                         generator=generator))


def _loss(vla, sample, **kwargs):
    return vla(actions=torch.randn(2, 4, 7),
               action_masks=torch.ones(2, 4, dtype=torch.bool),
               states=torch.randn(2, 8), **sample, **kwargs)


# ----------------------------------------------------------------------
# Zero-init behaviour, fusion and gradients
# ----------------------------------------------------------------------
def test_zero_init_fusion_matches_the_pretrained_original_exactly():
    aux = _aux_vla()
    plain = _plain_vla()
    plain.load_state_dict(aux.export_original_core_state_dict(), strict=True)
    sample, flow = _cached(), _flow(seed=5)
    actions, states = torch.randn(2, 4, 7), torch.randn(2, 8)
    plain_out = plain(actions=actions, states=states,
                      action_masks=torch.ones(2, 4, dtype=torch.bool),
                      **sample, **copy.deepcopy(flow))
    aux_out = aux(actions=actions, states=states,
                  action_masks=torch.ones(2, 4, dtype=torch.bool),
                  **sample, **copy.deepcopy(flow))
    # B/C and every bias start at zero: Ydoubleprime == Y exactly.
    torch.testing.assert_close(aux_out['loss_action'], plain_out['loss_action'],
                               rtol=0, atol=0)
    reference = sample['reference_tokens']
    states = torch.randn(2, 8)
    with torch.no_grad():
        aux_action = aux.predict_action(reference_tokens=reference,
                                       text_tokens=sample['text_tokens'],
                                       states=states, seed=13)
        plain_action = plain.predict_action(reference_tokens=reference,
                                           text_tokens=sample['text_tokens'],
                                           states=states, seed=13)
    torch.testing.assert_close(aux_action, plain_action, rtol=0, atol=0)


def test_radio_only_gradients_reach_a_b_c_selected_block_and_final_projection():
    aux = _aux_vla(TinyRadioTeacher())
    head = aux.vla_head
    with torch.no_grad():
        # Break the zero symmetry so every branch participates.
        head.radio_b.weight.normal_(0, 0.1)
        head.radio_c.weight.normal_(0, 0.1)
    out = _loss(aux, _cached(), **_flow(seed=9))
    out['loss'].backward()
    for name in ('radio_a.weight', 'radio_b.weight', 'radio_c.weight'):
        grad = dict(head.named_parameters())[name].grad
        assert grad is not None and grad.abs().sum() > 0, name
    assert head.denoiser.blocks[-1].qkv.weight.grad.abs().sum() > 0
    assert head.denoiser.token_output.weight.grad.abs().sum() > 0
    assert aux.vlm_backbone.radio_teacher.kernel.grad is None
    # Injected accessors used by the fused export.
    assert set(head.adapter_keys) == set(
        name.split('.', 1)[1] for name in aux.RADIO_PROJECTION_KEYS
        if name.startswith('vla_head.radio_'))


def test_no_vae_velocity_mse_is_computed(monkeypatch):
    from fluxvla.models.third_party_models.patchwam import policy as policy_mod
    calls = []
    original = policy_mod.masked_sample_mse

    def spy(prediction, target, valid=None):
        calls.append(tuple(target.shape))
        return original(prediction, target, valid)

    monkeypatch.setattr(policy_mod, 'masked_sample_mse', spy)
    aux = _aux_vla()
    out = _loss(aux, _cached(), **_flow(seed=9))
    assert torch.isfinite(out['loss'])
    # Only the action MSE runs; the future VAE velocity MSE is never computed.
    assert len(calls) == 1 and calls[0][1] == 4
    assert not out['loss_video'].requires_grad
    assert torch.equal(out['loss_vae'], out['loss_video'])
    plain = _plain_vla()
    calls.clear()
    _loss(plain, _cached(), **_flow(seed=9))
    assert len(calls) == 2


def test_shared_noise_projection_target_is_fixed_seeded_and_unit_norm():
    matrix = make_radio_noise_projection(RADIO_DIM, 128, seed=0)
    assert matrix.dtype == torch.float32
    torch.testing.assert_close(matrix.norm(dim=1),
                               torch.ones(RADIO_DIM), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(make_radio_noise_projection(RADIO_DIM, 128, 0),
                               matrix, rtol=0, atol=0)
    aux = _aux_vla()
    assert 'vla_head.radio_noise_projection' in aux.state_dict()
    noise = torch.randn(2, 6, 128)
    projected = aux.vla_head.project_future_noise(noise)
    assert projected.shape == (2, 6, RADIO_DIM)
    torch.testing.assert_close(
        projected, torch.einsum('rc,btc->btr', matrix, noise), rtol=1e-5,
        atol=1e-5)
    with pytest.raises(ValueError, match='radio_noise_seed'):
        make_radio_noise_projection(RADIO_DIM, 128, seed=-1)


def test_radio_loss_uses_the_passed_parent_noise_and_masks_padding():
    aux = _aux_vla()
    sample = _cached()
    shared = _flow(seed=3)
    first = _loss(aux, sample, **shared)
    other_noise = dict(shared)
    other_noise['future_noise'] = torch.randn(2, 6, 128,
                                              generator=torch.Generator()
                                              .manual_seed(4))
    second = _loss(aux, sample, **other_noise)
    # The target is built from the actual parent future noise, so changing it
    # changes the RADIO objective.
    assert not torch.allclose(first['loss_radio'], second['loss_radio'])
    assert torch.isfinite(first['loss']) and torch.isfinite(second['loss'])

    masked = _cached()
    masked['future_valid'] = torch.ones(2, 6, dtype=torch.bool)
    masked['future_valid'][0, -1] = False
    masked['radio_future_tokens'] = masked['radio_future_tokens'].clone()
    masked['radio_future_tokens'][0, -1] = float('nan')
    out = _loss(aux, masked, **_flow(seed=3))
    assert torch.isfinite(out['loss_radio'])
    poisoned = copy.deepcopy(masked)
    poisoned['radio_future_tokens'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='radio_future_tokens'):
        _loss(aux, poisoned, **_flow(seed=3))


def test_zero_radio_weight_is_action_only_but_keeps_the_velocity_residual():
    aux = _aux_vla(vla_head=_aux_head(radio_weight=0.0))
    head = aux.vla_head
    assert head.capture_layer is None
    sample = _cached()
    sample.pop('radio_future_tokens')
    out = _loss(aux, sample, **_flow(seed=6))
    assert torch.isfinite(out['loss'])
    torch.testing.assert_close(out['loss'],
                               head.action_weight * out['loss_action'],
                               rtol=0, atol=0)
    assert not out['loss_radio'].requires_grad
    # The trained residual still executes: perturbing C changes the velocity.
    with torch.no_grad():
        head.radio_c.weight.normal_(0, 0.5)
    perturbed = _loss(aux, sample, **_flow(seed=6))
    assert not torch.allclose(perturbed['loss_action'], out['loss_action'])
    # No teacher, no target requirement, no capture.
    unusable = _aux_vla(None, vlm_backbone=_backbone(
        None, radio_model_path='/nonexistent/radio'),
        vla_head=_aux_head(radio_weight=0.0))
    assert torch.isfinite(_loss(unusable, sample, **_flow(seed=6))['loss'])


# ----------------------------------------------------------------------
# Export, strict resume, legacy initialization
# ----------------------------------------------------------------------
def test_unfused_export_loads_strictly_and_matches_parameter_count():
    aux = _aux_vla()
    runtime = aux.export_unfused_inference_state_dict()
    core = aux.export_original_core_state_dict()
    report = aux.exported_inference_keys_report()
    assert set(report['removed']) == set(aux.RADIO_TRAINING_KEYS)
    assert report['teacher_keys'] == []
    assert not any('radio_noise_projection' in key or
                   key.startswith('vla_head.radio_') for key in core)
    # The non-deployable core export is initialization-only and explicitly
    # loses the trained residual; the runtime export keeps it and strict-loads.
    assert all(key in runtime for key in aux.RADIO_TRAINING_KEYS)
    again = _aux_vla()
    again.load_state_dict(runtime, strict=True)
    torch.testing.assert_close(again.vla_head.radio_c.bias,
                               aux.vla_head.radio_c.bias)
    plain = _plain_vla()
    plain.load_state_dict(core, strict=True)
    assert (sum(t.numel() for t in core.values())
            == sum(t.numel() for t in plain.state_dict().values()))
    assert (sum(t.numel() for t in aux.state_dict().values())
            == sum(t.numel() for t in plain.state_dict().values())
            + sum(aux.state_dict()[key].numel()
                  for key in aux.RADIO_TRAINING_KEYS))


def test_native_resume_requires_adapters_and_noise_buffer():
    source = _aux_vla()
    state = copy.deepcopy(source.state_dict())
    target = _aux_vla()
    target.load_state_dict(copy.deepcopy(state))
    torch.testing.assert_close(target.vla_head.radio_a.weight,
                               source.vla_head.radio_a.weight)
    broken = copy.deepcopy(state)
    broken.pop('vla_head.radio_c.weight')
    fresh = target.vla_head.radio_c.weight.detach().clone()
    with pytest.raises(ValueError, match='missing'):
        target.load_state_dict(copy.deepcopy(broken), strict=False)
    torch.testing.assert_close(target.vla_head.radio_c.weight, fresh)
    no_buffer = copy.deepcopy(state)
    no_buffer.pop('vla_head.radio_noise_projection')
    with pytest.raises(ValueError, match='radio_noise_projection'):
        target.load_state_dict(copy.deepcopy(no_buffer), strict=False)
    bad_buffer = copy.deepcopy(state)
    bad_buffer['vla_head.radio_noise_projection'] = torch.full_like(
        bad_buffer['vla_head.radio_noise_projection'], float('nan'))
    with pytest.raises(ValueError, match='radio_noise_projection'):
        target.load_state_dict(copy.deepcopy(bad_buffer), strict=False)
    with pytest.raises(ValueError, match='missing='):
        target.load_patchwam_state_dict(copy.deepcopy(broken), strict=True)


def test_legacy_initializer_leaves_only_the_new_tensors_fresh():
    source = _aux_vla()
    # Explicitly named non-deployable core export, plus a real plain-model
    # state to prove the legacy path works from an original checkpoint.
    legacy = source.export_original_core_state_dict()
    assert 'vla_head.radio_noise_projection' not in legacy
    fresh = _aux_vla()
    report = fresh.initialize_from_legacy_state_dict(copy.deepcopy(legacy),
                                                    strict=True)
    assert report['left_fresh'] == sorted(
        fresh.RADIO_PROJECTION_KEYS + fresh.RADIO_BUFFER_KEYS)
    # The declared new tensors stay at their fresh (random) initialization.
    assert not torch.allclose(fresh.vla_head.radio_a.weight,
                              source.vla_head.radio_a.weight)
    torch.testing.assert_close(fresh.vla_head.state_projection.weight,
                               source.vla_head.state_projection.weight)
    torch.testing.assert_close(fresh.vla_head.denoiser.token_output.weight,
                               source.vla_head.denoiser.token_output.weight)
    corrupt = copy.deepcopy(legacy)
    corrupt['vla_head.state_projection.weight'] = torch.zeros(3, 3)
    with pytest.raises(ValueError, match='shape differs'):
        fresh.initialize_from_legacy_state_dict(copy.deepcopy(corrupt),
                                                strict=True)

    # A real plain PatchWAM state initializes the shared core identically.
    plain_state = _plain_vla().state_dict()
    initial = _aux_vla()
    report = initial.initialize_from_legacy_state_dict(
        copy.deepcopy(plain_state), strict=True)
    assert report['left_fresh'] == sorted(
        initial.RADIO_PROJECTION_KEYS + initial.RADIO_BUFFER_KEYS)


def test_fused_export_matches_the_training_model_and_rejects_early_layers():
    aux = _aux_vla()
    head = aux.vla_head
    with torch.no_grad():
        head.radio_a.weight.normal_(0, 0.2)
        head.radio_a.bias.normal_(0, 0.05)
        head.radio_b.weight.normal_(0, 0.2)
        head.radio_b.bias.normal_(0, 0.05)
        head.radio_c.weight.normal_(0, 0.2)
        head.radio_c.bias.normal_(0, 0.05)
    fused = _plain_vla()
    fused.vla_head.denoiser.enable_output_bias()
    deployed_state = aux.export_inference_state_dict()  # public default
    torch.testing.assert_close(aux.fused_output_state_dict()[
        'vla_head.denoiser.token_output.weight'],
        deployed_state['vla_head.denoiser.token_output.weight'], rtol=0,
        atol=0)
    assert 'vla_head.denoiser.token_output.bias' in deployed_state
    # Exactly one added 128-wide bias tensor for the fused deployable head.
    assert (sum(t.numel() for t in deployed_state.values())
            == sum(t.numel() for t in _plain_vla().state_dict().values())
            + 128)
    fused.load_state_dict(deployed_state, strict=True)
    before = {name: parameter.detach().clone()
              for name, parameter in aux.named_parameters()}
    aux.export_inference_state_dict()
    for name, parameter in aux.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    sample, states = _cached(), torch.randn(2, 8)
    with torch.no_grad():
        trained = aux.predict_action(reference_tokens=sample['reference_tokens'],
                                     text_tokens=sample['text_tokens'],
                                     states=states, seed=21)
        deployed = fused.predict_action(
            reference_tokens=sample['reference_tokens'],
            text_tokens=sample['text_tokens'], states=states, seed=21)
    torch.testing.assert_close(deployed, trained, rtol=1e-4, atol=1e-4)

    early = _aux_vla(vla_head=_aux_head(adapter_layer_idex=0))
    assert not early.vla_head.is_final_selection()
    with pytest.raises(ValueError, match='final-layer'):
        early.fused_output_state_dict()
    with pytest.raises(ValueError, match='final-layer'):
        early.export_inference_state_dict()  # public default also refuses
    # The documented unfused teacher-free path loads strictly and reproduces
    # identical seeded behaviour for an early selection.
    runtime = _aux_vla(None, vla_head=_aux_head(adapter_layer_idex=0))
    runtime.load_state_dict(early.export_unfused_inference_state_dict(),
                            strict=True)
    with torch.no_grad():
        early_action = early.predict_action(
            reference_tokens=sample['reference_tokens'],
            text_tokens=sample['text_tokens'], states=states, seed=21)
        runtime_action = runtime.predict_action(
            reference_tokens=sample['reference_tokens'],
            text_tokens=sample['text_tokens'], states=states, seed=21)
    torch.testing.assert_close(runtime_action, early_action, rtol=0, atol=0)


def test_teacher_is_never_used_for_inference_or_export():
    teacher = ExplodingTeacher()
    aux = _aux_vla(teacher)
    aux.eval()
    with torch.no_grad():
        actions = aux.predict_action(images=_video(), states=torch.randn(2, 8),
                                     task_description=['a', 'b'], seed=4)
    assert actions.shape == (2, 4, 7) and torch.isfinite(actions).all()
    assert teacher.calls == 0
    aux.export_inference_state_dict()
    aux.fused_output_state_dict()
    assert teacher.calls == 0


def test_invalid_radio_configuration_is_rejected():
    with pytest.raises(ValueError, match='video_weight'):
        build_vla_from_cfg(dict(
            type='PatchWAMRadioAuxVLA', vlm_backbone=_backbone(),
            vla_head=_aux_head(video_weight=0.5), action_horizon=4,
            num_inference_steps=2))
    with pytest.raises(ValueError, match='radio_weight'):
        build_vla_from_cfg(dict(
            type='PatchWAMRadioAuxVLA', vlm_backbone=_backbone(),
            vla_head=_aux_head(radio_weight=-1.0), action_horizon=4,
            num_inference_steps=2))
    with pytest.raises(ValueError, match='radio_dim'):
        build_vla_from_cfg(dict(
            type='PatchWAMRadioAuxVLA', vlm_backbone=_backbone(),
            vla_head=_aux_head(radio_dim=0), action_horizon=4,
            num_inference_steps=2))
    with pytest.raises(ValueError, match='adapter_layer_idex'):
        build_vla_from_cfg(dict(
            type='PatchWAMRadioAuxVLA', vlm_backbone=_backbone(),
            vla_head=_aux_head(adapter_layer_idex=5), action_horizon=4,
            num_inference_steps=2))


# ----------------------------------------------------------------------
# Official reduced FLUX.2 blocks
# ----------------------------------------------------------------------
def _small_transformer(depth=1, single=1):
    if str(FLUX_SOURCE) not in sys.path:
        sys.path.insert(0, str(FLUX_SOURCE))
    model_api = importlib.import_module('flux2.model')
    return model_api.Flux2(model_api.Klein4BParams(
        in_channels=128, context_in_dim=48, hidden_size=64, num_heads=4,
        depth=depth, depth_single_blocks=single, axes_dim=[4, 4, 4, 4],
        mlp_ratio=2.0, use_guidance_embed=False))


flux_required = pytest.mark.skipif(
    not (FLUX_SOURCE / 'flux2' / 'model.py').is_file(),
    reason='local official flux2 source is unavailable')


@flux_required
def test_official_flux_fused_gradients_with_checkpointing():
    denoiser = OfficialFluxDenoiser(_small_transformer(depth=2, single=1),
                                    gradient_checkpointing=True)
    policy = RadioAuxFlowPolicy(denoiser, action_dim=7, text_dim=48,
                                proprio_dim=None, token_dim=128,
                                radio_dim=RADIO_DIM, radio_weight=1.0,
                                video_weight=0.0, action_weight=0.0)
    with torch.no_grad():
        policy.radio_b.weight.normal_(0, 0.1)
        policy.radio_c.weight.normal_(0, 0.1)
    sample = {
        'reference_tokens': torch.randn(2, 3, 128),
        'future_tokens': torch.randn(2, 4, 128),
        'radio_future_tokens': torch.randn(2, 4, RADIO_DIM),
        'text_tokens': torch.randn(2, 5, 48),
        'action': torch.randn(2, 5, 7),
    }
    generator = torch.Generator().manual_seed(31)
    out = policy(sample, sigma=torch.tensor([0.4, 0.6]),
                 future_noise=torch.randn(2, 4, 128, generator=generator),
                 action_noise=torch.randn(2, 5, 128, generator=generator))
    assert torch.isfinite(out['loss']) and torch.isfinite(out['loss_radio'])
    out['loss'].backward()
    for parameter in (policy.radio_a.weight, policy.radio_b.weight,
                      policy.radio_c.weight,
                      denoiser.transformer.double_blocks[-1].img_attn.qkv.weight,
                      denoiser.transformer.final_layer.linear.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


# ----------------------------------------------------------------------
# Restored teacher/target coverage (unchanged semantics)
# ----------------------------------------------------------------------
class _RadioOutput:
    def __init__(self, summary, spatial):
        self.summary, self.spatial_features = summary, spatial


@pytest.mark.parametrize('api', ['tuple', 'dataclass', 'dict', 'tensor'])
def test_teacher_spatial_unpacking_accepts_the_official_returns(api):
    from fluxvla.models.third_party_models.patchwam.radio import \
        radio_feature_targets, radio_spatial_features
    teacher = TinyRadioTeacher()
    original = teacher.encode

    def encode(images):
        summary, spatial = original(images)
        if api == 'tuple':
            return summary, spatial
        if api == 'dataclass':
            return _RadioOutput(summary, spatial)
        if api == 'dict':
            return {'summary': summary, 'spatial_features': spatial}
        return spatial

    teacher.encode = encode
    targets = radio_feature_targets(torch.zeros(1, 3, 4, 6), teacher=teacher,
                                    raster=(4, 6), patch_size=2,
                                    minimum_resolution=2,
                                    expected_dim=RADIO_DIM)
    assert targets.shape == (1, 24, RADIO_DIM)
    with pytest.raises(TypeError, match='spatial'):
        radio_spatial_features({'summary': torch.zeros(2, RADIO_DIM)})


def test_backbone_targets_are_frozen_per_view_and_skippable():
    from fluxvla.models.backbones.vlms.patchwam_backbone import \
        PatchWAMBackbone
    from fluxvla.models.third_party_models.patchwam.radio import \
        load_local_radio_teacher
    teacher = TinyRadioTeacher()
    backbone = build_vla_from_cfg(dict(
        type='PatchWAMRadioAuxVLA', vlm_backbone=_backbone(teacher),
        vla_head=_aux_head(), action_horizon=4, num_inference_steps=2)
    ).vlm_backbone
    views = torch.zeros(1, 2, 3, 2, 4, 6)
    views[0, 0], views[0, 1] = -1.0, 1.0
    encoded = backbone.encode(dict(video=views, prompt=['move'],
                                   need_radio=True), target=True)
    tokens = encoded['radio_future_tokens'][0]
    one = backbone.encode(dict(video=views[:, :1], prompt=['move'],
                               need_radio=True),
                          target=True)['radio_future_tokens'][0]
    two = backbone.encode(dict(video=views[:, 1:], prompt=['move'],
                               need_radio=True),
                          target=True)['radio_future_tokens'][0]
    torch.testing.assert_close(tokens[:24], one)
    torch.testing.assert_close(tokens[24:], two)
    assert not tokens.requires_grad
    assert all(not p.requires_grad
               for p in backbone.radio_teacher.parameters())
    skipped = backbone.encode(dict(video=views, prompt=['move'],
                                   need_radio=False), target=True)
    assert 'radio_future_tokens' not in skipped
    # Local asset failures stay clear and never fall back.
    with pytest.raises(FileNotFoundError, match='local Hugging Face export'):
        load_local_radio_teacher('/nonexistent/radio')
    from fluxvla.models.backbones.vlms.patchwam_radio_backbone import \
        PatchWAMRadioAuxBackbone
    with pytest.raises(ValueError, match='not both'):
        PatchWAMRadioAuxBackbone(autoencoder=TinyAutoencoder(),
                                 text_encoder=TinyTextEncoder(), token_dim=128,
                                 text_dim=32, history_pool_size=2,
                                 radio_teacher=teacher,
                                 radio_model_path='/somewhere')


def test_invalid_padding_is_sanitized_before_the_teacher():
    backbone = build_vla_from_cfg(dict(
        type='PatchWAMRadioAuxVLA', vlm_backbone=_backbone(TinyRadioTeacher()),
        vla_head=_aux_head(), action_horizon=4, num_inference_steps=2)
    ).vlm_backbone
    video = _video().clone()
    video[1, 1] = float('nan')
    frame_masks = torch.tensor([[True, True], [True, False]])
    img_masks = torch.tensor([[True, True], [True, False]])
    encoded = backbone.encode(dict(video=video, prompt=['a', 'b'],
                                   frame_masks=frame_masks,
                                   img_masks=img_masks, need_radio=True),
                              target=True)
    assert not encoded['future_valid'][1].any()
    assert torch.count_nonzero(encoded['radio_future_tokens'][1]) == 0
    assert torch.isfinite(encoded['radio_future_tokens']).all()


def test_cached_targets_align_and_reject_supervised_nans():
    from fluxvla.models.third_party_models.patchwam.radio import \
        align_supervised_features
    tokens = torch.randn(2, 3, RADIO_DIM, dtype=torch.float64,
                         requires_grad=True)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    aligned = align_supervised_features(tokens, valid,
                                        name='radio_future_tokens',
                                        dtype=torch.bfloat16)
    assert aligned.dtype == torch.bfloat16 and not aligned.requires_grad
    padded = torch.randn(2, 3, RADIO_DIM)
    padded[0, 2] = float('nan')
    align_supervised_features(padded, valid, name='radio_future_tokens')
    poisoned = padded.clone()
    poisoned[0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        align_supervised_features(poisoned, valid,
                                  name='radio_future_tokens')


def test_fixed_projection_survives_dtype_conversion_and_bf16_training(monkeypatch):
    # This CPU's oneDNN backend lacks BF16 backward; exercise Torch's CPU path.
    monkeypatch.setattr(torch.backends.mkldnn, 'enabled', False)
    from fluxvla.models.third_party_models.patchwam.radio import \
        make_radio_noise_projection
    expected = make_radio_noise_projection(RADIO_DIM, 128, seed=0)
    aux = _aux_vla()
    torch.testing.assert_close(aux.vla_head.radio_noise_projection, expected,
                               rtol=0, atol=0)
    aux_bf16 = _aux_vla()
    aux_bf16.to(torch.bfloat16)
    buffer = aux_bf16.vla_head.radio_noise_projection
    assert buffer.dtype == torch.float32
    torch.testing.assert_close(buffer, expected, rtol=0, atol=0)
    aux_bf16.to(torch.float16).to(torch.float32)
    torch.testing.assert_close(aux_bf16.vla_head.radio_noise_projection,
                               expected, rtol=0, atol=0)
    # CPU bfloat16 forward/backward with a float32 cached target.
    aux_bf16.to(torch.bfloat16)
    assert aux_bf16.vla_head.denoiser.token_output.weight.dtype == torch.bfloat16
    sample = _cached()
    flow = _flow(seed=2)
    flow = {key: value.bfloat16() if key.endswith('_noise') else value
            for key, value in flow.items()}
    out = aux_bf16(actions=torch.randn(2, 4, 7).bfloat16(),
                   action_masks=torch.ones(2, 4, dtype=torch.bool),
                   states=torch.randn(2, 8).bfloat16(),
                   **{key: (value.bfloat16() if key != 'radio_future_tokens'
                            else value) for key, value in sample.items()},
                   **flow)
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    assert aux_bf16.vla_head.radio_a.weight.grad is not None
    # Checkpoint round trip keeps the exact float32 projection values.
    restored = _aux_vla()
    restored.load_state_dict(copy.deepcopy(aux_bf16.state_dict()), strict=True)
    torch.testing.assert_close(restored.vla_head.radio_noise_projection,
                               expected, rtol=0, atol=0)


@pytest.mark.parametrize('mode,joint,ee', [
    ('joint_only', 7, None), ('end_effector_only', None, 6), ('dual', 7, 6)])
def test_fused_sampling_preserves_future_and_actions_in_every_mode(mode, joint, ee):
    width = (joint or 0) + (ee or 0)
    layout = dict(action_mode=mode, joint_action_dim=joint,
                  end_effector_action_dim=ee)
    aux = _aux_vla(vla_head=_aux_head(action_dim=width, **layout))
    plain = build_vla_from_cfg(dict(
        type='PatchWAMVLA', action_horizon=4, num_inference_steps=3,
        vlm_backbone=dict(type='PatchWAMBackbone', autoencoder=TinyAutoencoder(),
                          text_encoder=TinyTextEncoder(), token_dim=128,
                          text_dim=32, history_pool_size=2),
        vla_head=dict(type='PatchWAMHead', action_dim=width, text_dim=32,
                      proprio_dim=8, token_dim=128, **layout,
                      denoiser=dict(type='PatchWAMTinyDenoiser', **TINY,
                                    output_bias=True))))
    with torch.no_grad():
        for adapter in (aux.vla_head.radio_a, aux.vla_head.radio_b,
                        aux.vla_head.radio_c):
            adapter.weight.normal_(0, 0.03)
            adapter.bias.normal_(0, 0.02)
    plain.load_state_dict(aux.export_inference_state_dict(), strict=True)
    batch = _cached()
    batch['proprio'] = torch.randn(2, 8)
    options = dict(horizon=4, future_length=6, steps=3,
                   guidance_scale=1.4, action_guidance_scale=1.2)
    trained = aux.vla_head.sample_actions(
        batch, generator=torch.Generator().manual_seed(99), **options)
    deployed = plain.vla_head.sample_actions(
        batch, generator=torch.Generator().manual_seed(99), **options)
    for key in ('action', 'future_tokens'):
        torch.testing.assert_close(deployed[key], trained[key], rtol=1e-5,
                                   atol=1e-5)
    assert set(plain.vla_head.codec.split(deployed['action'])) == set(
        aux.vla_head.codec.coordinate_layout)


@flux_required
@pytest.mark.parametrize('selected', [0, 1, 2, -1])
def test_official_selected_conditioned_hidden_and_biased_fusion(selected, monkeypatch):
    denoiser = OfficialFluxDenoiser(_small_transformer(depth=2, single=1),
                                    gradient_checkpointing=True)
    denoiser.enable_output_bias()
    policy = RadioAuxFlowPolicy(denoiser, action_dim=7, text_dim=48,
                                radio_dim=RADIO_DIM, adapter_layer_idex=selected)
    raw = {}
    paired, unified = denoiser._paired_block, denoiser._unified_block
    core = denoiser.transformer

    def capture_pair(block, *args):
        result = paired(block, *args)
        raw[list(core.double_blocks).index(block)] = result[1][:, 3:]
        return result

    def capture_single(block, *args):
        result = unified(block, *args)
        raw[2] = result[:, 5 + 3:]
        return result

    monkeypatch.setattr(denoiser, '_paired_block', capture_pair)
    monkeypatch.setattr(denoiser, '_unified_block', capture_single)
    with torch.no_grad():
        core.final_layer.linear.bias.normal_(0, 0.03)
        for adapter in (policy.radio_a, policy.radio_b, policy.radio_c):
            adapter.weight.normal_(0, 0.03)
            adapter.bias.normal_(0, 0.02)
    batch = dict(reference_tokens=torch.randn(2, 3, 128),
                 text_tokens=torch.randn(2, 5, 48))
    future, action = torch.randn(2, 4, 128), torch.randn(2, 5, 128)
    sigma = torch.tensor([0.3, 0.7])
    prediction_x, prediction_u, context = policy._velocity(batch, future, action, sigma)
    time = core.time_in(denoiser._embed_time(sigma, 256))
    shift, scale = core.final_layer.adaLN_modulation(time).chunk(2, -1)
    expected = (1 + scale[:, None]) * core.final_layer.norm_final(
        raw[2 if selected == -1 else selected]) + shift[:, None]
    torch.testing.assert_close(context['selected_hidden'], expected, rtol=0, atol=0)
    if selected in (-1, 2):
        weight, bias = policy.fused_output_affine()
        expected_velocity = F.linear(context['final_hidden'], weight, bias)
        torch.testing.assert_close(torch.cat((prediction_x, prediction_u), 1),
                                   expected_velocity, rtol=1e-5, atol=1e-5)
    prediction_x.square().mean().backward()
    assert policy.radio_b.weight.grad is not None
