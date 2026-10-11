# Copyright 2026 Limx Dynamics
"""Native integration for independent visual widths and staged adaptation."""

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from fluxvla.engines import HEADS, VLM_BACKBONES, build_vla_from_cfg
from fluxvla.models.backbones.visions import RadioViTBackbone
from fluxvla.models.backbones.vlms import patchwam_backbone  # noqa: F401
from fluxvla.models.blocks.dit_lora import lora_parameter_names
from fluxvla.models.heads import patchwam_head  # noqa: F401
from fluxvla.models.third_party_models.patchwam.policy import PatchFlowPolicy
from fluxvla.models.third_party_models.patchwam.tiny import \
    SmallJointTransformer
from fluxvla.models.vlas import patchwam_vla  # noqa: F401

VISUAL_DIM = 6
TOKEN_DIM = 16
TEXT_DIM = 8
ACTION_DIM = 7


class StubRadio(nn.Module):
    """Frozen C-RADIO stand-in with the official call contract."""

    def __init__(self, feature_dim=VISUAL_DIM, patch_size=16):
        super().__init__()
        self.feature_dim = feature_dim
        self.stem = nn.Conv2d(3, feature_dim, patch_size, patch_size)
        self.seen = []

    @staticmethod
    def get_nearest_supported_resolution(height, width):
        return (height // 16 * 16, width // 16 * 16)

    def forward(self, images, feature_fmt='NLC'):
        self.seen.append((feature_fmt, images.detach().clone()))
        spatial = self.stem(images)
        return spatial.mean(dim=(2, 3))[:, None], spatial


class TinyTextEncoder(nn.Module):
    requires_images = False
    trainable_adapters = False

    def __init__(self, dim=TEXT_DIM):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(3, dim))
        self.feature_dim = dim

    def forward(self, prompts):
        return self.embed.unsqueeze(0).expand(len(prompts), -1, -1)


def tiny_policy(*, visual_dim=None, token_dim=TOKEN_DIM):
    torch.manual_seed(5)
    denoiser = SmallJointTransformer(token_dim=token_dim, text_dim=TEXT_DIM,
                                     width=32, heads=4, depth=2,
                                     visual_dim=visual_dim)
    return PatchFlowPolicy(denoiser, action_dim=ACTION_DIM, text_dim=TEXT_DIM,
                           proprio_dim=ACTION_DIM, token_dim=token_dim,
                           visual_dim=visual_dim)


def staged_head(*, visual_dim=VISUAL_DIM, token_dim=TOKEN_DIM, stage='stage1',
                lora=None):
    denoiser = dict(type='PatchWAMTinyDenoiser', token_dim=token_dim,
                    text_dim=TEXT_DIM, width=32, heads=4, depth=2,
                    visual_dim=visual_dim)
    return dict(type='PatchWAMHead', action_dim=ACTION_DIM, text_dim=TEXT_DIM,
                proprio_dim=ACTION_DIM, token_dim=token_dim,
                visual_dim=visual_dim, stage=stage, lora=lora,
                denoiser=denoiser)


#: The CPU-backed joint transformer keeps its own block/layer names, so the
#: LoRA target list is supplied explicitly (the default list is the official
#: FLUX.2 double/single block layout).
TINY_LORA = dict(rank=2, alpha=4, targets=('qkv', 'mix', 'ff.0', 'ff.2'),
                 blocks=('blocks',))


def visual_backbone(visual_encoder):
    return dict(type='PatchWAMBackbone', autoencoder=None,
                text_encoder=TinyTextEncoder(), token_dim=TOKEN_DIM,
                text_dim=TEXT_DIM, history_pool_size=2,
                visual_encoder=visual_encoder)


def batch(visual_dim=VISUAL_DIM, token_dim=TOKEN_DIM):
    generator = torch.Generator().manual_seed(11)

    def draw(*shape):
        return torch.randn(*shape, generator=generator)

    return {
        'reference_tokens': draw(2, 2, visual_dim),
        'future_tokens': draw(2, 2, visual_dim),
        'text_tokens': draw(2, 3, TEXT_DIM),
        'action': draw(2, 4, ACTION_DIM),
        'proprio': draw(2, ACTION_DIM),
    }


def test_registered_head_opens_the_stage_groups():
    head = HEADS.build(staged_head(stage='stage1'))
    trainable = {name for name, param in head.named_parameters()
                 if param.requires_grad}
    assert trainable == set(head.visual_parameter_names())
    assert [group['name'] for group in head.parameter_groups()] == \
        ['visual_projection']
    assert not head.state_projection.weight.requires_grad
    assert not head.denoiser.token_input.weight.requires_grad
    stage2 = HEADS.build(staged_head(stage='stage2',
                                     lora=TINY_LORA))
    adapters = {f'denoiser.{name}'
                for name in lora_parameter_names(stage2.denoiser)}
    assert adapters
    trainable2 = {name for name, param in stage2.named_parameters()
                  if param.requires_grad}
    assert trainable2 == set(stage2.visual_parameter_names()) | adapters
    assert [group['name'] for group in stage2.parameter_groups()] == \
        ['visual_projection', 'dit_lora']


def test_fused_velocity_transform_is_rejected_for_distinct_widths():
    policy = tiny_policy(visual_dim=VISUAL_DIM)
    with pytest.raises(ValueError, match='shared token width'):
        policy.install_visual_width_check(fused=True)
    policy.install_visual_width_check(fused=False)
    head = HEADS.build(staged_head())
    head.fused_hidden_layer = 1
    head.velocity_transform = lambda velocity, selected: velocity
    with pytest.raises(ValueError, match='shared token width'):
        head(batch())


def test_distinct_width_forward_and_sampling_in_the_vendored_core():
    policy = tiny_policy(visual_dim=VISUAL_DIM)
    data = batch()
    result = policy(data)
    assert torch.isfinite(result['loss']) and result['loss'] > 0
    result['loss'].backward()
    assert policy.denoiser.visual_in.weight.grad.abs().sum() > 0
    assert policy.denoiser.visual_out.weight.grad.abs().sum() > 0
    sampled = policy.sample_actions(data, horizon=4, future_length=2, steps=2,
                                    generator=torch.Generator().manual_seed(3))
    assert sampled['future_tokens'].shape == (2, 2, VISUAL_DIM)
    assert sampled['action'].shape == (2, 4, ACTION_DIM)
    with pytest.raises(ValueError, match='reference_tokens must be'):
        policy(dict(data, reference_tokens=torch.randn(2, 2, TOKEN_DIM)))
    with pytest.raises(ValueError, match='future_tokens must be'):
        policy(dict(data, future_tokens=torch.randn(2, 2, TOKEN_DIM)))


def test_shared_width_default_keeps_the_legacy_layout():
    policy = tiny_policy()
    assert policy.visual_dim == policy.codec.token_dim == TOKEN_DIM
    assert policy.denoiser.visual_in is None
    assert policy.denoiser.visual_out is None
    assert policy.visual_tokens_are_shared_width


def test_backbone_uses_the_shared_vision_wrapper_for_tokens_and_geometry():
    encoder = RadioViTBackbone('c-radio_v4-h', model=StubRadio(),
                               feature_dim=VISUAL_DIM, patch_size=16,
                               resolution_step=16)
    model = VLM_BACKBONES.build(visual_backbone(encoder))
    assert model.autoencoder is None and model.visual_dim == VISUAL_DIM
    video = torch.rand(2, 3, 3, 2, 32, 32) * 2 - 1
    encoded = model.encode(
        {'images': video, 'prompt': ['a', 'b']}, target=True)
    grid = encoder.feature_grid(32, 32)
    tokens_per_view = grid[0] * grid[1]
    assert encoded['reference_tokens'].shape == (2, 3 * tokens_per_view,
                                                 VISUAL_DIM)
    assert encoded['future_tokens'].shape == (2, 3 * tokens_per_view,
                                              VISUAL_DIM)
    # One positional group per camera view, for the current and future rasters.
    ids = sorted(set(encoded['reference_ids'][..., 0].flatten().tolist()))
    assert ids == [10., 11., 12.]
    assert sorted(set(encoded['future_ids'][..., 0].flatten().tolist())) == \
        [0., 1., 2.]
    mode, seen = encoder.model.seen[0]
    assert mode == 'NCHW'
    current = video[:, :, :, 0].flatten(0, 1)
    torch.testing.assert_close(seen.float(), (current + 1) / 2, atol=2e-3,
                               rtol=0)
    model.train()
    assert not model.visual_encoder.training
    assert not any(param.requires_grad
                   for param in model.visual_encoder.parameters())


def stage_model(*, stage, visual_dim=VISUAL_DIM, lora=None, pretrained=None,
                stage1_checkpoint=None, token_dim=TOKEN_DIM):
    encoder = RadioViTBackbone('c-radio_v4-h', model=StubRadio(),
                               feature_dim=visual_dim, patch_size=16,
                               resolution_step=16)
    return build_vla_from_cfg(dict(
        type='PatchWAMVLA',
        action_horizon=4,
        num_inference_steps=2,
        vlm_backbone=visual_backbone(encoder),
        vla_head=staged_head(visual_dim=visual_dim, token_dim=token_dim,
                             stage=stage, lora=lora),
        visual_stage=stage,
        pretrained_name_or_path=pretrained,
        stage1_checkpoint=stage1_checkpoint,
    ))


def test_stage1_initialization_is_exact_and_only_projections_are_new(tmp_path):
    pretrained = tiny_policy()
    path = tmp_path / 'pretrained.safetensors'
    save_file({key: value.clone()
               for key, value in pretrained.state_dict().items()}, str(path))
    model = stage_model(stage='stage1', pretrained=str(path))
    before = {name: value.clone() for name, value in
              model.vla_head.denoiser.visual_in.state_dict().items()}
    report = model.initialize_visual_stage()
    expected_new = sorted(
        f'vla_head.{name}' for name in model.vla_head.visual_parameter_names())
    assert report['new_keys'] == expected_new
    for key, value in pretrained.state_dict().items():
        torch.testing.assert_close(model.state_dict()['vla_head.' + key],
                                   value, rtol=0, atol=0)
    for key, value in before.items():
        torch.testing.assert_close(
            model.vla_head.denoiser.visual_in.state_dict()[key], value,
            rtol=0, atol=0)


def test_stage2_inherits_stage1_and_keeps_adapters_zero(tmp_path):
    stage1_model = stage_model(stage='stage1')
    stage1_path = tmp_path / 'stage1.safetensors'
    save_file({key: value.clone() for key, value in
               stage1_model.vla_head.state_dict().items()}, str(stage1_path))
    model = stage_model(stage='stage2', lora=TINY_LORA,
                        stage1_checkpoint=str(stage1_path))
    adapters = {name: value.clone() for name, value in
                model.vla_head.named_parameters() if 'dit_lora.' in name}
    report = model.initialize_visual_stage()
    assert report['stage'] == 'stage2'
    expected_new = sorted(
        f'vla_head.denoiser.{name}'
        for name in lora_parameter_names(model.vla_head.denoiser))
    assert report['new_keys'] == expected_new
    for key, value in stage1_model.vla_head.state_dict().items():
        torch.testing.assert_close(model.vla_head.state_dict()[key], value,
                                   rtol=0, atol=0)
    parameters = dict(model.vla_head.named_parameters())
    for name, value in adapters.items():
        torch.testing.assert_close(parameters[name], value, rtol=0, atol=0)


def test_stage_initialization_is_atomic_and_load_stays_strict(tmp_path):
    pretrained = tiny_policy()
    weights = {key: value.clone()
               for key, value in pretrained.state_dict().items()}
    weights[next(iter(weights))] = torch.zeros(1)
    path = tmp_path / 'broken.safetensors'
    save_file(weights, str(path))
    model = stage_model(stage='stage1', pretrained=str(path))
    before = {name: value.clone()
              for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match='shape differs'):
        model.initialize_visual_stage()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    good = tmp_path / 'good.safetensors'
    save_file({key: value.clone()
               for key, value in pretrained.state_dict().items()}, str(good))
    model = stage_model(stage='stage1', pretrained=str(good))
    state = {key: value.clone()
             for key, value in pretrained.state_dict().items()}
    with pytest.raises(ValueError, match='Strict PatchWAM checkpoint load'):
        model.load_patchwam_state_dict(state)


def test_stage_initialization_requires_the_matching_sources():
    model = stage_model(stage='stage1')
    with pytest.raises(ValueError, match='pretrained_name_or_path'):
        model.initialize_visual_stage()
    with pytest.raises(ValueError, match='visual_stage=stage2'):
        stage_model(stage='stage2', lora=TINY_LORA)


def test_stage_policies_cooperate_with_requires_grad_optimizer_selection():
    model = stage_model(stage='stage2', lora=TINY_LORA,
                        stage1_checkpoint='/tmp/stage1.pth')
    model.apply_visual_stage_policy()
    trainable = {name for name, param in model.named_parameters()
                 if param.requires_grad}
    assert trainable
    assert all('visual_in' in name or 'visual_out' in name
               or 'dit_lora' in name for name in trainable)
    assert not any(param.requires_grad
                   for param in model.vlm_backbone.parameters())
    assert not model.vla_head.denoiser.token_input.weight.requires_grad
    result = model.vla_head(batch())
    result['loss'].backward()
    gradients = dict(model.named_parameters())
    assert gradients['vla_head.denoiser.visual_in.weight'].grad.abs().sum() > 0
    visual_out = gradients['vla_head.denoiser.visual_out.weight']
    assert visual_out.grad.abs().sum() > 0
    adapters = [name for name in gradients if 'dit_lora.' in name
                and name.endswith('.up.weight')]
    assert adapters and all(gradients[name].grad is not None
                            and gradients[name].grad.abs().sum() > 0
                            for name in adapters)
    assert gradients['vla_head.denoiser.token_input.weight'].grad is None
    assert gradients['vla_head.state_projection.weight'].grad is None


def trainable_names(model):
    return sorted(name for name, param in model.named_parameters()
                  if param.requires_grad)


def test_runner_lifecycle_orders_load_the_stage_source_and_keep_the_freeze(
        tmp_path):
    """Both runner orders must end with the same frozen/trainable split.

    The DDP runner calls ``freeze_backbones()`` before ``from_pretrained()``
    and the FSDP runner calls them in the opposite order, so the stage policy
    and the load have to be order independent.
    """
    torch.manual_seed(0)
    pretrained = tiny_policy()
    path = tmp_path / 'pretrained.safetensors'
    save_file({key: value.clone()
               for key, value in pretrained.state_dict().items()}, str(path))
    ddp = stage_model(stage='stage1', pretrained=str(path))
    ddp.freeze_backbones()
    ddp.from_pretrained()
    fsdp = stage_model(stage='stage1', pretrained=str(path))
    fsdp.from_pretrained()
    fsdp.freeze_backbones()
    expected = sorted(f'vla_head.{name}'
                      for name in ddp.vla_head.visual_parameter_names())
    assert expected
    assert trainable_names(ddp) == trainable_names(fsdp) == expected
    for key, value in pretrained.state_dict().items():
        torch.testing.assert_close(ddp.state_dict()['vla_head.' + key], value,
                                   rtol=0, atol=0)
        torch.testing.assert_close(fsdp.state_dict()['vla_head.' + key], value,
                                   rtol=0, atol=0)
    # Stage 2 must inherit the stage 1 checkpoint, never the base pretrained
    # path, even when a recipe leaves that path set.
    stage1_path = tmp_path / 'stage1.safetensors'
    save_file({key: value.clone()
               for key, value in ddp.vla_head.state_dict().items()},
              str(stage1_path))
    stage2 = stage_model(stage='stage2', lora=TINY_LORA,
                         pretrained=str(path),
                         stage1_checkpoint=str(stage1_path))
    stage2.freeze_backbones()
    stage2.from_pretrained()
    assert stage2._stage_initialization['source'] == str(stage1_path)
    assert stage2._stage_initialization['stage'] == 'stage2'
    for key, value in ddp.vla_head.state_dict().items():
        torch.testing.assert_close(stage2.vla_head.state_dict()[key], value,
                                   rtol=0, atol=0)
    assert [group['name'] for group in stage2.vla_head.parameter_groups()] == \
        ['visual_projection', 'dit_lora']
    assert trainable_names(stage2) != trainable_names(ddp)


@pytest.mark.parametrize('dropped', [
    'denoiser.blocks.0.qkv.weight', 'denoiser.token_input.weight',
    'state_projection.weight',
])
def test_stage1_load_requires_frozen_inherited_tensors(tmp_path, dropped):
    """Stage 1 freezes the DiT/action/state maps; a load still needs them."""
    pretrained = tiny_policy()
    weights = {key: value.clone()
               for key, value in pretrained.state_dict().items()}
    del weights[dropped]
    path = tmp_path / 'damaged.safetensors'
    save_file(weights, str(path))
    model = stage_model(stage='stage1', pretrained=str(path))
    model.freeze_backbones()
    before = {name: value.clone()
              for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match='Strict PatchWAM checkpoint load'):
        model.from_pretrained()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_stage_loads_reject_an_empty_payload_atomically(tmp_path):
    path = tmp_path / 'empty.safetensors'
    save_file({}, str(path))
    model = stage_model(stage='stage1', pretrained=str(path))
    model.freeze_backbones()
    before = {name: value.clone()
              for name, value in model.state_dict().items()}
    with pytest.raises(ValueError):
        model.from_pretrained()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_stage2_requires_the_stage1_visual_projections(tmp_path):
    """Stage 2 may only be missing its own adapters, never stage 1's work."""
    stage1_model = stage_model(stage='stage1')
    stripped = {key: value.clone() for key, value in
                stage1_model.vla_head.state_dict().items()
                if 'visual_in' not in key and 'visual_out' not in key}
    path = tmp_path / 'stripped.safetensors'
    save_file(stripped, str(path))
    model = stage_model(stage='stage2', lora=TINY_LORA,
                        stage1_checkpoint=str(path))
    model.freeze_backbones()
    with pytest.raises(ValueError, match='Strict PatchWAM checkpoint load'):
        model.from_pretrained()


def test_stage_configuration_requires_a_matching_head_stage():
    with pytest.raises(ValueError, match='requires a matching'):
        build_vla_from_cfg(dict(
            type='PatchWAMVLA', num_inference_steps=2,
            vlm_backbone=visual_backbone(_stub_encoder()),
            vla_head=dict(staged_head(stage='stage1'), stage=None),
            visual_stage='stage1'))
    with pytest.raises(ValueError, match='conflicts with'):
        build_vla_from_cfg(dict(
            type='PatchWAMVLA', num_inference_steps=2,
            vlm_backbone=visual_backbone(_stub_encoder()),
            vla_head=staged_head(stage='stage1'),
            visual_stage='stage2'))


def _stub_encoder():
    return RadioViTBackbone('c-radio_v4-h', model=StubRadio(),
                            feature_dim=VISUAL_DIM, patch_size=16,
                            resolution_step=16)
