# Copyright 2026 Limx Dynamics
"""CPU units for the native PatchWAM backbone plus the VLA integration."""

import copy

import pytest
import torch
import torch.nn as nn

from fluxvla.engines import (HEADS, VLM_BACKBONES, build_head_from_cfg,
                             build_vla_from_cfg)
from fluxvla.models.backbones.vlms.patchwam_backbone import PatchWAMBackbone
from fluxvla.models.third_party_models.patchwam.vision_language import (
    LowRankResidualLinear, VisualInstructionEncoder)
from fluxvla.models.vlas.patchwam_vla import (DEFAULT_PATCHWAM_NAME_MAPPING,
                                              DEFAULT_PATCHWAM_PROMPT_TEMPLATE,
                                              PatchWAMVLA)


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


class TinyLoRAEncoder(nn.Module):
    """Stub text encoder exposing the LoRA residual naming contract."""

    requires_images = False
    trainable_adapters = True

    def __init__(self, dim: int = 32):
        super().__init__()
        self.base = nn.Linear(dim, dim)
        self.residual_in = nn.Linear(dim, 4, bias=False)
        self.residual_out = nn.Linear(4, dim, bias=False)

    def forward(self, prompts):
        return self.base.weight[:1].expand(len(prompts), 3, -1)


class SubtaskTextEncoder(nn.Module):
    requires_images = False
    trainable_adapters = False
    auxiliary_weight = 0.25

    def __init__(self):
        super().__init__()
        self.aux = nn.Parameter(torch.tensor(2.0))

    def forward(self, prompts):
        batch = len(prompts)
        return {
            'text_tokens': torch.zeros(batch, 3, 32),
            'text_valid': torch.ones(batch, 3, dtype=torch.bool),
            'auxiliary_loss': self.aux * 1.0,
        }


class RecordingTextEncoder(nn.Module):
    """Prompt-insensitive encoder that records the prompts it receives."""

    requires_images = False
    trainable_adapters = False

    def __init__(self, dim: int = 32, length: int = 3):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(length, dim))
        self.feature_dim = dim
        self.seen_prompts = []

    def forward(self, prompts):
        prompts = list(prompts)
        self.seen_prompts.append(prompts)
        return self.embed.unsqueeze(0).expand(len(prompts), -1, -1)


class VLRecordingTextEncoder(RecordingTextEncoder):
    """VL-style encoder that records the prompts and kwargs it receives."""

    requires_images = True

    def forward(self, prompts, current=None, history=None, history_valid=None,
                subtasks=None):
        self.seen_kwargs = dict(current=current, history=history,
                                history_valid=history_valid, subtasks=subtasks)
        return super().forward(prompts)


def tiny_head_cfg(**overrides):
    cfg = dict(
        type='PatchWAMHead',
        denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128, text_dim=32,
                      width=32, heads=4, depth=2),
        action_dim=14,
        text_dim=32,
        proprio_dim=14,
        token_dim=128,
    )
    cfg.update(overrides)
    return cfg


def build_backbone(**overrides):
    cfg = dict(autoencoder=TinyAutoencoder(),
               text_encoder=TinyTextEncoder(),
               token_dim=128,
               text_dim=32,
               history_pool_size=2)
    cfg.update(overrides)
    return PatchWAMBackbone(**cfg)


def build_video(batch=2, views=2, frames=5, height=4, width=6):
    return torch.rand(batch, views, 3, frames, height, width)


def build_vla(**overrides):
    cfg = dict(
        type='PatchWAMVLA',
        vlm_backbone=dict(
            type='PatchWAMBackbone',
            autoencoder=TinyAutoencoder(),
            text_encoder=TinyTextEncoder(),
            token_dim=128,
            text_dim=32,
            history_pool_size=2,
        ),
        vla_head=tiny_head_cfg(),
        action_horizon=16,
        num_inference_steps=2,
    )
    cfg.update(overrides)
    return build_vla_from_cfg(cfg)


def test_encode_raster_ids_masks_and_sampling_length():
    backbone = build_backbone()
    video = build_video()
    frame_masks = torch.tensor([[True, True], [True, False]])
    img_masks = torch.tensor([[True, True], [True, False]])
    encoded = backbone.encode(
        dict(video=video, prompt=['move', 'lift'], frame_masks=frame_masks,
             img_masks=img_masks), target=True)
    token_dim = 128
    tokens = 2 * 4 * 6
    assert encoded['reference_tokens'].shape == (2, tokens, token_dim)
    assert encoded['future_tokens'].shape == (2, tokens, token_dim)
    # Reference ids use group 10+view, future ids use group=view.
    assert torch.equal(encoded['reference_ids'][0, :, 0],
                       torch.cat([torch.full((24, ), 10.0),
                                  torch.full((24, ), 11.0)]))
    assert torch.equal(encoded['future_ids'][0, :, 0],
                       torch.cat([torch.zeros(24), torch.ones(24)]))
    assert encoded['sampling_future_length'] == tokens
    assert encoded['text_tokens'].shape == (2, 3, 32)
    # Sample 0 keeps both cameras; sample 1 drops camera 1 and the future row.
    assert encoded['reference_valid'][0].all()
    assert not encoded['reference_valid'][1, 24:].any()
    assert encoded['future_valid'][0].all()
    assert not encoded['future_valid'][1].any()


def test_history_pooling_extends_reference_only():
    backbone = build_backbone()
    video = build_video()
    history = torch.rand(2, 3, 3, 4, 6)
    history_valid = torch.tensor([[True, True, False], [False, True, True]])
    encoded = backbone.encode(
        dict(video=video, prompt=['move', 'lift'], history_video=history,
             history_valid=history_valid), target=True)
    tokens = 2 * 4 * 6
    history_tokens = 3 * 2 * 2
    assert encoded['reference_tokens'].shape == (2, tokens + history_tokens, 128)
    assert encoded['future_tokens'].shape == (2, tokens, 128)
    assert encoded['sampling_future_length'] == tokens
    assert encoded['reference_valid'].shape == (2, tokens + history_tokens)
    assert not encoded['reference_valid'][1, tokens:tokens + 4].any()
    assert encoded['reference_valid'][1, tokens + 4:].all()


def test_missing_history_mask_is_an_error():
    backbone = build_backbone()
    with pytest.raises(ValueError, match='validity mask'):
        backbone.encode(
            dict(video=build_video(), prompt=['a', 'b'],
                 history_video=torch.rand(2, 2, 3, 4, 6), history_valid=None),
            target=False)


def test_freeze_policy_keeps_lora_and_freezes_base():
    encoder = TinyLoRAEncoder()
    backbone = PatchWAMBackbone(autoencoder=TinyAutoencoder(),
                                text_encoder=encoder, token_dim=128,
                                text_dim=32)
    assert all(not p.requires_grad for p in backbone.autoencoder.parameters())
    assert not encoder.base.weight.requires_grad
    assert encoder.residual_in.weight.requires_grad
    assert encoder.residual_out.weight.requires_grad
    trainable = set(backbone.trainable_parameter_names())
    assert 'text_encoder.residual_in.weight' in trainable
    assert 'text_encoder.base.weight' not in trainable


def test_visual_instruction_encoder_attaches_lora_and_contract():
    class _Visual(nn.Module):
        pass

    class _Language(nn.Module):
        def __init__(self, width=8):
            super().__init__()
            self.q_proj = nn.Linear(width, width)
            self.v_proj = nn.Linear(width, width)

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = _Language()
            self.visual = _Visual()
            self.config = type('Cfg', (), {'text_config': type(
                'Text', (), {'num_hidden_layers': 40, 'hidden_size': 8})})

    class _Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()
            self.config = self.model.config

    encoder = VisualInstructionEncoder(encoder=_Encoder(), processor=object(),
                                       instruction_layers=(9,),
                                       summary_layers=(18,),
                                       trainable_adapters=True)
    assert encoder.adapter_count == 2
    assert encoder.feature_dim == 8
    assert encoder.checkpoint_contract()['kind'] == 'causal_qwen3_vl'
    adapters = [
        module for module in encoder.modules()
        if isinstance(module, LowRankResidualLinear)
    ]
    assert len(adapters) == 2


def test_backbone_config_registered():
    assert VLM_BACKBONES.get('PatchWAMBackbone') is not None


def test_config_built_training_and_prediction_with_backbone():
    vla = build_vla()
    video = build_video()
    actions = torch.randn(2, 16, 14)
    action_masks = torch.ones(2, 16, dtype=torch.bool)
    action_masks[1, -2:] = False
    frame_masks = torch.tensor([[True, True], [True, False]])
    img_masks = torch.ones(2, 2, dtype=torch.bool)
    out = vla(video=video, states=torch.randn(2, 16, 14), actions=actions,
              action_masks=action_masks, frame_masks=frame_masks,
              img_masks=img_masks, task_description=['move', 'lift'])
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    assert vla.vla_head.state_projection.weight.grad is not None
    assert vla.vlm_backbone.autoencoder.proj.weight.grad is None

    states = torch.zeros(2, 14)
    prediction = vla.predict_action(video=video[:, :, :, 0:1], states=states,
                                    task_description=['move', 'lift'], seed=5)
    assert prediction.shape == (2, 16, 14)
    repeat = vla.predict_action(video=video[:, :, :, 0:1], states=states,
                                task_description=['move', 'lift'], seed=5)
    torch.testing.assert_close(prediction, repeat, rtol=0, atol=0)


def test_prediction_future_length_ignores_history_prefix():
    vla = build_vla()
    video = build_video(frames=1)
    history = torch.rand(2, 2, 3, 4, 6)
    history_valid = torch.ones(2, 2, dtype=torch.bool)
    out = vla.predict_action(video=video, states=torch.randn(2, 14),
                             task_description=['move', 'lift'],
                             history_video=history, history_valid=history_valid,
                             initial_future=torch.zeros(2, 48, 128),
                             initial_action=torch.zeros(2, 16, 128),
                             num_inference_steps=1)
    assert out.shape == (2, 16, 14)


def test_checkpoint_mapping_report_and_strict_validation():
    vla = build_vla()
    native_keys = list(vla.state_dict())
    source_keys = [
        name.replace('vlm_backbone.autoencoder.', 'autoencoder.').replace(
            'vlm_backbone.text_encoder.', 'text_encoder.').replace(
                'vla_head.', 'policy.') for name in native_keys
    ]
    report = vla.checkpoint_mapping_report(source_keys)
    assert not report['missing']
    assert not report['unexpected']
    assert len(report['resolved']) == len(native_keys)
    # Remove one key -> strict validation must fail.
    strict = build_vla(strict_mapping=True)
    with pytest.raises(ValueError, match='Strict PatchWAM checkpoint mapping'):
        strict.checkpoint_mapping_report(source_keys[:-1])


def test_fsdp_policy_and_disjoint_ownership():
    vla = build_vla()
    policy = vla.get_fsdp_wrapping_policy()
    assert callable(policy)
    head_keys = {name for name, _ in vla.vla_head.named_parameters()}
    backbone_keys = {name for name, _ in vla.vlm_backbone.named_parameters()}
    assert head_keys.isdisjoint(backbone_keys)
    state_keys = set(vla.state_dict())
    assert any(key.startswith('vla_head.') for key in state_keys)
    assert any(key.startswith('vlm_backbone.') for key in state_keys)


def test_deferred_ema_and_self_flow_options_are_rejected():
    with pytest.raises(ValueError, match='deferred'):
        build_vla(self_flow_variant=1)
    with pytest.raises(ValueError, match='deferred'):
        build_vla(ema_decay=0.99)
    with pytest.raises(TypeError, match='Unexpected PatchWAMVLA arguments'):
        build_vla(unknown_option=True)


def test_native_save_and_resume_round_trip():
    vla = build_vla()
    video = build_video()
    out = vla(video=video, states=torch.randn(2, 16, 14),
              actions=torch.randn(2, 16, 14),
              task_description=['move', 'lift'])
    out['loss'].backward()
    optimizer = torch.optim.SGD(vla.parameters(), lr=0.01)
    optimizer.step()
    vla.eval()
    state = copy.deepcopy(vla.state_dict())
    states = torch.zeros(2, 14)
    with torch.no_grad():
        reference = vla.predict_action(video=video[:, :, :, 0:1],
                                       states=states,
                                       task_description=['move', 'lift'], seed=3)
    clone = build_vla()
    clone.load_state_dict(state)
    clone.eval()
    with torch.no_grad():
        restored = clone.predict_action(video=video[:, :, :, 0:1],
                                        states=states,
                                        task_description=['move', 'lift'], seed=3)
    torch.testing.assert_close(reference, restored, rtol=0, atol=0)


def test_flux_denoiser_requires_local_assets():
    assert HEADS.get('PatchWAMFluxDenoiser') is not None
    with pytest.raises(FileNotFoundError):
        build_head_from_cfg(
            dict(type='PatchWAMHead',
                 denoiser=dict(type='PatchWAMFluxDenoiser',
                               transformer_path='/nonexistent/flux.safetensors'),
                 action_dim=14, text_dim=32, proprio_dim=14, token_dim=128))


def test_subtask_loss_is_added_once_with_its_weight():
    vla = build_vla()
    vla.vlm_backbone.text_encoder = SubtaskTextEncoder()
    video = build_video()
    actions = torch.randn(2, 16, 14)
    sigma = torch.full((2, ), 0.5)
    generator = torch.Generator().manual_seed(9)
    future_noise = torch.randn(2, 48, 128, generator=generator)
    action_noise = torch.randn(2, 16, 128, generator=generator)
    out = vla(video=video, states=torch.zeros(2, 14), actions=actions,
              task_description=['a', 'b'], sigma=sigma,
              future_noise=future_noise, action_noise=action_noise)
    assert torch.isfinite(out['loss_subtask'])
    assert torch.isclose(out['loss_subtask'], torch.tensor(2.0))
    expected = out['loss'] - 0.25 * out['loss_subtask']
    vla.vlm_backbone.text_encoder.aux.data.zero_()
    baseline = vla(video=video, states=torch.zeros(2, 14), actions=actions,
                   task_description=['a', 'b'], sigma=sigma,
                   future_noise=future_noise, action_noise=action_noise)
    torch.testing.assert_close(baseline['loss'], expected, rtol=1e-6,
                               atol=1e-6)


def test_explicit_generator_controls_prediction():
    vla = build_vla()
    video = build_video(frames=1)
    states = torch.zeros(2, 14)
    first = vla.predict_action(video=video, states=states,
                               task_description=['a', 'b'],
                               generator=torch.Generator().manual_seed(4))
    second = vla.predict_action(video=video, states=states,
                                task_description=['a', 'b'],
                                generator=torch.Generator().manual_seed(4))
    third = vla.predict_action(video=video, states=states,
                               task_description=['a', 'b'],
                               generator=torch.Generator().manual_seed(5))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.allclose(first, third)


def test_invalid_future_slots_are_masked_before_attention():
    vla = build_vla()
    video = build_video()
    actions = torch.randn(2, 16, 14)
    frame_masks = torch.tensor([[True, True], [True, False]])
    img_masks = torch.ones(2, 2, dtype=torch.bool)
    sigma = torch.full((2, ), 0.5)
    generator = torch.Generator().manual_seed(17)
    future_noise = torch.randn(2, 48, 128, generator=generator)
    action_noise = torch.randn(2, 16, 128, generator=generator)
    kwargs = dict(states=torch.zeros(2, 14), actions=actions,
                  frame_masks=frame_masks, img_masks=img_masks,
                  task_description=['a', 'b'], sigma=sigma,
                  future_noise=future_noise, action_noise=action_noise)
    clean = vla(video=video, **kwargs)
    corrupted_video = video.clone()
    # Sample 1's future endpoint is invalid; NaN pixels must not reach the loss.
    corrupted_video[1, :, :, -1] = float('nan')
    corrupted = vla(video=corrupted_video, **kwargs)
    assert torch.isfinite(corrupted['loss'])
    torch.testing.assert_close(corrupted['loss'], clean['loss'], rtol=0,
                               atol=0)
    torch.testing.assert_close(corrupted['loss_action'], clean['loss_action'],
                               rtol=0, atol=0)
    corrupted['loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in vla.parameters()
               if p.grad is not None)


def test_width_validation_rejects_mismatched_modules():
    with pytest.raises(ValueError, match='token_input'):
        build_head_from_cfg(dict(
            type='PatchWAMHead',
            denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=64,
                          text_dim=32, width=32, heads=4, depth=2),
            action_dim=14, text_dim=32, proprio_dim=14, token_dim=128))
    with pytest.raises(ValueError, match='features'):
        PatchWAMBackbone(autoencoder=TinyAutoencoder(),
                         text_encoder=TinyTextEncoder(dim=16), token_dim=128,
                         text_dim=32)


# ----------------------------------------------------------------------
# Instruction prompt template (source-compatible) at the encoder boundary
# ----------------------------------------------------------------------

#: The exact prefix the source PatchWAM data pipeline prepends to every task.
PROMPT_PREFIX = DEFAULT_PATCHWAM_PROMPT_TEMPLATE.split('{task}')[0]


def default_prompt(task: str) -> str:
    """Format one raw instruction with the source template."""
    return DEFAULT_PATCHWAM_PROMPT_TEMPLATE.format(task=task)


def build_recording_vla(**overrides):
    """Registry-built VLA whose real text encoder records its prompts."""
    vlm_backbone = dict(
        type='PatchWAMBackbone',
        autoencoder=TinyAutoencoder(),
        text_encoder=RecordingTextEncoder(),
        token_dim=128,
        text_dim=32,
        history_pool_size=2,
    )
    vlm_backbone.update(overrides.pop('vlm_backbone', {}))
    return build_vla(vlm_backbone=vlm_backbone, **overrides)


def spy_encode(vla):
    """Record every request handed to the real backbone ``encode``."""
    seen = []
    original = vla.vlm_backbone.encode

    def spy(batch, target, **kwargs):
        seen.append((dict(batch), target))
        return original(batch, target=target, **kwargs)

    vla.vlm_backbone.encode = spy
    return seen


def test_forward_formats_instruction_for_the_encoder():
    # Pin the source text independently of the production constant.
    assert DEFAULT_PATCHWAM_PROMPT_TEMPLATE == (
        "A video recorded from a robot's point of view executing the following "
        'instruction: {task}')
    vla = build_recording_vla()
    encoder = vla.vlm_backbone.text_encoder
    requests = spy_encode(vla)
    instructions = ['pick the red cup', 'open the top drawer']
    out = vla(video=build_video(), states=torch.zeros(2, 14),
              actions=torch.randn(2, 16, 14), task_description=instructions)
    assert torch.isfinite(out['loss'])

    expected = [default_prompt(text) for text in instructions]
    assert encoder.seen_prompts == [expected]
    # The raw metadata is forwarded unchanged; the template lives in `prompt`.
    request, target = requests[-1]
    assert target is True
    assert request['task_description'] == instructions
    assert request['prompt'] == expected
    for prompt, text in zip(expected, instructions):
        assert prompt.count(PROMPT_PREFIX) == 1
        assert prompt.startswith(PROMPT_PREFIX) and prompt.endswith(text)


def test_predict_action_formats_instruction_for_the_encoder():
    vla = build_recording_vla()
    encoder = vla.vlm_backbone.text_encoder
    requests = spy_encode(vla)
    instructions = ['pick the red cup', 'open the top drawer']
    prediction = vla.predict_action(video=build_video(frames=1),
                                    states=torch.zeros(2, 14),
                                    task_description=instructions, seed=4)
    assert prediction.shape == (2, 16, 14)

    expected = [default_prompt(text) for text in instructions]
    assert encoder.seen_prompts[-1] == expected
    request, target = requests[-1]
    assert target is False
    assert request['task_description'] == instructions
    assert request['prompt'] == expected
    for prompt in expected:
        assert prompt.count(PROMPT_PREFIX) == 1


def test_distinct_instructions_preserve_their_contents():
    vla = build_recording_vla()
    encoder = vla.vlm_backbone.text_encoder
    video = build_video(batch=1, frames=1)
    states = torch.zeros(1, 14)
    vla.predict_action(video=video, states=states,
                       task_description=['move the cup'], seed=4)
    vla.predict_action(video=video, states=states,
                       task_description=['open the drawer'], seed=4)
    assert encoder.seen_prompts == [[default_prompt('move the cup')],
                                    [default_prompt('open the drawer')]]


def test_string_and_tuple_instructions_become_one_prompt_list():
    vla = build_recording_vla()
    encoder = vla.vlm_backbone.text_encoder
    video = build_video(batch=1, frames=1)
    states = torch.zeros(1, 14)
    vla.predict_action(video=video, states=states,
                       task_description='pick the cup', seed=4)
    vla.predict_action(video=video, states=states,
                       task_description=('pick the cup',), seed=4)
    # A single string never iterates per character: both forms give one prompt.
    assert encoder.seen_prompts == [[default_prompt('pick the cup')],
                                    [default_prompt('pick the cup')]]


def test_already_formatted_instruction_is_not_wrapped_twice():
    vla = build_recording_vla()
    encoder = vla.vlm_backbone.text_encoder
    formatted = default_prompt('pick the cup')
    vla.predict_action(video=build_video(batch=1, frames=1),
                       states=torch.zeros(1, 14),
                       task_description=[formatted], seed=4)
    assert encoder.seen_prompts[-1] == [formatted]
    assert encoder.seen_prompts[-1][0].count(PROMPT_PREFIX) == 1


def test_explicit_prompt_is_preserved_verbatim():
    vla = build_recording_vla()
    encoder = vla.vlm_backbone.text_encoder
    video = build_video(batch=1, frames=1)
    states = torch.zeros(1, 14)
    prepared = ['an already prepared prompt']
    vla.predict_action(video=video, states=states,
                       task_description=['pick the cup'], prompt=prepared,
                       seed=4)
    assert encoder.seen_prompts[-1] == prepared
    vla.predict_action(video=video, states=states, prompt=prepared, seed=4)
    assert encoder.seen_prompts[-1] == prepared


def test_vl_style_encoder_receives_the_formatted_prompt():
    vla = build_recording_vla(vlm_backbone=dict(
        text_encoder=VLRecordingTextEncoder()))
    encoder = vla.vlm_backbone.text_encoder
    vla.predict_action(video=build_video(batch=1, frames=1),
                       states=torch.zeros(1, 14),
                       task_description=['pick the cup'],
                       vl_current=torch.zeros(1, 3, 8, 8), seed=4)
    assert encoder.seen_prompts[-1] == [default_prompt('pick the cup')]
    assert encoder.seen_kwargs['current'] is not None


def test_pre_encoded_text_tokens_bypass_prompt_building():
    vla = build_recording_vla()
    encoder = vla.vlm_backbone.text_encoder
    video = build_video(batch=1, views=2, frames=1, height=4, width=6)
    tokens = 2 * 1 * 4 * 6
    prediction = vla.predict_action(
        video=video, states=torch.zeros(1, 14),
        reference_tokens=torch.zeros(1, tokens, 128),
        text_tokens=torch.zeros(1, 3, 32), seed=4)
    assert prediction.shape == (1, 16, 14)
    assert encoder.seen_prompts == []
