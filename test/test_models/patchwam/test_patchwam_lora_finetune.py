"""CPU units for ordinary (unstaged) PatchWAM DiT LoRA finetuning.

The mode is selected by ``vla_head.lora`` plus ``vla_head.stage = None``: the
DiT transformer blocks stay frozen behind the canonical ``attach_dit_lora``
adapters, the vision/language frontend stays frozen (including any encoder LoRA
residuals), and every other policy layer stays fully trainable.
"""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from mmengine.config import Config
from safetensors.torch import save_file

from fluxvla.engines import HEADS, VLAS, build_head_from_cfg, build_vla_from_cfg
from fluxvla.models.blocks.dit_lora import (LORA_ATTRIBUTE,
                                            lora_parameter_names)
from fluxvla.models.backbones.vlms import patchwam_backbone  # noqa: F401
from fluxvla.models.heads import patchwam_head  # noqa: F401  (registration)
from fluxvla.models.heads.patchwam_head import PatchWAMHead
from fluxvla.models.third_party_models.patchwam.policy import PatchFlowPolicy
from fluxvla.models.third_party_models.patchwam.flux import OfficialFluxDenoiser
from fluxvla.models.third_party_models.patchwam.tiny import \
    SmallJointTransformer
from fluxvla.models.vlas import patchwam_vla  # noqa: F401  (registration)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / 'configs' / 'patchwam'
TRAIN_CONFIG = 'patchwam_robotwin_vlm_lora_finetune.py'
EVAL_CONFIG = 'patchwam_robotwin_vlm_lora_eval.py'

#: Local official FLUX.2 checkout; the real-naming units skip without it.
FLUX_SOURCE = Path('/home/chuong/workspace/wam/ImageWAM/third_party/flux2/src')

VISUAL_DIM = 6
TOKEN_DIM = 16
TEXT_DIM = 8
ACTION_DIM = 7

#: The tiny joint transformer keeps its own block/layer names, so the LoRA
#: target list is explicit (the default list is the official FLUX.2 layout).
TINY_LORA = dict(rank=2, alpha=4,
                 targets=('qkv', 'mix', 'ff.0', 'ff.2'),
                 blocks=('blocks',))


class StubVisualEncoder(nn.Module):
    """Frozen visual-frontend stand-in with the shared feature contract."""

    feature_dim = VISUAL_DIM

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, VISUAL_DIM, 16, 16)

    def encode_features(self, frames):
        spatial = self.stem(frames)
        return SimpleNamespace(tokens=spatial.flatten(2).transpose(1, 2),
                               grid=(spatial.shape[-2], spatial.shape[-1]),
                               feature_dim=VISUAL_DIM)


class AdapterTextEncoder(nn.Module):
    """Text encoder whose optional language LoRA residuals are trainable."""

    requires_images = False
    trainable_adapters = True

    def __init__(self, dim=TEXT_DIM):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(3, dim))
        self.feature_dim = dim
        self.residual_in = nn.Linear(dim, 4, bias=False)
        self.residual_out = nn.Linear(4, dim, bias=False)

    def forward(self, prompts):
        return self.embed.unsqueeze(0).expand(len(prompts), -1, -1)


class FrozenTextEncoder(nn.Module):
    requires_images = False
    trainable_adapters = False

    def __init__(self, dim=TEXT_DIM):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(3, dim))
        self.feature_dim = dim

    def forward(self, prompts):
        return self.embed.unsqueeze(0).expand(len(prompts), -1, -1)


class TinyAutoencoder(nn.Module):
    """Frozen autoencoder stand-in for the pre-encoded CPU forward path."""

    def __init__(self, channels: int = 128):
        super().__init__()
        self.proj = nn.Conv2d(3, channels, 1)

    def encode(self, video):
        return self.proj(video)


def tiny_denoiser(*, visual_dim=VISUAL_DIM, token_dim=TOKEN_DIM):
    return dict(type='PatchWAMTinyDenoiser', token_dim=token_dim,
                text_dim=TEXT_DIM, width=32, heads=4, depth=2,
                visual_dim=visual_dim)


def finetune_head_cfg(*, lora=TINY_LORA, stage=None, visual_dim=VISUAL_DIM,
                      token_dim=TOKEN_DIM):
    return dict(type='PatchWAMHead', action_dim=ACTION_DIM, text_dim=TEXT_DIM,
                proprio_dim=ACTION_DIM, token_dim=token_dim,
                visual_dim=visual_dim, stage=stage, lora=lora,
                denoiser=tiny_denoiser(visual_dim=visual_dim,
                                       token_dim=token_dim))


def backbone_cfg(*, text_encoder=None, visual_encoder=None, token_dim=TOKEN_DIM):
    cfg = dict(type='PatchWAMBackbone', autoencoder=None, token_dim=token_dim,
               text_dim=TEXT_DIM, history_pool_size=2)
    if text_encoder is not None:
        cfg['text_encoder'] = text_encoder
    if visual_encoder is not None:
        cfg['visual_encoder'] = visual_encoder
    return cfg


def finetune_model(*, pretrained=None, backbone=None, lora=TINY_LORA,
                   visual_dim=VISUAL_DIM, token_dim=TOKEN_DIM, **head_kwargs):
    return build_vla_from_cfg(dict(
        type='PatchWAMVLA',
        action_horizon=4,
        num_inference_steps=2,
        vlm_backbone=backbone,
        vla_head=finetune_head_cfg(lora=lora, visual_dim=visual_dim,
                                   token_dim=token_dim, **head_kwargs),
        pretrained_name_or_path=pretrained,
    ))


def tiny_policy(*, visual_dim=VISUAL_DIM, token_dim=TOKEN_DIM):
    torch.manual_seed(5)
    denoiser = SmallJointTransformer(token_dim=token_dim, text_dim=TEXT_DIM,
                                     width=32, heads=4, depth=2,
                                     visual_dim=visual_dim)
    return PatchFlowPolicy(denoiser, action_dim=ACTION_DIM, text_dim=TEXT_DIM,
                           proprio_dim=ACTION_DIM, token_dim=token_dim,
                           visual_dim=visual_dim)


def preencoded_batch(*, batch=2, reference=2, future=2, text=3, horizon=4,
                     visual_dim=VISUAL_DIM, text_dim=TEXT_DIM):
    generator = torch.Generator().manual_seed(11)

    def draw(*shape):
        return torch.randn(*shape, generator=generator)

    return {
        'reference_tokens': draw(batch, reference, visual_dim),
        'future_tokens': draw(batch, future, visual_dim),
        'text_tokens': draw(batch, text, text_dim),
        'action': draw(batch, horizon, ACTION_DIM),
        'proprio': draw(batch, ACTION_DIM),
    }


def trainable_names(model):
    return sorted(name for name, param in model.named_parameters()
                  if param.requires_grad)


def block_parameter_names(model, prefix):
    return sorted(name for name, _ in model.named_parameters()
                  if name.startswith(prefix))


def test_lora_is_rejected_for_stage1_and_kept_for_stage2():
    """Only the new unstaged mode is added; the staged split is unchanged."""
    with pytest.raises(ValueError, match='LoRA'):
        build_head_from_cfg(finetune_head_cfg(lora=TINY_LORA,
                                              stage='stage1'))
    head = build_head_from_cfg(finetune_head_cfg(lora=TINY_LORA,
                                                 stage='stage2'))
    assert head.stage == 'stage2'
    assert head.lora_finetune is False


def test_finetune_policy_freezes_blocks_and_opens_everything_else():
    head = build_head_from_cfg(finetune_head_cfg())
    assert head.stage is None
    assert head.lora_finetune is True

    parameters = dict(head.named_parameters())
    block_names = block_parameter_names(head, 'denoiser.blocks.')
    # The whole block is frozen: attention, feed-forward, modulation and norm.
    assert any(name.endswith('modulation.weight') for name in block_names)
    assert any(name.endswith('ff_norm.weight') for name in block_names)
    assert block_names
    assert not any(parameters[name].requires_grad for name in block_names)

    adapters = [f'denoiser.{name}'
                for name in lora_parameter_names(head.denoiser)]
    assert adapters
    assert all(parameters[name].requires_grad for name in adapters)
    for name in ('denoiser.token_input.weight', 'denoiser.text_input.weight',
                 'denoiser.time_input.0.weight',
                 'denoiser.token_output.weight',
                 'denoiser.visual_in.weight', 'denoiser.visual_out.weight',
                 'state_projection.weight', 'state_projection.bias'):
        assert parameters[name].requires_grad, name
    trainable = {name for name, param in head.named_parameters()
                 if param.requires_grad}
    assert trainable == ({name for name in parameters if name not in block_names}
                         | set(adapters))
    assert [group['name'] for group in head.parameter_groups()] == \
        ['policy', 'dit_lora']
    grouped = [param for group in head.parameter_groups()
               for param in group['params']]
    assert len(grouped) == len({id(param) for param in grouped})


def test_finetune_head_backward_shows_the_gradient_split():
    head = build_head_from_cfg(finetune_head_cfg())
    result = head(preencoded_batch())
    assert torch.isfinite(result['loss']) and result['loss'] > 0
    result['loss'].backward()
    parameters = dict(head.named_parameters())

    frozen = block_parameter_names(head, 'denoiser.blocks.')
    assert frozen
    assert all(parameters[name].grad is None for name in frozen)

    adapters = [name for name in parameters
                if f'denoiser.{LORA_ATTRIBUTE}.' in name]
    assert adapters
    for name in adapters:
        grad = parameters[name].grad
        assert grad is not None and torch.isfinite(grad).all()
    up_projections = [name for name in adapters if name.endswith('.up.weight')]
    assert up_projections
    assert all(parameters[name].grad.abs().sum() > 0
               for name in up_projections)

    for name in ('denoiser.token_input.weight', 'denoiser.text_input.weight',
                 'denoiser.visual_in.weight', 'denoiser.visual_out.weight',
                 'denoiser.token_output.weight', 'state_projection.weight'):
        grad = parameters[name].grad
        assert grad is not None and grad.abs().sum() > 0, name


def test_finetune_report_lists_fully_qualified_names_and_patterns():
    head = build_head_from_cfg(finetune_head_cfg())
    report = head.trainable_parameter_report()
    assert report['mode'] == 'lora_finetune'
    assert report['dit_block_patterns'] == ['denoiser.blocks.*']
    assert report['trainable_parameter_names'] == \
        sorted(name for name, param in head.named_parameters()
               if param.requires_grad)
    patterns = report['trainable_module_patterns']
    assert f'denoiser.{LORA_ATTRIBUTE}.*' in patterns
    assert 'denoiser.token_input.*' in patterns
    assert 'state_projection.*' in patterns
    # The frozen DiT blocks never appear as a trainable pattern.
    assert not any(pattern.startswith('denoiser.blocks.') for pattern in patterns)
    blocked = [name for name in report['trainable_parameter_names']
               if name.startswith('denoiser.blocks.')]
    assert blocked == []
    assert len(report['trainable_parameter_names']) == \
        report['num_trainable_parameters']
    adapters = [name for name in report['trainable_parameter_names']
                if f'denoiser.{LORA_ATTRIBUTE}.' in name]
    assert sorted(adapters) == report['adapter_parameter_names']
    # The same names are reported through the model-level API.
    model = finetune_model()
    model.freeze_backbones()
    model_report = model.trainable_parameter_report()
    assert model_report['mode'] == 'lora_finetune'
    assert model_report['head']['dit_block_patterns'] == \
        ['denoiser.blocks.*']
    assert model_report['trainable_parameter_names'] == \
        [f'vla_head.{name}'
         for name in head.trainable_parameter_report()[
             'trainable_parameter_names']]


def test_finetune_mode_freezes_the_frontend_including_encoder_adapters():
    backbone = backbone_cfg(text_encoder=AdapterTextEncoder(),
                            visual_encoder=StubVisualEncoder())
    model = finetune_model(backbone=backbone)
    # Building the backbone opens its language residuals, so the freeze has to
    # happen after the backbone's own trainable policy runs.
    encoder = model.vlm_backbone.text_encoder
    assert encoder.residual_in.weight.requires_grad
    model.freeze_backbones()
    assert all(not param.requires_grad
               for param in model.vlm_backbone.parameters())
    assert not encoder.residual_in.weight.requires_grad
    assert not encoder.residual_out.weight.requires_grad
    assert not model.vlm_backbone.visual_encoder.stem.weight.requires_grad
    parameters = dict(model.vla_head.named_parameters())
    assert not parameters['denoiser.blocks.0.qkv.weight'].requires_grad
    assert parameters['denoiser.token_input.weight'].requires_grad
    adapter = f'denoiser.{LORA_ATTRIBUTE}.blocks_0_qkv.down.weight'
    assert parameters[adapter].requires_grad
    trainable = trainable_names(model)
    assert trainable and all(name.startswith('vla_head.') for name in trainable)


def test_unstaged_without_lora_keeps_every_head_parameter_trainable():
    head = build_head_from_cfg(finetune_head_cfg(lora=None, stage=None))
    assert head.lora_finetune is False
    assert head.stage is None
    assert head.parameter_groups() == []
    assert all(param.requires_grad for param in head.parameters())
    model = finetune_model(lora=None,
                           backbone=backbone_cfg(
                               text_encoder=FrozenTextEncoder(),
                               visual_encoder=StubVisualEncoder()))
    model.freeze_backbones()
    assert all(param.requires_grad
               for param in model.vla_head.parameters())
    assert not any(param.requires_grad
                   for param in model.vlm_backbone.parameters())
    assert not hasattr(model.vla_head.denoiser, LORA_ATTRIBUTE)
    assert model.vla_head.denoiser.blocks[0].qkv.weight.requires_grad


def _small_official_transformer():
    """Reduced official FLUX.2 ``Flux2`` module (no pretrained weights)."""
    if str(FLUX_SOURCE) not in sys.path:
        sys.path.insert(0, str(FLUX_SOURCE))
    import flux2.model as model_api

    params = model_api.Klein4BParams(
        in_channels=128, context_in_dim=48, hidden_size=64, num_heads=4,
        depth=1, depth_single_blocks=1, axes_dim=[4, 4, 4, 4], mlp_ratio=2.0,
        use_guidance_embed=False)
    return OfficialFluxDenoiser(model_api.Flux2(params))


@pytest.mark.skipif(not (FLUX_SOURCE / 'flux2' / 'model.py').is_file(),
                    reason='local official flux2 source is unavailable')
def test_official_flux_blocks_freeze_behind_live_adapters():
    """The real ``double_blocks``/``single_blocks`` naming and execution."""
    head = PatchWAMHead(_small_official_transformer(), action_dim=ACTION_DIM,
                        text_dim=48, proprio_dim=ACTION_DIM, token_dim=128,
                        lora=dict())
    assert head.lora_finetune is True
    assert head.dit_block_patterns() == [
        'denoiser.transformer.double_blocks.*',
        'denoiser.transformer.single_blocks.*']
    parameters = dict(head.named_parameters())
    frozen = head.dit_block_parameter_names()
    # Query/key norm scales live inside the blocks, so they freeze with them.
    assert any(name.endswith('norm.query_norm.scale') for name in frozen)
    assert not any(parameters[name].requires_grad for name in frozen)
    for name in ('denoiser.transformer.img_in.weight',
                 'denoiser.transformer.txt_in.weight',
                 'denoiser.transformer.final_layer.linear.weight'):
        assert parameters[name].requires_grad, name

    batch = preencoded_batch(visual_dim=128, text_dim=48)
    result = head(batch)
    assert torch.isfinite(result['loss']) and result['loss'] > 0
    result['loss'].backward()
    assert all(parameters[name].grad is None for name in frozen)
    upward = [name for name in parameters
              if f'denoiser.{LORA_ATTRIBUTE}.' in name
              and name.endswith('.up.weight')]
    assert upward
    assert all(parameters[name].grad.abs().sum() > 0 for name in upward)
    for name in ('denoiser.transformer.img_in.weight',
                 'denoiser.transformer.txt_in.weight',
                 'denoiser.transformer.final_layer.linear.weight',
                 'state_projection.weight'):
        assert parameters[name].grad.abs().sum() > 0, name


def test_finetune_base_initialization_allows_only_the_new_adapters(tmp_path):
    base = tiny_policy()
    path = tmp_path / 'base.safetensors'
    save_file({key: value.clone() for key, value in base.state_dict().items()},
              str(path))
    model = finetune_model(pretrained=str(path))
    adapter_values = {name: value.clone() for name, value in
                      model.vla_head.named_parameters()
                      if f'denoiser.{LORA_ATTRIBUTE}.' in name}
    assert adapter_values
    model.freeze_backbones()
    model.from_pretrained()
    report = model._finetune_initialization
    assert report['format'] == 'bare'
    assert report['new_keys'] == sorted(
        f'vla_head.denoiser.{name}'
        for name in lora_parameter_names(model.vla_head.denoiser))
    for key, value in base.state_dict().items():
        torch.testing.assert_close(model.state_dict()['vla_head.' + key],
                                   value, rtol=0, atol=0)
    # A base payload carries no adapters, so they keep their zero residual.
    for name, value in adapter_values.items():
        torch.testing.assert_close(
            model.vla_head.state_dict()[name], value, rtol=0, atol=0)
    weights = dict(model.vla_head.named_parameters())
    assert all(weights[name].abs().sum() == 0
               for name in adapter_values if name.endswith('.up.weight'))


@pytest.mark.parametrize('dropped', [
    'denoiser.blocks.0.qkv.weight',
    'denoiser.blocks.0.modulation.bias',
    'denoiser.token_input.weight',
    'state_projection.weight',
])
def test_finetune_load_requires_inherited_head_tensors(tmp_path, dropped):
    """Frozen DiT blocks and every other inherited tensor stay mandatory."""
    base = tiny_policy()
    weights = {key: value.clone() for key, value in base.state_dict().items()}
    del weights[dropped]
    path = tmp_path / 'damaged.safetensors'
    save_file(weights, str(path))
    model = finetune_model(pretrained=str(path))
    model.freeze_backbones()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match='Strict PatchWAM checkpoint load'):
        model.from_pretrained()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_finetune_load_rejects_unexpected_and_malformed_payloads(tmp_path):
    base = tiny_policy()
    weights = {key: value.clone() for key, value in base.state_dict().items()}
    stray = tmp_path / 'stray.safetensors'
    save_file(dict(weights, **{'denoiser.blocks.99.qkv.weight': torch.zeros(3)}),
              str(stray))
    model = finetune_model(pretrained=str(stray))
    with pytest.raises(ValueError, match='unexpected'):
        model.from_pretrained()

    adapter = next(name for name in
                   lora_parameter_names(finetune_model().vla_head.denoiser)
                   if name.endswith('down.weight'))
    full = dict(finetune_model().vla_head.state_dict())
    full[f'denoiser.{adapter}'] = torch.zeros(1, 1)
    malformed = tmp_path / 'malformed.safetensors'
    save_file({f'vla_head.{key}': value for key, value in full.items()},
              str(malformed))
    model = finetune_model(pretrained=str(malformed))
    with pytest.raises(ValueError, match='shape'):
        model.from_pretrained()


def test_finetune_native_roundtrip_restores_the_adapters(tmp_path):
    model = finetune_model()
    adapter_names = [name for name, _ in model.vla_head.named_parameters()
                     if f'denoiser.{LORA_ATTRIBUTE}.' in name]
    with torch.no_grad():
        for name in adapter_names:
            model.vla_head.get_parameter(name).add_(0.25)
    path = tmp_path / 'native.safetensors'
    save_file({key: value.clone() for key, value in model.state_dict().items()},
              str(path))

    restored = finetune_model(pretrained=str(path))
    restored.freeze_backbones()
    restored.from_pretrained()
    assert restored._finetune_initialization['format'] == 'native'
    assert restored._finetune_initialization['new_keys'] == sorted(
        f'vla_head.denoiser.{name}'
        for name in lora_parameter_names(restored.vla_head.denoiser))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0,
                                   atol=0)
    # Loading a strict native payload is what the eval runner does.
    restored.load_state_dict(
        {key: value.clone() for key, value in model.state_dict().items()},
        strict=True)


def test_runner_lifecycle_orders_keep_the_finetune_policy(tmp_path):
    """DDP freezes then loads; FSDP loads then freezes; both end identical."""
    base = tiny_policy()
    path = tmp_path / 'base.safetensors'
    save_file({key: value.clone() for key, value in base.state_dict().items()},
              str(path))
    ddp = finetune_model(pretrained=str(path))
    ddp.freeze_backbones()
    ddp.from_pretrained()
    fsdp = finetune_model(pretrained=str(path))
    fsdp.from_pretrained()
    fsdp.freeze_backbones()

    assert trainable_names(ddp) == trainable_names(fsdp)
    assert ddp.lora_finetune is fsdp.lora_finetune is True
    assert all(name.startswith('vla_head.') for name in trainable_names(ddp))
    assert any(f'denoiser.{LORA_ATTRIBUTE}.' in name
               for name in trainable_names(ddp))
    assert not any('denoiser.blocks.' in name for name in trainable_names(ddp))
    for key, value in base.state_dict().items():
        torch.testing.assert_close(ddp.state_dict()['vla_head.' + key], value,
                                   rtol=0, atol=0)
        torch.testing.assert_close(fsdp.state_dict()['vla_head.' + key], value,
                                   rtol=0, atol=0)


def test_finetune_recipe_configs_load_and_match_the_base_recipe(monkeypatch):
    monkeypatch.delenv('PATCHWAM_PRETRAINED_CHECKPOINT', raising=False)
    cfg = Config.fromfile(str(CONFIG_DIR / TRAIN_CONFIG))
    assert VLAS.get(cfg.model.type) is not None
    assert HEADS.get(cfg.model.vla_head.type) is not None
    assert HEADS.get(cfg.model.vla_head.denoiser.type) is not None
    assert cfg.model.vla_head.lora == dict(rank=32, alpha=64, dropout=0.0)
    assert 'stage' not in cfg.model.vla_head
    assert 'visual_stage' not in cfg.model
    assert cfg.model.vla_head.get('visual_dim') is None
    assert 'step_140000' in cfg.model.pretrained_name_or_path
    assert cfg.model.freeze_vlm_backbone is True
    # Everything else comes from the RobotWin VLM full-finetune recipe.
    assert cfg.model.vla_head.action_dim == 14
    assert cfg.model.vla_head.proprio_dim == 14
    assert cfg.model.vla_head.text_dim == 7680
    assert cfg.model.vlm_backbone.text_encoder_kind == 'qwen3_vl'
    assert cfg.model.vla_head.denoiser.variant == 'klein-base-4b'
    assert cfg.model.vla_head.video_weight == 0.5
    assert cfg.runner.auto_scale_lr is True
    assert cfg.inference_model.vla_head.lora == cfg.model.vla_head.lora

    monkeypatch.setenv('PATCHWAM_PRETRAINED_CHECKPOINT', '/tmp/other_base.pt')
    override = Config.fromfile(str(CONFIG_DIR / TRAIN_CONFIG))
    assert override.model.pretrained_name_or_path == '/tmp/other_base.pt'

    monkeypatch.delenv('PATCHWAM_PRETRAINED_CHECKPOINT', raising=False)
    eval_cfg = Config.fromfile(str(CONFIG_DIR / EVAL_CONFIG))
    assert 'runner' not in eval_cfg
    assert 'auto_scale_lr' not in eval_cfg
    assert eval_cfg.model.vla_head.lora == cfg.model.vla_head.lora
    assert eval_cfg.inference_model.vla_head.lora == cfg.model.vla_head.lora
    assert eval_cfg.eval.type == 'RobotwinEvalRunner'
    assert eval_cfg.model.vla_head.action_dim == 14
    # The LoRA mode freezes the frontend, so the eval model must keep the flag
    # that the constructor validates.
    assert eval_cfg.inference_model.freeze_vlm_backbone is True


def test_finetune_recipe_head_builds_and_trains_with_tiny_modules():
    cfg = Config.fromfile(str(CONFIG_DIR / TRAIN_CONFIG))
    model_cfg = copy.deepcopy(Config(cfg.model).to_dict())
    model_cfg.pop('pretrained_name_or_path', None)
    model_cfg['vlm_backbone'] = dict(
        type='PatchWAMBackbone', autoencoder=TinyAutoencoder(),
        text_encoder=FrozenTextEncoder(dim=32), token_dim=128, text_dim=32,
        history_pool_size=2)
    model_cfg['vla_head'] = dict(
        type='PatchWAMHead',
        denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128, text_dim=32,
                      width=32, heads=4, depth=2),
        action_dim=14, text_dim=32, proprio_dim=14, token_dim=128,
        lora=TINY_LORA)
    vla = build_vla_from_cfg(model_cfg)
    assert vla.lora_finetune is True
    vla.freeze_backbones()
    out = vla(images=torch.rand(2, 3, 2, 32, 64), states=torch.randn(2, 14),
              actions=torch.randn(2, 16, 14), task_description=['a', 'b'])
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    parameters = dict(vla.named_parameters())
    assert parameters['vla_head.denoiser.blocks.0.qkv.weight'].grad is None
    assert parameters['vla_head.state_projection.weight'].grad is not None
    adapter = f'vla_head.denoiser.{LORA_ATTRIBUTE}.blocks_0_qkv.down.weight'
    assert parameters[adapter].grad is not None
    assert not any(param.requires_grad
                   for param in vla.vlm_backbone.parameters())


_MISSING = object()


def _config_diff(left, right, prefix=''):
    """Dotted keys whose config values differ between two dicts."""
    if isinstance(left, dict) and isinstance(right, dict):
        differing = []
        for key in sorted(set(left) | set(right)):
            differing += _config_diff(left.get(key, _MISSING),
                                      right.get(key, _MISSING),
                                      f'{prefix}{key}.')
        return differing
    if left is _MISSING or right is _MISSING or left != right:
        return [prefix.rstrip('.')]
    return []


def test_finetune_recipes_change_only_the_adaptation_policy():
    """The opt-in recipes must not touch dataset, optimizer, or base settings."""
    base = Config.fromfile(
        str(CONFIG_DIR / 'patchwam_robotwin_vlm_full_finetune.py')).to_dict()
    cfg = Config.fromfile(str(CONFIG_DIR / TRAIN_CONFIG)).to_dict()
    assert _config_diff(cfg['model'], base['model']) == [
        'pretrained_name_or_path', 'vla_head.lora']
    for section in ('train_dataloader', 'val_dataloader', 'runner'):
        assert cfg[section] == base[section]
    assert cfg['inference_model']['vla_head']['lora'] == \
        cfg['model']['vla_head']['lora']

    base_eval = Config.fromfile(
        str(CONFIG_DIR / 'patchwam_robotwin_eval.py')).to_dict()
    eval_cfg = Config.fromfile(str(CONFIG_DIR / EVAL_CONFIG)).to_dict()
    assert _config_diff(eval_cfg['model'], base_eval['model']) == \
        ['vla_head.lora']
    assert _config_diff(eval_cfg['inference_model'],
                        base_eval['inference_model']) == ['vla_head.lora']
    assert _config_diff(eval_cfg['eval'], base_eval['eval']) == \
        ['norm_stats_path']


