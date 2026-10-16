"""CPU units for ordinary (unstaged) PatchWAM DiT PEFT finetuning.

The mode is selected by ``vla_head.peft`` plus ``vla_head.stage = None``: the
DiT transformer blocks stay frozen behind the configured PEFT adapter, the
vision/language frontend stays frozen (including any encoder LoRA residuals),
and every other policy layer stays fully trainable.
"""

import copy
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from mmengine.config import Config
from safetensors.torch import save_file

from fluxvla.engines import (HEADS, VLAS, build_head_from_cfg,
                             build_vla_from_cfg)
import warnings

from fluxvla.models.blocks.dit_peft import (DIT_ADAPTER_TARGETS,
                                            adapter_module_paths,
                                            adapter_parameter_names,
                                            attach_dit_adapters,
                                            legacy_lora_config,
                                            normalize_adapter_config)
from fluxvla.models.backbones.vlms import patchwam_backbone  # noqa: F401
from fluxvla.models.heads import patchwam_head  # noqa: F401  (registration)
from fluxvla.models.heads.patchwam_head import PatchWAMHead
from fluxvla.models.third_party_models.patchwam.policy import PatchFlowPolicy
from fluxvla.models.third_party_models.patchwam.flux import \
    OfficialFluxDenoiser
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

#: The tiny joint transformer keeps its own block/layer names, so the PEFT
#: targets are explicit (the default list is the official FLUX.2 layout).
TINY_PEFT = dict(peft_type='LORA', r=2, lora_alpha=4,
                 target_modules=('qkv', 'mix', 'ff.0', 'ff.2'),
                 blocks=('blocks',))

#: The same recipe with another module-level PEFT tuner.
TINY_LOHA = dict(peft_type='LOHA', r=2, alpha=4,
                 target_modules=('qkv', 'mix', 'ff.0', 'ff.2'),
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


def finetune_head_cfg(*, peft=TINY_PEFT, lora=None, stage=None,
                      visual_dim=VISUAL_DIM, token_dim=TOKEN_DIM):
    cfg = dict(type='PatchWAMHead', action_dim=ACTION_DIM, text_dim=TEXT_DIM,
               proprio_dim=ACTION_DIM, token_dim=token_dim,
               visual_dim=visual_dim, stage=stage, peft=peft,
               denoiser=tiny_denoiser(visual_dim=visual_dim,
                                      token_dim=token_dim))
    if lora is not None:
        cfg['lora'] = lora
    return cfg


def backbone_cfg(*, text_encoder=None, visual_encoder=None,
                 token_dim=TOKEN_DIM):
    cfg = dict(type='PatchWAMBackbone', autoencoder=None, token_dim=token_dim,
               text_dim=TEXT_DIM, history_pool_size=2)
    if text_encoder is not None:
        cfg['text_encoder'] = text_encoder
    if visual_encoder is not None:
        cfg['visual_encoder'] = visual_encoder
    return cfg


def finetune_model(*, pretrained=None, backbone=None, peft=TINY_PEFT,
                   visual_dim=VISUAL_DIM, token_dim=TOKEN_DIM, **head_kwargs):
    return build_vla_from_cfg(dict(
        type='PatchWAMVLA',
        action_horizon=4,
        num_inference_steps=2,
        vlm_backbone=backbone,
        vla_head=finetune_head_cfg(peft=peft, visual_dim=visual_dim,
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


def adapter_names(head):
    """Head-relative PEFT adapter parameter names."""
    return [f'denoiser.{name}'
            for name in adapter_parameter_names(head.denoiser)]


def base_model_key(state, name):
    """Key in ``state`` holding the base tensor for the unadapted ``name``."""
    if name in state:
        return name
    module, _, parameter = name.rpartition('.')
    wrapped = f'{module}.base_layer.{parameter}'
    assert wrapped in state, name
    return wrapped


def test_peft_is_rejected_for_stage1_and_kept_for_stage2():
    """Only the new unstaged mode is added; the staged split is unchanged."""
    with pytest.raises(ValueError, match='PEFT'):
        build_head_from_cfg(finetune_head_cfg(peft=TINY_PEFT, stage='stage1'))
    head = build_head_from_cfg(finetune_head_cfg(peft=TINY_PEFT,
                                                 stage='stage2'))
    assert head.stage == 'stage2'
    assert head.peft_finetune is False
    assert head.peft_type == 'LORA'


def test_finetune_policy_freezes_blocks_and_opens_everything_else():
    head = build_head_from_cfg(finetune_head_cfg())
    assert head.stage is None
    assert head.peft_finetune is True

    parameters = dict(head.named_parameters())
    block_names = block_parameter_names(head, 'denoiser.blocks.')
    # The whole block is frozen: attention, feed-forward, modulation and norm.
    assert any(name.endswith('modulation.weight') for name in block_names)
    assert any(name.endswith('ff_norm.weight') for name in block_names)
    assert block_names
    adapters = adapter_names(head)
    assert adapters
    block_bases = [name for name in block_names if name not in set(adapters)]
    assert block_bases
    assert not any(parameters[name].requires_grad for name in block_bases)
    assert all(parameters[name].requires_grad for name in adapters)
    for name in ('denoiser.token_input.weight', 'denoiser.text_input.weight',
                 'denoiser.time_input.0.weight',
                 'denoiser.token_output.weight',
                 'denoiser.visual_in.weight', 'denoiser.visual_out.weight',
                 'state_projection.weight', 'state_projection.bias'):
        assert parameters[name].requires_grad, name
    trainable = {name for name, param in head.named_parameters()
                 if param.requires_grad}
    assert trainable == set(adapters) | {
        name for name in parameters if not name.startswith('denoiser.blocks.')}
    assert [group['name'] for group in head.parameter_groups()] == \
        ['policy', 'peft']
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
    adapters = adapter_names(head)
    assert adapters
    assert all(parameters[name].grad is None
               for name in frozen if name not in set(adapters))
    for name in adapters:
        grad = parameters[name].grad
        assert grad is not None and torch.isfinite(grad).all()
    upward = [name for name in adapters
              if name.endswith('lora_B.default.weight')]
    assert upward
    assert all(parameters[name].grad.abs().sum() > 0 for name in upward)

    for name in ('denoiser.token_input.weight', 'denoiser.text_input.weight',
                 'denoiser.visual_in.weight', 'denoiser.visual_out.weight',
                 'denoiser.token_output.weight', 'state_projection.weight'):
        grad = parameters[name].grad
        assert grad is not None and grad.abs().sum() > 0, name


def test_finetune_report_lists_fully_qualified_names_and_patterns():
    head = build_head_from_cfg(finetune_head_cfg())
    report = head.trainable_parameter_report()
    assert report['mode'] == 'peft_finetune'
    assert report['peft_type'] == 'LORA'
    assert report['dit_block_patterns'] == ['denoiser.blocks.*']
    assert report['trainable_parameter_names'] == \
        sorted(name for name, param in head.named_parameters()
               if param.requires_grad)
    patterns = report['trainable_module_patterns']
    assert 'denoiser.token_input.*' in patterns
    assert 'state_projection.*' in patterns
    # The frozen DiT blocks never appear as a module whose own weights train;
    # the adapters are reported through their own patterns.
    assert not any(pattern.startswith('denoiser.blocks.')
                   for pattern in patterns)
    attached = set(adapter_names(head))
    blocked = [name for name in report['trainable_parameter_names']
               if name.startswith('denoiser.blocks.') and name not in attached]
    assert blocked == []
    assert report['adapter_module_patterns'] == [
        'denoiser.blocks.*.ff.0', 'denoiser.blocks.*.ff.2',
        'denoiser.blocks.*.mix', 'denoiser.blocks.*.qkv']
    assert len(report['trainable_parameter_names']) == \
        report['num_trainable_parameters']
    adapters = sorted(name for name in report['trainable_parameter_names']
                      if name in attached)
    assert adapters == report['adapter_parameter_names']
    assert report['num_adapter_parameters'] == len(adapters)
    # The same names are reported through the model-level API.
    model = finetune_model()
    model.freeze_backbones()
    model_report = model.trainable_parameter_report()
    assert model_report['mode'] == 'peft_finetune'
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
    # No adapter may land outside the head's DiT blocks, the frozen frontend
    # included.
    assert adapter_module_paths(model.vlm_backbone) == ()
    assert all(path.startswith('transformer.') or path.startswith('blocks.')
               for path in adapter_module_paths(model.vla_head.denoiser))
    parameters = dict(model.vla_head.named_parameters())
    assert not parameters[
        'denoiser.blocks.0.qkv.base_layer.weight'].requires_grad
    assert parameters['denoiser.token_input.weight'].requires_grad
    adapter = 'denoiser.blocks.0.qkv.lora_A.default.weight'
    assert parameters[adapter].requires_grad
    trainable = trainable_names(model)
    assert trainable
    assert all(name.startswith('vla_head.') for name in trainable)


def test_unstaged_without_adapters_keeps_every_head_parameter_trainable():
    head = build_head_from_cfg(finetune_head_cfg(peft=None, stage=None))
    assert head.peft_finetune is False
    assert head.stage is None
    assert head.peft_type is None
    assert head.parameter_groups() == []
    assert all(param.requires_grad for param in head.parameters())
    model = finetune_model(peft=None,
                           backbone=backbone_cfg(
                               text_encoder=FrozenTextEncoder(),
                               visual_encoder=StubVisualEncoder()))
    model.freeze_backbones()
    assert all(param.requires_grad
               for param in model.vla_head.parameters())
    assert not any(param.requires_grad
                   for param in model.vlm_backbone.parameters())
    assert adapter_module_paths(model.vla_head.denoiser) == ()
    assert model.vla_head.denoiser.blocks[0].qkv.weight.requires_grad


def test_alternate_tuner_loha_uses_the_same_config_path():
    """A second real module-level tuner works through ``vla_head.peft``."""
    head = build_head_from_cfg(finetune_head_cfg(peft=TINY_LOHA))
    assert head.peft_type == 'LOHA'
    adapters = adapter_names(head)
    assert adapters
    assert all('.hada_w' in name for name in adapters)
    parameters = dict(head.named_parameters())
    frozen = [name for name in block_parameter_names(head, 'denoiser.blocks.')
              if name not in set(adapters)]
    assert frozen
    assert all(parameters[name].requires_grad for name in adapters)
    assert not any(parameters[name].requires_grad for name in frozen)
    assert [group['name'] for group in head.parameter_groups()] == \
        ['policy', 'peft']
    result = head(preencoded_batch())
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    assert all(parameters[name].grad is None for name in frozen)
    assert any(parameters[name].grad is not None
               and parameters[name].grad.abs().sum() > 0
               for name in adapters)
    assert head.trainable_parameter_report()['peft_type'] == 'LOHA'


def test_legacy_lora_configuration_converts_with_a_deprecation_warning():
    legacy = dict(rank=2, alpha=4, dropout=0.0,
                  targets=('qkv', 'mix', 'ff.0', 'ff.2'),
                  blocks=('blocks',))
    with pytest.deprecated_call(match='vla_head.peft'):
        options = legacy_lora_config(legacy)
    assert options['peft_type'] == 'LORA'
    assert options['r'] == 2 and options['lora_alpha'] == 4
    assert options['lora_dropout'] == 0.0
    assert options['target_modules'] == ('qkv', 'mix', 'ff.0', 'ff.2')

    with pytest.deprecated_call(match='vla_head.peft'):
        head = build_head_from_cfg(finetune_head_cfg(peft=None, lora=legacy))
    assert head.peft_type == 'LORA'
    assert head.peft_finetune is True
    assert adapter_names(head) == adapter_names(
        build_head_from_cfg(finetune_head_cfg()))

    both = finetune_head_cfg()
    both['lora'] = legacy
    with pytest.raises(ValueError, match='not both'):
        build_head_from_cfg(both)


def test_legacy_lora_alias_always_warns_and_keeps_the_old_defaults():
    """A partial (or empty) alias must not adopt PEFT's LoRA defaults."""
    with pytest.deprecated_call(match='lora_alpha=64'):
        empty = legacy_lora_config({})
    assert empty['r'] == 32 and empty['lora_alpha'] == 64
    assert empty['lora_dropout'] == 0.0
    assert empty['target_modules'] == list(DIT_ADAPTER_TARGETS)
    assert empty['peft_type'] == 'LORA'

    with pytest.deprecated_call(match='vla_head.lora'):
        partial = legacy_lora_config(dict(rank=2))
    assert partial['r'] == 2
    assert partial['lora_alpha'] == 64 and partial['lora_dropout'] == 0.0

    with pytest.deprecated_call(match='vla_head.lora'):
        alpha_only = legacy_lora_config(dict(alpha=8, dropout=0.1))
    assert alpha_only['r'] == 32
    assert alpha_only['lora_alpha'] == 8
    assert alpha_only['lora_dropout'] == 0.1

    # Building a head through the alias warns exactly once, from the alias
    # itself; the converted mapping is already PEFT-native.
    tiny_alias = dict(alpha=8,
                      targets=('qkv', 'mix', 'ff.0', 'ff.2'),
                      blocks=('blocks',))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        head = build_head_from_cfg(finetune_head_cfg(peft=None,
                                                     lora=tiny_alias))
        head(preencoded_batch())['loss'].backward()
    deprecations = [item for item in caught
                    if issubclass(item.category, DeprecationWarning)]
    assert len(deprecations) == 1
    assert 'vla_head.lora' in str(deprecations[0].message)
    assert head.peft_type == 'LORA'
    parameters = dict(head.named_parameters())
    assert parameters['denoiser.blocks.0.qkv.lora_B.default.weight'].grad \
        is not None
    assert not parameters[
        'denoiser.blocks.0.qkv.base_layer.weight'].requires_grad

    # The alias is LoRA only; another tuner goes on `vla_head.peft`.
    with pytest.raises(ValueError, match='use `vla_head.peft`'):
        legacy_lora_config(dict(peft_type='LOHA', r=8, alpha=16))


def test_peft_surface_rejects_the_legacy_option_names():
    """`vla_head.peft` is PEFT's own surface: no silent reinterpretation."""
    with pytest.raises(ValueError, match='legacy PatchWAM option names'):
        normalize_adapter_config(dict(peft_type='LORA', rank=2))
    with pytest.raises(ValueError, match='legacy PatchWAM option names'):
        build_head_from_cfg(finetune_head_cfg(
            peft=dict(rank=2, alpha=4, dropout=0.0, targets=('qkv',),
                      blocks=('blocks',))))
    # `alpha` stays a native LoHa option.
    assert normalize_adapter_config(
        dict(peft_type='LOHA', r=8, alpha=16))['alpha'] == 16


def test_invalid_and_unsupported_adapter_configs_fail_clearly():
    def fresh_denoiser():
        return build_head_from_cfg(
            finetune_head_cfg(peft=None, stage=None)).denoiser

    with pytest.raises(ValueError, match='Unsupported PEFT adapter type'):
        attach_dit_adapters(fresh_denoiser(),
                            dict(peft_type='NOT_A_TUNER', r=2,
                                 blocks=('blocks',)))

    with pytest.raises(ValueError, match='prompt-learning'):
        attach_dit_adapters(fresh_denoiser(),
                            dict(peft_type='PROMPT_TUNING',
                                 num_virtual_tokens=4, blocks=('blocks',)))

    with pytest.raises(ValueError, match='not found'):
        attach_dit_adapters(fresh_denoiser(),
                            dict(peft_type='LORA', r=2,
                                 target_modules=('nope',),
                                 blocks=('blocks',)))

    # A target that exists but sits outside the DiT block containers is
    # rejected instead of silently adapting a nonblock projection.
    with pytest.raises(ValueError, match='outside the DiT blocks'):
        attach_dit_adapters(fresh_denoiser(),
                            dict(peft_type='LORA', r=2,
                                 target_modules=('token_input',),
                                 blocks=('blocks',)))

    # The same denoiser cannot carry two adapter tuners.
    adapted = build_head_from_cfg(finetune_head_cfg()).denoiser
    assert adapter_module_paths(adapted)
    with pytest.raises(ValueError, match='already carries PEFT adapters'):
        attach_dit_adapters(adapted, dict(peft_type='LORA', r=2,
                                          target_modules=('qkv',),
                                          blocks=('blocks',)))


def test_adapter_module_reports_a_missing_peft_install():
    """A missing ``peft`` install names the dependency and the install step."""
    probe = '\n'.join([
        'import importlib.util',
        'import sys',
        '',
        "path = sys.argv[1]",
        'class Blocker:',
        '    def find_spec(self, name, path=None, target=None):',
        "        if name == 'peft' or name.startswith('peft.'):",
        "            raise ImportError('blocked for the test')",
        '        return None',
        '',
        'sys.meta_path.insert(0, Blocker())',
        'spec = importlib.util.spec_from_file_location('
        "'dit_peft_probe', path)",
        'module = importlib.util.module_from_spec(spec)',
        'try:',
        '    spec.loader.exec_module(module)',
        'except ImportError as exc:',
        '    print(exc)',
        'else:',
        "    raise SystemExit('dit_peft imported without peft')",
    ])
    completed = subprocess.run(
        [sys.executable, '-c', probe,
         str(REPO_ROOT / 'fluxvla' / 'models' / 'blocks' / 'dit_peft.py')],
        capture_output=True, text=True, check=True)
    assert 'pip install peft' in completed.stdout


def test_peft_dependency_is_declared_for_cpu_tests():
    import peft

    manifest = (REPO_ROOT / 'test' / 'requirements-cpu.txt').read_text()
    declared = [line.strip() for line in manifest.splitlines()
                if line.strip().startswith('peft==')]
    assert declared == [f'peft=={peft.__version__}']
    base = (REPO_ROOT / 'requirements-base.txt').read_text()
    assert f'peft=={peft.__version__}' in base


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
@pytest.mark.parametrize('peft', [
    dict(peft_type='LORA', r=2, lora_alpha=4),
    dict(peft_type='LOHA', r=2, alpha=4),
])
def test_official_flux_blocks_freeze_behind_live_adapters(peft):
    """The real ``double_blocks``/``single_blocks`` naming and execution."""
    head = PatchWAMHead(_small_official_transformer(), action_dim=ACTION_DIM,
                        text_dim=48, proprio_dim=ACTION_DIM, token_dim=128,
                        peft=peft)
    assert head.peft_finetune is True
    assert head.peft_type == peft['peft_type']
    assert head.dit_block_patterns() == [
        'denoiser.transformer.double_blocks.*',
        'denoiser.transformer.single_blocks.*']
    assert head.trainable_parameter_report()['adapter_module_patterns'] == [
        'denoiser.transformer.double_blocks.*.img_attn.proj',
        'denoiser.transformer.double_blocks.*.img_attn.qkv',
        'denoiser.transformer.double_blocks.*.img_mlp.0',
        'denoiser.transformer.double_blocks.*.img_mlp.2',
        'denoiser.transformer.double_blocks.*.txt_attn.proj',
        'denoiser.transformer.double_blocks.*.txt_attn.qkv',
        'denoiser.transformer.double_blocks.*.txt_mlp.0',
        'denoiser.transformer.double_blocks.*.txt_mlp.2',
        'denoiser.transformer.single_blocks.*.linear1',
        'denoiser.transformer.single_blocks.*.linear2',
    ]
    parameters = dict(head.named_parameters())
    frozen = head.dit_block_parameter_names()
    # Query/key norm scales live inside the blocks, so they freeze with them,
    # while the adapter tensors that live there stay trainable.
    assert any(name.endswith('norm.query_norm.scale') for name in frozen)
    assert not any(parameters[name].requires_grad for name in frozen)
    adapters = adapter_names(head)
    assert adapters
    assert all(parameters[name].requires_grad for name in adapters)
    for name in ('denoiser.transformer.img_in.weight',
                 'denoiser.transformer.txt_in.weight',
                 'denoiser.transformer.final_layer.linear.weight'):
        assert parameters[name].requires_grad, name

    batch = preencoded_batch(visual_dim=128, text_dim=48)
    result = head(batch)
    assert torch.isfinite(result['loss']) and result['loss'] > 0
    result['loss'].backward()
    assert all(parameters[name].grad is None for name in frozen)
    updatable = [name for name in adapters
                 if name.endswith(('.lora_B.default.weight',
                                   '.hada_w2_b.default'))]
    assert updatable
    assert all(parameters[name].grad.abs().sum() > 0 for name in updatable)
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
    attached = set(adapter_names(model.vla_head))
    adapter_values = {name: value.clone() for name, value in
                      model.vla_head.named_parameters() if name in attached}
    assert adapter_values
    model.freeze_backbones()
    model.from_pretrained()
    report = model._finetune_initialization
    assert report['format'] == 'bare'
    assert report['new_keys'] == sorted(
        f'vla_head.{name}' for name in adapter_names(model.vla_head))
    for key, value in base.state_dict().items():
        # The released tensors keep their values under the PEFT base_layer.
        head_state = model.vla_head.state_dict()
        torch.testing.assert_close(head_state[base_model_key(head_state, key)],
                                   value, rtol=0, atol=0)
    # A base payload carries no adapters, so they keep their zero residual.
    for name, value in adapter_values.items():
        torch.testing.assert_close(
            model.vla_head.state_dict()[name], value, rtol=0, atol=0)
    weights = dict(model.vla_head.named_parameters())
    for name in adapter_values:
        if name.endswith('lora_B.default.weight'):
            assert weights[name].abs().sum() == 0, name


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
    before = {name: value.clone()
              for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match='Strict PatchWAM checkpoint load'):
        model.from_pretrained()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_finetune_load_rejects_unexpected_and_malformed_payloads(tmp_path):
    base = tiny_policy()
    weights = {key: value.clone() for key, value in base.state_dict().items()}
    stray = tmp_path / 'stray.safetensors'
    save_file(
        dict(weights, **{'denoiser.blocks.99.qkv.weight': torch.zeros(3)}),
        str(stray))
    model = finetune_model(pretrained=str(stray))
    with pytest.raises(ValueError, match='unexpected'):
        model.from_pretrained()

    adapter = next(name for name in adapter_names(finetune_model().vla_head)
                   if name.endswith('lora_A.default.weight'))
    full = dict(finetune_model().vla_head.state_dict())
    full[adapter] = torch.zeros(1, 1)
    malformed = tmp_path / 'malformed.safetensors'
    save_file({f'vla_head.{key}': value for key, value in full.items()},
              str(malformed))
    model = finetune_model(pretrained=str(malformed))
    with pytest.raises(ValueError, match='shape'):
        model.from_pretrained()


def test_finetune_load_rejects_the_removed_private_lora_payload(tmp_path):
    """A bespoke ``dit_lora`` checkpoint fails with an actionable message."""
    weights = {key: value.clone()
               for key, value in tiny_policy().state_dict().items()}
    weights['denoiser.dit_lora.blocks_0_qkv.down.weight'] = torch.zeros(2, 32)
    path = tmp_path / 'legacy_lora.safetensors'
    save_file(weights, str(path))
    model = finetune_model(pretrained=str(path))
    with pytest.raises(ValueError, match='removed private LoRA adapters'):
        model.from_pretrained()
    with pytest.raises(ValueError, match='vla_head.peft'):
        model.from_pretrained()


def test_finetune_native_roundtrip_restores_the_adapters(tmp_path):
    model = finetune_model()
    names = adapter_names(model.vla_head)
    with torch.no_grad():
        for name in names:
            model.vla_head.get_parameter(name).add_(0.25)
    path = tmp_path / 'native.safetensors'
    save_file(
        {key: value.clone() for key, value in model.state_dict().items()},
        str(path))

    restored = finetune_model(pretrained=str(path))
    restored.freeze_backbones()
    restored.from_pretrained()
    assert restored._finetune_initialization['format'] == 'native'
    assert restored._finetune_initialization['new_keys'] == sorted(
        f'vla_head.{name}' for name in adapter_names(restored.vla_head))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0,
                                   atol=0)
    # Loading a strict native payload is what the eval runner does.
    restored.load_state_dict(
        {key: value.clone() for key, value in model.state_dict().items()},
        strict=True)


def test_finetune_native_load_stays_strict_about_adapter_keys(tmp_path):
    """Only this tuner's own adapter tensors are allowed to be new."""
    model = finetune_model()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    adapters = sorted(f'vla_head.{name}'
                      for name in adapter_names(model.vla_head))
    # A payload without the adapter tensors still loads: they are exactly the
    # keys a base payload predates, and every base tensor is present.
    stripped = {key: value for key, value in state.items()
                if key not in set(adapters)}
    path = tmp_path / 'base_only.safetensors'
    save_file(stripped, str(path))
    restored = finetune_model(pretrained=str(path))
    restored.freeze_backbones()
    restored.from_pretrained()
    assert restored._finetune_initialization['new_keys'] == adapters

    stray = dict(stripped)
    stray['vla_head.denoiser.blocks.0.qkv.lora_A.other.weight'] = torch.zeros(
        2, 32)
    path = tmp_path / 'other_adapter.safetensors'
    save_file(stray, str(path))
    restored = finetune_model(pretrained=str(path))
    restored.freeze_backbones()
    with pytest.raises(ValueError, match='unexpected'):
        restored.from_pretrained()


def test_stage2_inherits_a_stage1_payload_across_the_base_layer_split(
        tmp_path):
    """Stage 2 reads a stage 1 payload written before PEFT rewrapped layers."""
    stage1 = build_vla_from_cfg(dict(
        type='PatchWAMVLA', action_horizon=4, num_inference_steps=2,
        vlm_backbone=backbone_cfg(text_encoder=FrozenTextEncoder(),
                                  visual_encoder=StubVisualEncoder()),
        vla_head=finetune_head_cfg(peft=None, stage='stage1'),
        visual_stage='stage1'))
    stage1.freeze_backbones()
    stage1_state = {key: value.clone()
                    for key, value in stage1.vla_head.state_dict().items()}
    path = tmp_path / 'stage1.safetensors'
    save_file(stage1_state, str(path))

    stage2 = build_vla_from_cfg(dict(
        type='PatchWAMVLA', action_horizon=4, num_inference_steps=2,
        vlm_backbone=backbone_cfg(text_encoder=FrozenTextEncoder(),
                                  visual_encoder=StubVisualEncoder()),
        vla_head=finetune_head_cfg(peft=TINY_PEFT, stage='stage2'),
        visual_stage='stage2', stage1_checkpoint=str(path)))
    stage2.freeze_backbones()
    stage2.from_pretrained()
    report = stage2._stage_initialization
    assert report['stage'] == 'stage2'
    assert report['new_keys'] == sorted(
        f'vla_head.{name}' for name in adapter_names(stage2.vla_head))
    head_state = stage2.vla_head.state_dict()
    for key, value in stage1_state.items():
        torch.testing.assert_close(head_state[base_model_key(head_state, key)],
                                   value, rtol=0, atol=0)
    # The stage's own adapters keep their zero residual.
    for name in adapter_names(stage2.vla_head):
        if name.endswith('lora_B.default.weight'):
            assert head_state[name].abs().sum() == 0
    trainable = trainable_names(stage2)
    assert any('lora_A' in name for name in trainable)
    assert not any('base_layer' in name for name in trainable)
    assert not any(name.startswith('vlm_backbone.') for name in trainable)


def test_runner_rejects_a_second_adapter_injection():
    """`model.use_lora` must not stack on top of component-level adapters."""
    from fluxvla.engines.runners.ddp_train_runner import (
        DDPTrainRunner, _existing_adapter_modules)

    model = finetune_model()
    assert _existing_adapter_modules(model)
    assert _existing_adapter_modules(nn.Linear(2, 2)) == []

    runner = DDPTrainRunner.__new__(DDPTrainRunner)
    runner.vla = model
    runner.cfg = SimpleNamespace(model=SimpleNamespace(
        use_lora=True, lora_rank=2, lora_dropout=0.0,
        lora_target_modules=['qkv']))
    with pytest.raises(ValueError, match='already carries PEFT adapters'):
        runner._apply_lora()


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
    assert ddp.peft_finetune is fsdp.peft_finetune is True
    assert all(name.startswith('vla_head.') for name in trainable_names(ddp))
    attached = {f'vla_head.{name}' for name in adapter_names(ddp.vla_head)}
    assert attached & set(trainable_names(ddp))
    assert not any('qkv.base_layer' in name for name in trainable_names(ddp))
    for key, value in base.state_dict().items():
        ddp_head = ddp.vla_head.state_dict()
        fsdp_head = fsdp.vla_head.state_dict()
        torch.testing.assert_close(
            ddp_head[base_model_key(ddp_head, key)], value, rtol=0, atol=0)
        torch.testing.assert_close(
            fsdp_head[base_model_key(fsdp_head, key)], value, rtol=0, atol=0)


def test_finetune_recipe_configs_load_and_match_the_base_recipe(monkeypatch):
    monkeypatch.delenv('PATCHWAM_PRETRAINED_CHECKPOINT', raising=False)
    cfg = Config.fromfile(str(CONFIG_DIR / TRAIN_CONFIG))
    assert VLAS.get(cfg.model.type) is not None
    assert HEADS.get(cfg.model.vla_head.type) is not None
    assert HEADS.get(cfg.model.vla_head.denoiser.type) is not None
    assert cfg.model.vla_head.peft == dict(peft_type='LORA', r=32,
                                           lora_alpha=64, lora_dropout=0.0)
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
    assert cfg.inference_model.vla_head.peft == cfg.model.vla_head.peft

    monkeypatch.setenv('PATCHWAM_PRETRAINED_CHECKPOINT', '/tmp/other_base.pt')
    override = Config.fromfile(str(CONFIG_DIR / TRAIN_CONFIG))
    assert override.model.pretrained_name_or_path == '/tmp/other_base.pt'

    monkeypatch.delenv('PATCHWAM_PRETRAINED_CHECKPOINT', raising=False)
    eval_cfg = Config.fromfile(str(CONFIG_DIR / EVAL_CONFIG))
    assert 'runner' not in eval_cfg
    assert 'auto_scale_lr' not in eval_cfg
    assert eval_cfg.model.vla_head.peft == cfg.model.vla_head.peft
    assert eval_cfg.inference_model.vla_head.peft == cfg.model.vla_head.peft
    assert eval_cfg.eval.type == 'RobotwinEvalRunner'
    assert eval_cfg.model.vla_head.action_dim == 14
    # The adapter mode freezes the frontend, so the eval model must keep the
    # flag that the constructor validates.
    assert eval_cfg.inference_model.freeze_vlm_backbone is True


def _loha_recipe(tmp_path, *, delete):
    """A derived recipe that swaps the tuner, with or without ``_delete_``."""
    derived = tmp_path / ('loha_delete.py' if delete else 'loha_merge.py')
    delete_key = '_delete_=True, ' if delete else ''
    derived.write_text(
        f"""
_base_ = {str(CONFIG_DIR / TRAIN_CONFIG)!r}

_loha = dict({delete_key}peft_type='LOHA', r=8, alpha=16,
             target_modules=['qkv', 'mix', 'ff.0', 'ff.2'],
             blocks=['blocks'])
model = dict(vla_head=dict(peft=dict(_loha)))
inference_model = dict(vla_head=dict(peft=dict(_loha)))
""")
    return Config.fromfile(str(derived))


def test_recipe_swaps_the_adapter_type_and_trains_the_new_tuner(tmp_path):
    """A derived recipe swaps the tuner; `_delete_` is what makes it work."""
    cfg = _loha_recipe(tmp_path, delete=True)
    assert cfg.model.vla_head.peft == dict(
        peft_type='LOHA', r=8, alpha=16,
        target_modules=['qkv', 'mix', 'ff.0', 'ff.2'], blocks=['blocks'])
    assert cfg.inference_model.vla_head.peft == cfg.model.vla_head.peft
    # The inherited recipe settings survive the override.
    assert cfg.model.vla_head.text_dim == 7680
    assert cfg.runner.auto_scale_lr is True

    # Build a real head from the merged recipe mapping. Only the tiny denoiser
    # and the widths it consumes are substituted; `peft` is the recipe's.
    head_cfg = copy.deepcopy(Config(cfg.model.vla_head).to_dict())
    head_cfg.update(denoiser=tiny_denoiser(visual_dim=TOKEN_DIM),
                    text_dim=TEXT_DIM,
                    action_dim=ACTION_DIM, proprio_dim=ACTION_DIM,
                    token_dim=TOKEN_DIM)
    head = build_head_from_cfg(head_cfg)
    assert head.peft_type == 'LOHA'
    assert head.adapter_report['peft_type'] == 'LOHA'
    assert head.adapter_report['target_modules'] == [
        'blocks.0.ff.0', 'blocks.0.ff.2', 'blocks.0.mix', 'blocks.0.qkv',
        'blocks.1.ff.0', 'blocks.1.ff.2', 'blocks.1.mix', 'blocks.1.qkv']
    adapters = adapter_names(head)
    assert adapters and all('.hada_w' in name for name in adapters)
    parameters = dict(head.named_parameters())
    result = head(preencoded_batch(visual_dim=TOKEN_DIM))
    assert torch.isfinite(result['loss']) and result['loss'] > 0
    result['loss'].backward()
    hada = parameters['denoiser.blocks.0.qkv.hada_w2_b.default']
    assert hada.grad is not None and hada.grad.abs().sum() > 0
    assert not parameters[
        'denoiser.blocks.0.qkv.base_layer.weight'].requires_grad
    for name in ('denoiser.token_input.weight', 'denoiser.text_input.weight',
                 'denoiser.token_output.weight'):
        assert parameters[name].grad.abs().sum() > 0, name


def test_derived_recipe_without_delete_keeps_the_inherited_lora_options(
        tmp_path):
    """A plain override deep-merges, and the mixed mapping fails clearly."""
    cfg = _loha_recipe(tmp_path, delete=False)
    merged = dict(cfg.model.vla_head.peft)
    assert merged['peft_type'] == 'LOHA' and merged['alpha'] == 16
    # The inherited LoRA options are still there, which LoHa cannot accept.
    assert merged['lora_alpha'] == 64 and merged['lora_dropout'] == 0.0

    head_cfg = finetune_head_cfg(peft=merged)
    with pytest.raises(ValueError, match='Invalid PEFT adapter options'):
        build_head_from_cfg(head_cfg)


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
        peft=TINY_PEFT)
    vla = build_vla_from_cfg(model_cfg)
    assert vla.peft_finetune is True
    vla.freeze_backbones()
    out = vla(images=torch.rand(2, 3, 2, 32, 64), states=torch.randn(2, 14),
              actions=torch.randn(2, 16, 14), task_description=['a', 'b'])
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    parameters = dict(vla.named_parameters())
    frozen = 'vla_head.denoiser.blocks.0.qkv.base_layer.weight'
    assert parameters[frozen].grad is None
    assert parameters['vla_head.state_projection.weight'].grad is not None
    adapter = 'vla_head.denoiser.blocks.0.qkv.lora_A.default.weight'
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
    """The opt-in recipes only change the adaptation policy."""
    base = Config.fromfile(
        str(CONFIG_DIR / 'patchwam_robotwin_vlm_full_finetune.py')).to_dict()
    cfg = Config.fromfile(str(CONFIG_DIR / TRAIN_CONFIG)).to_dict()
    assert _config_diff(cfg['model'], base['model']) == [
        'pretrained_name_or_path', 'vla_head.peft']
    for section in ('train_dataloader', 'val_dataloader', 'runner'):
        assert cfg[section] == base[section]
    assert cfg['inference_model']['vla_head']['peft'] == \
        cfg['model']['vla_head']['peft']

    base_eval = Config.fromfile(
        str(CONFIG_DIR / 'patchwam_robotwin_eval.py')).to_dict()
    eval_cfg = Config.fromfile(str(CONFIG_DIR / EVAL_CONFIG)).to_dict()
    assert _config_diff(eval_cfg['model'], base_eval['model']) == \
        ['vla_head.peft']
    assert _config_diff(eval_cfg['inference_model'],
                        base_eval['inference_model']) == ['vla_head.peft']
    assert _config_diff(eval_cfg['eval'], base_eval['eval']) == \
        ['norm_stats_path']
