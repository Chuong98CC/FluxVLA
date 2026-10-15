# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from fluxvla.engines import HEADS
from fluxvla.engines.utils.builder import build_from_cfg
from ..third_party_models.patchwam import (PatchFlowPolicy,
                                           SmallJointTransformer)
from ..blocks.dit_lora import (LORA_ATTRIBUTE, attach_dit_lora,
                               lora_block_prefixes, lora_parameter_names)
from ..third_party_models.patchwam.flux import OfficialFluxDenoiser

__all__ = ['PatchWAMFluxDenoiser', 'PatchWAMHead', 'PatchWAMTinyDenoiser']


@HEADS.register_module()
class PatchWAMTinyDenoiser(SmallJointTransformer):
    """Offline, randomly initialized joint transformer for CPU tests.

    Registered under ``HEADS`` so ``PatchWAMHead`` configs can build the tiny
    denoiser through the same ``type``-based construction used everywhere else
    in FluxVLA. It carries no benchmark weights and is never a substitute for
    the official FLUX.2 denoiser.
    """

    def __init__(self, *args, output_bias: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if output_bias:
            # Fused RADIO inference expects a bias-enabled output projection;
            # the bias starts at zero so original weights stay exact.
            self.enable_output_bias()


@HEADS.register_module()
class PatchWAMFluxDenoiser(OfficialFluxDenoiser):
    """Registered loader for the official FLUX.2 prefix-masked denoiser.

    Owns the loaded transformer and preserves the official parameter names
    (``transformer.*``) and the prefix-masked attention execution. Loading is
    strict and local: the transformer weights are read from ``transformer_path``
    and the official ``flux2`` package is imported (optionally from
    ``flux_source``). No network access, no remote code.

    ``OfficialFluxDenoiser.__init__`` validates the Klein base layer contract
    (``img_in``/``txt_in``/``double_blocks``/``single_blocks``/... and
    ``use_guidance_embed == False``), so an incompatible checkpoint fails loudly
    instead of silently running the wrong attention path.
    """

    def __init__(
        self,
        transformer_path: str,
        *,
        flux_source: Optional[str] = None,
        variant: str = 'klein-base-4b',
        device: Union[str, torch.device] = 'cuda',
        dtype: str = 'bfloat16',
        gradient_checkpointing: bool = False,
        output_bias: bool = False,
        visual_dim: Optional[int] = None,
    ) -> None:
        transformer, asset_signature = _load_official_transformer(
            transformer_path=transformer_path,
            flux_source=flux_source,
            variant=variant,
            device=device,
            dtype=dtype,
        )
        super().__init__(transformer, gradient_checkpointing=gradient_checkpointing,
                         visual_dim=visual_dim)
        self.variant = variant
        self.asset_signature = asset_signature
        if output_bias:
            # Adds the single fused output bias after the original strict asset
            # load; it is zero-initialized.
            self.enable_output_bias()


@HEADS.register_module()
class PatchWAMHead(PatchFlowPolicy):
    """FluxVLA head wrapping the vendored PatchWAM joint flow policy.

    The head is a thin, config-aware subclass of the copied
    :class:`~fluxvla.models.third_party_models.patchwam.policy.PatchFlowPolicy`.
    It owns the denoiser, proprio projection, parameter-free action codec, and
    the flow objective; the encoded ``reference_tokens`` / ``future_tokens`` /
    ``text_tokens`` and masks are produced upstream by the backbone (or passed
    through as cached tensors). All numerical behavior, mask polarity, and
    random draw order come from the vendored core.

    ``denoiser`` may be an already-constructed ``nn.Module`` (used by the
    parity tests to load identical weights) or a ``type``-keyed config. A dict
    config is built through ``HEADS``; ``PatchWAMTinyDenoiser`` is the offline
    default, and the official FLUX.2 denoiser is added in a later phase.

    Self-Flow and EMA training require engine-owned optimizer/teacher hooks that
    are not part of the first native integration. A nonzero ``self_flow_variant``
    is rejected here instead of silently running an unsupervised variant.

    ``lora`` without ``stage`` selects ordinary (unstaged) finetuning: the DiT
    block bases stay frozen behind the ``dit_lora`` adapters, and every other
    policy layer stays fully trainable. ``stage`` keeps its staged meaning
    (stage 1 the new visual projections, stage 2 those projections plus the
    adapters), and leaving both unset keeps the previous fully trainable head.
    """

    def __init__(
        self,
        denoiser: Union[nn.Module, Mapping[str, Any]],
        *,
        action_dim: int,
        text_dim: int,
        proprio_dim: Optional[int] = None,
        token_dim: int = 128,
        action_scale: float = 1.0,
        shift: float = 5.0,
        video_weight: float = 0.5,
        action_weight: float = 1.0,
        isolate_actions: bool = False,
        condition_dropout: float = 0.0,
        flow_normalization: str = 'continuous',
        self_flow_variant: int = 0,
        pack_proprio_after_text: bool = False,
        action_mode: str = 'joint_only',
        joint_action_dim: Optional[int] = None,
        end_effector_action_dim: Optional[int] = None,
        visual_dim: Optional[int] = None,
        stage: Optional[str] = None,
        lora: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self_flow_variant:
            raise ValueError(
                'Self-Flow training is deferred from the first native PatchWAM '
                'integration; set self_flow_variant=0. EMA/self-flow optimizer '
                'hooks and checkpointed teacher state are not wired up.')
        if stage is not None and stage not in self.STAGE_GROUPS:
            raise ValueError(f'stage must be one of {sorted(self.STAGE_GROUPS)}')
        if lora is not None and stage not in (None, 'stage2'):
            raise ValueError(
                'DiT LoRA adapters belong to stage 2 or to an unstaged LoRA '
                'finetune (stage=None); stage 1 trains only the new visual '
                'projections')
        if stage is not None and visual_dim in (None, token_dim):
            raise ValueError(
                'A staged recipe needs new visual projections, so visual_dim '
                'must differ from token_dim')
        denoiser = _build_denoiser(denoiser, token_dim=token_dim,
                                   text_dim=text_dim, visual_dim=visual_dim)
        _validate_denoiser_dims(denoiser, token_dim=token_dim,
                                text_dim=text_dim, visual_dim=visual_dim)
        super().__init__(
            denoiser,
            action_dim=action_dim,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            token_dim=token_dim,
            visual_dim=visual_dim,
            action_scale=action_scale,
            shift=shift,
            video_weight=video_weight,
            action_weight=action_weight,
            isolate_actions=isolate_actions,
            condition_dropout=condition_dropout,
            flow_normalization=flow_normalization,
            pack_proprio_after_text=pack_proprio_after_text,
            action_mode=action_mode,
            joint_action_dim=joint_action_dim,
            end_effector_action_dim=end_effector_action_dim,
        )
        self.stage: Optional[str] = None
        # True for the unstaged LoRA finetune policy (adapters set, no stage).
        self.lora_finetune = bool(lora is not None and stage is None)
        if lora is not None:
            options = dict(lora)
            unknown = sorted(set(options) -
                             {'rank', 'alpha', 'dropout', 'targets', 'blocks'})
            if unknown:
                raise ValueError(f'Unknown LoRA options: {unknown}')
            attach_dit_lora(self.denoiser, **options)
        if stage is not None:
            self.apply_stage_policy(stage)
        elif self.lora_finetune:
            self.apply_lora_finetune_policy()

    #: Parameter groups each staged recipe opens.
    STAGE_GROUPS = {'stage1': ('visual_projection',),
                    'stage2': ('visual_projection', 'dit_lora')}

    #: Parameter groups the unstaged LoRA policy opens. ``policy`` holds every
    #: trainable head parameter outside the frozen DiT blocks and the adapters.
    FINETUNE_GROUPS = ('policy', 'dit_lora')

    #: Groups a stage introduces on top of the previous checkpoint.
    STAGE_NEW_GROUPS = {'stage1': ('visual_projection',), 'stage2': ('dit_lora',)}

    def visual_parameter_names(self) -> list:
        """Names of the new visual input/output projections, if present."""
        names = set()
        for module_name, module in self.named_modules():
            if module_name and module_name.rsplit('.', 1)[-1] in {'visual_in',
                                                                  'visual_out'}:
                names.update(f'{module_name}.{name}'
                             for name, _ in module.named_parameters())
        return sorted(names)

    def _group_parameter_names(self, stage: str) -> dict:
        if stage not in self.STAGE_GROUPS:
            raise ValueError(f'stage must be one of {sorted(self.STAGE_GROUPS)}')
        resolved = {'visual_projection': self.visual_parameter_names()}
        lora_names = [f'denoiser.{name}'
                      for name in lora_parameter_names(self.denoiser)]
        resolved['dit_lora'] = sorted(lora_names)
        return {group: names
                for group, names in resolved.items()
                if group in self.STAGE_GROUPS[stage]}

    def apply_stage_policy(self, stage: str) -> dict:
        """Freeze the head and open only this stage's trainable groups.

        Stage 1 trains the new visual projections over the frozen pretrained
        policy; stage 2 adds the DiT LoRA adapters. The frozen autoencoder /
        visual backbone, text encoder, state projection, and pretrained action
        maps stay frozen in both stages.
        """
        resolved = self._group_parameter_names(stage)
        for group, names in resolved.items():
            if not names:
                raise ValueError(
                    f'stage {stage} requires {group} parameters, but none exist')
        self.requires_grad_(False)
        parameters = dict(self.named_parameters())
        for names in resolved.values():
            for name in names:
                parameters[name].requires_grad_(True)
        self.stage = stage
        return resolved

    def parameter_groups(self) -> list:
        """Optimizer groups for the active policy, in configured order."""
        parameters = dict(self.named_parameters())
        if self.lora_finetune:
            groups = []
            for group in self.FINETUNE_GROUPS:
                names = self._finetune_group_names(group)
                if names:
                    groups.append({
                        'name': group,
                        'params': [parameters[name] for name in names],
                    })
            return groups
        if self.stage is None:
            return []
        groups = []
        for group, names in self._group_parameter_names(self.stage).items():
            if names:
                groups.append({'name': group,
                               'params': [parameters[name] for name in names]})
        return groups

    def _finetune_group_names(self, group: str) -> list:
        """Head-relative parameter names of one unstaged-LoRA optimizer group."""
        if group not in self.FINETUNE_GROUPS:
            raise ValueError(
                f'group must be one of {list(self.FINETUNE_GROUPS)}')
        adapters = set(self._lora_parameter_names())
        if group == 'dit_lora':
            return sorted(adapters)
        return sorted(name for name, param in self.named_parameters()
                      if param.requires_grad and name not in adapters)

    def _lora_parameter_names(self) -> list:
        """Head-relative names of the attached DiT LoRA adapter parameters."""
        return [f'denoiser.{name}'
                for name in lora_parameter_names(self.denoiser)]

    def dit_block_patterns(self) -> list:
        """Dotted patterns of the adapted (frozen) DiT block subtrees."""
        return sorted(f'denoiser.{prefix}.*'
                      for prefix in lora_block_prefixes(self.denoiser))

    def dit_block_parameter_names(self) -> list:
        """Head-relative base parameters of the DiT block subtrees."""
        parameters = dict(self.named_parameters())
        prefixes = tuple(f'denoiser.{prefix}.'
                         for prefix in lora_block_prefixes(self.denoiser))
        if not prefixes:
            return []
        return sorted(name for name in parameters if name.startswith(prefixes))

    def apply_lora_finetune_policy(self) -> dict:
        """Open every policy parameter except the frozen DiT block bases.

        Ordinary (unstaged) finetuning keeps the released policy trainable and
        freezes only the adapted DiT transformer blocks: the entire
        ``double_blocks``/``single_blocks`` subtree (``blocks`` for the tiny CPU
        denoiser), modulation and normalization parameters included. Every other
        head layer - the denoiser's input, output, time, text and visual
        projections, the shared stream modulations, the parameter-free action
        codec and the state/action maps - stays fully trainable beside the
        ``dit_lora`` adapters.
        """
        if not self.lora_finetune:
            raise ValueError(
                'apply_lora_finetune_policy requires an unstaged LoRA head '
                'configuration (`lora` set with `stage=None`)')
        frozen = self.dit_block_parameter_names()
        if not frozen:
            raise ValueError(
                'The LoRA finetune policy needs the DiT block parameters the '
                'attached adapters were resolved from, but none were found')
        self.requires_grad_(True)
        parameters = dict(self.named_parameters())
        for name in frozen:
            parameters[name].requires_grad_(False)
        return {'mode': 'lora_finetune',
                'dit_block_patterns': self.dit_block_patterns(),
                'frozen_parameters': frozen}

    def finetune_new_keys(self) -> list:
        """Head-relative keys that only a base pretrained payload may omit.

        Empty unless this head runs the unstaged LoRA policy: the adapters are
        the only tensors a pretrained PatchWAM payload cannot carry.
        """
        if not self.lora_finetune:
            return []
        return sorted(self._lora_parameter_names())

    def trainable_parameter_report(self) -> dict:
        """Fully qualified trainable names and patterns for this policy.

        The names are head-relative; ``trainable_module_patterns`` collapses the
        adapters to one ``denoiser.dit_lora.*`` pattern so the report stays
        readable for the full-size FLUX.2 transformer.
        """
        parameters = dict(self.named_parameters())
        adapter_prefix = f'denoiser.{LORA_ATTRIBUTE}.'
        names = sorted(name for name, param in parameters.items()
                       if param.requires_grad)
        modules = set()
        for name in names:
            module = name.rsplit('.', 1)[0]
            if module.startswith(adapter_prefix):
                module = adapter_prefix[:-1]
            modules.add(module)
        return {
            'mode':
            self.stage or ('lora_finetune' if self.lora_finetune else 'none'),
            'dit_block_patterns': self.dit_block_patterns(),
            'trainable_parameter_names': names,
            'trainable_module_patterns': sorted(f'{module}.*'
                                                for module in modules),
            'adapter_parameter_names': [
                name for name in names if name.startswith(adapter_prefix)
            ],
            'num_trainable_parameters': len(names),
        }

    def staged_new_keys(self, stage: str) -> list:
        """Head-relative checkpoint keys this stage introduces."""
        if stage not in self.STAGE_NEW_GROUPS:
            raise ValueError(
                f'stage must be one of {sorted(self.STAGE_NEW_GROUPS)}')
        parameters = dict(self.named_parameters())
        resolved = self._group_parameter_names(stage)
        return sorted(name for group in self.STAGE_NEW_GROUPS[stage]
                      for name in resolved[group] if name in parameters)

    @property
    def condition_features(self) -> int:
        """Width of the encoded text context consumed by the denoiser."""
        return self.text_dim


def _build_denoiser(denoiser: Union[nn.Module, Mapping[str, Any]], *,
                    token_dim: int, text_dim: int,
                    visual_dim: Optional[int] = None) -> nn.Module:
    """Return an ``nn.Module`` denoiser from a module or ``type`` config."""
    if isinstance(denoiser, nn.Module):
        return denoiser
    if not isinstance(denoiser, Mapping):
        raise TypeError(
            'PatchWAMHead.denoiser must be an nn.Module or a type-keyed '
            f'config mapping, got {type(denoiser)!r}')
    cfg = dict(denoiser)
    denoiser_type = cfg.setdefault('type', 'PatchWAMTinyDenoiser')
    if visual_dim is not None:
        cfg.setdefault('visual_dim', visual_dim)
    if denoiser_type == 'PatchWAMTinyDenoiser':
        # Only the tiny joint transformer shares the policy token/text widths.
        # The official FLUX.2 wrapper takes explicit asset arguments instead of
        # silently absorbing tiny-only kwargs.
        cfg.setdefault('token_dim', token_dim)
        cfg.setdefault('text_dim', text_dim)
    built = build_from_cfg(cfg, HEADS)
    if not isinstance(built, nn.Module):
        raise TypeError('denoiser config must build an nn.Module, got '
                        f'{type(built)!r}')
    return built


def _load_official_transformer(*, transformer_path: str,
                               flux_source: Optional[str], variant: str,
                               device: Union[str, torch.device], dtype: str):
    """Strictly load a local FLUX.2 transformer and its asset identity."""
    import importlib
    import sys

    from ..third_party_models.patchwam.flux import _asset_file_identity

    if variant not in {'klein-base-4b', 'klein-base-9b'}:
        raise ValueError('variant must be klein-base-4b or klein-base-9b')
    if dtype not in {'float32', 'float16', 'bfloat16'}:
        raise ValueError('Unsupported torch dtype')
    if not Path(transformer_path).is_file():
        raise FileNotFoundError(transformer_path)
    package_root = None
    if flux_source is not None:
        package_root = Path(flux_source).expanduser().resolve() / 'src'
        if not (package_root / 'flux2' / 'model.py').is_file():
            raise FileNotFoundError(
                'flux_source must be the official FLUX.2 repository root')
        sys.path.insert(0, str(package_root))
    try:
        model_api = importlib.import_module('flux2.model')
    except ImportError as exc:
        raise ImportError(
            'Install the official black-forest-labs/flux2 package or supply '
            'flux_source') from exc
    if package_root is not None and not Path(model_api.__file__).resolve().is_relative_to(
            package_root / 'flux2'):
        raise RuntimeError(
            'Another FLUX.2 source is already imported; start a fresh process '
            'for flux_source')
    from safetensors.torch import load_file

    precision = getattr(torch, dtype)
    config = (model_api.Klein4BParams()
              if variant.endswith('4b') else model_api.Klein9BParams())
    with torch.device('meta'):
        transformer = model_api.Flux2(config).to(dtype=precision)
    transformer.load_state_dict(
        load_file(transformer_path, device=str(device)), strict=True,
        assign=True)
    transformer = transformer.to(device=device, dtype=precision)
    return transformer, _asset_file_identity(Path(transformer_path))


def _validate_denoiser_dims(denoiser: nn.Module, *, token_dim: int,
                            text_dim: int,
                            visual_dim: Optional[int] = None) -> None:
    """Check the denoiser's real input projections match the policy widths."""
    transformer = getattr(denoiser, 'transformer', None)
    if transformer is not None:
        txt_in = getattr(transformer, 'txt_in', None)
        img_in = getattr(transformer, 'img_in', None)
        if txt_in is not None and txt_in.in_features != text_dim:
            raise ValueError(
                f'transformer.txt_in expects {txt_in.in_features} context '
                f'features but text_dim={text_dim}')
        if img_in is not None and img_in.in_features != token_dim:
            raise ValueError(
                f'transformer.img_in expects {img_in.in_features} latent '
                f'channels but token_dim={token_dim}')
        if visual_dim is not None and visual_dim != token_dim:
            visual_in = getattr(denoiser, 'visual_in', None)
            if visual_in is None or visual_in.in_features != visual_dim:
                raise ValueError(
                    f'The denoiser does not provide a {visual_dim}-wide visual '
                    'input projection; pass visual_dim to the denoiser config')
        return
    text_input = getattr(denoiser, 'text_input', None)
    token_input = getattr(denoiser, 'token_input', None)
    if text_input is not None and text_input.in_features != text_dim:
        raise ValueError(
            f'denoiser.text_input expects {text_input.in_features} context '
            f'features but text_dim={text_dim}')
    if token_input is not None and token_input.in_features != token_dim:
        raise ValueError(
            f'denoiser.token_input expects {token_input.in_features} latent '
            f'channels but token_dim={token_dim}')
    if visual_dim is not None and visual_dim != token_dim:
        visual_in = getattr(denoiser, 'visual_in', None)
        if visual_in is None or visual_in.in_features != visual_dim:
            raise ValueError(
                f'The denoiser does not provide a {visual_dim}-wide visual '
                'input projection; pass visual_dim to the denoiser config')
