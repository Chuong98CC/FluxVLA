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
    ) -> None:
        transformer, asset_signature = _load_official_transformer(
            transformer_path=transformer_path,
            flux_source=flux_source,
            variant=variant,
            device=device,
            dtype=dtype,
        )
        super().__init__(transformer, gradient_checkpointing=gradient_checkpointing)
        self.variant = variant
        self.asset_signature = asset_signature


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
    ) -> None:
        if self_flow_variant:
            raise ValueError(
                'Self-Flow training is deferred from the first native PatchWAM '
                'integration; set self_flow_variant=0. EMA/self-flow optimizer '
                'hooks and checkpointed teacher state are not wired up.')
        denoiser = _build_denoiser(denoiser, token_dim=token_dim,
                                   text_dim=text_dim)
        _validate_denoiser_dims(denoiser, token_dim=token_dim,
                                text_dim=text_dim)
        super().__init__(
            denoiser,
            action_dim=action_dim,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            token_dim=token_dim,
            action_scale=action_scale,
            shift=shift,
            video_weight=video_weight,
            action_weight=action_weight,
            isolate_actions=isolate_actions,
            condition_dropout=condition_dropout,
            flow_normalization=flow_normalization,
        )

    @property
    def condition_features(self) -> int:
        """Width of the encoded text context consumed by the denoiser."""
        return self.text_dim


def _build_denoiser(denoiser: Union[nn.Module, Mapping[str, Any]], *,
                    token_dim: int, text_dim: int) -> nn.Module:
    """Return an ``nn.Module`` denoiser from a module or ``type`` config."""
    if isinstance(denoiser, nn.Module):
        return denoiser
    if not isinstance(denoiser, Mapping):
        raise TypeError(
            'PatchWAMHead.denoiser must be an nn.Module or a type-keyed '
            f'config mapping, got {type(denoiser)!r}')
    cfg = dict(denoiser)
    denoiser_type = cfg.setdefault('type', 'PatchWAMTinyDenoiser')
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
                            text_dim: int) -> None:
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
