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

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.nn as nn

from fluxvla.engines import VLM_BACKBONES, initialize_overwatch
from ...third_party_models.patchwam.encoding import encode_observations
from ...third_party_models.patchwam.vision_language import \
    LowRankResidualLinear

overwatch = initialize_overwatch(__name__)

__all__ = ['PatchWAMBackbone']


@VLM_BACKBONES.register_module()
class PatchWAMBackbone(nn.Module):
    """PatchWAM observation frontend: frozen autoencoder + instruction encoder.

    Owns only the packed-latent autoencoder and the language/visual instruction
    encoder. It exposes ``encode(batch, target) -> dict`` producing the
    ``reference_tokens`` / ``future_tokens`` / ids / masks consumed by
    :class:`~fluxvla.models.heads.patchwam_head.PatchWAMHead`, mirroring the
    vendored ``FluxAssetPolicy._encode`` contract. The denoiser, codec, and flow
    objective stay in the head, so backbone and head parameters are disjoint.

    ``autoencoder`` / ``text_encoder`` may be injected modules (used by tests) or
    loaded from local assets via the ``*_path`` arguments. Loading requires the
    official ``flux2`` package and local weights; nothing is downloaded.
    """

    def __init__(
        self,
        autoencoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        *,
        token_dim: int = 128,
        text_dim: int = 32,
        history_pool_size: int = 4,
        visual_backbone: Optional[Mapping[str, Any]] = None,
        visual_encoder: Optional[nn.Module] = None,
        visual_dim: Optional[int] = None,
        autoencoder_path: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        flux_source: Optional[str] = None,
        text_encoder_kind: str = 'qwen3',
        device: Union[str, torch.device] = 'cpu',
        dtype: str = 'bfloat16',
        context_length: int = 128,
        vl_trainable_adapters: bool = False,
        vl_lora_rank: int = 64,
        vl_lora_alpha: float = 128,
        vl_lora_dropout: float = 0.05,
        vl_instruction_layers=(9, 18, 27),
        vl_summary_layers=(18, 27, 36),
        vl_history_pool_size: int = 4,
        vl_image_size: int = 448,
        vl_auxiliary_weight: float = 0.1,
        vl_label_smoothing: float = 0.1,
        vl_require_subtasks: bool = False,
        vl_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if history_pool_size < 1:
            raise ValueError('history_pool_size must be positive')
        if visual_backbone is not None and visual_encoder is not None:
            raise ValueError(
                'Configure either a `visual_backbone` spec or an injected '
                '`visual_encoder`, not both')
        self.token_dim = int(token_dim)
        self.text_dim = int(text_dim)
        self.history_pool_size = int(history_pool_size)
        # A configured visual backbone replaces the frozen autoencoder as the
        # observation frontend; the shared wrappers own the range conversion
        # and the raster geometry.
        self.visual_encoder = _build_visual_encoder(visual_backbone,
                                                   visual_encoder)
        self.visual_dim = (None if self.visual_encoder is None else int(
            _declared_visual_dim(self.visual_encoder, visual_dim)))
        needs_autoencoder = self.visual_encoder is None and autoencoder is None
        if needs_autoencoder or text_encoder is None:
            if text_encoder_path is None or (needs_autoencoder
                                             and autoencoder_path is None):
                raise ValueError(
                    'PatchWAMBackbone requires an injected text encoder or '
                    '`text_encoder_path` (plus `autoencoder_path` when no '
                    'visual backbone is configured).')
            loaded_autoencoder, text_encoder = _load_local_modules(
                # A configured visual backbone never allocates or loads the VAE.
                autoencoder_path=(autoencoder_path if needs_autoencoder else
                                  None),
                text_encoder_path=text_encoder_path,
                tokenizer_path=tokenizer_path,
                flux_source=flux_source,
                text_encoder_kind=text_encoder_kind,
                device=device,
                dtype=dtype,
                context_length=context_length,
                token_dim=self.token_dim,
                text_dim=self.text_dim,
                vl_trainable_adapters=vl_trainable_adapters,
                vl_lora_rank=vl_lora_rank,
                vl_lora_alpha=vl_lora_alpha,
                vl_lora_dropout=vl_lora_dropout,
                vl_instruction_layers=vl_instruction_layers,
                vl_summary_layers=vl_summary_layers,
                vl_history_pool_size=vl_history_pool_size,
                vl_image_size=vl_image_size,
                vl_auxiliary_weight=vl_auxiliary_weight,
                vl_label_smoothing=vl_label_smoothing,
                vl_require_subtasks=vl_require_subtasks,
                vl_gradient_checkpointing=vl_gradient_checkpointing,
            )
            if needs_autoencoder:
                autoencoder = loaded_autoencoder
        self.autoencoder = None if self.visual_encoder is not None else autoencoder
        self.text_encoder = text_encoder
        feature_dim = getattr(text_encoder, 'feature_dim', None)
        if feature_dim is not None and feature_dim != self.text_dim:
            raise ValueError(
                f'text_encoder provides {feature_dim} features but text_dim='
                f'{self.text_dim}')
        self.apply_trainable_policy()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(self, batch: Mapping[str, Any], target: bool, *,
               radio_encoder=None) -> Dict[str, Any]:
        """Return the encoded-batch contract consumed by the PatchWAM head.

        ``radio_encoder`` optionally adds detached RADIO feature targets beside
        the unchanged frozen VAE latents (used by the auxiliary
        representation-loss variant); the default ``None`` keeps the original
        behavior.
        """
        return encode_observations(
            batch,
            autoencoder=self.autoencoder,
            text_encoder=self.text_encoder,
            token_dim=self.token_dim,
            text_dim=self.text_dim,
            history_pool_size=self.history_pool_size,
            target=target,
            radio_encoder=radio_encoder,
            visual_encoder=self.visual_encoder,
            visual_dim=self.visual_dim,
        )

    def forward(self,
                images: Optional[torch.Tensor] = None,
                target: bool = False,
                **kwargs) -> Dict[str, Any]:
        request = dict(kwargs)
        if images is not None:
            request['images'] = images
        return self.encode(request, target=target)

    # ------------------------------------------------------------------
    # Freeze / train policy
    # ------------------------------------------------------------------
    def apply_trainable_policy(self) -> 'PatchWAMBackbone':
        """Freeze the AE and encoder base while keeping language LoRA live."""
        if self.autoencoder is not None:
            self.autoencoder.requires_grad_(False)
            self.autoencoder.eval()
        if self.visual_encoder is not None:
            self.visual_encoder.requires_grad_(False)
            self.visual_encoder.eval()
        encoder = self.text_encoder
        for param in encoder.parameters():
            param.requires_grad_(False)
        if getattr(encoder, 'trainable_adapters', False):
            enabled = 0
            for name, param in encoder.named_parameters():
                if 'residual_in' in name or 'residual_out' in name:
                    param.requires_grad_(True)
                    enabled += 1
            if not enabled:
                raise ValueError(
                    'trainable_adapters is set but no LoRA residual parameters '
                    'were found')
        else:
            encoder.eval()
        return self

    def train(self, mode: bool = True) -> 'PatchWAMBackbone':
        super().train(mode)
        if self.autoencoder is not None:
            self.autoencoder.eval()
        if self.visual_encoder is not None:
            # Frozen encoders never leave evaluation mode.
            self.visual_encoder.eval()
        if getattr(self.text_encoder, 'requires_images', False):
            self.text_encoder.train(mode)
        else:
            self.text_encoder.eval()
        return self

    def trainable_parameter_names(self) -> list:
        return [
            name for name, param in self.named_parameters()
            if param.requires_grad
        ]

    def checkpoint_contract(self) -> Dict[str, Any]:
        contract = getattr(self.text_encoder, 'checkpoint_contract', None)
        return {
            'history_pool_size': self.history_pool_size,
            'token_dim': self.token_dim,
            'text_dim': self.text_dim,
            'visual_backbone': _visual_contract(self.visual_encoder),
            'text_encoder':
            contract() if callable(contract) else
            type(self.text_encoder).__qualname__,
        }


def _build_visual_encoder(visual_backbone: Optional[Mapping[str, Any]],
                          visual_encoder: Optional[nn.Module]):
    """Resolve the optional shared visual backbone from a spec or module.

    Both paths come from ``fluxvla.models.backbones.visions``: a ``kind`` of
    ``c-radio_v4-h``/``c-radio_v4-so400m`` builds the frozen C-RADIO wrapper
    from a local export, and ``vae`` wraps an injected autoencoder. Nothing is
    downloaded and no encoder is re-implemented here.
    """
    if visual_encoder is not None:
        return visual_encoder
    if visual_backbone is None:
        return None
    from fluxvla.models.backbones.visions import (RadioViTBackbone,
                                                  VAELatentVisionBackbone)
    options = dict(visual_backbone)
    kind = options.pop('kind', None)
    if kind in ('c-radio_v4-h', 'c-radio_v4-so400m', 'radio', 'c-radio'):
        if kind in ('c-radio_v4-h', 'c-radio_v4-so400m'):
            options.setdefault('version', kind)
        return RadioViTBackbone(**options)
    if kind in ('vae', 'vae-latents', 'vae_latents'):
        if 'autoencoder' not in options:
            raise ValueError(
                'A VAE visual backbone requires a local `autoencoder` module')
        return VAELatentVisionBackbone(**options)
    raise ValueError(f'Unknown visual backbone kind: {kind!r}')


def _declared_visual_dim(visual_encoder: nn.Module,
                         configured: Optional[int]) -> int:
    declared = getattr(visual_encoder, 'feature_dim', None)
    if declared is None:
        if configured is None:
            raise ValueError(
                'The visual backbone must declare `feature_dim` or the recipe '
                'must set `visual_dim`')
        return int(configured)
    if configured is not None and int(configured) != int(declared):
        raise ValueError(
            f'The visual backbone produces {int(declared)}-wide features but '
            f'visual_dim={int(configured)} was configured')
    return int(declared)


def _visual_contract(visual_encoder: Optional[nn.Module]):
    if visual_encoder is None:
        return None
    contract = getattr(visual_encoder, 'checkpoint_contract', None)
    if callable(contract):
        return contract()
    return {
        'identifier': getattr(visual_encoder, 'identifier', None),
        'feature_dim': getattr(visual_encoder, 'feature_dim', None),
        'resolution_step': getattr(visual_encoder, 'resolution_step', None),
    }


def _load_local_modules(*, autoencoder_path: str, text_encoder_path: str,
                        flux_source: Optional[str], text_encoder_kind: str,
                        tokenizer_path: Optional[str] = None,
                        device: Union[str, torch.device], dtype: str,
                        context_length: int, token_dim: int, text_dim: int,
                        vl_trainable_adapters: bool, vl_lora_rank: int,
                        vl_lora_alpha: float, vl_lora_dropout: float,
                        vl_instruction_layers, vl_summary_layers,
                        vl_history_pool_size: int, vl_image_size: int,
                        vl_auxiliary_weight: float, vl_label_smoothing: float,
                        vl_require_subtasks: bool,
                        vl_gradient_checkpointing: bool):
    """Strictly load the frozen autoencoder and the local instruction encoder."""
    import importlib
    import sys

    if autoencoder_path is not None and not Path(autoencoder_path).is_file():
        raise FileNotFoundError(autoencoder_path)
    if not Path(text_encoder_path).is_dir():
        raise FileNotFoundError(
            'text_encoder_path must be a local Hugging Face model directory')
    if dtype not in {'float32', 'float16', 'bfloat16'}:
        raise ValueError('Unsupported torch dtype')
    if text_encoder_kind not in {'qwen3', 'qwen3_vl'}:
        raise ValueError('text_encoder_kind must be qwen3 or qwen3_vl')
    package_root = None
    if autoencoder_path is not None and flux_source is not None:
        package_root = Path(flux_source).expanduser().resolve() / 'src'
        if not (package_root / 'flux2' / 'model.py').is_file():
            raise FileNotFoundError(
                'flux_source must be the official FLUX.2 repository root')
        sys.path.insert(0, str(package_root))
    from safetensors.torch import load_file

    precision = getattr(torch, dtype)
    autoencoder = None
    if autoencoder_path is not None:
        try:
            ae_api = importlib.import_module('flux2.autoencoder')
        except ImportError as exc:
            raise ImportError(
                'Install the official black-forest-labs/flux2 package or '
                'supply flux_source') from exc
        if package_root is not None and not Path(
                ae_api.__file__).resolve().is_relative_to(package_root /
                                                          'flux2'):
            raise RuntimeError(
                'Another FLUX.2 source is already imported; start a fresh '
                'process for flux_source')
        with torch.device('meta'):
            autoencoder = ae_api.AutoEncoder(ae_api.AutoEncoderParams()).to(
                dtype=precision)
        autoencoder.load_state_dict(
            load_file(autoencoder_path, device=str(device)), strict=True,
            assign=True)
        autoencoder = autoencoder.to(device=device, dtype=precision)

    from ...third_party_models.patchwam.flux import LocalInstructionEncoder
    if text_encoder_kind == 'qwen3':
        encoder = LocalInstructionEncoder(text_encoder_path,
                                          tokenizer_path=tokenizer_path,
                                          context_length=context_length,
                                          dtype=precision).to(device)
    else:
        from ...third_party_models.patchwam.vision_language import \
            VisualInstructionEncoder
        encoder = VisualInstructionEncoder(
            text_encoder_path,
            dtype=precision,
            trainable_adapters=vl_trainable_adapters,
            lora_rank=vl_lora_rank,
            lora_alpha=vl_lora_alpha,
            lora_dropout=vl_lora_dropout,
            instruction_layers=vl_instruction_layers,
            summary_layers=vl_summary_layers,
            history_pool_size=vl_history_pool_size,
            image_size=vl_image_size,
            auxiliary_weight=vl_auxiliary_weight,
            label_smoothing=vl_label_smoothing,
            require_subtasks=vl_require_subtasks,
            gradient_checkpointing=vl_gradient_checkpointing,
        ).to(device)
    if getattr(encoder, 'feature_dim', None) != text_dim:
        raise ValueError(
            f'text encoder provides {getattr(encoder, "feature_dim", None)} '
            f'features; the head expects {text_dim}')
    return autoencoder, encoder
