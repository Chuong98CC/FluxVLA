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
"""C-RADIOv4-H targets, the training-only teacher and the auxiliary objective.

:class:`RadioAuxFlowPolicy` keeps the 128-wide VAE-noised future state and adds
a linear velocity residual supervised through a RADIO readout. The auxiliary
target uses a fixed projection of the same future noise minus clean teacher
features; future VAE velocity MSE is disabled.

The teacher is a training dependency only: it is held outside the module tree,
is always frozen/eval/no-grad, never enters ``state_dict`` and is never built
for inference.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import math
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .flow import masked_sample_mse
from .policy import PatchFlowPolicy

__all__ = [
    'DEFAULT_RADIO_FEATURE_DIM',
    'DEFAULT_RADIO_PATCH_SIZE',
    'LocalRadioTeacher',
    'RADIO_V4_VERSION',
    'RadioAuxFlowPolicy',
    'align_supervised_features',
    'load_local_radio_teacher',
    'radio_feature_targets',
    'radio_spatial_features',
    'radio_target_encoder',
    'resolve_capture_layer',
]

RADIO_V4_VERSION = 'c-radio_v4-h'
DEFAULT_RADIO_FEATURE_DIM = 1280
DEFAULT_RADIO_PATCH_SIZE = 16
ADAPTER_LAYER_IDEX_NAME = 'adapter_layer_idex'

_SPATIAL_KEYS = ('spatial_features', 'last_hidden_state', 'features', 'spatial')


def _normalize_version(value: Any) -> str:
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


def radio_spatial_features(result: Any) -> Tensor:
    """Extract the ``[N,S,D]`` spatial stream from an official RADIO return.

    The upstream API yields ``(summary, spatial_features)``, a ``RadioOutput``
    dataclass, a Hugging Face ``ModelOutput`` or a plain mapping. Summary and
    adaptor outputs are never used as supervision.
    """
    if torch.is_tensor(result):
        return result
    for name in _SPATIAL_KEYS:
        value = getattr(result, name, None)
        if torch.is_tensor(value):
            return value
    if isinstance(result, Mapping):
        for name in _SPATIAL_KEYS:
            value = result.get(name)
            if torch.is_tensor(value):
                return value
    if isinstance(result, (tuple, list)):
        if len(result) == 1 and torch.is_tensor(result[0]):
            return result[0]
        if len(result) == 2 and torch.is_tensor(result[1]):
            return result[1]
    raise TypeError(
        'The C-RADIO teacher must return spatial features [N,S,D], a '
        '(summary, spatial_features) pair, a RadioOutput or a mapping with a '
        f'spatial key; got {type(result)!r}')


def _teacher_call(teacher: Any, images: Tensor) -> Tensor:
    """Run an injected teacher or an ``nn.Module`` forward."""
    encode = getattr(teacher, 'encode', None)
    result = encode(images) if callable(encode) else teacher(images)
    return radio_spatial_features(result)


class LocalRadioTeacher(nn.Module):
    """Frozen ``c-radio_v4-h`` spatial features from a local HF export.

    ``images`` are ``[N,3,H,W]`` in ``[0,1]``: the export's own input
    conditioner performs the normalization, so no CLIP/image processor is
    applied here. Calling code holds this module outside the trained module
    tree.
    """

    def __init__(self, model: nn.Module, *, feature_dim: int, patch_size: int,
                 minimum_resolution: int = DEFAULT_RADIO_PATCH_SIZE,
                 min_resolution_step: Optional[int] = None,
                 version: str = RADIO_V4_VERSION,
                 model_path: Optional[str] = None) -> None:
        super().__init__()
        if feature_dim < 1 or patch_size < 1:
            raise ValueError('RADIO feature_dim and patch_size must be positive')
        self.model = model
        self.feature_dim = int(feature_dim)
        self.patch_size = int(patch_size)
        self.minimum_resolution = int(minimum_resolution)
        self.min_resolution_step = (None if not min_resolution_step else
                                    int(min_resolution_step))
        self.version = str(version)
        self.model_path = model_path
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def encode(self, images: Tensor) -> Tensor:
        self.model.eval()
        return _teacher_call(self.model, images)


def _import_export_code(path: Path) -> Any:
    """Import a local export's own model code as a package.

    ``transformers``' remote-code loader copies only the entry module's direct
    relative imports into its cache, so an export whose modules import a
    further sibling cannot be loaded through ``trust_remote_code``. Importing
    the directory as a normal package executes exactly the provided files
    without writing to any cache. The package name is a hash of the resolved
    directory, so two directories that sanitise to the same identifier stay
    distinct.
    """
    resolved = path.resolve()
    digest = hashlib.sha1(str(resolved).encode('utf-8')).hexdigest()[:16]
    package = f'_fluxvla_local_export_{digest}'
    if package not in sys.modules:
        spec = importlib.machinery.ModuleSpec(package, loader=None,
                                              is_package=True)
        module = importlib.util.module_from_spec(spec)
        module.__path__ = [str(resolved)]
        sys.modules[package] = module
    return importlib.import_module(f'{package}.hf_model')


def _load_export_directly(path: Path) -> tuple[Any, nn.Module]:
    """Build a local export from its own code and safetensors weights.

    ``PreTrainedModel.from_pretrained`` builds the model inside a meta-device
    context in recent transformers, which the official RADIO ``__init__``
    cannot survive because its timm backbone reads scalar values off tensors.
    Constructing the class directly and loading the checkpoint's safetensors
    outside that context keeps the provided code unchanged. Weight keys are
    checked both ways so a mismatched checkpoint fails instead of running with
    missing or ignored tensors.
    """
    from safetensors.torch import load_file
    hf_model = _import_export_code(path)
    config = hf_model.RADIOConfig.from_pretrained(str(path),
                                                  local_files_only=True)
    model = hf_model.RADIOModel(config)
    weights = load_file(str(path / 'model.safetensors'))
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if unexpected:
        raise RuntimeError(
            'The local RADIO export has weights this model does not use: '
            f'{sorted(unexpected)[:8]}')
    unresolved = [key for key in missing if '.num_batches_tracked' not in key]
    if unresolved:
        raise RuntimeError(
            'The local RADIO export is missing weights the model needs: '
            f'{sorted(unresolved)[:8]}')
    return config, model


def load_local_radio_teacher(
    model_path: str,
    *,
    version: str = RADIO_V4_VERSION,
    feature_dim: int = DEFAULT_RADIO_FEATURE_DIM,
    patch_size: int = DEFAULT_RADIO_PATCH_SIZE,
    minimum_resolution: int = DEFAULT_RADIO_PATCH_SIZE,
    device: Optional[Any] = None,
    dtype: Optional[torch.dtype] = None,
) -> LocalRadioTeacher:
    """Strictly load a local Hugging Face C-RADIOv4-H export.

    The official export is resolved with ``AutoConfig``/``AutoModel`` using
    ``local_files_only=True`` and ``trust_remote_code=True`` so locally supplied
    model code can run. Nothing is downloaded, no package is installed and there
    is no fallback to an older RADIO implementation: a missing or mismatched
    asset fails with the offending path.
    """
    path = Path(model_path).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            'The C-RADIOv4-H teacher must be a local Hugging Face export '
            f'directory: {path}')
    config_path = path / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(
            f'The C-RADIOv4-H export is missing config.json: {config_path}')
    config = json.loads(config_path.read_text())
    asset_version = config.get('version')
    if _normalize_version(asset_version) != _normalize_version(version):
        raise ValueError(
            f'The local RADIO export reports version {asset_version!r}; this '
            f'variant requires {version!r}')
    asset_patch = config.get('patch_size')
    if asset_patch is not None and int(asset_patch) != int(patch_size):
        raise ValueError(
            f'The local RADIO export uses patch_size={asset_patch}; the model '
            f'is configured for {patch_size}')
    asset_width = config.get('hidden_size') or config.get('embed_dim')
    if asset_width is not None and int(asset_width) != int(feature_dim):
        raise ValueError(
            f'The local RADIO export produces {int(asset_width)}-wide features;'
            f' the model is configured for radio_dim={feature_dim}')
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as error:  # pragma: no cover - transformers is required
        raise RuntimeError(
            'transformers is required to load the local C-RADIOv4-H export'
        ) from error
    try:
        hf_config = AutoConfig.from_pretrained(str(path), local_files_only=True,
                                               trust_remote_code=True)
        model = AutoModel.from_pretrained(str(path), config=hf_config,
                                          local_files_only=True,
                                          trust_remote_code=True)
    except Exception as error:  # noqa: BLE001 - one clear failure
        hf_config = None
        try:
            hf_config, model = _load_export_directly(path)
        except Exception as direct_error:  # noqa: BLE001 - one clear failure
            raise RuntimeError(
                'Unable to build the local C-RADIOv4-H export at '
                f'{path} with AutoModel/AutoConfig (local_files_only=True, '
                'trust_remote_code=True) or by importing its own model code '
                'directly. Confirm the directory is a complete official '
                f'export. AutoModel error: {error!r}. Direct error: '
                f'{direct_error!r}') from direct_error
    declared_version = getattr(hf_config, 'version', None)
    if declared_version is not None and (
            _normalize_version(declared_version)
            != _normalize_version(version)):
        raise ValueError(
            f'The loaded RADIO config reports version {declared_version!r}; '
            f'this variant requires {version!r}')
    declared_width = (getattr(model, 'embed_dim', None)
                      or getattr(hf_config, 'hidden_size', None))
    if declared_width is not None and int(declared_width) != int(feature_dim):
        raise ValueError(
            f'The loaded RADIO model produces {int(declared_width)}-wide '
            f'features; the model is configured for radio_dim={feature_dim}')
    step = (getattr(model, 'min_resolution_step', None)
            or getattr(hf_config, 'min_resolution_step', None)
            or getattr(hf_config, 'window_size', None))
    if step is not None:
        # A window grid multiplies the patch size; keep it in pixel units.
        step = int(step)
        if step < patch_size:
            step *= patch_size
    if device is not None:
        model = model.to(device=device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return LocalRadioTeacher(model, feature_dim=feature_dim,
                             patch_size=patch_size,
                             minimum_resolution=minimum_resolution,
                             min_resolution_step=step,
                             version=version, model_path=str(path))


def _teacher_geometry(teacher: Any, *, patch_size: int,
                      minimum_resolution: int) -> tuple[int, int]:
    """Patch size and padded-input step honoured for this teacher."""
    patch = int(getattr(teacher, 'patch_size', None) or patch_size)
    if patch < 1:
        raise ValueError('The RADIO patch size must be positive')
    step = max(patch, int(minimum_resolution or 0))
    teacher_step = getattr(teacher, 'min_resolution_step', None)
    if teacher_step:
        step = max(step, int(teacher_step))
    return patch, step


def radio_feature_targets(
    video: Tensor,
    *,
    teacher: Any,
    raster: tuple[int, int],
    patch_size: int = DEFAULT_RADIO_PATCH_SIZE,
    minimum_resolution: int = DEFAULT_RADIO_PATCH_SIZE,
    expected_dim: int = DEFAULT_RADIO_FEATURE_DIM,
) -> Tensor:
    """Align per-view RADIO spatial features to the existing future raster.

    ``video`` is ``[N,3,H,W]`` in the native [-1,1] world endpoint range; each
    element is one camera view, so views are never mixed. Valid nonfinite values
    raise; invalid frames/views must already be sanitized to zero by the caller.
    The endpoint is converted to [0,1], padded to the teacher's step grid,
    encoded, and the teacher grid is spatially interpolated to ``raster`` only
    when the two differ. Returns ``[N, raster_h*raster_w, D]``.
    """
    if video.ndim != 4 or video.shape[1] != 3:
        raise ValueError('RADIO future images must be [N,3,H,W]')
    raster_h, raster_w = int(raster[0]), int(raster[1])
    if raster_h < 1 or raster_w < 1:
        raise ValueError('The future raster must be positive')
    height, width = video.shape[-2:]
    if not torch.isfinite(video).all():
        raise ValueError(
            'RADIO future images must be finite on valid coordinates; '
            'sanitize invalid frames/views before the teacher runs')
    # The world model feeds the explicit [-1,1] endpoint; RADIO consumes [0,1]
    # and applies its own input conditioner exactly once.
    images = ((video.float() + 1) / 2).clamp_(0, 1)
    patch, step = _teacher_geometry(teacher, patch_size=patch_size,
                                    minimum_resolution=minimum_resolution)
    padded_h = math.ceil(height / step) * step
    padded_w = math.ceil(width / step) * step
    if (padded_h, padded_w) != (height, width):
        images = F.pad(images, (0, padded_w - width, 0, padded_h - height))
    parameter = (next(teacher.parameters(), None)
                 if isinstance(teacher, nn.Module) else None)
    if parameter is not None:
        # Match the teacher's device/dtype explicitly so mixed precision never
        # silently promotes or mismatches the export's parameters.
        images = images.to(device=parameter.device, dtype=parameter.dtype)
    # The teacher is frozen: targets never carry gradient into its weights or
    # into the encoder, even when a caller injects a plain ``nn.Module``.
    with torch.no_grad():
        features = _teacher_call(teacher, images)
    if features.ndim == 4 and features.shape[1] == int(expected_dim):
        features = features.flatten(2).transpose(1, 2)
    if features.ndim != 3 or features.shape[0] != images.shape[0]:
        raise ValueError('The RADIO teacher must return [N,S,D] spatial features')
    if features.shape[-1] != int(expected_dim):
        raise ValueError(
            f'The RADIO teacher produced {features.shape[-1]} features; the '
            f'model is configured for radio_dim={expected_dim}')
    grid_h, grid_w = padded_h // patch, padded_w // patch
    if features.shape[1] != grid_h * grid_w:
        raise ValueError(
            f'The RADIO teacher returned {features.shape[1]} tokens; the '
            f'{grid_h}x{grid_w} grid derived from patch_size={patch} and step='
            f'{step} on a {padded_h}x{padded_w} input expects '
            f'{grid_h * grid_w}')
    # Crop the tokens that only cover padding back to the real image extent.
    needed_h = math.ceil(height / patch)
    needed_w = math.ceil(width / patch)
    if (needed_h, needed_w) != (grid_h, grid_w):
        features = features.reshape(-1, grid_h, grid_w, features.shape[-1])
        features = features[:, :needed_h, :needed_w].reshape(
            -1, needed_h * needed_w, features.shape[-1])
    if (needed_h, needed_w) != (raster_h, raster_w):
        features = F.interpolate(
            features.transpose(1, 2).reshape(-1, features.shape[-1],
                                             needed_h, needed_w),
            size=(raster_h, raster_w), mode='bilinear', align_corners=False,
        ).flatten(2).transpose(1, 2)
    return features


def radio_target_encoder(
    teacher_provider: Callable[[torch.device], Any],
    *,
    patch_size: int = DEFAULT_RADIO_PATCH_SIZE,
    minimum_resolution: int = DEFAULT_RADIO_PATCH_SIZE,
    expected_dim: int = DEFAULT_RADIO_FEATURE_DIM,
) -> Callable[[Tensor, tuple[int, int]], Tensor]:
    """Build the teacher hook consumed by ``encode_observations``."""

    def encode_future(video: Tensor, raster: tuple[int, int]) -> Tensor:
        teacher = teacher_provider(video.device)
        return radio_feature_targets(
            video, teacher=teacher, raster=raster, patch_size=patch_size,
            minimum_resolution=minimum_resolution, expected_dim=expected_dim)

    return encode_future


def align_supervised_features(tokens: Tensor, valid: Optional[Tensor], *,
                              name: str, device: Optional[torch.device] = None,
                              dtype: Optional[torch.dtype] = None) -> Tensor:
    """Detach, align and finiteness-check a supervised RADIO target.

    ``valid`` may arrive on another device (it is not stored with the cached
    features); invalid padding is allowed to hold anything, but every supervised
    coordinate must be finite so a NaN target fails clearly instead of turning
    the loss into NaN.
    """
    if not torch.is_tensor(tokens):
        raise TypeError(f'`{name}` must be a tensor')
    target = tokens.detach()
    if device is not None or dtype is not None:
        target = target.to(device=device, dtype=dtype)
    if valid is None:
        supervised = target
    else:
        if not torch.is_tensor(valid):
            raise TypeError(f'`{name}` validity must be a tensor')
        valid = valid.to(device=target.device, dtype=torch.bool)
        if valid.shape != target.shape[:2]:
            raise ValueError(f'`{name}` validity must be [B,S]')
        supervised = target[valid]
    if not torch.isfinite(supervised).all():
        raise ValueError(
            f'`{name}` must be finite on valid coordinates; masked padding may '
            'hold any value, supervised coordinates may not')
    return target


def _denoiser_reference(denoiser: nn.Module) -> Optional[torch.Tensor]:
    """First denoiser parameter used as the device/dtype reference."""
    return next(denoiser.parameters(), None)


def make_radio_noise_projection(radio_dim: int, token_dim: int,
                                seed: int = 0) -> Tensor:
    """Fixed shared-noise lifting matrix ``P`` (float32, unit-norm rows).

    Built from a private seeded generator so no global RNG draw is consumed.
    Each row is normalized to unit L2 norm; the matrix is training-only and is
    not a learned parameter.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f'radio_noise_seed must be a non-negative integer, got {seed!r}')
    generator = torch.Generator().manual_seed(int(seed))
    # Explicit float32 so the matrix never follows the global default dtype.
    matrix = torch.randn(int(radio_dim), int(token_dim),
                         dtype=torch.float32, generator=generator)
    return matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)


def resolve_capture_layer(adapter_layer_idex: int, depth: int, *,
                          name: str = ADAPTER_LAYER_IDEX_NAME) -> int:
    """Translate the public zero-based layer selector into capture order.

    ``adapter_layer_idex`` indexes block outputs from zero (``0..depth-1``) with
    ``-1`` meaning the last block. Transformer order is all double blocks
    followed by all single blocks, which is exactly the order the existing
    capture walks. Returns the 1-based ``representation_layer`` value the
    transformer capture expects.
    """
    if isinstance(adapter_layer_idex, bool) or not isinstance(
            adapter_layer_idex, int):
        raise ValueError(
            f'`{name}` must be an integer block index in [-1, {depth - 1}] '
            f'(-1 selects the last block); got {adapter_layer_idex!r}')
    if adapter_layer_idex < -1 or adapter_layer_idex >= depth:
        raise ValueError(
            f'`{name}` must be -1 (last block) or a zero-based block index in '
            f'[0, {depth - 1}]; got {adapter_layer_idex}')
    return depth if adapter_layer_idex == -1 else adapter_layer_idex + 1


class RadioAuxFlowPolicy(PatchFlowPolicy):
    """Fused linear RADIO velocity auxiliary on the original flow.

    Keeps the original 128-wide VAE future state, noising and sampler and adds
    three biased linear adapters plus one fixed shared-noise projection:

    * ``radio_a``: ``token_dim -> radio_dim`` (nonzero init, bias zero)
    * ``radio_b``: ``hidden_dim -> radio_dim`` (weight and bias zero)
    * ``radio_c``: ``radio_dim -> token_dim`` (weight and bias zero)
    * ``radio_noise_projection``: fixed float32 ``P [radio_dim, token_dim]``

    ``Yprime = A(Y) + B(H_selected)`` and ``Ydoubleprime = Y + C(Yprime)`` is
    the shared velocity head for every generated future/action token, while the
    training-only readout ``R = A(Ydoubleprime) + B(H_selected)`` is supervised
    on future tokens against ``P @ epsilon_z - r0`` (the same sanitized parent
    future noise). Zero-initialized B/C weights and all-zero biases make
    ``Ydoubleprime == Y`` at startup, so pretrained outputs are unchanged.
    """

    def __init__(self, denoiser: nn.Module, *, action_dim: int, text_dim: int,
                 radio_dim: int = DEFAULT_RADIO_FEATURE_DIM,
                 radio_weight: float = 1.0,
                 radio_noise_seed: int = 0,
                 adapter_layer_idex: int = -1,
                 video_weight: float = 0.0,
                 **policy_options) -> None:
        super().__init__(denoiser, action_dim=action_dim, text_dim=text_dim,
                         video_weight=video_weight,
                         **policy_options)
        if policy_options.get('self_flow_variant'):
            raise ValueError(
                'Self-Flow is not supported for the RADIO auxiliary variant')
        if isinstance(radio_dim, bool) or not isinstance(radio_dim, int) \
                or radio_dim < 1:
            raise ValueError(
                f'radio_dim must be a positive integer, got {radio_dim!r}')
        if not math.isfinite(float(radio_weight)) or radio_weight < 0:
            raise ValueError('radio_weight must be finite and non-negative')
        if float(self.video_weight) != 0:
            raise ValueError(
                'video_weight must be 0 for the RADIO auxiliary variant: the '
                'future VAE velocity MSE is not computed; use radio_weight and '
                'action_weight instead')
        width = getattr(denoiser, 'representation_dim', None)
        if width is None:
            raise ValueError(
                'The RADIO auxiliary head requires a feature-capable denoiser '
                'exposing representation_dim')
        depth = getattr(denoiser, 'representation_layers', None)
        if not depth:
            raise ValueError(
                'The RADIO auxiliary head requires a denoiser exposing '
                'representation_layers so `adapter_layer_idex` can be '
                'validated')
        # Public zero-based selector, kept as a plain configuration attribute
        # (never a parameter or buffer) next to the derived 1-based layer.
        self.adapter_layer_idex = adapter_layer_idex
        self._capture_layer = resolve_capture_layer(adapter_layer_idex,
                                                    int(depth))
        self.radio_dim = int(radio_dim)
        self.radio_weight = float(radio_weight)
        self.radio_noise_seed = radio_noise_seed
        token_dim = int(self.codec.token_dim)
        self.radio_a = nn.Linear(token_dim, int(radio_dim), bias=True)
        self.radio_b = nn.Linear(int(width), int(radio_dim), bias=True)
        self.radio_c = nn.Linear(int(radio_dim), token_dim, bias=True)
        with torch.no_grad():
            # B/C start at exactly zero so the fused velocity equals the
            # original output at initialization; all biases stay zero.
            self.radio_b.weight.zero_()
            self.radio_c.weight.zero_()
            for adapter in (self.radio_a, self.radio_b, self.radio_c):
                adapter.bias.zero_()
        self.register_buffer(
            'radio_noise_projection',
            make_radio_noise_projection(int(radio_dim), token_dim,
                                        radio_noise_seed))
        reference = _denoiser_reference(denoiser)
        if reference is not None:
            # Match the transformer's device/dtype instead of staying on the
            # default device in float32.
            for adapter in (self.radio_a, self.radio_b, self.radio_c):
                adapter.to(device=reference.device, dtype=reference.dtype)
            self.radio_noise_projection = (
                self.radio_noise_projection.to(device=reference.device))
        # Fuse every generated token through the shared Ydoubleprime head.
        self.fused_hidden_layer = self._capture_layer
        self.velocity_transform = self.radio_fuse

    def _apply(self, fn, recurse: bool = True):
        """Keep the fixed projection in float32 across dtype conversions."""
        keep = self._buffers.get('radio_noise_projection')
        keep = None if keep is None else keep.detach().clone()
        super()._apply(fn, recurse=recurse)
        buffer = self._buffers.get('radio_noise_projection')
        if keep is not None and buffer is not None:
            # ``.half()`` / ``.bfloat16()`` must not lose the fixed target
            # precision: restore the exact float32 values on the new device.
            self._buffers['radio_noise_projection'] = keep.to(
                device=buffer.device)
        return self

    @property
    def capture_layer(self) -> Optional[int]:
        """1-based transformer layer whose conditioned hidden B consumes."""
        if self.radio_weight == 0:
            return None
        return self._capture_layer

    @property
    def adapter_keys(self) -> tuple:
        """Trainable adapter parameter names (relative to the policy)."""
        names = []
        for module in ('radio_a', 'radio_b', 'radio_c'):
            names.extend((f'{module}.weight', f'{module}.bias'))
        return tuple(names)

    def is_final_selection(self) -> bool:
        layers = getattr(self.denoiser, 'representation_layers', None)
        return bool(layers) and int(self._capture_layer) == int(layers)

    def radio_fuse(self, velocity: Tensor, hidden: Tensor) -> Tensor:
        """``Ydoubleprime``: the shared fused velocity head."""
        return velocity + self.radio_c(
            self.radio_a(velocity) + self.radio_b(hidden))

    def radio_readout(self, fused_velocity: Tensor, hidden: Tensor) -> Tensor:
        """Training-only ``R = A(Ydoubleprime) + B(H_selected)``."""
        return self.radio_a(fused_velocity) + self.radio_b(hidden)

    def project_future_noise(self, noise: Tensor) -> Tensor:
        """``epsilon_R = P epsilon_z`` in float32 (no target gradients)."""
        matrix = self.radio_noise_projection.to(device=noise.device,
                                                dtype=torch.float32)
        # P [radio_dim, token_dim] lifted over the token_dim axis of the
        # [B, S, token_dim] parent noise.
        return torch.einsum('rc,btc->btr', matrix, noise.float())

    def final_output_affine(self):
        """Original final ``(weight, bias)`` in float32 for fusion algebra."""
        affine = getattr(self.denoiser, 'final_output_affine', None)
        if not callable(affine):
            raise ValueError(
                'The RADIO fused export requires a denoiser exposing '
                'final_output_affine')
        return affine()

    def fused_output_affine(self):
        """Fold A/B/C into the final affine output projection (float32)."""
        final_weight, final_bias = self.final_output_affine()
        a_w, a_b = self.radio_a.weight.detach().float(), self.radio_a.bias.detach().float()
        b_w, b_b = self.radio_b.weight.detach().float(), self.radio_b.bias.detach().float()
        c_w, c_b = self.radio_c.weight.detach().float(), self.radio_c.bias.detach().float()
        weight = final_weight + c_w @ (a_w @ final_weight + b_w)
        bias = final_bias + c_w @ (a_w @ final_bias + a_b + b_b) + c_b
        return weight, bias

    def forward(self, batch, **kwargs):
        if kwargs.get('representation_only') or kwargs.get('teacher_sampling'):
            raise ValueError(
                'The RADIO auxiliary loss is only defined for the training '
                'flow forward')
        result = super().forward(batch, supervise_video=False, **kwargs)
        result['loss_vae'] = result['loss_video']
        if self.radio_weight == 0:
            # Action-only objective: no teacher, no readout, no target, but the
            # learned velocity residual still executes.
            result.pop('fused_context', None)
            result['loss_radio'] = result['loss_video'].detach()
            result['loss'] = self.action_weight * result['loss_action']
            return result
        context = result.pop('fused_context', None)
        target = batch.get('radio_future_tokens')
        if context is None or not torch.is_tensor(target):
            raise ValueError(
                'radio_weight>0 requires `radio_future_tokens` and the fused '
                'RADIO flow context')
        selected = context['selected_hidden']
        future_length = int(batch['future_tokens'].shape[1])
        if selected.ndim != 3 or selected.shape[1] < future_length:
            raise ValueError(
                'The selected transformer states must cover the future tokens')
        hidden = selected[:, :future_length]
        clean = align_supervised_features(
            target, batch.get('future_valid'), name='radio_future_tokens',
            device=hidden.device)
        noise = context['future_noise'].detach()
        velocity_target = (self.project_future_noise(noise)
                           - clean.float()).detach()
        readout = self.radio_readout(
            context['future_velocity'][:, :future_length], hidden)
        if readout.shape != velocity_target.shape:
            raise ValueError(
                f'The RADIO readout {tuple(readout.shape)} does not match the '
                f'velocity target {tuple(velocity_target.shape)}')
        valid = context['valid_future']
        per_sample = masked_sample_mse(readout.float(), velocity_target,
                                       valid)
        loss_radio = (self.flow.weight(context['sigma']) * per_sample).mean()
        result['loss_radio'] = loss_radio
        result['loss'] = (self.radio_weight * loss_radio
                          + self.action_weight * result['loss_action'])
        return result
