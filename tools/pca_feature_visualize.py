#!/usr/bin/env python3
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
"""Spatial-feature PCA visualization for vision backbones.

Three backbones are supported: C-RADIO through the frozen PyTorch wrapper
(``cradio``), C-RADIO through a serialized TensorRT engine (``cradio-trt``) and
a local Hugging Face DINOv2/DINOv3 ViT (``dino``). Each one produces finite
``[N, D]`` spatial tokens plus a truthful ``(h, w)`` grid, which is reduced to
three PCA components and written next to a plot of the resized input.

Nothing heavy is imported at module import, so ``--help`` works everywhere and
no model, network or proxy setting is touched until a backend is actually run.
Local assets only: remote model fetching requires ``--allow-remote``.

Examples:

    CKPT=checkpoints/Pretrained_models/C-RADIOv4-H
    ENG=$CKPT/exported/256x256/c_radio_v4_h_256.engine

    python tools/pca_feature_visualize.py --backbone cradio \
        --model-path $CKPT --image assets/frame_000000.jpg --image-size 256 \
        --output runs/visualization/c_radio_pca_256/pytorch_h

    python tools/pca_feature_visualize.py --backbone cradio-trt \
        --engine-path $ENG --image assets/frame_000000.jpg --image-size 256 \
        --pca-reference runs/visualization/c_radio_pca_256/pytorch_h.pt \
        --output runs/visualization/c_radio_pca_256/fp16_trt
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Canonical backbone names and the aliases accepted on the command line.
BACKBONE_CHOICES: Tuple[str, ...] = ('cradio', 'cradio-trt', 'dino')
_BACKBONE_ALIASES: Dict[str, str] = {
    'cradio': 'cradio',
    'c-radio': 'cradio',
    'radio': 'cradio',
    'cradio-pytorch': 'cradio',
    'cradio-pt': 'cradio',
    'cradio-trt': 'cradio-trt',
    'c-radio-trt': 'cradio-trt',
    'radio-trt': 'cradio-trt',
    'trt': 'cradio-trt',
    'dino': 'dino',
    'dinov2': 'dino',
    'dinov3': 'dino',
    'dino-v2': 'dino',
    'dino-v3': 'dino',
}

DEFAULT_IMAGE_SIZE = 256
DEFAULT_VERSION = 'c-radio_v4-h'
DEFAULT_OUTPUT_ROOT = 'runs/visualization'

#: Canonical C-RADIO variants and the spellings accepted for each one.
RADIO_VARIANT_FORMS: Dict[str, Tuple[str, ...]] = {
    'c-radio_v4-h': (
        'c-radio_v4-h',
        'c-radio-v4-h',
        'cradiov4h',
        'nvidia/C-RADIOv4-H',
        'C-RADIOv4-H',
    ),
    'c-radio_v4-so400m': (
        'c-radio_v4-so400m',
        'c-radio-v4-so400m',
        'cradiov4so400m',
        'nvidia/C-RADIOv4-SO400M',
        'C-RADIOv4-SO400M',
    ),
}


def _identifier_key(value: Any) -> str:
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


_RADIO_VARIANT_BY_FORM: Dict[str, str] = {
    _identifier_key(form): variant
    for variant, forms in RADIO_VARIANT_FORMS.items()
    for form in forms
}


class PcaVisualizeError(RuntimeError):
    """A user-facing failure with a message meant to be printed."""


def normalize_backbone(value: str) -> str:
    """Canonical backbone name for a user-supplied identifier."""
    key = str(value).strip().lower()
    if key not in _BACKBONE_ALIASES:
        raise PcaVisualizeError(f'Unknown backbone {value!r}; expected one of '
                                f'{sorted(BACKBONE_CHOICES)} (aliases: '
                                f'{sorted(_BACKBONE_ALIASES)})')
    return _BACKBONE_ALIASES[key]


def parse_image_size(values: Optional[Sequence[Any]],
                     default: int = DEFAULT_IMAGE_SIZE) -> Tuple[int, int]:
    """``(height, width)`` from one square size or an explicit ``H W`` pair."""
    if not values:
        return (int(default), int(default))
    if len(values) not in (1, 2):
        raise PcaVisualizeError(
            '--image-size takes one square size or an explicit height and '
            f'width; got {list(values)}')
    numbers = []
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise PcaVisualizeError(
                f'--image-size values must be integers; got {value!r}')
        if number < 1:
            raise PcaVisualizeError(
                f'--image-size values must be positive; got {number}')
        numbers.append(number)
    if len(numbers) == 1:
        return (numbers[0], numbers[0])
    return (numbers[0], numbers[1])


def resolve_dtype(name: str) -> Any:
    """Torch dtype for ``fp32``/``fp16``/``bf16``."""
    import torch
    dtypes = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
    }
    if name not in dtypes:
        raise PcaVisualizeError(f'Unsupported dtype {name!r}; expected one of '
                                f'{sorted(dtypes)}')
    return dtypes[name]


def resolve_radio_variant(model_name: Optional[str],
                          version: Optional[str]) -> str:
    """The C-RADIO variant selected by ``--model-name`` and/or ``--version``.

    ``--model-name`` accepts the canonical identifiers and the official
    NVIDIA names (``nvidia/C-RADIOv4-H``, ``nvidia/C-RADIOv4-SO400M``);
    ``--version`` is the same selector kept for scripts written against the
    earlier interface. Giving both is fine only when they agree, and no
    selector at all keeps the historical H default.
    """
    chosen: List[Tuple[str, str]] = []
    for option, value in (('--model-name', model_name), ('--version',
                                                         version)):
        if not value:
            continue
        variant = _RADIO_VARIANT_BY_FORM.get(_identifier_key(value))
        if variant is None:
            raise PcaVisualizeError(
                f'{option} {value!r} is not a C-RADIO variant; expected one '
                f'of {sorted(RADIO_VARIANT_FORMS)} or an official name such '
                "as 'nvidia/C-RADIOv4-H'")
        chosen.append((option, variant))
    if not chosen:
        return DEFAULT_VERSION
    variants = sorted({variant for _, variant in chosen})
    if len(variants) > 1:
        detail = ', '.join(f'{option}={variant}' for option, variant in chosen)
        raise PcaVisualizeError(
            f'Conflicting C-RADIO selection: {detail}. Pass one variant.')
    return variants[0]


# -- PCA --------------------------------------------------------------------


def fit_spatial_pca(tokens: Any,
                    *,
                    n_components: int = 3,
                    reference: Optional[Dict[str, Any]] = None,
                    grid: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    """Reduce ``[N, D]`` tokens to three displayable PCA components.

    With ``reference`` the projection uses that bundle's mean, components and
    scaling instead of refitting, so two backends are compared in the same
    axes rather than in independently rotated ones. Components are ordered by
    singular value and each one's sign is fixed deterministically (the largest
    magnitude loading is made positive), so repeated runs agree. Constant
    components scale to zero instead of dividing by zero.
    """
    import numpy as np
    matrix = np.asarray(tokens, dtype=np.float64)
    if matrix.ndim != 2:
        raise PcaVisualizeError(
            f'Spatial tokens must be [N, D]; got shape {matrix.shape}')
    if not np.isfinite(matrix).all():
        raise PcaVisualizeError('Spatial tokens contain non-finite values')
    count, width = matrix.shape
    components = (
        int(reference['components'].shape[0])
        if reference is not None else int(n_components))
    if reference is None and min(count, width) < components:
        raise PcaVisualizeError(
            f'PCA needs at least {components} tokens and {components} feature '
            f'dimensions; got {count} tokens of width {width}')
    if reference is None:
        mean = matrix.mean(axis=0)
        centered = matrix - mean
        _, singular, right = np.linalg.svd(centered, full_matrices=False)
        basis = right[:components]
        signs = np.where(
            basis[np.arange(components),
                  np.abs(basis).argmax(axis=1)] < 0, -1.0, 1.0)
        basis = basis * signs[:, None]
        total = float((singular**2).sum())
        variance = (singular[:components]**2 / max(count - 1, 1))
        ratio = ((singular[:components]**2) /
                 total).tolist() if total else [0.0] * components
    else:
        mean = np.asarray(reference['mean'], dtype=np.float64)
        basis = np.asarray(reference['components'], dtype=np.float64)
        variance = np.asarray(
            reference.get('variance', np.zeros(components)), dtype=np.float64)
        ratio = list(reference.get('variance_ratio', [0.0] * components))
        if mean.shape[0] != width or basis.shape[1] != width:
            raise PcaVisualizeError(
                f'The PCA reference expects width {basis.shape[1]} but the '
                f'tokens are {width}-wide')
    scores = (matrix - mean) @ basis.T
    if reference is not None and 'scale_min' in reference:
        low = np.asarray(reference['scale_min'], dtype=np.float64)
        high = np.asarray(reference['scale_max'], dtype=np.float64)
    else:
        low, high = scores.min(axis=0), scores.max(axis=0)
    report = {
        'mean': mean,
        'components': basis,
        'variance': variance,
        'variance_ratio': ratio,
        'scale_min': low,
        'scale_max': high,
        'tokens': count,
        'feature_dim': width,
        'components_used': components,
        'refit': reference is None,
    }
    report['scores'] = scores
    report['image'] = pca_image(scores, low, high, grid)
    return report


def pca_image(scores: Any, low: Any, high: Any,
              grid: Optional[Tuple[int, int]]) -> Any:
    """Map PCA scores to a unit-range RGB raster of shape ``(h, w, 3)``."""
    import numpy as np
    span = np.asarray(
        high, dtype=np.float64) - np.asarray(
            low, dtype=np.float64)
    scaled = np.where(span > 0, (np.asarray(scores, dtype=np.float64) - low) /
                      np.where(span > 0, span, 1.0), 0.0)
    scaled = np.clip(scaled, 0.0, 1.0)
    if grid is None:
        return scaled
    height, width = int(grid[0]), int(grid[1])
    if scaled.shape[0] != height * width:
        raise PcaVisualizeError(
            f'PCA produced {scaled.shape[0]} tokens for a {height}x{width} '
            'grid')
    return scaled.reshape(height, width, scaled.shape[1])


# -- preprocessing and backends --------------------------------------------


def load_wrapper_image(path: Any, height: int,
                       width: int) -> Tuple[Any, Dict[str, Any]]:
    """The shared C-RADIO preprocessing: bicubic RGB resize to ``[-1, 1]``."""
    from tools import c_radio_export
    return c_radio_export.load_image(path, height=height, width=width)


def unit_rgb(wrapper: Any) -> Any:
    """The decoded image as displayable unit-range ``(H, W, 3)`` RGB."""
    from tools import c_radio_export
    return c_radio_export.unit_range(wrapper)[0].permute(1, 2, 0).numpy()


def backbone_patch_size(backbone: Any) -> int:
    """Patch size of a C-RADIO backbone, from the wrapper or its metadata."""
    for source in (backbone, getattr(backbone, 'metadata', None)):
        patch = getattr(source, 'patch_size', None)
        if patch:
            return int(patch)
    raise PcaVisualizeError(
        'The backbone exposes no patch size; cannot report the token grid')


def encoded_size(backbone: Any, height: int, width: int) -> Tuple[int, int]:
    """The geometry the encoder would actually consume for a request.

    The official ``get_nearest_supported_resolution`` helper is authoritative
    when the wrapped model provides one; otherwise the wrapper's own
    patch-aligned raster is used (and a fixed-geometry engine raises the
    compiled-resolution error here).
    """
    model = getattr(backbone, 'model', None)
    nearest = getattr(model, 'get_nearest_supported_resolution', None)
    if callable(nearest):
        try:
            resolved = nearest(int(height), int(width))
        except TypeError:
            resolved = nearest((int(height), int(width)))
        if hasattr(resolved, 'height') and hasattr(resolved, 'width'):
            return (int(resolved.height), int(resolved.width))
        if isinstance(resolved, (tuple, list)) and len(resolved) == 2:
            return (int(resolved[0]), int(resolved[1]))
    grid = backbone.feature_grid(int(height), int(width))
    patch = backbone_patch_size(backbone)
    return (int(grid[0]) * patch, int(grid[1]) * patch)


def check_geometry(backbone: Any, height: int, width: int,
                   backend: str) -> Tuple[Tuple[int, int], int]:
    """Exact token grid and patch size, or a clear geometry failure.

    A request the encoder would round to another resolution is rejected: the
    plotted input has to be the image the encoder actually consumed, so the
    supported size is named rather than silently substituted.
    """
    request = (int(height), int(width))
    try:
        encoded = encoded_size(backbone, height, width)
        if encoded != request:
            raise PcaVisualizeError(
                f'{backend} encodes {request[0]}x{request[1]} at '
                f'{encoded[0]}x{encoded[1]}; request a supported size exactly,'
                ' because this tool plots the input the encoder actually saw')
        grid = backbone.feature_grid(request[0], request[1])
        patch = backbone_patch_size(backbone)
    except PcaVisualizeError:
        raise
    except ValueError as error:
        raise PcaVisualizeError(f'{backend} cannot encode {request[0]}'
                                f'x{request[1]}: {error}') from error
    return (int(grid[0]), int(grid[1])), patch


def encode_cradio(backbone: Any, wrapper: Any, *, device: Any,
                  dtype: str) -> Tuple[Any, Tuple[int, int], Dict[str, Any]]:
    """Encode ``[-1, 1]`` images with a frozen C-RADIO wrapper.

    FP32 runs under :func:`strict_fp32` (no TF32). Lower precision uses
    autocast rather than half-precision weights, because the export's input
    conditioner emits FP32 and a half-precision parameter set would mismatch
    it.
    """
    import torch
    from tools import c_radio_export
    images = wrapper.to(device)
    target = resolve_dtype(dtype)
    with torch.no_grad():
        if target is torch.float32:
            with c_radio_export.strict_fp32() as precision:
                output = backbone.encode_features(images)
            precision_note = {
                'weights': 'float32',
                'autocast': None,
                'tf32': bool(precision['matmul_tf32']),
            }
        else:
            autocast_dtype = target if images.is_cuda else torch.bfloat16
            with torch.autocast(
                    device_type='cuda' if images.is_cuda else 'cpu',
                    dtype=autocast_dtype):
                output = backbone.encode_features(images)
            precision_note = {
                'weights': 'float32',
                'autocast': str(autocast_dtype),
                'tf32': None,
            }
    tokens = output.tokens[0].detach().float().cpu().numpy()
    grid = (int(output.grid[0]), int(output.grid[1]))
    diagnostics = {
        'dtype': dtype,
        'precision': precision_note,
        'tokens': int(tokens.shape[0]),
        'feature_dim': int(tokens.shape[1]),
        'grid': [grid[0], grid[1]],
        'finite': bool(torch.isfinite(output.tokens).all()),
    }
    return tokens, grid, diagnostics


def check_dtype_for_device(dtype: str, device: Any, backend: str) -> None:
    """Reject a precision the selected device cannot actually run."""
    if str(device) == 'cpu' and dtype == 'fp16':
        raise PcaVisualizeError(
            f'CPU autocast does not support fp16 (requested for {backend}); '
            'use --dtype fp32 or bf16, or --device cuda')


def _load_cradio_pytorch(args: argparse.Namespace, device: Any) -> Any:
    import torch
    from fluxvla.models.backbones.visions.radio_vit import RadioViTBackbone
    if not args.model_path:
        raise PcaVisualizeError(
            '--backbone cradio needs --model-path pointing at a complete '
            'local C-RADIO export')
    if not Path(args.model_path).is_dir():
        raise PcaVisualizeError(
            f'The C-RADIO export directory does not exist: {args.model_path}')
    # FP32 weights: autocast handles a requested lower precision.
    return RadioViTBackbone.from_local_export(
        args.model_path,
        version=args.version,
        dtype=torch.float32,
        device=device)


def _load_cradio_trt(args: argparse.Namespace, device: Any) -> Any:
    from fluxvla.models.backbones.visions.radio_trt import (
        RadioViTBackboneTensorRT)
    if not args.engine_path:
        raise PcaVisualizeError('--backbone cradio-trt needs --engine-path')
    if not Path(args.engine_path).is_file():
        raise PcaVisualizeError(
            f'The TensorRT engine does not exist: {args.engine_path}')
    return RadioViTBackboneTensorRT.from_engine(
        args.engine_path, metadata_path=args.metadata_path, device=device)


def _load_dino(args: argparse.Namespace, device: Any) -> Tuple[Any, Any]:
    source = args.model_path or args.model_name
    if not source:
        raise PcaVisualizeError(
            '--backbone dino needs --model-path (a local directory) or '
            '--model-name with --allow-remote')
    if args.model_path and not Path(args.model_path).is_dir():
        raise PcaVisualizeError(
            f'The DINO model directory does not exist: {args.model_path}')
    if not args.model_path and not args.allow_remote:
        raise PcaVisualizeError(
            f'--model-name {args.model_name!r} would be fetched from the '
            'network; pass --model-path for a local model or --allow-remote '
            'to opt in')
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as error:
        raise PcaVisualizeError(
            'transformers is required for --backbone dino; install it or use '
            'a C-RADIO backbone') from error
    local_only = not args.allow_remote
    try:
        processor = AutoImageProcessor.from_pretrained(
            source, local_files_only=local_only)
        model = AutoModel.from_pretrained(source, local_files_only=local_only)
    except Exception as error:  # noqa: BLE001 - one clear failure
        raise PcaVisualizeError(
            f'Unable to load the DINO model {source!r} '
            f'(local_files_only={local_only}). Pass a complete local '
            'directory with --model-path, or --allow-remote to fetch '
            f'{source!r} by name. Original error: {error!r}') from error
    return model.eval().to(device), processor


def dino_geometry(config: Any, height: int, width: int) -> Dict[str, Any]:
    """Patch size, prefix-token count and token grid for a DINO ViT.

    The patch size and the number of register tokens come from the model
    config rather than being assumed, and the requested size must be a patch
    multiple so the token count is exact.
    """
    patch = int(getattr(config, 'patch_size', 0) or 0)
    if patch < 1:
        raise PcaVisualizeError(
            'The DINO config exposes no usable patch_size; cannot derive the '
            'token grid')
    if int(height) % patch or int(width) % patch:
        raise PcaVisualizeError(
            f'The DINO patch size {patch} requires image sizes that are '
            f'multiples of {patch}; got {int(height)}x{int(width)}')
    registers = int(getattr(config, 'num_register_tokens', 0) or 0)
    grid = (int(height) // patch, int(width) // patch)
    return {
        'patch_size': patch,
        'num_register_tokens': registers,
        'prefix_tokens': 1 + registers,
        'grid': grid,
        'expected_tokens': grid[0] * grid[1],
    }


def dino_pixel_values(processor: Any, image: Any, height: int,
                      width: int) -> Any:
    """Processor-normalized ``[1, 3, H, W]`` at the requested size.

    The resize is done here (bicubic, no crop) so a processor that would
    resize or centre-crop on its own is explicitly told not to.
    """
    import torch
    from PIL import Image
    resized = image.convert('RGB').resize((int(width), int(height)),
                                          Image.BICUBIC)
    try:
        batch = processor(
            images=resized,
            return_tensors='pt',
            do_resize=False,
            do_center_crop=False)
    except TypeError:  # a processor without those switches
        batch = processor(images=resized, return_tensors='pt')
    values = batch['pixel_values'] if isinstance(batch, dict) else getattr(
        batch, 'pixel_values', None)
    if values is None:
        raise PcaVisualizeError(
            'The DINO image processor returned no pixel_values')
    if tuple(values.shape[-2:]) != (int(height), int(width)):
        raise PcaVisualizeError(
            'The DINO image processor changed the requested size: asked for '
            f'{int(height)}x{int(width)}, got {tuple(values.shape[-2:])}. Run '
            'the processor with do_resize=False and do_center_crop=False, or '
            'pick the size the processor expects')
    return values


def dino_tokens(hidden: Any, geometry: Dict[str, Any]) -> Any:
    """Strip the CLS and register prefix tokens and check the token count."""
    import torch
    if not torch.is_tensor(hidden) or hidden.ndim != 3:
        raise PcaVisualizeError(
            f'DINO hidden states must be [B, 1+registers+N, D]; got '
            f'{getattr(hidden, "shape", None)}')
    prefix = int(geometry['prefix_tokens'])
    if hidden.shape[1] < prefix:
        raise PcaVisualizeError(
            f'DINO hidden states have {hidden.shape[1]} tokens, fewer than '
            f'the {prefix} prefix tokens (CLS plus registers)')
    tokens = hidden[0, prefix:, :]
    expected = int(geometry['expected_tokens'])
    if int(tokens.shape[0]) != expected:
        raise PcaVisualizeError(
            f'DINO produced {int(tokens.shape[0])} spatial tokens for a '
            f'{geometry["grid"][0]}x{geometry["grid"][1]} grid; expected '
            f'{expected}. Check the patch size and the requested image size')
    return tokens.detach().float().cpu().numpy()


def encode_dino(model: Any, processor: Any, image: Any, *, height: int,
                width: int, device: Any,
                dtype: str) -> Tuple[Any, Tuple[int, int], Dict[str, Any]]:
    """Extract DINOv2/v3 spatial tokens and their grid."""
    import torch
    geometry = dino_geometry(model.config, height, width)
    values = dino_pixel_values(processor, image, height, width).to(device)
    target = resolve_dtype(dtype)
    with torch.no_grad():
        if target is torch.float32:
            output = model(pixel_values=values)
        else:
            autocast_dtype = target if values.is_cuda else torch.bfloat16
            with torch.autocast(
                    device_type='cuda' if values.is_cuda else 'cpu',
                    dtype=autocast_dtype):
                output = model(pixel_values=values)
    hidden = getattr(output, 'last_hidden_state', None)
    if hidden is None:
        raise PcaVisualizeError(
            'The DINO model returned no last_hidden_state; this tool supports '
            'plain DINOv2/DINOv3 ViT models, not fused towers')
    tokens = dino_tokens(hidden, geometry)
    grid = (int(geometry['grid'][0]), int(geometry['grid'][1]))
    diagnostics = {
        'dtype': dtype,
        'patch_size': geometry['patch_size'],
        'num_register_tokens': geometry['num_register_tokens'],
        'prefix_tokens': geometry['prefix_tokens'],
        'tokens': int(tokens.shape[0]),
        'feature_dim': int(tokens.shape[1]),
        'grid': [grid[0], grid[1]],
        'finite': bool(torch.isfinite(hidden).all()),
    }
    return tokens, grid, diagnostics


# -- orchestration ---------------------------------------------------------


def resolve_device(name: str) -> Any:
    """``cuda``/``cpu``/``cuda:1`` as a torch device, checked for presence."""
    import torch
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise PcaVisualizeError(
            'CUDA was requested but is not available; pass --device cpu')
    return device


def default_output(args: argparse.Namespace, size: Tuple[int, int]) -> str:
    """A stable default artifact path."""
    stem = f'{args.backbone}_{size[0]}x{size[1]}'
    if args.backbone == 'dino' and (args.model_name or args.model_path):
        source = Path(
            args.model_path).name if args.model_path else (args.model_name)
        stem = f'dino_{source}'
    return str(Path(DEFAULT_OUTPUT_ROOT) / stem)


def extract(args: argparse.Namespace, size: Tuple[int, int],
            device: Any) -> Dict[str, Any]:
    """Run the selected backend and return tokens, grid and diagnostics."""
    from PIL import Image
    from tools import c_radio_export
    height, width = size
    if args.backbone == 'cradio-trt':
        if args.dtype != 'fp32':
            raise PcaVisualizeError(
                '--dtype does not apply to cradio-trt: the engine\'s tactic '
                'precision was fixed when it was built (see the engine '
                'metadata). Use --backbone cradio for a requested precision.')
    else:
        check_dtype_for_device(
            args.dtype, device,
            'C-RADIO' if args.backbone == 'cradio' else 'DINO')
    if args.backbone == 'cradio':
        with c_radio_export.upstream_attention() as attention:
            backbone = _load_cradio_pytorch(args, device)
            grid, patch = check_geometry(backbone, height, width, 'C-RADIO')
            wrapper, preprocess = load_wrapper_image(args.image, height, width)
            tokens, grid, diagnostics = encode_cradio(
                backbone, wrapper, device=device, dtype=args.dtype)
        diagnostics['attention'] = attention
        diagnostics['patch_size'] = patch
        model = args.model_path
    elif args.backbone == 'cradio-trt':
        backbone = _load_cradio_trt(args, device)
        try:
            metadata = backbone.metadata
            if metadata.variant != args.version:
                raise PcaVisualizeError(
                    f'The engine was built for {metadata.variant!r} but '
                    f'--model-name/--version selects {args.version!r}; pass '
                    'the engine that matches the requested variant')
            grid, patch = check_geometry(backbone, height, width,
                                         'C-RADIO TensorRT')
            wrapper, preprocess = load_wrapper_image(args.image, height, width)
            tokens, grid, diagnostics = encode_cradio(
                backbone, wrapper, device=device, dtype=args.dtype)
            diagnostics['patch_size'] = patch
            # The compiled tactic precision is fixed at build time and the
            # --dtype flag is rejected for this backend, so the precision
            # report comes from the engine metadata rather than from the
            # eager autocast path.
            diagnostics['precision'] = {
                'source': 'engine metadata',
                'engine_precision': metadata.precision,
                'input_dtype': metadata.input_dtype,
                'output_dtype': metadata.output_dtype,
                'autocast': None,
            }
            diagnostics['tensorrt_version'] = metadata.tensorrt_version
        finally:
            backbone.close()
        model = args.engine_path
    else:
        model, processor = _load_dino(args, device)
        with Image.open(args.image) as handle:
            image = handle.convert('RGB')
        tokens, grid, diagnostics = encode_dino(
            model,
            processor,
            image,
            height=height,
            width=width,
            device=device,
            dtype=args.dtype)
        wrapper, preprocess = load_wrapper_image(args.image, height, width)
        model = args.model_path or args.model_name
    return {
        'tokens': tokens,
        'grid': grid,
        'input_rgb': unit_rgb(wrapper),
        'preprocess': preprocess,
        'diagnostics': diagnostics,
        'model': str(model),
    }


def save_outputs(args: argparse.Namespace, result: Dict[str, Any],
                 pca: Dict[str, Any], report: Dict[str, Any]) -> List[str]:
    """Write the figure, the tensor bundle and the JSON diagnostics.

    The artifact manifest is settled before anything is written, so the saved
    JSON, the bundle's diagnostics and the returned/stdout report all list the
    same files. Array fields are saved as CPU tensors so the bundle can be
    reloaded with ``weights_only=True``.
    """
    import torch
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle_path = Path(f'{output}.pt')
    json_path = Path(f'{output}.json')
    figure_path = (None if args.no_figure else str(
        plot_outputs(args, result, pca, output)))
    report['artifacts'] = [str(bundle_path), figure_path, str(json_path)]
    report['artifacts'] = [path for path in report['artifacts'] if path]
    bundle = {
        'format': 'pca-feature-bundle',
        'format_version': 2,
        'tokens': torch.as_tensor(result['tokens']),
        'grid': torch.as_tensor(result['grid'], dtype=torch.int64),
        'input_rgb': torch.as_tensor(result['input_rgb']),
        'pca_image': torch.as_tensor(pca['image']),
        'scores': torch.as_tensor(pca['scores']),
        'mean': torch.as_tensor(pca['mean']),
        'components': torch.as_tensor(pca['components']),
        'variance': torch.as_tensor(pca['variance']),
        'variance_ratio': torch.as_tensor(pca['variance_ratio']),
        'scale_min': torch.as_tensor(pca['scale_min']),
        'scale_max': torch.as_tensor(pca['scale_max']),
        'diagnostics': report,
    }
    torch.save(bundle, bundle_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report['artifacts']


def plot_outputs(args: argparse.Namespace, result: Dict[str, Any],
                 pca: Dict[str, Any], output: Path) -> Path:
    """Side-by-side resized input and PCA raster.

    The non-interactive Agg backend is forced only when no window is wanted;
    ``--show`` keeps whatever interactive backend the environment configured.
    """
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    height, width = result['input_rgb'].shape[:2]
    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(result['input_rgb'])
    axes[0].set_title(f'Input {width}x{height} '
                      f'({Path(args.image).name})')
    axes[0].axis('off')
    axes[1].imshow(pca['image'])
    axes[1].set_title(f'{args.backbone} PCA-3 '
                      f'{result["grid"][0]}x{result["grid"][1]} tokens')
    axes[1].axis('off')
    figure.tight_layout()
    path = Path(f'{output}.png')
    figure.savefig(path, dpi=150, bbox_inches='tight')
    if args.show:
        try:
            plt.show()
        except Exception as error:  # noqa: BLE001 - headless display
            raise PcaVisualizeError(
                '--show needs an interactive matplotlib backend and a '
                f'display ({error!r}); omit --show to write the figure only'
            ) from error
    plt.close(figure)
    return path


def load_pca_reference(path: Any) -> Optional[Dict[str, Any]]:
    """Read a saved bundle's PCA basis for a shared projection.

    Bundles are tensor-only and are loaded with ``weights_only=True``, so a
    pickle payload supplied through ``--pca-reference`` never executes. An
    older pickle bundle is rejected with a request to regenerate it; there is
    no unsafe fallback.
    """
    if not path:
        return None
    import numpy as np
    import torch
    source = Path(path).expanduser()
    if not source.is_file():
        raise PcaVisualizeError(f'The PCA reference bundle does not exist: '
                                f'{source}')
    try:
        bundle = torch.load(source, map_location='cpu', weights_only=True)
    except Exception as error:  # noqa: BLE001 - one clear failure
        raise PcaVisualizeError(
            f'{source} is not a tensor-only PCA bundle ({error!r}). Bundles '
            'written by an older version contained pickled arrays and are not '
            'loaded; regenerate the reference with this tool.') from error
    if not isinstance(bundle, dict):
        raise PcaVisualizeError(f'{source} is not a PCA bundle (expected a '
                                f'mapping, got {type(bundle).__name__})')
    required = ('mean', 'components', 'scale_min', 'scale_max')
    missing = [key for key in required if key not in bundle]
    if missing:
        raise PcaVisualizeError(
            f'The PCA reference bundle {source} is missing {missing}; it must '
            'be written by this tool')
    reference = {
        key: _reference_array(bundle[key], key, source)
        for key in required
    }
    for key in ('variance', 'variance_ratio'):
        if key in bundle:
            reference[key] = _reference_array(bundle[key], key, source)
    if 'tokens' in bundle:
        reference['tokens'] = _reference_array(bundle['tokens'], 'tokens',
                                               source)
    _validate_reference(reference, source)
    reference['path'] = str(source)
    return reference


def _reference_array(value: Any, key: str, source: Path) -> Any:
    """One bundle field as a float64 NumPy array."""
    import numpy as np
    import torch
    if not torch.is_tensor(value):
        raise PcaVisualizeError(
            f'{source} field {key!r} is a {type(value).__name__}, not a '
            'tensor; regenerate the reference with this tool')
    return np.asarray(value.detach().to(torch.float64).cpu().numpy())


def _validate_reference(reference: Dict[str, Any], source: Path) -> None:
    """Check the reference basis, scaling and optional tokens."""
    import numpy as np
    mean = reference['mean']
    basis = reference['components']
    low = reference['scale_min']
    high = reference['scale_max']
    if mean.ndim != 1:
        raise PcaVisualizeError(f'{source} mean must be [D]; got {mean.shape}')
    if basis.ndim != 2 or basis.shape[0] != 3:
        raise PcaVisualizeError(
            f'{source} must hold exactly 3 components of shape [3, D]; got '
            f'{basis.shape}')
    if basis.shape[1] != mean.shape[0]:
        raise PcaVisualizeError(
            f'{source} components are {basis.shape[1]}-wide but the mean is '
            f'{mean.shape[0]}-wide; regenerate the reference')
    if low.shape != (3, ) or high.shape != (3, ):
        raise PcaVisualizeError(
            f'{source} scaling must be three values per component; got '
            f'{low.shape} and {high.shape}')
    for name, array in (('mean', mean), ('components', basis),
                        ('scale_min', low), ('scale_max', high)):
        if not np.isfinite(array).all():
            raise PcaVisualizeError(
                f'{source} field {name!r} contains non-finite values')
    if (high < low).any():
        raise PcaVisualizeError(
            f'{source} has scale_max below scale_min; regenerate it')
    tokens = reference.get('tokens')
    if tokens is not None:
        if tokens.ndim != 2 or tokens.shape[1] != basis.shape[1]:
            raise PcaVisualizeError(
                f'{source} reference tokens must be [N, {basis.shape[1]}]; '
                f'got {tokens.shape}')
        if not np.isfinite(tokens).all():
            raise PcaVisualizeError(
                f'{source} reference tokens contain non-finite values')


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Extract, reduce and save one visualization."""
    import numpy as np
    size = parse_image_size(args.image_size)
    if not Path(args.image).is_file():
        raise PcaVisualizeError(f'Image not found: {args.image}')
    device = resolve_device(args.device)
    if args.backbone == 'dino':
        if args.version is not None:
            raise PcaVisualizeError(
                '--version selects a C-RADIO variant; omit it for --backbone '
                'dino and pass the model with --model-path or --model-name')
    else:
        args.version = resolve_radio_variant(args.model_name, args.version)
    reference = load_pca_reference(args.pca_reference)
    result = extract(args, size, device)
    pca = fit_spatial_pca(
        result['tokens'], reference=reference, grid=result['grid'])
    report: Dict[str, Any] = {
        'backbone':
        args.backbone,
        'model':
        result['model'],
        'image':
        result['preprocess'],
        'requested_size_hw': [size[0], size[1]],
        'patch_size':
        int(result['diagnostics']['patch_size']),
        'effective_size_hw': [
            result['grid'][0] * int(result['diagnostics']['patch_size']),
            result['grid'][1] * int(result['diagnostics']['patch_size'])
        ],
        'device':
        str(device),
        'grid': [int(result['grid'][0]),
                 int(result['grid'][1])],
        'tokens':
        int(result['tokens'].shape[0]),
        'feature_dim':
        int(result['tokens'].shape[1]),
        'finite':
        bool(np.isfinite(result['tokens']).all()),
        'pca_components':
        int(pca['components_used']),
        'pca_refit':
        bool(pca['refit']),
        'pca_variance_ratio':
        [float(value) for value in pca['variance_ratio']],
        'pca_variance': [float(value) for value in pca['variance']],
        'pca_reference':
        reference['path'] if reference else None,
        'pca_reference_used':
        bool(reference is not None),
        'backend_diagnostics':
        result['diagnostics'],
    }
    if reference is not None and 'tokens' in reference:
        from tools import c_radio_export
        import torch
        reference_tokens = np.asarray(reference['tokens'], dtype=np.float64)
        if reference_tokens.shape != result['tokens'].shape:
            report['comparison_skipped'] = (
                f'the reference holds {list(reference_tokens.shape)} values '
                f'and this run produced {list(result["tokens"].shape)}; the '
                'PCA basis and scaling are still reused, but elementwise '
                'features are not comparable across different token counts')
        else:
            metrics = c_radio_export.feature_metrics(
                torch.from_numpy(
                    np.asarray(result['tokens'], dtype=np.float64)),
                torch.from_numpy(reference_tokens),
                atol=1e-2,
                rtol=1e-2)
            report['comparison'] = metrics
    args.output = args.output or default_output(args, size)
    report['output'] = args.output
    report['artifacts'] = save_outputs(args, result, pca, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    """The CLI, built without importing torch or any model code."""
    parser = argparse.ArgumentParser(
        prog='pca_feature_visualize',
        description='Visualize spatial features with 3-component PCA.',
        epilog='Backbones: ' + ', '.join(BACKBONE_CHOICES) + '.')
    parser.add_argument('--image', required=True, help='Input image file.')
    parser.add_argument(
        '--backbone',
        default='cradio',
        help='cradio, cradio-trt or dino (aliases accepted).')
    parser.add_argument(
        '--model-name',
        default=None,
        help='C-RADIO variant (canonical or official name, for example '
        'c-radio_v4-h or nvidia/C-RADIOv4-SO400M), or a Hugging Face model '
        'id for DINO, which needs --allow-remote to fetch.')
    parser.add_argument(
        '--model-path',
        default=None,
        help='Local model directory (C-RADIO export or DINO).')
    parser.add_argument(
        '--engine-path',
        default=None,
        help='Serialized TensorRT engine (cradio-trt).')
    parser.add_argument(
        '--metadata-path',
        default=None,
        help='Engine metadata sidecar (default <engine>.json).')
    parser.add_argument(
        '--version',
        default=None,
        help='C-RADIO variant; an alias for --model-name that must agree with '
        'it when both are given. Omit both to keep the '
        f'{DEFAULT_VERSION} default.')
    parser.add_argument(
        '--image-size',
        nargs='+',
        default=None,
        metavar='N',
        help='Square size or explicit height width '
        f'(default {DEFAULT_IMAGE_SIZE}).')
    parser.add_argument(
        '--device', default='auto', help='auto, cpu, cuda or cuda:N.')
    parser.add_argument(
        '--dtype',
        default='fp32',
        choices=['fp32', 'fp16', 'bf16'],
        help='Compute precision; lower precision uses '
        'autocast over FP32 weights.')
    parser.add_argument(
        '--output', default=None, help='Artifact stem for .pt/.json/.png.')
    parser.add_argument(
        '--pca-reference',
        default=None,
        help='Bundle to reuse for the PCA basis, sign and '
        'scaling (and to compare features against).')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Also open an interactive window (default is '
        'headless).')
    parser.add_argument(
        '--no-figure', action='store_true', help='Skip the matplotlib figure.')
    parser.add_argument(
        '--allow-remote',
        action='store_true',
        help='Allow fetching a model by name. Off by default: '
        'this tool never sets proxies or endpoints.')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments, run one visualization, and report failures."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.backbone = normalize_backbone(args.backbone)
    except PcaVisualizeError as error:
        parser.error(str(error))
    try:
        report = run(args)
    except PcaVisualizeError as error:
        print(f'error: {error}', file=sys.stderr)
        return 1
    except Exception as error:  # noqa: BLE001 - one clear failure
        print(f'error: {type(error).__name__}: {error}', file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
