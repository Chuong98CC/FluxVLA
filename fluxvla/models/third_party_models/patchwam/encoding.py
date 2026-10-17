"""Observation encoding shared by the native PatchWAM backbone.

This mirrors ``FluxAssetPolicy._encode`` from the vendored ``flux.py`` but takes
the flow-policy dimensions (``token_dim``/``text_dim``) explicitly instead of
reaching into a policy instance. The numerical contract - packed 128-channel
latents, ``group=10+view`` reference raster ids, ``group=view`` future ids,
history pooling, and the ``sampling_future_length`` metadata - is unchanged.
"""

from collections.abc import Mapping
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .geometry import flatten_image_latents, raster_coordinates


def resolve_endpoint_video(batch: Mapping[str, Any], *, target: bool):
    """Return sanitized current/future endpoint rasters in world RGB.

    The returned tensors are ``[B,V,3,H,W]`` and share the exact augmented
    mosaic the VAE frontend consumes. Invalid camera or frame slots are zeroed
    before any encoder sees them.
    """
    video = batch.get('camera_video',
                      batch.get('video', batch.get('images')))
    if video is None:
        raise KeyError(
            'A video or separate camera_video observation is required')
    if video.ndim == 4:
        video = video.unsqueeze(2)
    if video.ndim == 5:
        video = video.unsqueeze(1)
    if (video.ndim != 6 or video.shape[2] != 3
            or video.shape[3] < (2 if target else 1)):
        raise ValueError(
            'Observation video must be [B,V,3,T,H,W], with T>=2 for training')
    batch_size, views = video.shape[:2]
    if not 1 <= views <= 10:
        raise ValueError(
            'Separate camera observations support one to ten views')
    current_video = video[:, :, :, 0]
    future_video = video[:, :, :, -1] if target else None
    camera_valid = batch.get('img_masks')
    frame_valid = batch.get('frame_masks')
    if camera_valid is not None:
        camera_valid = camera_valid.to(
            device=video.device, dtype=torch.bool)
        if camera_valid.shape != (batch_size, views):
            raise ValueError('img_masks must be [B, V]')
        mask = camera_valid[:, :, None, None, None]
        current_video = torch.where(mask, current_video, 0)
        if future_video is not None:
            future_video = torch.where(mask, future_video, 0)
    if frame_valid is not None:
        frame_valid = frame_valid.to(
            device=video.device, dtype=torch.bool)
        if (frame_valid.ndim != 2 or frame_valid.shape[0] != batch_size
                or frame_valid.shape[1] < 2):
            raise ValueError('frame_masks must be [B, T>=2]')
        current_video = torch.where(
            frame_valid[:, 0, None, None, None, None],
            current_video, 0)
        if future_video is not None:
            future_video = torch.where(
                frame_valid[:, -1, None, None, None, None],
                future_video, 0)
    return current_video, future_video, batch_size, views


def raster_layout_metadata(*, views: int, video_size, grid):
    """Serializable view-major raster geometry for teacher descriptors."""
    height, width = int(grid[0]), int(grid[1])
    return {
        'views': int(views),
        'video_size': [int(video_size[0]), int(video_size[1])],
        'grid': [height, width],
        'view_order': 'view_major_row_major',
        'tokens_per_view': height * width,
        'token_count': int(views) * height * width,
    }


def _raster_validity(batch: Mapping[str, Any], batch_size: int, views: int,
                     height: int, width: int, device: torch.device):
    """Per-sample current/future raster validity from native masks.

    ``img_masks`` ``[B, V]`` marks valid cameras (some benchmarks pad views);
    ``frame_masks`` ``[B, T]`` marks valid sampled world frames - index 0 is the
    current frame and the last index the future endpoint. Either may be absent.
    """
    tokens_per_view = height * width
    reference = torch.ones(batch_size, views * tokens_per_view, dtype=torch.bool,
                           device=device)
    img_masks = batch.get('img_masks')
    if img_masks is not None:
        if img_masks.shape != (batch_size, views):
            raise ValueError('img_masks must be [B, V] matching camera views')
        reference = img_masks.to(device=device, dtype=torch.bool).repeat_interleave(
            tokens_per_view, dim=1)
    future = reference.clone()
    frame_masks = batch.get('frame_masks')
    if frame_masks is not None:
        if frame_masks.ndim != 2 or frame_masks.shape[0] != batch_size or frame_masks.shape[1] < 2:
            raise ValueError('frame_masks must be [B, T>=2] with the future endpoint last')
        frame_masks = frame_masks.to(device=device, dtype=torch.bool)
        reference = reference & frame_masks[:, 0:1]
        future = future & frame_masks[:, -1:]
    return reference, future


def _encoder_parameter(autoencoder, visual_encoder):
    """A parameter that defines the encoded token device and dtype."""
    for module in (visual_encoder, autoencoder):
        if module is None:
            continue
        parameter = next(module.parameters(), None)
        if parameter is not None:
            return parameter
    return None


def _declared_feature_dim(visual_encoder, default: int) -> int:
    width = getattr(visual_encoder, 'feature_dim', None)
    return default if width is None else int(width)


def _frame_encoder(autoencoder, visual_encoder):
    """Return ``frames [N,3,H,W] -> [N,C,h,w]`` for the configured frontend.

    The shared vision wrapper returns ``[B, N, D]`` spatial tokens; they are
    transposed back to the ``[N, D, h, w]`` packing the rest of this module
    (raster ids, flattened reference/future tokens, pooled history) already
    assumes, so both frontends share one code path downstream.
    """
    if visual_encoder is None:
        if autoencoder is None:
            raise ValueError('An autoencoder or a visual backbone is required')
        parameter = next(autoencoder.parameters())

        def encode(frames):
            return autoencoder.encode(frames.to(parameter))

        return encode
    def encode(frames):
        output = visual_encoder.encode_features(frames)
        tokens = getattr(output, 'tokens', None)
        grid = getattr(output, 'grid', None)
        if tokens is None or grid is None:
            raise ValueError(
                'The visual backbone must return tokens and a feature grid')
        height, width = int(grid[0]), int(grid[1])
        if tokens.ndim != 3 or tokens.shape[1] != height * width:
            raise ValueError(
                'Visual backbone tokens must be [N, h*w, D] for the returned '
                'grid')
        return tokens.transpose(1, 2).reshape(tokens.shape[0],
                                              tokens.shape[-1], height, width)

    return encode


def encode_observations(
    batch: Mapping[str, Any],
    *,
    autoencoder: Optional[torch.nn.Module],
    text_encoder: torch.nn.Module,
    token_dim: int,
    text_dim: int,
    history_pool_size: int,
    target: bool,
    radio_encoder: Optional[Any] = None,
    visual_encoder: Optional[Any] = None,
    visual_dim: Optional[int] = None,
) -> dict[str, Any]:
    """Build the encoded-batch contract consumed by ``PatchFlowPolicy``.

    ``radio_encoder`` optionally *adds* detached ``radio_future_tokens`` beside
    the unchanged frozen future-VAE latents. It is called as
    ``radio_encoder(video_flat, raster)`` with ``video_flat`` ``[B*V,3,H,W]`` in
    the native [-1,1] range and the packed reference raster ``(height, width)``
    the teacher features must align to, and returns ``[B*V, height*width, D]``.

    ``visual_encoder`` replaces the frozen autoencoder as the observation
    frontend: the shared wrapper converts the ``[-1, 1]`` endpoint frames and
    returns ``[B, N, D]`` spatial tokens with their raster, which this function
    reshapes to the same ``[N, D, h, w]`` packing the VAE path produces. The
    RADIO auxiliary target stays tied to the VAE path and is rejected here when
    both are configured.
    """
    if visual_encoder is not None and radio_encoder is not None:
        raise ValueError(
            'The RADIO auxiliary target requires the VAE latent frontend; a '
            'visual backbone recipe must not also configure radio_encoder')
    encode_frames = _frame_encoder(autoencoder, visual_encoder)
    cache_formats = batch.get('text_cache_format')
    if cache_formats is not None:
        cache_formats = ([cache_formats] if isinstance(cache_formats, str) else
                         cache_formats)
        if getattr(text_encoder, 'requires_images', False):
            raise ValueError(
                'VL conditioning requires current/past images rather than a '
                'text-only cache')
        if any(value != 'qwen3_flux2' for value in cache_formats) or 'text_tokens' not in batch:
            raise ValueError('This policy requires a native Qwen3 FLUX.2 text cache')

    result = dict(batch)
    encoder_parameter = _encoder_parameter(autoencoder, visual_encoder)
    future_valid = None
    if 'reference_tokens' in batch:
        current = batch['reference_tokens']
        if current.ndim != 3:
            raise ValueError('reference_tokens must be [B,R,token_dim]')
        if encoder_parameter is not None:
            current = current.to(encoder_parameter)
        result['reference_tokens'] = current
        batch_size = current.shape[0]
        channels = current.shape[-1]
        if target and 'future_tokens' not in batch:
            raise ValueError('Cached training observations require future_tokens')
        if 'future_tokens' in batch:
            result['future_tokens'] = batch['future_tokens'].to(current)
        if 'radio_future_tokens' in batch:
            # Cached RADIO targets follow the same device/dtype convention as
            # the cached reference/future latents.
            result['radio_future_tokens'] = batch['radio_future_tokens'].to(
                current)
        for name in ('reference_ids', 'future_ids'):
            if name in batch:
                result[name] = batch[name].to(device=current.device)
        if 'reference_valid' in batch:
            result['reference_valid'] = batch['reference_valid'].to(
                device=current.device, dtype=torch.bool)
        if 'future_valid' in batch:
            future_valid = batch['future_valid'].to(device=current.device,
                                                    dtype=torch.bool)
    else:
        current_video, future_video, batch_size, views = (
            resolve_endpoint_video(batch, target=target))
        with torch.no_grad():
            current = encode_frames(current_video.flatten(0, 1))
        _, channels, height, width = current.shape
        if visual_encoder is not None:
            expected = int(_declared_feature_dim(visual_encoder, token_dim)
                           if visual_dim is None else visual_dim)
            if channels != expected:
                raise ValueError(
                    f'The visual backbone produced {channels}-wide features but '
                    f'{expected} were configured')
        elif channels != token_dim:
            raise ValueError('Official AE must produce already packed 128-channel latents')
        result['reference_tokens'] = flatten_image_latents(current).reshape(
            batch_size, views * height * width, channels)
        result['reference_ids'] = torch.cat([
            raster_coordinates(batch_size, height, width, group=10 + view,
                               device=current.device) for view in range(views)
        ], 1)
        result['future_ids'] = torch.cat([
            raster_coordinates(batch_size, height, width, group=view,
                               device=current.device) for view in range(views)
        ], 1)
        layout = raster_layout_metadata(
            views=views, video_size=current_video.shape[-2:],
            grid=(height, width))
        result['current_raster_length'] = views * height * width
        result['current_raster_layout'] = layout
        if target:
            result['future_raster_layout'] = dict(layout)
        reference_valid, future_valid = _raster_validity(
            batch, batch_size, views, height, width, current.device)
        result['reference_valid'] = reference_valid
        if target:
            with torch.no_grad():
                future = encode_frames(future_video.flatten(0, 1))
            if future.shape != current.shape:
                raise ValueError('Current and future raster shapes differ')
            result['future_tokens'] = flatten_image_latents(future).reshape(
                batch_size, views * height * width, channels)
            result['future_valid'] = future_valid
            if radio_encoder is not None:
                radio_tokens = radio_encoder(future_video.flatten(0, 1),
                                             (height, width))
                if (not isinstance(radio_tokens, torch.Tensor)
                        or radio_tokens.ndim != 3
                        or radio_tokens.shape[0] != batch_size * views
                        or radio_tokens.shape[1] != height * width):
                    raise ValueError(
                        'radio_encoder must return [B*V,height*width,D] '
                        'features aligned to the reference raster')
                result['radio_future_tokens'] = radio_tokens.detach().reshape(
                    batch_size, views * height * width, radio_tokens.shape[-1])

    history = batch.get('history_video')
    if history is not None:
        if history.ndim != 5 or history.shape[0] != batch_size or history.shape[2] != 3:
            raise ValueError('World-model history must be [B,K,3,H,W]')
        valid = batch.get('history_valid')
        if valid is None or valid.shape != history.shape[:2]:
            raise ValueError('History observations require an explicit [B,K] validity mask')
        if history.shape[1]:
            if 'reference_ids' not in result:
                raise ValueError(
                    'Cached observations need raster reference_ids before adding history')
            valid = valid.to(device=current.device, dtype=torch.bool)
            safe_history = torch.where(
                valid[..., None, None, None].to(history.device), history, 0)
            with torch.no_grad():
                encoded_history = encode_frames(safe_history.flatten(0, 1))
                pooled = F.adaptive_avg_pool2d(
                    encoded_history, (history_pool_size,) * 2)
            history_tokens = flatten_image_latents(pooled).reshape(
                batch_size, history.shape[1] * history_pool_size**2, channels)
            token_valid = valid.repeat_interleave(history_pool_size**2, dim=1)
            history_tokens = history_tokens.masked_fill(~token_valid[..., None], 0)
            history_ids = torch.cat([
                raster_coordinates(batch_size, history_pool_size,
                                   history_pool_size, group=-(slot + 1),
                                   device=current.device)
                for slot in range(history.shape[1])
            ], 1)
            reference_valid = result.get(
                'reference_valid',
                torch.ones(result['reference_tokens'].shape[:2], dtype=torch.bool,
                           device=current.device))
            result['reference_valid'] = torch.cat(
                (reference_valid.to(current.device), token_valid), 1)
            result['reference_tokens'] = torch.cat(
                (result['reference_tokens'], history_tokens.to(result['reference_tokens'])), 1)
            result['reference_ids'] = torch.cat(
                (result['reference_ids'], history_ids), 1)
    # Pooled history is context; predict only the current raster.
    if 'future_ids' in result:
        result['sampling_future_length'] = result['future_ids'].shape[1]
    text = batch.get('text_tokens')
    if text is None:
        prompt = batch.get('prompt')
        if not isinstance(prompt, (list, tuple)) or len(prompt) != batch_size:
            raise ValueError('prompt must contain one instruction per sample')
        if getattr(text_encoder, 'requires_images', False):
            vl_current = batch.get('vl_current')
            if vl_current is None:
                raise ValueError(
                    'VL conditioning requires an unaugmented vl_current observation')
            encoded = text_encoder(
                list(prompt), current=vl_current,
                history=batch.get('vl_history'),
                history_valid=batch.get('vl_history_valid'),
                subtasks=batch.get('subtask') if target else None,
            )
        else:
            encoded = text_encoder(list(prompt))
        if isinstance(encoded, tuple):
            text, result['text_valid'] = encoded
        elif isinstance(encoded, Mapping):
            text = encoded['text_tokens']
            result['text_valid'] = encoded['text_valid']
            if 'auxiliary_loss' in encoded:
                result['language_auxiliary_loss'] = encoded['auxiliary_loss']
        else:
            text = encoded
    elif getattr(text_encoder, 'trainable_adapters', False):
        raise ValueError('Trainable VL adapters cannot be bypassed with cached text features')
    result['text_tokens'] = text.to(current)
    if (result['text_tokens'].ndim != 3 or result['text_tokens'].shape[0] != batch_size
            or result['text_tokens'].shape[-1] != text_dim):
        raise ValueError(f'text_tokens must be [B,L,{text_dim}]')
    if 'text_valid' in result:
        result['text_valid'] = result['text_valid'].to(device=current.device,
                                                       dtype=torch.bool)
    return result
