# Copyright 2026 Limx Dynamics
"""Validate the copied official FLUX.2 adapter on real small official blocks.

Skips when the local ImageWAM ``flux2`` source checkout is unavailable. No
pretrained weights are loaded; the official ``Flux2`` module is built directly
with reduced dimensions.
"""

import sys
from pathlib import Path

import pytest
import torch

from fluxvla.models.heads.patchwam_head import _validate_denoiser_dims
from fluxvla.models.third_party_models.patchwam.flux import OfficialFluxDenoiser
from fluxvla.models.third_party_models.patchwam.geometry import (
    joint_visibility, sequence_coordinates)

FLUX_SOURCE = Path(
    '/home/chuong/workspace/wam/ImageWAM/third_party/flux2/src')
pytestmark = pytest.mark.skipif(
    not (FLUX_SOURCE / 'flux2' / 'model.py').is_file(),
    reason='local official flux2 source is unavailable')


def _small_transformer():
    if str(FLUX_SOURCE) not in sys.path:
        sys.path.insert(0, str(FLUX_SOURCE))
    import flux2.model as model_api
    params = model_api.Klein4BParams(
        in_channels=128,
        context_in_dim=48,
        hidden_size=64,
        num_heads=4,
        depth=1,
        depth_single_blocks=1,
        axes_dim=[4, 4, 4, 4],
        mlp_ratio=2.0,
        use_guidance_embed=False,
    )
    return model_api.Flux2(params)


def test_official_flux_denoiser_runs_real_blocks_and_backward():
    denoiser = OfficialFluxDenoiser(_small_transformer())
    batch, reference_tokens, future_tokens, context = 2, 3, 2, 4
    reference = torch.randn(batch, reference_tokens, 128)
    noisy = torch.randn(batch, future_tokens, 128)
    sigma = torch.rand(batch)
    text = torch.randn(batch, context, 48)
    reference_ids = sequence_coordinates(batch, reference_tokens, group=10)
    noisy_ids = sequence_coordinates(batch, future_tokens, group=0)
    context_ids = sequence_coordinates(batch, context, group=0)
    visibility = joint_visibility(context, reference_tokens, future_tokens, 0)
    prediction = denoiser(
        reference, noisy, sigma, text, reference_ids=reference_ids,
        noisy_ids=noisy_ids, context_ids=context_ids, visibility=visibility)
    assert prediction.shape == (batch, future_tokens, 128)
    prediction.square().mean().backward()
    assert denoiser.transformer.img_in.weight.grad.abs().sum() > 0
    assert denoiser.transformer.txt_in.weight.grad.abs().sum() > 0


def test_width_validation_against_real_transformer_projections():
    denoiser = OfficialFluxDenoiser(_small_transformer())
    _validate_denoiser_dims(denoiser, token_dim=128, text_dim=48)
    with pytest.raises(ValueError, match='txt_in'):
        _validate_denoiser_dims(denoiser, token_dim=128, text_dim=64)
    with pytest.raises(ValueError, match='img_in'):
        _validate_denoiser_dims(denoiser, token_dim=64, text_dim=48)
