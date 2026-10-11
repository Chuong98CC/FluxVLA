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

"""Swappable frozen visual backbones: geometry, range, and freezing."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fluxvla.engines import VISION_BACKBONES
from fluxvla.models.backbones.visions import (FrozenVisionFeatures,
                                              RadioViTBackbone,
                                              SpatialFeatureOutput,
                                              VAELatentVisionBackbone)
from fluxvla.models.backbones.visions.frozen_vision import to_unit_range


class StubRadio(nn.Module):
    """Minimal C-RADIO stand-in with the official call contract."""

    def __init__(self, feature_dim=8, patch_size=16, resolution_step=16,
                 stride=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.resolution_step = resolution_step
        stride = patch_size if stride is None else stride
        self.stem = nn.Conv2d(3, feature_dim, patch_size, stride)
        self.seen = []

    def get_nearest_supported_resolution(self, height, width):
        step = self.resolution_step
        return (-(-height // step) * step, -(-width // step) * step)

    def forward(self, images, feature_fmt='NLC'):
        self.seen.append((feature_fmt, images.detach().clone()))
        spatial = self.stem(images)
        summary = spatial.mean(dim=(2, 3))[:, None]
        if feature_fmt == 'NCHW':
            return summary, spatial
        return summary, spatial.flatten(2).transpose(1, 2)


class StubAutoencoder(nn.Module):
    """Latent stand-in for a frozen VAE."""

    def __init__(self, channels=4, step=8):
        super().__init__()
        self.step = step
        self.proj = nn.Conv2d(3, channels, step, step)
        self.seen = []

    def encode(self, images):
        self.seen.append(images.detach().clone())
        return self.proj(images)


class DownSamplingRadio(StubRadio):
    """Official-style helper that resolves to a *smaller* supported size."""

    def get_nearest_supported_resolution(self, height, width):
        step = self.resolution_step
        return (max(step, height // step * step),
                max(step, width // step * step))


class NHWCRadio(StubRadio):
    """Export that returns channels-last spatial features without the kwarg."""

    def forward(self, images):
        self.seen.append(('NHWC', images.detach().clone()))
        spatial = self.stem(images)
        summary = spatial.mean(dim=(2, 3))[:, None]
        return summary, spatial.permute(0, 2, 3, 1)


def _backbone(**kwargs):
    torch.manual_seed(0)
    return RadioViTBackbone('c-radio_v4-h',
                            model=StubRadio(),
                            feature_dim=8,
                            patch_size=16,
                            resolution_step=16,
                            default_image_resolution=(3, 32, 32),
                            **kwargs)


def test_radio_backbone_returns_tokens_and_raster_geometry():
    backbone = _backbone().eval()
    images = torch.rand(2, 3, 32, 32) * 2 - 1
    output = backbone.encode_features(images)
    assert isinstance(output, SpatialFeatureOutput)
    assert output.tokens.shape == (2, 4, 8)
    assert output.grid == (2, 2)
    assert output.feature_dim == backbone.feature_dim == 8
    assert backbone.embed_dim == 8 and backbone.num_patches == 4
    assert torch.allclose(backbone(images), output.tokens)
    assert backbone.model.seen[-1][0] == 'NCHW'


def test_radio_backbone_converts_range_once_and_pads_geometry():
    backbone = _backbone().eval()
    images = torch.full((1, 3, 20, 20), -1.0)
    images[:, 0] = 1.0
    output = backbone.encode_features(images)
    seen = backbone.model.seen[-1][1]
    assert seen.shape == (1, 3, 32, 32)
    # Only the unpadded 20x20 corner carries the endpoint frame.
    assert torch.allclose(seen[..., :20, :20][:, 0],
                          torch.ones_like(seen[..., :20, :20][:, 0]))
    assert not seen[..., :20, :20][:, 1].any()
    assert not seen[..., 20:, :].any() and not seen[..., :, 20:].any()
    assert output.grid == (2, 2)
    assert backbone.feature_grid(20, 20) == (2, 2)


def test_radio_backbone_is_frozen_and_never_accumulates_gradients():
    backbone = _backbone()
    assert not backbone.training
    assert not any(p.requires_grad for p in backbone.parameters())
    images = (torch.rand(1, 3, 32, 32) * 2 - 1).requires_grad_()
    output = backbone.encode_features(images)
    assert not output.tokens.requires_grad
    assert output.tokens.grad_fn is None


def test_frozen_backbones_stay_in_evaluation_mode_after_train():
    autoencoder = StubAutoencoder()
    wrappers = (_backbone(),
                VAELatentVisionBackbone('vae-test', autoencoder=autoencoder,
                                        feature_dim=4))
    for backbone in wrappers:
        backbone.train()
        assert not backbone.training
        assert all(not module.training for module in backbone.modules())
        backbone.train(False)
        assert not backbone.training


def test_radio_backbone_resizes_down_to_a_smaller_supported_resolution():
    backbone = RadioViTBackbone('c-radio_v4-h',
                               model=DownSamplingRadio(feature_dim=8,
                                                       patch_size=16),
                               feature_dim=8,
                               patch_size=16,
                               resolution_step=16)
    assert backbone.feature_grid(20, 20) == (1, 1)
    assert backbone.feature_grid(40, 40) == (2, 2)
    output = backbone.encode_features(torch.rand(1, 3, 20, 20) * 2 - 1)
    seen = backbone.model.seen[-1][1]
    # Padding cannot remove pixels, so the unsupported input is resampled down.
    assert seen.shape == (1, 3, 16, 16)
    assert output.grid == (1, 1) and output.tokens.shape == (1, 1, 8)


def test_radio_backbone_accepts_channels_last_spatial_features():
    backbone = RadioViTBackbone('c-radio_v4-h',
                               model=NHWCRadio(feature_dim=8, patch_size=16),
                               feature_dim=8,
                               patch_size=16,
                               resolution_step=16)
    output = backbone.encode_features(torch.rand(1, 3, 32, 32) * 2 - 1)
    assert backbone.model.seen[-1][0] == 'NHWC'
    assert output.grid == (2, 2)
    assert output.tokens.shape == (1, 4, 8)
    # Channels-last tokens must equal the channels-first reading of the raster.
    raster = backbone.model.stem(backbone.model.seen[-1][1])
    torch.testing.assert_close(output.tokens,
                               raster.flatten(2).transpose(1, 2))


def test_unsupported_geometry_that_is_not_patch_aligned_fails():
    class OddRadio(StubRadio):

        def get_nearest_supported_resolution(self, height, width):
            return (height, width)

    backbone = RadioViTBackbone('c-radio_v4-h',
                               model=OddRadio(feature_dim=8, patch_size=16),
                               feature_dim=8,
                               patch_size=16,
                               resolution_step=16)
    with pytest.raises(ValueError, match='patch-size multiple'):
        backbone.encode_features(torch.rand(1, 3, 20, 20) * 2 - 1)


def test_radio_backbone_accepts_a_model_without_the_format_kwarg():

    class ExportStyle(nn.Module):

        def __init__(self):
            super().__init__()
            self.stem = StubRadio()

        def forward(self, images):
            _, spatial = self.stem(images, feature_fmt='NCHW')
            tokens = spatial.flatten(2).transpose(1, 2)
            return SimpleNamespace(spatial_features=tokens)

    backbone = RadioViTBackbone('c-radio_v4-h',
                                model=ExportStyle(),
                                feature_dim=8,
                                patch_size=16,
                                resolution_step=16)
    output = backbone.encode_features(torch.rand(1, 3, 32, 32) * 2 - 1)
    assert output.tokens.shape == (1, 4, 8)


def test_radio_backbone_rejects_bad_inputs_dimensions_and_assets():
    backbone = _backbone()
    with pytest.raises(ValueError, match='three RGB channels'):
        backbone.encode_features(torch.rand(1, 4, 32, 32))
    with pytest.raises(ValueError, match='in .-1,1.'):
        backbone.encode_features(torch.rand(1, 3, 32, 32) * 4)
    with pytest.raises(ValueError, match='finite'):
        backbone.encode_features(torch.full((1, 3, 32, 32), float('nan')))
    with pytest.raises(ValueError, match='feature count'):
        mismatched = RadioViTBackbone('c-radio_v4-h',
                                      model=StubRadio(patch_size=16, stride=8),
                                      feature_dim=8,
                                      patch_size=16,
                                      resolution_step=16)
        mismatched.encode_features(torch.rand(1, 3, 32, 32) * 2 - 1)
    with pytest.raises(ValueError, match='exactly one of model or model_path'):
        RadioViTBackbone('c-radio_v4-h', feature_dim=8)
    with pytest.raises(ValueError, match='exactly one of model or model_path'):
        RadioViTBackbone('c-radio_v4-h',
                         model=StubRadio(),
                         model_path='/tmp/radio',
                         feature_dim=8)
    with pytest.raises(ValueError, match='Unsupported C-RADIO version'):
        RadioViTBackbone('c-radio_v5-h', model=StubRadio(), feature_dim=8)
    with pytest.raises(FileNotFoundError):
        RadioViTBackbone.from_local_export('/nonexistent/radio-export',
                                           version='c-radio_v4-h')


@pytest.mark.parametrize('version,expected', [('c-radio_v4-h', 1280),
                                              ('c-radio_v4-so400m', 1152)])
def test_radio_version_widths_are_declared_and_validated(version, expected):
    from fluxvla.models.backbones.visions.radio_vit import radio_feature_dim
    assert radio_feature_dim(version) == expected
    backbone = RadioViTBackbone(version,
                                model=StubRadio(feature_dim=expected),
                                patch_size=16,
                                resolution_step=16)
    assert backbone.feature_dim == expected
    wrong = RadioViTBackbone(version,
                             model=StubRadio(feature_dim=expected + 1),
                             feature_dim=expected,
                             patch_size=16,
                             resolution_step=16)
    with pytest.raises(ValueError, match='was configured'):
        wrong.encode_features(torch.rand(1, 3, 32, 32) * 2 - 1)


def test_vae_backbone_exposes_latents_as_tokens_and_stays_frozen():
    autoencoder = StubAutoencoder(channels=4, step=8)
    backbone = VAELatentVisionBackbone('vae-test',
                                       autoencoder=autoencoder,
                                       feature_dim=4,
                                       resolution_step=8)
    images = torch.rand(2, 3, 32, 32) * 2 - 1
    output = backbone.encode_features(images)
    assert output.tokens.shape == (2, 16, 4)
    assert output.grid == (4, 4)
    assert output.feature_dim == 4
    assert not any(p.requires_grad for p in backbone.parameters())
    assert torch.allclose(autoencoder.seen[-1], images)
    wrong_width = VAELatentVisionBackbone('vae-test',
                                         autoencoder=autoencoder,
                                         feature_dim=8)
    with pytest.raises(ValueError, match='was configured'):
        wrong_width.encode_features(images)
    with pytest.raises(ValueError, match='frozen autoencoder'):
        VAELatentVisionBackbone('vae-test', feature_dim=4)


def test_frozen_feature_contract_validates_shapes_and_ranges():
    with pytest.raises(ValueError, match='does not match the declared grid'):
        SpatialFeatureOutput(tokens=torch.zeros(1, 3, 4), grid=(2, 2))
    with pytest.raises(ValueError, match='positive'):
        SpatialFeatureOutput(tokens=torch.zeros(1, 0, 4), grid=(0, 1))
    unit = to_unit_range(torch.tensor([[[[-1.0]], [[0.0]], [[0.0]]]]))
    expected = torch.tensor([[[[0.0]], [[0.5]], [[0.5]]]])
    assert torch.allclose(unit, expected)
    assert to_unit_range(torch.ones(1, 3, 1, 1)).all()
    with pytest.raises(ValueError, match='finite'):
        to_unit_range(torch.full((1, 3, 2, 2), float('inf')))


def test_both_backbones_are_registered_and_share_the_contract():
    assert VISION_BACKBONES.get('RadioViTBackbone') is RadioViTBackbone
    assert (VISION_BACKBONES.get('VAELatentVisionBackbone')
            is VAELatentVisionBackbone)
    assert issubclass(RadioViTBackbone, FrozenVisionFeatures)
    assert issubclass(VAELatentVisionBackbone, FrozenVisionFeatures)
