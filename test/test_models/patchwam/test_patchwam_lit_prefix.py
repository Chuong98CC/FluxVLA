# Copyright 2026 Limx Dynamics
"""Prefix feature API tests on real (tiny-config) official FLUX blocks."""

import sys
from pathlib import Path

import pytest
import torch

from fluxvla.models.third_party_models.patchwam.flux import OfficialFluxDenoiser
from fluxvla.models.third_party_models.patchwam.geometry import (
    joint_visibility, sequence_coordinates)
from fluxvla.models.third_party_models.patchwam.tiny import SmallJointTransformer

FLUX_SOURCE = Path(
    '/home/chuong/workspace/wam/ImageWAM/third_party/flux2/src')


def _small_official_transformer():
    if str(FLUX_SOURCE) not in sys.path:
        sys.path.insert(0, str(FLUX_SOURCE))
    import flux2.model as model_api
    params = model_api.Klein4BParams(
        in_channels=128, context_in_dim=48, hidden_size=64, num_heads=4,
        depth=2, depth_single_blocks=1, axes_dim=[4, 4, 4, 4], mlp_ratio=2.0,
        use_guidance_embed=False)
    return model_api.Flux2(params)


def _prefix_inputs(batch=2, text_len=4, reference_len=3, text_dim=48,
                   token_dim=128):
    generator = torch.Generator().manual_seed(7)
    context = torch.randn(batch, text_len, text_dim, generator=generator)
    reference = torch.randn(batch, reference_len, token_dim,
                            generator=generator)
    context_ids = sequence_coordinates(batch, text_len, group=0)
    reference_ids = sequence_coordinates(batch, reference_len, group=10)
    visibility = joint_visibility(text_len, reference_len, 0, 0)
    sigma = torch.zeros(batch)
    return context, reference, context_ids, reference_ids, visibility, sigma


def _denoiser(kind):
    if kind == 'official':
        return OfficialFluxDenoiser(_small_official_transformer())
    return SmallJointTransformer(token_dim=128, text_dim=32, width=32, heads=4,
                                 depth=3)


def _dims(kind):
    if kind == 'official':
        return dict(text_dim=48, token_dim=128)
    return dict(text_dim=32, token_dim=128)


def _io_modules(kind, denoiser):
    if kind == 'official':
        return denoiser.transformer.img_in, denoiser.transformer.txt_in
    return denoiser.token_input, denoiser.text_input


def _blocks(kind, denoiser):
    if kind == 'official':
        return list(denoiser.transformer.double_blocks) + list(
            denoiser.transformer.single_blocks)
    return list(denoiser.blocks)


@pytest.mark.parametrize('kind', ['tiny', 'official'])
def test_forward_prefix_streams_captures_in_layer_order(kind, monkeypatch):
    if kind == 'official' and not (FLUX_SOURCE / 'flux2' / 'model.py').is_file():
        pytest.skip('local official flux2 source is unavailable')
    denoiser = _denoiser(kind)
    dims = _dims(kind)
    context, reference, context_ids, reference_ids, visibility, sigma = \
        _prefix_inputs(text_dim=dims['text_dim'])
    trace = []
    if kind == 'official':
        counter = {'index': 0}

        def _layer_index():
            counter['index'] += 1
            return counter['index']

        for name in ('_paired_block', '_unified_block'):
            original = getattr(denoiser, name)
            if name == '_paired_block':
                def wrapped(block, text, image, *args, _original=original,
                            **kwargs):
                    trace.append(f'block{_layer_index()}')
                    return _original(block, text, image, *args, **kwargs)
            else:
                def wrapped(states, *args, _original=original, **kwargs):
                    trace.append(f'block{_layer_index()}')
                    return _original(states, *args, **kwargs)
            monkeypatch.setattr(denoiser, name, wrapped)
    else:
        for index, block in enumerate(_blocks(kind, denoiser)):
            original = block.forward

            def wrapped(*args, _index=index, _original=original, **kwargs):
                trace.append(f'block{_index}')
                return _original(*args, **kwargs)

            monkeypatch.setattr(block, 'forward', wrapped)
    captured = []

    def capture(index, text_states, image_states):
        trace.append(f'capture{index}')
        captured.append((index, text_states, image_states))

    total_layers = denoiser.representation_layers
    text_states, image_states = denoiser.forward_prefix(
        context, reference, sigma, context_ids=context_ids,
        reference_ids=reference_ids, visibility=visibility,
        capture_layers=[1, total_layers], capture=capture)
    expected_width = 64 if kind == 'official' else 32
    assert text_states.shape == (2, 4, expected_width)
    assert image_states.shape[0] == 2 and image_states.shape[1] == 3
    assert [entry[0] for entry in captured] == [1, total_layers]
    # Memories stream out per layer: the first capture happens right after one
    # executed layer (not after the whole stack), and the final capture is the
    # last event.
    first_capture = trace.index('capture1')
    assert sum(1 for entry in trace[:first_capture]
               if entry.startswith('block')) == 1
    assert any(entry.startswith('block') for entry in trace[first_capture:])
    assert trace[-1] == 'capture%d' % total_layers
    assert trace.count('capture%d' % total_layers) == 1
    # The returned streams are the final ones, identical to the last capture.
    final_text, final_image = captured[-1][1], captured[-1][2]
    torch.testing.assert_close(text_states, final_text)
    torch.testing.assert_close(image_states, final_image)


@pytest.mark.parametrize('kind', ['tiny', 'official'])
def test_forward_prefix_semantic_stream_carries_final_block_gradients(kind):
    if kind == 'official' and not (FLUX_SOURCE / 'flux2' / 'model.py').is_file():
        pytest.skip('local official flux2 source is unavailable')
    denoiser = _denoiser(kind)
    dims = _dims(kind)
    context, reference, context_ids, reference_ids, visibility, sigma = \
        _prefix_inputs(text_dim=dims['text_dim'])
    total_layers = denoiser.representation_layers
    text_states, image_states = denoiser.forward_prefix(
        context, reference, sigma, context_ids=context_ids,
        reference_ids=reference_ids, visibility=visibility,
        capture_layers=[total_layers], capture=lambda *args: None)
    (text_states.square().mean() + image_states.square().mean()).backward()
    if kind == 'official':
        last_unified = denoiser.transformer.single_blocks[-1]
        last_paired = denoiser.transformer.double_blocks[-1]
        assert last_unified.linear1.weight.grad.abs().sum() > 0
        assert last_paired.img_mlp[0].weight.grad.abs().sum() > 0
    else:
        assert denoiser.blocks[-1].qkv.weight.grad.abs().sum() > 0
        assert denoiser.blocks[0].qkv.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('kind', ['tiny', 'official'])
def test_forward_prefix_gradients_reach_context_and_reference(kind):
    if kind == 'official' and not (FLUX_SOURCE / 'flux2' / 'model.py').is_file():
        pytest.skip('local official flux2 source is unavailable')
    denoiser = _denoiser(kind)
    dims = _dims(kind)
    context, reference, context_ids, reference_ids, visibility, sigma = \
        _prefix_inputs(text_dim=dims['text_dim'])
    context.requires_grad_(True)
    reference.requires_grad_(True)
    seen = {}

    def capture(index, text_states, image_states):
        seen[index] = (text_states, image_states)

    denoiser.forward_prefix(
        context, reference, sigma, context_ids=context_ids,
        reference_ids=reference_ids, visibility=visibility,
        capture_layers=[1], capture=capture)
    loss = sum(value.square().mean() for pair in seen.values() for value in pair)
    loss.backward()
    assert context.grad is not None and context.grad.abs().sum() > 0
    assert reference.grad is not None and reference.grad.abs().sum() > 0
    image_in, text_in = _io_modules(kind, denoiser)
    assert image_in.weight.grad.abs().sum() > 0
    assert text_in.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('kind', ['tiny', 'official'])
def test_forward_prefix_mask_polarity_is_consumed(kind):
    if kind == 'official' and not (FLUX_SOURCE / 'flux2' / 'model.py').is_file():
        pytest.skip('local official flux2 source is unavailable')
    denoiser = _denoiser(kind)
    dims = _dims(kind)
    context, reference, context_ids, reference_ids, _, sigma = _prefix_inputs(
        text_dim=dims['text_dim'])
    all_valid = torch.ones(2, 3, dtype=torch.bool)

    def run(values, valid):
        visibility = joint_visibility(4, 3, 0, 0, reference_valid=valid)
        return denoiser.forward_prefix(
            context, values, sigma, context_ids=context_ids,
            reference_ids=reference_ids, visibility=visibility)

    base = run(reference.clone(), all_valid)
    masked_valid = torch.ones(2, 3, dtype=torch.bool)
    masked_valid[:, 2] = False
    # Callers zero invalid slots (native PatchWAM contract) so a padded camera
    # cannot leak through the residual stream.
    zeroed = reference.clone()
    zeroed[:, 2] = 0.0
    masked = run(zeroed, masked_valid)
    assert any(not torch.allclose(left, right) for left, right in zip(base, masked))
    for value in masked:
        assert torch.isfinite(value).all()


@pytest.mark.parametrize('kind', ['tiny', 'official'])
def test_forward_prefix_rejects_bad_capture_specs(kind):
    if kind == 'official' and not (FLUX_SOURCE / 'flux2' / 'model.py').is_file():
        pytest.skip('local official flux2 source is unavailable')
    denoiser = _denoiser(kind)
    dims = _dims(kind)
    context, reference, context_ids, reference_ids, visibility, sigma = \
        _prefix_inputs(text_dim=dims['text_dim'])
    with pytest.raises(ValueError, match='capture_layers'):
        denoiser.forward_prefix(
            context, reference, sigma, context_ids=context_ids,
            reference_ids=reference_ids, visibility=visibility,
            capture_layers=[denoiser.representation_layers + 1],
            capture=lambda *args: None)
    with pytest.raises(ValueError, match='requires a `capture`'):
        denoiser.forward_prefix(
            context, reference, sigma, context_ids=context_ids,
            reference_ids=reference_ids, visibility=visibility,
            capture_layers=[1])
    with pytest.raises(ValueError, match='strictly increasing'):
        denoiser.forward_prefix(
            context, reference, sigma, context_ids=context_ids,
            reference_ids=reference_ids, visibility=visibility,
            capture_layers=[2, 1], capture=lambda *args: None)
    with pytest.raises(ValueError, match='visibility'):
        denoiser.forward_prefix(
            context, reference, sigma, context_ids=context_ids,
            reference_ids=reference_ids,
            visibility=torch.ones(2, 1, 4, 4, dtype=torch.bool))
