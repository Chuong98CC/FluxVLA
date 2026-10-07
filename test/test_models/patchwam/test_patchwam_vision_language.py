# Copyright 2026 Limx Dynamics
"""Real tiny Qwen3-VL forward/backward through the copied instruction encoder.

Uses the installed Transformers 5.3.0 Qwen3VL model and the local FluxVLA
Qwen3 tokenizer; no pretrained weights are downloaded.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from fluxvla.models.third_party_models.patchwam.vision_language import (
    LowRankResidualLinear, VisualInstructionEncoder, pool_visual_grid)

TOKENIZER_DIR = Path(
    '/home/chuong/workspace/wam/ImageWAM/checkpoints/flux2/'
    'FLUX.2-klein-base-4B/tokenizer')
pytestmark = pytest.mark.skipif(
    not TOKENIZER_DIR.is_dir(),
    reason='local Qwen3 tokenizer is unavailable')

IMAGE_TOKEN_ID = 151655
VISION_START = 151652
VISION_END = 151653


class _Patcher:
    def __call__(self, images, return_tensors='pt', do_resize=False):
        patches, grids = [], []
        for image in images:
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            channels, height, width = tensor.shape
            grid_h, grid_w = height // 16, width // 16
            chunk = tensor.reshape(channels, grid_h, 16, grid_w,
                                   16).permute(1, 3, 0, 2, 4).reshape(
                                       grid_h * grid_w, channels * 16 * 16)
            patches.append(chunk)
            grids.append([1, grid_h, grid_w])
        return {
            'pixel_values': torch.cat(patches),
            'image_grid_thw': torch.tensor(grids),
        }


class _Processor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        tokenizer.add_special_tokens({'additional_special_tokens': [
            '<|vision_start|>', '<|image_pad|>', '<|vision_end|>',
            '<|im_start|>', '<|im_end|>']})
        self.image_processor = _Patcher()

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        parts = []
        for item in messages[0]['content']:
            if item['type'] == 'image':
                parts.append(f'<|vision_start|><|image_pad|><|vision_end|>')
            elif item['type'] == 'text':
                parts.append(item['text'])
        text = ''.join(parts)
        if add_generation_prompt:
            text += '<|im_start|>assistant\n'
        return text


def _tiny_encoder(**overrides):
    from transformers import (AutoTokenizer, Qwen3VLConfig,
                              Qwen3VLForConditionalGeneration)
    config = Qwen3VLConfig(
        text_config=dict(hidden_size=32, intermediate_size=64,
                         num_hidden_layers=6, num_attention_heads=2,
                         num_key_value_heads=2, vocab_size=151700,
                         max_position_embeddings=1024),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_heads=2,
                           depth=2, patch_size=16, spatial_merge_size=2,
                           out_hidden_size=32, num_position_embeddings=64,
                           temporal_patch_size=1),
        image_token_id=IMAGE_TOKEN_ID, vision_start_token_id=VISION_START,
        vision_end_token_id=VISION_END, vocab_size=151700)
    model = Qwen3VLForConditionalGeneration(config)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR,
                                              local_files_only=True)
    settings = dict(instruction_layers=(1, 2, 3), summary_layers=(2, 3, 4),
                    history_pool_size=4, image_size=32, auxiliary_weight=0.1)
    settings.update(overrides)
    return VisualInstructionEncoder(encoder=model, processor=_Processor(
        tokenizer), **settings)


def test_real_qwen3_vl_forward_backward_and_lora_gradients():
    encoder = _tiny_encoder(trainable_adapters=True)
    encoder.train(True)
    current = torch.rand(1, 3, 32, 32) * 2 - 1
    history = torch.rand(1, 1, 3, 32, 32) * 2 - 1
    result = encoder(['pick the cup'], current=current, history=history,
                     history_valid=torch.ones(1, 1, dtype=torch.bool),
                     subtasks=['move the arm'])
    assert result['text_tokens'].shape[0] == 1
    assert result['text_tokens'].shape[-1] == 96  # 3 x hidden_size 32
    assert result['text_valid'].any()
    assert 'auxiliary_loss' in result
    (result['text_tokens'].sum() + result['auxiliary_loss']).backward()
    adapters = [module for module in encoder.modules()
                if isinstance(module, LowRankResidualLinear)]
    assert adapters
    # ``residual_out`` is zero-initialized, so its gradient is the live signal;
    # ``residual_in`` gradients are finite (zero at initialization).
    assert all(module.residual_out.weight.grad is not None
               and module.residual_out.weight.grad.abs().sum() > 0
               for module in adapters)
    assert all(module.residual_in.weight.grad is not None
               and torch.isfinite(module.residual_in.weight.grad).all()
               for module in adapters)
    assert all(module.base.weight.grad is None for module in adapters)
    assert all(part.grad is None for part in encoder.encoder.model.visual
               .parameters())


def test_real_qwen3_vl_conditioning_is_causal_to_subtask_labels():
    encoder = _tiny_encoder(trainable_adapters=False)
    encoder.train(True)
    current = torch.rand(1, 3, 32, 32) * 2 - 1
    kwargs = dict(current=current,
                  history_valid=torch.zeros(1, 0, dtype=torch.bool))
    first = encoder(['pick the cup'], subtasks=['open the drawer'], **kwargs)
    second = encoder(['pick the cup'], subtasks=['close the door'], **kwargs)
    torch.testing.assert_close(first['text_tokens'], second['text_tokens'],
                               rtol=0, atol=0)


def test_real_qwen3_vl_requires_subtask_labels_when_configured():
    encoder = _tiny_encoder(trainable_adapters=False, require_subtasks=True)
    encoder.train(True)
    with pytest.raises(ValueError, match='subtask'):
        encoder(['pick the cup'], current=torch.rand(1, 3, 32, 32) * 2 - 1)


def test_pool_visual_grid_shapes():
    tokens = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
    pooled = pool_visual_grid(tokens, 4, 4, size=2)
    assert pooled.shape == (4, 8)
