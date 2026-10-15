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

"""Zero-initialized LoRA adapters for the PatchWAM denoising transformer.

This is the one canonical LoRA implementation used by the native
``PatchWAMHead`` stage policy. Adapters attach to the existing attention and
feed-forward linear layers of the double and single DiT blocks: the base
layer keeps its name and its frozen weights, so checkpoints written before
adaptation still load strictly and only ``dit_lora.*`` keys are new.

``attach_dit_lora`` also records the dotted paths of the block containers it
resolved on the denoiser (``dit_lora_blocks``), so a freeze policy can keep
exactly those subtrees frozen without re-deriving the container layout.

The forward hook resolves the adapter registry through the layer it is
called on, so a deep-copied model (an EMA/Self-Flow teacher or an exported
snapshot) applies its own copied adapters instead of the original ones.
"""

import math
from collections.abc import Iterable, Sequence

from torch import Tensor, nn

#: Linear layers adapted inside each DiT block.
DIT_LORA_TARGETS: tuple = (
    'img_attn.qkv', 'img_attn.proj', 'txt_attn.qkv', 'txt_attn.proj',
    'img_mlp.0', 'img_mlp.2', 'txt_mlp.0', 'txt_mlp.2',
    'linear1', 'linear2',
)

#: Block containers searched for adapters (also resolved through a wrapper's
#: ``transformer`` attribute, so registered denoiser wrappers work unchanged).
DIT_LORA_BLOCKS: tuple = ('double_blocks', 'single_blocks')

#: Attribute holding the adapters, and the parameter-name marker.
LORA_ATTRIBUTE = 'dit_lora'

#: Attribute holding the resolved block-container paths, relative to the
#: denoiser (for example ``('transformer.double_blocks', ...)``).
LORA_BLOCKS_ATTRIBUTE = 'dit_lora_blocks'

#: Private per-layer link to the adapter registry.
LORA_ATTRIBUTE_FIELD = '_lora_adapter_registry'

__all__ = [
    'DIT_LORA_BLOCKS',
    'DIT_LORA_TARGETS',
    'LORA_ATTRIBUTE',
    'LORA_ATTRIBUTE_FIELD',
    'LORA_BLOCKS_ATTRIBUTE',
    'LoraResidual',
    'attach_dit_lora',
    'lora_block_prefixes',
    'lora_parameter_names',
    'lora_target_names',
]


def _validate_adapter_settings(rank: int, alpha: float,
                               dropout: float) -> None:
    if type(rank) is not int or rank < 1:
        raise ValueError('LoRA rank must be a positive integer')
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('LoRA alpha must be finite and positive')
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError('LoRA dropout must be finite and in [0,1)')


class LoraResidual(nn.Module):
    """Trainable low-rank residual added to a frozen linear layer."""

    def __init__(self, in_features: int, out_features: int, rank: int,
                 alpha: float, dropout: float, *, dtype=None) -> None:
        super().__init__()
        _validate_adapter_settings(rank, alpha, dropout)
        self.down = nn.Linear(in_features, rank, bias=False, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, dtype=dtype)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        # A zero residual output reproduces the frozen layer exactly at start.
        nn.init.zeros_(self.up.weight)
        self.dropout = nn.Dropout(dropout)
        self.multiplier = alpha / rank

    def forward(self, value: Tensor) -> Tensor:
        return self.up(self.down(self.dropout(value))) * self.multiplier


def _block_containers(denoiser: nn.Module, blocks: Iterable[str]):
    """Resolve each block container as ``(local name, dotted path, module)``.

    The local name keeps the historical adapter key naming, while the dotted
    path (``transformer.double_blocks`` for the registered denoiser wrapper)
    names the frozen subtree from the denoiser itself.
    """
    candidates = [(None, denoiser)]
    wrapped = getattr(denoiser, 'transformer', None)
    if isinstance(wrapped, nn.Module) and wrapped is not denoiser:
        candidates.append(('transformer', wrapped))
    found, seen = [], set()
    for wrapper, candidate in candidates:
        for name in blocks:
            container = getattr(candidate, name, None)
            if container is None or id(container) in seen:
                continue
            if not isinstance(container, nn.ModuleList) or not len(container):
                raise ValueError(
                    f'{name} must be a nonempty ModuleList of DiT blocks')
            path = name if wrapper is None else f'{wrapper}.{name}'
            found.append((name, path, container))
            seen.add(id(container))
    if not found:
        raise ValueError(
            f'No DiT block container found; expected one of {tuple(blocks)}')
    return found


def _target_suffix(name: str, targets: Sequence[str]):
    for target in targets:
        if name == target or name.endswith('.' + target):
            return target
    return None


def attach_dit_lora(denoiser: nn.Module, *, rank: int = 32, alpha: float = 64,
                    dropout: float = 0.0,
                    targets: Sequence[str] = DIT_LORA_TARGETS,
                    blocks: Sequence[str] = DIT_LORA_BLOCKS) -> dict:
    """Attach LoRA adapters to the denoiser's DiT attention and feed-forward.

    Validates the resolved structure before mutating anything: a missing block
    container, an unmatched target name, or an already adapted denoiser fails
    without changing any parameter.
    """
    _validate_adapter_settings(rank, alpha, dropout)
    if not targets or any(not isinstance(name, str) or not name.strip()
                          for name in targets):
        raise ValueError(
            'LoRA targets must be nonempty dotted module suffixes')
    if len(set(targets)) != len(targets):
        raise ValueError('LoRA target names must be unique')
    if getattr(denoiser, LORA_ATTRIBUTE, None) is not None:
        raise ValueError('The denoiser already carries LoRA adapters')
    containers = _block_containers(denoiser, blocks)
    matched, matched_targets = {}, set()
    for block_name, _, container in containers:
        for index, block in enumerate(container):
            for relative, module in block.named_modules():
                if relative and isinstance(module, nn.Linear):
                    target = _target_suffix(relative, targets)
                    if target is not None:
                        matched[f'{block_name}.{index}.{relative}'] = module
                        matched_targets.add(target)
    unmatched = [name for name in targets if name not in matched_targets]
    if unmatched:
        raise ValueError(
            f'LoRA targets do not exist in this transformer: {unmatched}')
    adapters = {
        name: LoraResidual(module.in_features, module.out_features, rank,
                           alpha, dropout, dtype=module.weight.dtype)
        for name, module in matched.items()
    }
    setattr(denoiser, LORA_ATTRIBUTE, nn.ModuleDict({
        name.replace('.', '_'): adapter
        for name, adapter in adapters.items()
    }))
    denoiser.lora_targets = {name.replace('.', '_'): name for name in adapters}
    # Dotted paths of the frozen block subtrees, from the denoiser down.
    setattr(denoiser, LORA_BLOCKS_ATTRIBUTE,
            tuple(path for _, path, _ in containers))
    registry = getattr(denoiser, LORA_ATTRIBUTE)
    for name, module in matched.items():
        key = name.replace('.', '_')

        def residual_hook(owner, inputs, output, key=key):
            # Resolve the adapter through the module the hook is called on so a
            # deep-copied model uses its own copied adapters.
            registry = getattr(owner, LORA_ATTRIBUTE_FIELD)
            return output + registry[key](inputs[0])

        # A plain attribute keeps the registry out of every layer's tree.
        object.__setattr__(module, LORA_ATTRIBUTE_FIELD, registry)
        module.register_forward_hook(residual_hook)
    return {
        'rank': rank,
        'alpha': float(alpha),
        'dropout': float(dropout),
        'targets': sorted(matched_targets),
        'adapters': {name: [module.in_features, module.out_features]
                     for name, module in matched.items()},
    }


def lora_parameter_names(model: nn.Module) -> list:
    """Names of every attached LoRA parameter, in module order."""
    marker = LORA_ATTRIBUTE + '.'
    return [name for name, _ in model.named_parameters() if marker in name]


def lora_target_names(model: nn.Module) -> dict:
    """Map adapter keys back to the dotted transformer module they adapt."""
    for module in model.modules():
        mapping = getattr(module, 'lora_targets', None)
        if mapping:
            return dict(mapping)
    return {}


def lora_block_prefixes(model: nn.Module) -> tuple:
    """Dotted paths of the adapted block containers, relative to the owner.

    Returns an empty tuple when no adapters are attached, so a caller can tell
    "no DiT LoRA" from "adapted blocks that happen to resolve to no path".
    """
    for module in model.modules():
        prefixes = getattr(module, LORA_BLOCKS_ATTRIBUTE, None)
        if prefixes:
            return tuple(prefixes)
    return ()
