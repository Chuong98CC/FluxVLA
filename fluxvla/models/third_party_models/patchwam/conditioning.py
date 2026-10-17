"""General denoiser conditioning objects shared by PatchWAM variants."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PreparedDenoiserPrefix:
    """Projected clean prefix reused across denoiser evaluations."""

    reference_hidden: Tensor
    context_hidden: Tensor

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.reference_hidden) or (
                self.reference_hidden.ndim != 3):
            raise ValueError('reference_hidden must be [B,R,D]')
        if not torch.is_tensor(self.context_hidden) or (
                self.context_hidden.ndim != 3):
            raise ValueError('context_hidden must be [B,L,D]')
        if self.reference_hidden.shape[0] != self.context_hidden.shape[0]:
            raise ValueError('Prepared prefix batch sizes differ')
        if self.reference_hidden.shape[-1] != self.context_hidden.shape[-1]:
            raise ValueError('Prepared prefix hidden widths differ')
        if (self.reference_hidden.device != self.context_hidden.device
                or self.reference_hidden.dtype != self.context_hidden.dtype):
            raise ValueError(
                'Prepared prefix tensors must share device and dtype')


__all__ = ['PreparedDenoiserPrefix']
