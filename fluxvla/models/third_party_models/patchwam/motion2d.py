"""2D patch-motion geometry, packing and masks for the PatchWAM variant.

A 16-pixel image patch supplies one signed ``(delta_row, delta_column)``
displacement per consecutive future step. One non-overlapping ``8x8`` region of
patches packs its ``[64,2]`` deltas into a single width-128 token with the local
row, local column and coordinate axes fastest in that order. The codec owns the
single ``(H, W)`` normalization and the inverse pixel decode, so public motion
stays in signed pixel units while the joint flow sees normalized tokens.

This mirrors the parameter-free spirit of :mod:`codec`: no learned weights, no
clamping and no sigmoid.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

DEFAULT_PATCH_SIZE = 16
DEFAULT_BLOCK_SIZE = 8
DEFAULT_TOKEN_DIM = 128
COORDINATE_ORDER = 'row_column'
MOTION_TOKEN_GROUP = 20.0


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')
    return int(value)


@dataclass(frozen=True)
class PatchMotion2DCodec:
    """Pack/unpack per-step 2D patch motion in the visual token space.

    Args:
        image_height: Fixed image height in pixels, divisible by ``patch_size``.
        image_width: Fixed image width in pixels, divisible by ``patch_size``.
        patch_size: Patch spacing in pixels (16).
        block_size: Patch count per block edge (8); the token packs
            ``block_size**2`` patch deltas of two coordinates each.
        token_dim: Motion token width; must equal ``block_size**2 * 2``.
        coordinate_order: Only ``row_column`` is supported: the first coordinate
            is the row (height) axis and the second the column (width) axis.

    Public motion tensors use ``[B, T, h, w, 2]`` with ``h = H / patch_size`` and
    ``w = W / patch_size``. Token order is ``timestep``, ``block row``,
    ``block column``, ``local row``, ``local column``, ``coordinate``.
    """

    image_height: int
    image_width: int
    patch_size: int = DEFAULT_PATCH_SIZE
    block_size: int = DEFAULT_BLOCK_SIZE
    token_dim: int = DEFAULT_TOKEN_DIM
    coordinate_order: str = COORDINATE_ORDER

    def __post_init__(self) -> None:
        _positive_int('image_height', self.image_height)
        _positive_int('image_width', self.image_width)
        _positive_int('patch_size', self.patch_size)
        _positive_int('block_size', self.block_size)
        if self.coordinate_order != COORDINATE_ORDER:
            raise ValueError(
                'coordinate_order must be "row_column"; the packed layout and '
                'the displacement convention are row,column')
        if self.image_height % self.patch_size or self.image_width % self.patch_size:
            raise ValueError(
                f'image size {self.image_height}x{self.image_width} must be '
                f'divisible by patch_size={self.patch_size}')
        expected = self.block_size * self.block_size * 2
        if self.token_dim != expected:
            raise ValueError(
                f'token_dim must equal block_size**2 * 2 = {expected} for '
                f'block_size={self.block_size}, got {self.token_dim}')
        rows = math.ceil(self.image_height / self.patch_size)
        columns = math.ceil(self.image_width / self.patch_size)
        object.__setattr__(self, '_patch_rows', rows)
        object.__setattr__(self, '_patch_columns', columns)
        object.__setattr__(self, '_block_rows', math.ceil(rows / self.block_size))
        object.__setattr__(
            self, '_block_columns', math.ceil(columns / self.block_size))

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    @property
    def grid_shape(self) -> tuple[int, int]:
        """Patch grid ``(h, w)``."""
        return self._patch_rows, self._patch_columns

    @property
    def block_grid(self) -> tuple[int, int]:
        """Block grid ``(block_rows, block_columns)``."""
        return self._block_rows, self._block_columns

    @property
    def blocks_per_timestep(self) -> int:
        """``K = ceil(h/block_size) * ceil(w/block_size)``."""
        return self._block_rows * self._block_columns

    @property
    def patches_per_timestep(self) -> int:
        rows, columns = self.grid_shape
        return rows * columns

    def token_count(self, horizon: int) -> int:
        """Number of motion tokens for ``horizon`` steps (``T * K``)."""
        _positive_int('horizon', horizon)
        return int(horizon) * self.blocks_per_timestep

    @property
    def padded_grid(self) -> tuple[int, int]:
        """Patch grid rounded up to whole blocks."""
        return (self._block_rows * self.block_size,
                self._block_columns * self.block_size)

    def pixel_scale(self, like: Tensor | None = None) -> Tensor:
        """``(H, W)`` normalization scale as float32 on ``like``'s device.

        The scale is built per call so encode/decode work on any device the
        caller uses (including ``meta`` for shape/device regression runs);
        ``.to(dtype=...)`` alone would have left a CPU tensor on CPU.
        """
        scale = torch.tensor(
            [float(self.image_height), float(self.image_width)],
            dtype=torch.float32)
        return scale if like is None else scale.to(device=like.device)

    def patch_centers(self) -> Tensor:
        """Source patch centers ``[h, w, 2]`` as ``(row, column)`` pixels."""
        rows, columns = self.grid_shape
        half = (self.patch_size - 1) / 2.0
        row_index = torch.arange(rows, dtype=torch.float32)
        column_index = torch.arange(columns, dtype=torch.float32)
        grid_row, grid_column = torch.meshgrid(row_index, column_index,
                                               indexing='ij')
        centers = torch.stack((grid_row * self.patch_size + half,
                               grid_column * self.patch_size + half), dim=-1)
        return centers

    def block_extent(self, block_row: int, block_column: int) -> tuple[int, int, int, int]:
        """Real patch extent ``(row_start, row_stop, column_start, column_stop)``."""
        rows, columns = self.grid_shape
        row_start = block_row * self.block_size
        column_start = block_column * self.block_size
        return (row_start, min(row_start + self.block_size, rows),
                column_start, min(column_start + self.block_size, columns))

    def block_centers(self) -> list[tuple[float, float]]:
        """Row/column center of every block ordered by block row then column.

        Partial edge blocks use the center of their real patch extent rather
        than the padded 8x8 footprint.
        """
        centers = []
        for block_row in range(self._block_rows):
            for block_column in range(self._block_columns):
                row_start, row_stop, column_start, column_stop = self.block_extent(
                    block_row, block_column)
                centers.append((
                    (row_start + row_stop - 1) / 2.0,
                    (column_start + column_stop - 1) / 2.0,
                ))
        return centers

    def token_ids(self, horizon: int) -> Tensor:
        """Spatial/time motion token IDs ``[T*K, 4]``.

        Layout per token: ``(20, block_center_row, block_center_column,
        zero_based_timestep)`` in the visual patch-grid coordinate system.
        """
        _positive_int('horizon', horizon)
        block_centers = self.block_centers()
        identifiers = torch.zeros(self.token_count(horizon), 4,
                                  dtype=torch.float32)
        for timestep in range(int(horizon)):
            offset = timestep * self.blocks_per_timestep
            for block_index, (row, column) in enumerate(block_centers):
                identifiers[offset + block_index, 0] = MOTION_TOKEN_GROUP
                identifiers[offset + block_index, 1] = row
                identifiers[offset + block_index, 2] = column
                identifiers[offset + block_index, 3] = float(timestep)
        return identifiers

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    def normalize(self, motion_pixels: Tensor) -> Tensor:
        """Divide signed pixel motion by ``(H, W)`` exactly once."""
        return motion_pixels.to(torch.float32) / self.pixel_scale(motion_pixels)

    def denormalize(self, motion_normalized: Tensor) -> Tensor:
        """Multiply normalized motion by ``(H, W)`` exactly once."""
        return motion_normalized.to(torch.float32) * self.pixel_scale(
            motion_normalized)

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------
    def _as_bthw2(self, motion: Tensor, *, name: str = 'motion') -> Tensor:
        if not torch.is_tensor(motion):
            raise TypeError(f'{name} must be a torch tensor')
        rows, columns = self.grid_shape
        failure = (f'{name} must be [B,T,h,w,2] or [B,T*h*w,2] with '
                   f'(h,w)={self.grid_shape}; got {tuple(motion.shape)}')
        if motion.ndim not in (3, 4, 5) or motion.shape[-1] != 2:
            raise ValueError(failure)
        if motion.ndim == 5:
            if tuple(motion.shape[-3:-1]) != (rows, columns):
                raise ValueError(failure)
            return motion
        if motion.ndim == 4:
            # [B, T, h*w, 2]
            if motion.shape[-2] != self.patches_per_timestep:
                raise ValueError(failure)
            batch, steps = motion.shape[:2]
            return motion.reshape(batch, steps, rows, columns, 2)
        # [B, T*h*w, 2]
        if motion.shape[1] % self.patches_per_timestep:
            raise ValueError(failure)
        batch = motion.shape[0]
        steps = motion.shape[1] // self.patches_per_timestep
        return motion.reshape(batch, steps, rows, columns, 2)

    def _as_bthw(self, valid: Tensor, *, name: str = 'motion_valid') -> Tensor:
        if not torch.is_tensor(valid):
            raise TypeError(f'{name} must be a torch tensor')
        if valid.dtype != torch.bool:
            raise ValueError(f'{name} must be a boolean tensor')
        rows, columns = self.grid_shape
        failure = (f'{name} must be [B,T,h,w] or [B,T,h*w] with '
                   f'(h,w)={self.grid_shape} and h*w='
                   f'{self.patches_per_timestep}; got {tuple(valid.shape)}')
        if valid.ndim == 4:
            if tuple(valid.shape[-2:]) != (rows, columns):
                raise ValueError(failure)
            return valid
        if valid.ndim == 3 and valid.shape[-1] == self.patches_per_timestep:
            # [B, T, h*w]: the middle axis is already the timestep axis.
            batch, steps = valid.shape[:2]
            return valid.reshape(batch, steps, rows, columns)
        raise ValueError(failure)

    # ------------------------------------------------------------------
    # Packing / unpacking
    # ------------------------------------------------------------------
    def _pack_grid(self, grid: Tensor) -> Tensor:
        """``[B,T,hp,wp,2]`` padded grid -> ``[B,T*K,128]`` tokens."""
        batch, steps = grid.shape[:2]
        block_rows, block_columns = self.block_grid
        block = self.block_size
        grouped = grid.reshape(batch, steps, block_rows, block, block_columns,
                               block, 2)
        # (b, t, br, bc, lr, lc, coord) with local row/column/coordinate fastest.
        grouped = grouped.permute(0, 1, 2, 4, 3, 5, 6)
        return grouped.reshape(batch, steps * block_rows * block_columns,
                               block * block * 2)

    def _unpack_tokens(self, tokens: Tensor) -> Tensor:
        """``[B,T*K,128]`` tokens -> padded ``[B,T,hp,wp,2]`` grid."""
        batch, token_count = tokens.shape[:2]
        steps = token_count // self.blocks_per_timestep
        block_rows, block_columns = self.block_grid
        block = self.block_size
        grouped = tokens.reshape(batch, steps, block_rows, block_columns, block,
                                 block, 2)
        grouped = grouped.permute(0, 1, 2, 4, 3, 5, 6)
        padded_rows, padded_columns = self.padded_grid
        return grouped.reshape(batch, steps, padded_rows, padded_columns, 2)

    def encode(self, motion_pixels: Tensor) -> Tensor:
        """Pack signed pixel motion into normalized motion tokens.

        Accepts ``[B,T,h,w,2]`` or ``[B,T*h*w,2]`` (the flattened layout the
        vendored policy hands to a codec) and returns ``[B,T*K,128]``. Invalid
        or padded patches are zero-filled; only the caller's validity mask
        decides what is supervised.
        """
        motion = self._as_bthw2(motion_pixels).to(torch.float32)
        batch, steps = motion.shape[:2]
        rows, columns = self.grid_shape
        padded_rows, padded_columns = self.padded_grid
        grid = motion.new_zeros(batch, steps, padded_rows, padded_columns, 2)
        grid[..., :rows, :columns, :] = self.normalize(motion)
        return self._pack_grid(grid)

    def decode(self, tokens: Tensor) -> Tensor:
        """Decode tokens to signed pixel motion ``[B,T,h,w,2]``."""
        if not torch.is_tensor(tokens):
            raise TypeError('tokens must be a torch tensor')
        if tokens.ndim != 3 or tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f'tokens must be [B,N,{self.token_dim}], got {tuple(tokens.shape)}')
        if tokens.shape[1] % self.blocks_per_timestep:
            raise ValueError(
                f'token count {tokens.shape[1]} is not a multiple of '
                f'blocks_per_timestep={self.blocks_per_timestep}')
        rows, columns = self.grid_shape
        grid = self._unpack_tokens(tokens.to(torch.float32))
        return self.denormalize(grid[..., :rows, :columns, :])

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------
    def coordinate_validity(self, coordinate_valid: Tensor) -> Tensor:
        """Expand per-patch coordinate validity to ``[B,T*K,128]``.

        ``coordinate_valid`` is ``[B,T,h,w,2]`` (or ``[B,T*h*w,2]``) and marks
        each coordinate that may be supervised. Padded local patches inside a
        partial block are forced invalid.
        """
        if not torch.is_tensor(coordinate_valid) or coordinate_valid.dtype != torch.bool:
            raise ValueError('coordinate_valid must be a boolean tensor')
        coordinate_valid = self._as_bthw2(coordinate_valid,
                                          name='coordinate_valid')
        batch, steps = coordinate_valid.shape[:2]
        rows, columns = self.grid_shape
        padded_rows, padded_columns = self.padded_grid
        grid = coordinate_valid.new_zeros(batch, steps, padded_rows,
                                         padded_columns, 2)
        grid[..., :rows, :columns, :] = coordinate_valid
        grouped = grid.reshape(batch, steps, self._block_rows, self.block_size,
                               self._block_columns, self.block_size, 2)
        grouped = grouped.permute(0, 1, 2, 4, 3, 5, 6)
        return grouped.reshape(batch, steps * self.blocks_per_timestep,
                               self.token_dim)

    def token_validity(self, motion_valid: Tensor) -> Tensor:
        """Shared per-patch validity -> coordinate-level ``[B,T*K,128]`` mask.

        ``motion_valid`` is ``[B,T,h,w]`` or its documented flattened form
        ``[B,T,h*w]``; the patch axis is always the last one.
        """
        valid = self._as_bthw(motion_valid)
        return self.coordinate_validity(valid[..., None].expand(*valid.shape, 2))

    # ------------------------------------------------------------------
    # Targets and reconstruction
    # ------------------------------------------------------------------
    def motion_from_traces(self, traces: Tensor, trace_valid: Tensor):
        """Consecutive-step displacement targets from absolute traces.

        ``motion[t] = trace[t+1] - trace[t]`` and
        ``motion_valid[t] = trace_valid[0] & trace_valid[t] &
        trace_valid[t+1]``, so a lost intermediate point invalidates both
        adjacent displacements. Valid positions must be finite; invalid
        placeholders are sanitized before subtraction.
        """
        if not torch.is_tensor(traces):
            raise TypeError('traces must be a torch tensor')
        rows, columns = self.grid_shape
        if (traces.ndim < 4
                or tuple(traces.shape[-3:]) != (rows, columns, 2)):
            raise ValueError(
                f'traces must be [...,T+1,h,w,2] with (h,w)={self.grid_shape}, '
                f'got {tuple(traces.shape)}')
        if traces.shape[-4] < 2:
            raise ValueError('traces must contain at least two timesteps')
        trace_valid = self._as_trace_valid(trace_valid, traces.shape)
        finite = torch.isfinite(traces)
        invalid_finite = (~finite) & trace_valid[..., None]
        if bool(invalid_finite.any()):
            raise ValueError(
                'valid trace positions must be finite; found a nonfinite point '
                'at a valid (trace_valid=True) position')
        safe = torch.where(finite, traces.to(torch.float32),
                           torch.zeros_like(traces, dtype=torch.float32))
        motion = safe[..., 1:, :, :, :] - safe[..., :-1, :, :, :]
        initial_valid = trace_valid[..., 0:1, :, :]
        motion_valid = (initial_valid & trace_valid[..., :-1, :, :]
                        & trace_valid[..., 1:, :, :])
        # Finite float32 endpoints can still overflow to infinity when
        # subtracted; that is a malformed valid displacement, not padding.
        overflow = (~torch.isfinite(motion)) & motion_valid[..., None]
        if bool(overflow.any()):
            raise ValueError(
                'valid trace displacements must be finite; finite float32 '
                'endpoints overflowed during the subtraction')
        motion = torch.where(motion_valid[..., None], motion,
                             torch.zeros_like(motion))
        return motion, motion_valid

    def _as_trace_valid(self, trace_valid: Tensor, trace_shape) -> Tensor:
        if not torch.is_tensor(trace_valid) or trace_valid.dtype != torch.bool:
            raise ValueError('trace_valid must be a boolean tensor')
        steps, rows, columns = trace_shape[-4], trace_shape[-3], trace_shape[-2]
        if tuple(trace_valid.shape) != tuple(trace_shape[:-1]):
            raise ValueError(
                f'trace_valid must be [...,T+1,h,w] matching traces, got '
                f'{tuple(trace_valid.shape)} for traces {tuple(trace_shape)}')
        if trace_valid.shape[-3] != steps or trace_valid.shape[-2] != rows \
                or trace_valid.shape[-1] != columns:
            raise ValueError('trace_valid shape does not match the trace raster')
        return trace_valid

    def traces_from_motion(self, motion_pixels: Tensor) -> Tensor:
        """Source-anchored absolute traces from signed pixel displacements.

        ``trace[0]`` repeats the patch centers; ``trace[t]`` is the centers plus
        the cumulative sum of the first ``t`` displacements.
        """
        motion = self._as_bthw2(motion_pixels).to(torch.float32)
        centers = self.patch_centers().to(motion)
        cumulative = torch.cumsum(motion, dim=-4)
        zero = motion.new_zeros(*motion.shape[:-4], 1, *motion.shape[-3:])
        offsets = torch.cat((zero, cumulative), dim=-4)
        return offsets + centers


__all__ = ['COORDINATE_ORDER', 'DEFAULT_BLOCK_SIZE', 'DEFAULT_PATCH_SIZE',
           'DEFAULT_TOKEN_DIM', 'MOTION_TOKEN_GROUP', 'PatchMotion2DCodec']
