"""Vendored PatchWAM data utilities (augmentation, appearance, history, cameras).

These modules are copied from the PatchWAM ``src/patchwam/data`` package and
retain their MIT (ImageWAM) attribution. The source ``episodes``,
``processing`` and ``scaling`` modules are intentionally not vendored: FluxVLA
reuses its native Parquet dataset, statistics, and normalization/denormalization
pipeline instead of the PatchWAM training engine.

Copyright (c) 2026 Yuyang "Alice.L". MIT-licensed; see ``LICENSE-MIT`` and
``PROVENANCE.json`` in this directory.
"""

from .appearance import AppearanceRandomizer
from .augmentation import ClipAugment
from .cameras import compose_cameras, prepare_rgb, resize_clip
from .history import CausalObservationBuffer, PastFrameSelector

__all__ = [
    'AppearanceRandomizer',
    'CausalObservationBuffer',
    'ClipAugment',
    'PastFrameSelector',
    'compose_cameras',
    'prepare_rgb',
    'resize_clip',
]
