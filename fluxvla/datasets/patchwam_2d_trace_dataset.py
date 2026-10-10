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

"""Offline manifest/NPZ/RGB dataset for the 2D patch-motion PatchWAM variant.

Layout::

    dataset/
      dataset.json
      train.jsonl
      val.jsonl
      rgb/<episode_id>/<frame_id>.png
      labels/<sample_id>.npz

Every sample returns native model inputs (current/future RGB in ``[-1, 1]``)
and *pixel* motion, so the codec owns the single ``(H, W)`` normalization. No
robot state, action or robot-physics statistic is produced here.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from fluxvla.engines import DATASETS
from ..models.third_party_models.patchwam.motion2d import (
    COORDINATE_ORDER, DEFAULT_BLOCK_SIZE, DEFAULT_PATCH_SIZE, PatchMotion2DCodec)

__all__ = ['PATCHWAM_2D_TRACE_SCHEMA_VERSION', 'PatchWAM2DTraceDataset']

PATCHWAM_2D_TRACE_SCHEMA_VERSION = 'patchwam_2d_trace_v1'

_SPLITS = ('train', 'val')
_ROW_KEYS = ('sample_id', 'episode_id', 'instruction', 'current_rgb',
             'future_rgb', 'labels')
_NPZ_KEYS = ('traces_2d', 'trace_valid', 'trace_t_seconds')
_CENTER_TOLERANCE = 1e-3
_TIME_TOLERANCE = 1e-6
_PATH_FIELDS = ('current_rgb', 'future_rgb', 'labels')


def _read_json(path: Path, what: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f'{what} not found: {path}')
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f'{what} is not valid JSON: {path}: {error}') from error


def _require_positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')
    return int(value)


def _optional_float(name: str, value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be a finite number, got {value!r}') from error
    if not np.isfinite(number):
        raise ValueError(f'{name} must be a finite number, got {value!r}')
    return number


@DATASETS.register_module()
class PatchWAM2DTraceDataset:
    """Manifest-driven 2D patch-motion trace dataset.

    The manifest fixes the geometry (``image_height``/``image_width``/
    ``patch_size=16``/``motion_block_size=8``), the row/column coordinate order
    and the strictly increasing ``timestep_offsets_s`` schedule. Constructor
    arguments may restate those values but never override them.

    Args:
        dataset_root: Dataset root directory containing ``dataset.json``.
        split: ``train`` or ``val``; the corresponding JSONL is loaded.
        check_split_disjoint: Require the train and val episodes to differ.
    """

    SCHEMA_VERSION = PATCHWAM_2D_TRACE_SCHEMA_VERSION

    def __init__(self, dataset_root: str, *, split: str = 'train',
                 horizon: Optional[int] = None,
                 patch_size: Optional[int] = None,
                 motion_block_size: Optional[int] = None,
                 coordinate_order: Optional[str] = None,
                 timestep_offsets_s: Optional[Sequence[float]] = None,
                 check_split_disjoint: bool = True) -> None:
        if split not in _SPLITS:
            raise ValueError(f'split must be one of {_SPLITS}, got {split!r}')
        root = Path(dataset_root).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f'dataset root not found: {root}')
        self.dataset_root = root.resolve()
        self.split = split
        manifest = _read_json(self.dataset_root / 'dataset.json',
                              'dataset.json')
        if not isinstance(manifest, dict):
            raise ValueError('dataset.json must contain a JSON object')
        self.manifest = manifest
        if manifest.get('schema_version') != self.SCHEMA_VERSION:
            raise ValueError(
                f'unsupported schema_version {manifest.get("schema_version")!r}; '
                f'expected {self.SCHEMA_VERSION!r}')

        declared_patch = manifest.get('patch_size')
        if declared_patch != DEFAULT_PATCH_SIZE:
            raise ValueError(
                f'patch_size must be {DEFAULT_PATCH_SIZE}, got '
                f'{declared_patch!r}')
        declared_block = manifest.get('motion_block_size')
        if declared_block != DEFAULT_BLOCK_SIZE:
            raise ValueError(
                f'motion_block_size must be {DEFAULT_BLOCK_SIZE}, got '
                f'{declared_block!r}')
        declared_order = manifest.get('coordinate_order')
        if declared_order != COORDINATE_ORDER:
            raise ValueError(
                f'coordinate_order must be {COORDINATE_ORDER!r}, got '
                f'{declared_order!r}')
        for name, value in (('patch_size', patch_size),
                            ('motion_block_size', motion_block_size),
                            ('coordinate_order', coordinate_order)):
            if value is not None and value != manifest.get(name):
                raise ValueError(
                    f'{name}={value!r} disagrees with the dataset manifest '
                    f'({manifest.get(name)!r})')

        self.image_height = _require_positive_int(
            'image_height', manifest.get('image_height'))
        self.image_width = _require_positive_int(
            'image_width', manifest.get('image_width'))
        self.horizon = _require_positive_int('horizon', manifest.get('horizon'))
        if horizon is not None and int(horizon) != self.horizon:
            raise ValueError(
                f'horizon={horizon!r} disagrees with the dataset manifest '
                f'({self.horizon})')
        offsets = manifest.get('timestep_offsets_s')
        if not isinstance(offsets, (list, tuple)) or len(offsets) != self.horizon:
            raise ValueError(
                f'timestep_offsets_s must contain exactly horizon='
                f'{self.horizon} values, got {offsets!r}')
        self.timestep_offsets_s = tuple(
            _optional_float('timestep_offsets_s', value) for value in offsets)
        if any(value <= 0 for value in self.timestep_offsets_s):
            raise ValueError(
                'timestep_offsets_s must be strictly positive sample times '
                f'relative to the current frame, got {self.timestep_offsets_s}')
        if any(second <= first for first, second in
               zip(self.timestep_offsets_s, self.timestep_offsets_s[1:])):
            raise ValueError(
                'timestep_offsets_s must be strictly increasing, got '
                f'{self.timestep_offsets_s}')
        if timestep_offsets_s is not None:
            declared = tuple(float(value) for value in timestep_offsets_s)
            if declared != self.timestep_offsets_s:
                raise ValueError(
                    f'timestep_offsets_s={declared} disagrees with the dataset '
                    f'manifest ({self.timestep_offsets_s})')
        self.patch_size = DEFAULT_PATCH_SIZE
        self.motion_block_size = DEFAULT_BLOCK_SIZE
        self.coordinate_order = COORDINATE_ORDER
        self.codec = PatchMotion2DCodec(
            image_height=self.image_height, image_width=self.image_width,
            patch_size=self.patch_size, block_size=self.motion_block_size)
        self.patch_centers = self.codec.patch_centers().numpy()
        self.rows = self._load_rows(split)
        self._check_split_sample_ids()
        if check_split_disjoint:
            self._check_split_disjoint()

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------
    def _load_rows(self, split: str) -> List[Dict[str, Any]]:
        path = self.dataset_root / f'{split}.jsonl'
        if not path.is_file():
            raise FileNotFoundError(f'{split} split not found: {path}')
        rows: List[Dict[str, Any]] = []
        seen = set()
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f'{split}.jsonl line {number} is not valid JSON: '
                    f'{error}') from error
            if not isinstance(row, dict):
                raise ValueError(
                    f'{split}.jsonl line {number} must be a JSON object')
            for key in _ROW_KEYS:
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError(
                        f'{split}.jsonl line {number} requires a non-empty '
                        f'{key!r}')
            sample_id = row['sample_id']
            if sample_id in seen:
                raise ValueError(
                    f'duplicate sample_id {sample_id!r} in {split}.jsonl')
            seen.add(sample_id)
            for field in _PATH_FIELDS:
                self._resolve(row[field], field=field, sample_id=sample_id)
            rows.append(row)
        return rows

    def _split_index(self, split: str) -> Optional[Dict[str, set]]:
        """Light ``(sample_ids, episode_ids)`` view of the other split."""
        path = self.dataset_root / f'{split}.jsonl'
        if not path.is_file():
            return None
        sample_ids, episode_ids = set(), set()
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (not isinstance(row, dict)
                    or not isinstance(row.get('episode_id'), str)
                    or not isinstance(row.get('sample_id'), str)):
                raise ValueError(
                    f'{split}.jsonl line {number} requires sample_id/episode_id')
            sample_ids.add(row['sample_id'])
            episode_ids.add(row['episode_id'])
        return {'sample_ids': sample_ids, 'episode_ids': episode_ids}

    def _check_split_sample_ids(self) -> None:
        """Sample ids are the manifest key, so they must be unique per dataset.

        Distinct episodes are not sufficient: a repeated id would make two rows
        point at one ``labels/<sample_id>.npz``.
        """
        other = 'val' if self.split == 'train' else 'train'
        theirs = self._split_index(other)
        if not theirs:
            return
        ours = {row['sample_id'] for row in self.rows}
        shared = sorted(ours & theirs['sample_ids'])
        if shared:
            raise ValueError(
                f'sample ids must be unique across the train and val splits; '
                f'duplicate sample_id(s): {shared}')

    def _check_split_disjoint(self) -> None:
        other = 'val' if self.split == 'train' else 'train'
        ours = {row['episode_id'] for row in self.rows}
        index = self._split_index(other)
        if not ours or not index:
            return
        shared = sorted(ours & index['episode_ids'])
        if shared:
            raise ValueError(
                'training and validation must use different episodes; shared '
                f'episode ids: {shared}')

    def _resolve(self, relative: str, *, field: str,
                 sample_id: str) -> Path:
        if not isinstance(relative, str) or not relative:
            raise ValueError(
                f'{field} for sample {sample_id!r} must be a non-empty '
                'relative path')
        candidate = Path(relative)
        if candidate.is_absolute():
            raise ValueError(
                f'{field} for sample {sample_id!r} must be relative to the '
                f'dataset root, got {relative!r}')
        resolved = (self.dataset_root / candidate).resolve()
        if not resolved.is_relative_to(self.dataset_root):
            raise ValueError(
                f'{field} for sample {sample_id!r} escapes the dataset root: '
                f'{relative!r}')
        return resolved

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def sample_ids(self) -> List[str]:
        return [row['sample_id'] for row in self.rows]

    @property
    def episode_ids(self) -> List[str]:
        return [row['episode_id'] for row in self.rows]

    @property
    def statistics(self) -> Dict[str, Any]:
        """Geometry/manifest metadata only; never robot joint statistics."""
        return {
            'schema_version': self.SCHEMA_VERSION,
            'split': self.split,
            'samples': len(self.rows),
            'image_height': self.image_height,
            'image_width': self.image_width,
            'patch_size': self.patch_size,
            'motion_block_size': self.motion_block_size,
            'horizon': self.horizon,
            'coordinate_order': self.coordinate_order,
            'timestep_offsets_s': list(self.timestep_offsets_s),
            'patch_grid': list(self.codec.grid_shape),
            'blocks_per_timestep': self.codec.blocks_per_timestep,
        }

    def __len__(self) -> int:
        return len(self.rows)

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------
    def _load_labels(self, row: Dict[str, Any]):
        sample_id = row['sample_id']
        path = self._resolve(row['labels'], field='labels',
                             sample_id=sample_id)
        if not path.is_file():
            raise FileNotFoundError(
                f'labels file for sample {sample_id!r} not found: {path}')
        try:
            with np.load(path, allow_pickle=False) as archive:
                keys = set(archive.files)
                missing = [key for key in _NPZ_KEYS if key not in keys]
                if missing:
                    raise ValueError(
                        f'labels archive for sample {sample_id!r} is missing '
                        f'{missing}')
                traces = archive['traces_2d']
                valid = archive['trace_valid']
                times = archive['trace_t_seconds']
                declared_order = (archive['coordinate_order']
                                  if 'coordinate_order' in keys else None)
        except ValueError:
            raise
        except Exception as error:  # pragma: no cover - defensive
            raise ValueError(
                f'labels archive for sample {sample_id!r} could not be read: '
                f'{error}') from error
        traces = np.asarray(traces)
        valid = np.asarray(valid)
        times = np.asarray(times)
        rows, columns = self.codec.grid_shape
        expected_traces = (self.horizon + 1, rows, columns, 2)
        if traces.shape != expected_traces:
            raise ValueError(
                f'traces_2d for sample {sample_id!r} must be [T+1,h,w,2] = '
                f'{expected_traces}, got {tuple(traces.shape)}')
        if not np.issubdtype(traces.dtype, np.floating):
            raise ValueError(
                f'traces_2d for sample {sample_id!r} must be float32, got '
                f'{traces.dtype}')
        if traces.dtype != np.float32:
            raise ValueError(
                f'traces_2d for sample {sample_id!r} must be float32, got '
                f'{traces.dtype}')
        if valid.shape != expected_traces[:-1]:
            raise ValueError(
                f'trace_valid for sample {sample_id!r} must be [T+1,h,w] = '
                f'{expected_traces[:-1]}, got {tuple(valid.shape)}')
        if valid.dtype != np.bool_:
            raise ValueError(
                f'trace_valid for sample {sample_id!r} must be boolean, got '
                f'{valid.dtype}')
        if times.shape != (self.horizon + 1,):
            raise ValueError(
                f'trace_t_seconds for sample {sample_id!r} must be [T+1] = '
                f'{(self.horizon + 1,)}, got {tuple(times.shape)}')
        if not np.issubdtype(times.dtype, np.floating):
            raise ValueError(
                f'trace_t_seconds for sample {sample_id!r} must be float64, '
                f'got {times.dtype}')
        if times.dtype != np.float64:
            raise ValueError(
                f'trace_t_seconds for sample {sample_id!r} must be float64, '
                f'got {times.dtype}')
        if declared_order is not None:
            declared = np.asarray(declared_order).reshape(-1)
            if declared.size != 1 or str(declared[0]) != COORDINATE_ORDER:
                raise ValueError(
                    f'coordinate_order for sample {sample_id!r} must be '
                    f'{COORDINATE_ORDER!r}, got {declared_order!r}')
        if not bool(np.isfinite(times).all()):
            raise ValueError(
                f'trace_t_seconds for sample {sample_id!r} must be finite and '
                'match the declared timestep_offsets_s')
        if abs(float(times[0])) > _TIME_TOLERANCE:
            raise ValueError(
                f'trace_t_seconds for sample {sample_id!r} must start at zero '
                'and match the declared timestep_offsets_s')
        if not np.allclose(times[1:], np.asarray(self.timestep_offsets_s),
                           rtol=0, atol=_TIME_TOLERANCE):
            raise ValueError(
                f'trace_t_seconds for sample {sample_id!r} must match the '
                f'declared timestep_offsets_s={self.timestep_offsets_s}, got '
                f'{times[1:].tolist()}')
        traces = traces.astype(np.float32)
        initial = valid[0]
        deviation = np.abs(traces[0] - self.patch_centers)
        if bool((initial[..., None] & (deviation > _CENTER_TOLERANCE)).any()):
            raise ValueError(
                f'traces_2d for sample {sample_id!r} must have trace zero '
                'anchored on the patch centers at every valid position')
        return traces, valid

    def _load_rgb(self, row: Dict[str, Any], field: str) -> np.ndarray:
        sample_id = row['sample_id']
        path = self._resolve(row[field], field=field, sample_id=sample_id)
        if not path.is_file():
            raise FileNotFoundError(
                f'{field} image for sample {sample_id!r} not found: {path}')
        try:
            with Image.open(path) as handle:
                if handle.mode != 'RGB':
                    raise ValueError(
                        f'{field} image for sample {sample_id!r} must be an '
                        f'8-bit RGB PNG, got mode {handle.mode!r}')
                array = np.asarray(handle.convert('RGB'), dtype=np.uint8)
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(
                f'{field} image for sample {sample_id!r} could not be read: '
                f'{error}') from error
        expected = (self.image_height, self.image_width, 3)
        if array.shape != expected:
            raise ValueError(
                f'{field} image for sample {sample_id!r} must match the '
                f'declared image geometry {expected}, got {tuple(array.shape)}')
        return array.astype(np.float32) / 127.5 - 1.0

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        sample_id = row['sample_id']
        traces, valid = self._load_labels(row)
        # The codec owns the displacement + validity contract and is torch-only;
        # convert once here and hand numpy back to the loader/collator.
        motion, motion_valid = self.codec.motion_from_traces(
            torch.from_numpy(traces), torch.from_numpy(valid))
        motion = motion.numpy()
        motion_valid = motion_valid.numpy()
        current = self._load_rgb(row, 'current_rgb')
        future = self._load_rgb(row, 'future_rgb')
        # [frames=2, H, W, 3] -> [view=1, channels=3, frames=2, H, W].
        frames = np.stack((current, future), axis=0)
        images = frames.transpose(3, 0, 1, 2)[None]
        return {
            'images': images.astype(np.float32),
            'motion_2d': motion.astype(np.float32),
            'motion_valid': np.asarray(motion_valid, dtype=np.bool_),
            'task_description': row['instruction'],
            'sample_id': sample_id,
            'episode_id': row['episode_id'],
        }
