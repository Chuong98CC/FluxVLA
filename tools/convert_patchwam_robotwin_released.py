"""Convert the released PatchWAM RoboTwin checkpoint and statistics to native form.

The published ``step_140000.pt`` is an ImageWAM-lineage payload
(``mot.mixtures.video.transformer.*`` plus ``proprio_encoder.*``). The shared
RoboTwin evaluator's policy builds the model and then calls
``nn.Module.load_state_dict(..., strict=True)`` on the raw file, which that
payload cannot satisfy. This tool writes *new* files that can:

* a native-format safetensors whose keys are exactly the model's
  ``state_dict()``, produced with the repository's own released-key mapping and
  verified tensor-by-tensor against the source, and
* a native ``dataset_statistics.json`` converted from the released
  ``dataset_stats.json`` z-score fields.

Neither the released checkpoint nor the released statistics file is modified.

Example::

    python tools/convert_patchwam_robotwin_released.py \
        --config configs/patchwam/patchwam_robotwin_released_eval.py
"""

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from mmengine.config import Config
from safetensors.torch import save_file

from fluxvla.datasets.utils.statistics import (
    convert_released_robotwin_statistics_file, reject_conversion_overwrite)
from fluxvla.engines import build_vla_from_cfg
from fluxvla.models.vlas.patchwam_vla import released_action_patch_state


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_released_state(path: str):
    try:
        payload = torch.load(
            path, map_location='cpu', weights_only=True, mmap=True)
    except RuntimeError:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(payload, dict) and 'model' in payload:
        payload = payload['model']
    return payload


def convert_checkpoint(config: str, ckpt_path: str, out_path: str,
                       device: str) -> dict:
    """Write a native-format checkpoint and return a verification report."""
    # Checked before the model is built or anything is written: the released
    # checkpoint must survive the conversion.
    reject_conversion_overwrite(ckpt_path, out_path)
    cfg = Config.fromfile(config)
    model_cfg = copy.deepcopy(
        cfg.inference_model if hasattr(cfg, 'inference_model') else cfg.model)
    # ``PatchWAMVLA`` takes no top-level ``device``; the runtime device lives in
    # the backbone and denoiser blocks.
    if device:
        if isinstance(model_cfg.get('vlm_backbone'), dict):
            model_cfg['vlm_backbone']['device'] = device
        if isinstance(model_cfg.get('vla_head'), dict) and isinstance(
                model_cfg['vla_head'].get('denoiser'), dict):
            model_cfg['vla_head']['denoiser']['device'] = device

    model = build_vla_from_cfg(model_cfg).eval()
    state = _read_released_state(ckpt_path)
    mapped = released_action_patch_state(state)
    if mapped is None:
        raise ValueError(
            f'{ckpt_path} is not a released PatchWAM Action-as-Patch '
            'payload; this converter only handles that format')

    report = model.load_patchwam_state_dict(state, strict=True)

    # Tensor-level parity: every mapped source tensor must be present with the
    # same shape and exactly the same values in the built model.
    current = model.state_dict()
    absent = sorted(set(mapped) - set(current))
    if absent:
        raise ValueError(f'Mapped keys absent from the model: {absent[:8]}')
    for target, tensor in mapped.items():
        built = current[target]
        if tuple(built.shape) != tuple(tensor.shape):
            raise ValueError(
                f'Shape mismatch for {target}: {tuple(tensor.shape)} vs '
                f'{tuple(built.shape)}')
        if not torch.equal(built.detach().to('cpu'),
                           tensor.to(built.dtype).detach().to('cpu')):
            raise ValueError(
                f'Value mismatch for {target}: the built model does not '
                'reproduce the released tensor')

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for name, tensor in current.items():
        value = tensor.detach()
        if value.is_floating_point() and value.dtype not in (
                torch.bfloat16, torch.float16, torch.float32):
            value = value.float()
        flat[name] = value.to('cpu').contiguous().clone()
    save_file(
        flat, out_path,
        metadata={
            'format': 'fluxvla-native',
            'source_checkpoint': str(Path(ckpt_path).resolve()),
            'source_sha256': _sha256(ckpt_path),
            'released_step': str(state.get('step', '')),
            'config': str(Path(config).resolve()),
        })
    del flat
    return {
        'format': report.get('format'),
        'loaded_tensors': report.get('loaded_tensors'),
        'model_tensors': len(current),
        'out_path': str(out_path),
        'out_bytes': os.path.getsize(out_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config', default='configs/patchwam/patchwam_robotwin_released_eval.py')
    parser.add_argument('--ckpt', default=None,
                        help='Released checkpoint; defaults to the config value')
    parser.add_argument('--out', default=None,
                        help='Native output; defaults to the config value')
    parser.add_argument('--stats', default=None,
                        help='Released statistics JSON')
    parser.add_argument('--stats-out', default=None,
                        help='Native statistics JSON output')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--skip-stats', action='store_true')
    parser.add_argument('--skip-ckpt', action='store_true')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    released_ckpt = args.ckpt or getattr(cfg, 'PATCHWAM_RELEASED_CHECKPOINT',
                                         None)
    out_ckpt = args.out or getattr(cfg, 'PATCHWAM_CONVERTED_CHECKPOINT', None)
    released_stats = args.stats or getattr(cfg, 'PATCHWAM_RELEASED_STATS', None)
    out_stats = args.stats_out or getattr(cfg, 'PATCHWAM_CONVERTED_STATS', None)

    if released_ckpt is None or out_ckpt is None:
        raise SystemExit(
            'The config does not expose PATCHWAM_RELEASED_CHECKPOINT / '
            'PATCHWAM_CONVERTED_CHECKPOINT; pass --ckpt and --out')

    result = {}
    if not args.skip_stats:
        if released_stats is None or out_stats is None:
            raise SystemExit('Pass --stats/--stats-out or use a config that '
                             'exposes both paths')
        Path(out_stats).parent.mkdir(parents=True, exist_ok=True)
        converted = convert_released_robotwin_statistics_file(
            released_stats, out_stats)
        result['statistics'] = {
            'source': released_stats,
            'out_path': out_stats,
            'fields': sorted(converted[list(converted)[0]]),
        }

    if not args.skip_ckpt:
        result['checkpoint'] = convert_checkpoint(
            args.config, released_ckpt, out_ckpt, args.device)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
