"""Bounded RoboTwin-Clean five-task smoke for the PatchWAM ablation recipes.

The supplied release is read-only: this script never regenerates data, never
writes into the dataset root, and never downloads assets. Every mode prints a
compact JSON report so a reviewer can reproduce the result independently.

Modes:
  selection  Preflight the five configured per-task roots: report per-folder
             episodes/FPS/cameras and the first/last instruction as provenance,
             and fail with a nonzero exit listing exactly which requested
             folders the release does not contain.
  dataset    Build the training dataset exactly as ``scripts/train.py`` does
             for the five-task recipe, decode real head/wrist video, and
             collate one batch with the recipe's own collator.
  train      Additionally build the configured model from the local assets, load
             the recipe's initial weights in the DDP runner's order, and run one
             forward/loss (``--backward`` for a backward pass).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG = (REPO_ROOT / 'configs' / 'patchwam'
          / 'patchwam_robotwin_vlm_5tasks_full_finetune.py')


def _config():
    from mmengine.config import Config
    return Config.fromfile(str(CONFIG))


def dataset_roots(config):
    """The recipe's configured per-task LeRobot roots, in order."""
    node = config.train_dataloader.dataset.datasets
    return [Path(item) for item in node['data_root_path']]


def missing_roots(roots):
    """Configured roots that are not present on disk."""
    return [str(root) for root in roots
            if not (root / 'meta' / 'info.json').is_file()]


def _missing_message(missing):
    return ('Missing RoboTwin-Clean task roots: ' + ', '.join(missing)
            + '. Publish the remaining task folders or point PATCHWAM_DATA_ROOT '
              'at a release that contains all five requested tasks.')


def require_roots(roots):
    """Fail loudly rather than training on a smaller subset than configured."""
    missing = missing_roots(roots)
    if missing:
        raise SystemExit(_missing_message(missing))


def selection_report(roots):
    """Per-folder metadata summary; missing folders are reported, never skipped."""
    started = time.time()
    folders, missing = [], []
    for root in roots:
        if not (root / 'meta' / 'info.json').is_file():
            missing.append(str(root))
            folders.append({'root': str(root), 'exists': False})
            continue
        info = json.loads((root / 'meta' / 'info.json').read_text(encoding='utf-8'))
        declared, first, last = 0, None, None
        with open(root / 'meta' / 'episodes.jsonl', encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                declared += 1
                instructions = row.get('tasks') or []
                if first is None:
                    first = instructions[0] if instructions else None
                last = instructions[0] if instructions else None
        folders.append({
            'root': str(root),
            'exists': True,
            'task': root.name,
            'episodes': declared,
            'fps': info.get('fps'),
            'codebase_version': info.get('codebase_version'),
            'cameras': sorted(key for key in info.get('features', {})
                              if key.startswith('observation.images')),
            'first_instruction': first,
            'last_instruction': last,
        })
    return {
        'mode': 'selection',
        'configured_roots': [str(root) for root in roots],
        'folders': folders,
        'missing': missing,
        'status': 'ok' if not missing else 'missing_folders',
        'seconds': round(time.time() - started, 2),
    }


def run_selection(args):
    roots = dataset_roots(_config())
    return selection_report(roots)


def _tensor_summary(tensor):
    import torch
    entry = {
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'finite': bool(torch.isfinite(tensor.float()).all()),
    }
    if tensor.numel() == 0:
        entry['empty'] = True
        return entry
    if tensor.dtype == torch.bool:
        entry['true_count'] = int(tensor.sum())
    else:
        entry.update(min=round(float(tensor.min()), 4),
                     max=round(float(tensor.max()), 4),
                     mean=round(float(tensor.float().mean()), 4))
    return entry


def _batch_summary(batch):
    import torch
    return {key: (_tensor_summary(value) if isinstance(value, torch.Tensor)
                  else {'type': type(value).__name__, 'value': str(value)[:120]})
            for key, value in batch.items()}


def _first_sample(dataset):
    """One configured sample for either dataset protocol the recipe uses."""
    try:
        return next(iter(dataset))
    except TypeError:
        return dataset[0]


def run_dataset(args):
    import torch
    from fluxvla.datasets.utils.task_selection import dataset_task_selection
    from fluxvla.engines import build_collator_from_cfg, build_dataset_from_cfg

    config = _config()
    require_roots(dataset_roots(config))
    started = time.time()
    dataset = build_dataset_from_cfg(config.train_dataloader.dataset)
    build_seconds = round(time.time() - started, 2)
    selection = dataset_task_selection(dataset)
    sample = _first_sample(dataset)
    batch = build_collator_from_cfg(config.runner.collator)([sample])
    return {
        'mode': 'dataset',
        'dataset_type': type(dataset).__name__,
        'length': len(dataset),
        'selection_source': None if selection is None else selection.get('source'),
        'selection_episodes': None if selection is None else selection.get('episodes'),
        'task_description': str(sample.get('task_description'))[:160],
        'sample': _batch_summary(sample),
        'batch': _batch_summary(batch),
        'torch': torch.__version__,
        'seconds': build_seconds,
    }


def run_train(args):
    import torch
    from fluxvla.engines import (build_collator_from_cfg, build_dataset_from_cfg,
                                 build_vla_from_cfg)

    config = _config()
    require_roots(dataset_roots(config))
    dataset = build_dataset_from_cfg(config.train_dataloader.dataset)
    batch = build_collator_from_cfg(config.runner.collator)([_first_sample(dataset)])
    started = time.time()
    model = build_vla_from_cfg(config.model)
    # The DDP runner freezes the backbones and loads the recipe's initial
    # weights before the mixed-precision device move; mirror both so the smoke
    # trains the same model the recipe does.
    model.freeze_backbones()
    model.from_pretrained()
    if torch.cuda.is_available():
        # Mirrors DDPTrainRunner._move_vla_to_device for the recipe's bf16
        # mixed-precision setting (keep_params_fp32 is not set).
        model = model.to(device='cuda', dtype=torch.bfloat16)
    model.train()
    outputs = model(**batch)
    # BaseTrainRunner._training_step trains on ``outputs['loss']`` alone; every
    # other entry is a diagnostic and must not be summed into the objective.
    if 'loss' not in outputs:
        raise RuntimeError(f'Model returned no objective: {sorted(outputs)}')
    objective = outputs['loss']
    if objective.ndim != 0 or not bool(torch.isfinite(objective.detach())):
        raise RuntimeError(f'Objective is not a finite scalar: {objective!r}')
    diagnostics = {key: round(float(value.detach()), 6)
                   for key, value in outputs.items()
                   if key != 'loss' and isinstance(value, torch.Tensor)
                   and value.ndim == 0}
    level = 'forward'
    gradients = {'checked': False}
    if args.backward:
        objective.backward()
        level = 'forward+backward'
        populated, nonzero, nonfinite = 0, 0, []
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            populated += 1
            if not bool(torch.isfinite(parameter.grad).all()):
                nonfinite.append(name)
            elif parameter.requires_grad and bool(parameter.grad.abs().sum() > 0):
                nonzero += 1
        gradients = {
            'checked': True,
            'trainable_parameters': sum(1 for parameter in model.parameters()
                                        if parameter.requires_grad),
            'populated_gradients': populated,
            'nonzero_trainable_gradients': nonzero,
            'nonfinite_gradients': nonfinite[:10],
        }
        if nonzero < 1:
            raise RuntimeError('Backward produced no nonzero trainable gradient')
        if nonfinite:
            raise RuntimeError(f'Non-finite trainable gradients: {nonfinite[:5]}')
    return {
        'mode': 'train',
        'level_reached': level,
        'initial_weights': getattr(model, 'pretrained_name_or_path', None),
        'objective': round(float(objective.detach()), 6),
        'objective_is_finite_scalar': True,
        'diagnostics': diagnostics,
        'gradients': gradients,
        'device': str(next(model.parameters()).device),
        'vram_allocated_gb': round(torch.cuda.memory_allocated() / 2 ** 30, 2),
        'vram_reserved_gb': round(torch.cuda.memory_reserved() / 2 ** 30, 2),
        'vram_peak_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
        'seconds': round(time.time() - started, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('selection', 'dataset', 'train'))
    parser.add_argument('--report', type=Path, default=None)
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        os.environ.setdefault(name, '1')
    report = {'selection': run_selection, 'dataset': run_dataset,
              'train': run_train}[args.mode](args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + '\n', encoding='utf-8')
    print(text)
    if report.get('missing'):
        raise SystemExit(_missing_message(report['missing']))


if __name__ == '__main__':
    main()
