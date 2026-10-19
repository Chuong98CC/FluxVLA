"""Run a bounded PatchWAM FK/action forward on the configured training data.

This is a diagnostic smoke, not training. It builds the recipe's real training
dataset, loads the local released checkpoint when requested, and reports the
loss components, TCP errors, and active-arm counts for a small batch.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from mmengine import Config

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fluxvla.datasets  # noqa: E402, F401
import fluxvla.transforms  # noqa: E402, F401
from fluxvla.engines import (build_collator_from_cfg,  # noqa: E402
                             build_dataset_from_cfg, build_vla_from_cfg)
from fluxvla.engines.runners.base_train_runner import (  # noqa: E402
    BaseTrainRunner)
from fluxvla.engines.runners.ddp_train_runner import (  # noqa: E402
    DDPTrainRunner)
from fluxvla.engines.utils.name_map import str_to_dtype  # noqa: E402
from fluxvla.models.backbones.vlms import patchwam_backbone  # noqa: E402
from fluxvla.models.heads import patchwam_head  # noqa: E402, F401
from fluxvla.models.vlas import patchwam_vla  # noqa: E402, F401


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        default='configs/patchwam/robotwin/patchwam_robotwin.py')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-checkpoint', action='store_true')
    parser.add_argument('--output', default=None)
    return parser.parse_args()


def _scalar_metrics(output):
    metrics = {}
    for key, value in output.items():
        if torch.is_tensor(value) and value.numel() == 1:
            metrics[key] = float(value.detach().float().cpu())
    return metrics


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = Config.fromfile(str(REPO_ROOT / args.config))
    dataset = build_dataset_from_cfg(cfg.train_dataloader.dataset)
    samples = [
        dataset._get_item_from_global_idx(index)
        for index in range(args.samples)
    ]
    collator = build_collator_from_cfg(cfg.runner.collator)
    batch = collator(samples)
    runner = object.__new__(DDPTrainRunner)
    runner.batch_fp32_keys = BaseTrainRunner._normalize_batch_fp32_keys(
        cfg.runner.get('batch_fp32_keys'))
    batch_dtype = (
        str_to_dtype(cfg.runner.get('mixed_precision_dtype', 'bf16'))
        if cfg.runner.get('enable_mixed_precision_training', True)
        else None)
    batch = runner._prepare_batch(batch, args.device, batch_dtype)

    model_cfg = copy.deepcopy(Config(cfg.model).to_dict())
    if args.no_checkpoint:
        model_cfg['pretrained_name_or_path'] = None
    elif args.checkpoint:
        model_cfg['pretrained_name_or_path'] = args.checkpoint
    vla = build_vla_from_cfg(model_cfg)
    vla.to(args.device)

    checkpoint_report = None
    checkpoint = model_cfg.get('pretrained_name_or_path')
    if checkpoint and not args.no_checkpoint:
        state = vla._read_checkpoint_state(checkpoint)
        _, format_name, resolved, missing, unexpected = (
            vla._resolve_patchwam_state(state))
        checkpoint_report = {
            'path': str(checkpoint),
            'format': format_name,
            'resolved_tensors': len(resolved),
            'missing': list(missing),
            'unexpected': list(unexpected),
        }
        vla.load_patchwam_state_dict(state, strict=True)
        del state
        gc.collect()

    vla.eval()
    with torch.no_grad():
        device_type = torch.device(args.device).type
        with torch.autocast(
                device_type=device_type, dtype=torch.bfloat16,
                enabled=device_type == 'cuda'):
            output = vla(**batch)
    metrics = _scalar_metrics(output)
    report = {
        'config': str(REPO_ROOT / args.config),
        'checkpoint': checkpoint_report,
        'samples': args.samples,
        'seed': args.seed,
        'batch_shapes': {
            key: list(value.shape) for key, value in batch.items()
            if torch.is_tensor(value) and key in (
                'video', 'actions', 'states', 'active_arm',
                'fk_action_mean', 'fk_action_std')},
        'batch_dtypes': {
            key: str(batch[key].dtype) for key in (
                'video', 'actions', 'states', 'fk_action_mean',
                'fk_action_std', 'fk_action_epsilon')
            if key in batch and torch.is_tensor(batch[key])
        },
        'active_arm': (
            batch['active_arm'].detach().cpu().tolist()
            if 'active_arm' in batch else None),
        'metrics': metrics,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
