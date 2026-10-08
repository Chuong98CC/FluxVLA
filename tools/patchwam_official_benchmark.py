#!/usr/bin/env python3
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
"""Official PatchWAM RoboTwin 2.0 benchmark launcher.

Drives the native evaluation entrypoint (``scripts/eval.py``) once per
(task, condition) case over the released PatchWAM C2R protocol: 50 tasks,
``demo_clean``/``demo_randomized``, 100 episodes each, unseen instructions,
action horizon 16, replan 16, 10 denoising steps, expert seed validation and
seed 0. That is 100 cases and 10,000 episodes.

This module is deliberately small and dependency-free: it only builds and runs
plain ``scripts/eval.py`` subprocesses, one case at a time on a single GPU, and
aggregates the per-case ``results.json`` files into a suite summary. It never
imports the model, torch or the simulator, so ``--dry-run``/``--list`` cannot
load a model or spawn a simulator.

The released per-task success rates are *not* copied into this repository; the
static task list lives in ``patchwam_robotwin_official_benchmark_tasks.json``.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (REPO_ROOT / 'configs' / 'patchwam' /
                    'patchwam_robotwin_official_benchmark_tasks.json')
DEFAULT_CONFIG = (REPO_ROOT / 'configs' / 'patchwam' /
                  'patchwam_robotwin_official_benchmark.py')
DEFAULT_ASSETS_ROOT = Path(
    os.environ.get('PATCHWAM_ASSETS_ROOT',
                   '/home/chuong/workspace/wam/ImageWAM'))
DEFAULT_SUITE_PARENT = (Path('runs') / 'evaluation' /
                        'patchwam_robotwin_official')

# cfg-option keys the launcher owns. Forwarding them to every case would
# silently point all cases at the same directory or run id.
OWNED_CFG_KEYS = ('eval.output_dir', 'eval.run_id')


class BenchmarkError(Exception):
    """Invalid launcher input; always reported before any GPU work starts."""


@dataclass(frozen=True)
class Case:
    """One (task, condition) evaluation case."""

    task: str
    condition: str
    task_config: str

    @property
    def slug(self) -> str:
        return f'{self.task}__{self.condition}'


@dataclass
class Plan:
    """Resolved, validated suite plan."""

    manifest: Dict[str, Any]
    cases: List[Case]
    config: Path
    python: str
    episodes: int
    suite_root: Path
    ckpt_path: Optional[str]
    forwarded_cfg_options: List[str]
    protocol_label: str
    custom_reasons: List[str]
    robotwin_root: Path
    cuda_visible_devices: str
    timestamp: str

    def command_for(self, case: Case) -> List[str]:
        command = [self.python, str(REPO_ROOT / 'scripts' / 'eval.py'),
                   '--config', str(self.config)]
        if self.ckpt_path:
            command += ['--ckpt-path', str(self.ckpt_path)]
        cfg_options = [
            f'eval.task_name={case.task}',
            f'eval.task_config={case.task_config}',
            f'eval.episodes_per_task={self.episodes}',
            f'eval.robotwin_root={self.robotwin_root}',
            f'eval.output_dir={self.suite_root}',
            f'eval.run_id={case.slug}',
            *self.forwarded_cfg_options,
        ]
        return command + ['--cfg-options', *cfg_options]

    def case_dir(self, case: Case) -> Path:
        return self.suite_root / case.slug


def _as_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_manifest(path: Path) -> Dict[str, Any]:
    """Load and structurally validate the static task manifest."""
    with open(path, 'r', encoding='utf-8') as stream:
        manifest = json.load(stream)
    for key in ('tasks', 'conditions', 'protocol'):
        if key not in manifest:
            raise BenchmarkError(f'manifest {path} is missing {key!r}')
    tasks = list(manifest['tasks'])
    if not tasks:
        raise BenchmarkError(f'manifest {path} lists no tasks')
    if len(tasks) != len(set(tasks)):
        raise BenchmarkError(f'manifest {path} has duplicate task names')
    conditions = manifest['conditions']
    if not conditions or not isinstance(conditions, dict):
        raise BenchmarkError(f'manifest {path} has no conditions')
    for name, spec in conditions.items():
        if not isinstance(spec, dict) or not spec.get('task_config'):
            raise BenchmarkError(
                f'manifest condition {name!r} needs a task_config')
    return manifest


def default_robotwin_root() -> Path:
    override = os.environ.get('PATCHWAM_ROBOTWIN_ROOT')
    if override:
        return Path(override)
    return DEFAULT_ASSETS_ROOT / 'third_party' / 'RoboTwin'


def validate_environment(manifest: Dict[str, Any], root: Path) -> None:
    """Check the current checkout actually has the modules a case needs."""
    if not root.is_dir():
        raise BenchmarkError(
            f'RoboTwin checkout not found: {root} (set PATCHWAM_ROBOTWIN_ROOT '
            'or --robotwin-root)')
    missing: List[str] = []
    for task in manifest['tasks']:
        if not (root / 'envs' / f'{task}.py').is_file():
            missing.append(f'envs/{task}.py')
        if not (root / 'description' / 'task_instruction' /
                f'{task}.json').is_file():
            missing.append(f'description/task_instruction/{task}.json')
    for spec in manifest['conditions'].values():
        config_name = spec['task_config']
        if not (root / 'task_config' / f'{config_name}.yml').is_file():
            missing.append(f'task_config/{config_name}.yml')
    if missing:
        preview = ', '.join(missing[:8])
        extra = '' if len(missing) <= 8 else f' (+{len(missing) - 8} more)'
        raise BenchmarkError(
            f'{len(missing)} RoboTwin files listed by the benchmark manifest '
            f'are missing from {root}: {preview}{extra}')


def condition_from_task_config(task_config: str,
                               manifest: Dict[str, Any]) -> str:
    for name, spec in manifest['conditions'].items():
        if spec['task_config'] == task_config:
            return name
    expected = sorted(spec['task_config']
                      for spec in manifest['conditions'].values())
    raise BenchmarkError(
        f'unsupported eval.task_config={task_config!r}; expected one of '
        f'{expected}')


def parse_cfg_options(
    tokens: Sequence[str],
    manifest: Dict[str, Any],
) -> tuple:
    """Split ``--cfg-options`` into (task filter, condition filter, episodes,
    forwarded, robotwin_root) while intercepting the keys the launcher owns.

    ``eval.task_name``/``eval.task_config`` are interpreted as suite filters
    instead of being forwarded to every case verbatim (which would turn all
    cases into the same task). ``eval.episodes_per_task`` overrides the episode
    count and is labelled as a custom protocol. ``eval.robotwin_root`` selects
    the simulator checkout the launcher validates *and* forwards, so the two
    can never diverge. Everything else is forwarded unchanged.
    """
    task_filter: List[str] = []
    condition_filter: List[str] = []
    episodes: Optional[int] = None
    robotwin_root: Optional[str] = None
    forwarded: List[str] = []
    for token in tokens:
        if '=' not in token:
            raise BenchmarkError(
                f'--cfg-options expects KEY=VALUE pairs, got {token!r}')
        key, _, value = token.partition('=')
        key = key.strip()
        value = value.strip()
        if not key:
            raise BenchmarkError(f'invalid --cfg-options entry {token!r}')
        if key in OWNED_CFG_KEYS:
            raise BenchmarkError(
                f'--cfg-options {key} is owned by the benchmark launcher; use '
                '--suite-root instead so every case gets its own directory')
        if key == 'eval.task_name':
            task_filter.append(value)
        elif key == 'eval.task_config':
            condition_filter.append(
                condition_from_task_config(value, manifest))
        elif key == 'eval.episodes_per_task':
            try:
                episodes = int(value)
            except ValueError as error:
                raise BenchmarkError(
                    f'eval.episodes_per_task must be an integer, got '
                    f'{value!r}') from error
            if episodes < 1:
                raise BenchmarkError('eval.episodes_per_task must be positive')
        elif key == 'eval.robotwin_root':
            if not value:
                raise BenchmarkError('eval.robotwin_root must not be empty')
            robotwin_root = value
        else:
            forwarded.append(token)
    return task_filter, condition_filter, episodes, forwarded, robotwin_root


def resolve_cases(
    manifest: Dict[str, Any],
    task_filter: Iterable[str],
    condition_filter: Iterable[str],
) -> List[Case]:
    tasks = list(manifest['tasks'])
    conditions = list(manifest['conditions'])
    requested_tasks = list(dict.fromkeys(task_filter))
    requested_conditions = list(dict.fromkeys(condition_filter))
    unknown_tasks = [task for task in requested_tasks if task not in tasks]
    if unknown_tasks:
        raise BenchmarkError(
            f'unknown benchmark task(s): {unknown_tasks}; see '
            f'{DEFAULT_MANIFEST.name} for the released list')
    unknown_conditions = [
        name for name in requested_conditions if name not in conditions
    ]
    if unknown_conditions:
        raise BenchmarkError(
            f'unknown condition(s): {unknown_conditions}; expected '
            f'{conditions}')
    selected_tasks = requested_tasks or tasks
    selected_conditions = requested_conditions or conditions
    return [
        Case(task=task,
             condition=condition,
             task_config=manifest['conditions'][condition]['task_config'])
        for task in selected_tasks for condition in selected_conditions
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='patchwam_official_benchmark.py',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG),
                        help='Native eval config (default: benchmark config).')
    parser.add_argument('--manifest', default=str(DEFAULT_MANIFEST),
                        help='Static task manifest JSON.')
    parser.add_argument('--python', default=sys.executable,
                        help='Policy interpreter used for scripts/eval.py.')
    parser.add_argument('--task', action='append', default=[],
                        metavar='NAME',
                        help='Restrict to a task (repeatable).')
    parser.add_argument('--condition', action='append', default=[],
                        choices=('clean', 'random'),
                        help='Restrict to a condition (repeatable).')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Episodes per case (default: 100, the protocol).')
    parser.add_argument('--smoke', action='store_true',
                        help='Single first task/clean case, one episode.')
    parser.add_argument('--suite-root', default=None,
                        help='Suite output directory (default: timestamped).')
    parser.add_argument('--ckpt-path', default=None,
                        help='Checkpoint override forwarded to '
                             'scripts/eval.py.')
    parser.add_argument('--robotwin-root', default=None,
                        help='RoboTwin checkout used for the static checks.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the plan and per-case commands; '
                             'run nothing.')
    parser.add_argument('--list', dest='list_only', action='store_true',
                        help='List the planned cases only.')
    parser.add_argument('--cfg-options', nargs='+', action='append',
                        default=None, metavar='KEY=VALUE',
                        help='Forwarded to scripts/eval.py (put it last).')
    return parser


def build_plan(args: argparse.Namespace,
               manifest: Dict[str, Any]) -> Plan:
    cfg_tokens = [token for group in (args.cfg_options or [])
                  for token in group]
    (cfg_tasks, cfg_conditions, cfg_episodes, forwarded,
     cfg_robotwin_root) = parse_cfg_options(cfg_tokens, manifest)

    cli_tasks = list(args.task)
    cli_conditions = list(args.condition)
    cli_episodes = args.episodes
    if args.smoke:
        cli_tasks = cli_tasks or [manifest['tasks'][0]]
        cli_conditions = cli_conditions or ['clean']
        if cli_episodes is None:
            cli_episodes = 1

    if cli_episodes is not None and cfg_episodes is not None \
            and cli_episodes != cfg_episodes:
        raise BenchmarkError(
            f'--episodes {cli_episodes} conflicts with --cfg-options '
            f'eval.episodes_per_task={cfg_episodes}')
    protocol_episodes = int(manifest['protocol'].get('episodes_per_task', 100))
    episodes = cli_episodes if cli_episodes is not None else (
        cfg_episodes if cfg_episodes is not None else protocol_episodes)
    if episodes < 1:
        raise BenchmarkError('episodes per case must be positive')

    cases = resolve_cases(manifest, cli_tasks + cfg_tasks,
                          cli_conditions + cfg_conditions)

    cli_root = Path(args.robotwin_root).expanduser() if args.robotwin_root \
        else None
    cfg_root = Path(cfg_robotwin_root).expanduser() if cfg_robotwin_root \
        else None
    if cli_root is not None and cfg_root is not None \
            and cli_root.resolve() != cfg_root.resolve():
        raise BenchmarkError(
            f'conflicting RoboTwin roots: --robotwin-root {cli_root} vs '
            f'--cfg-options eval.robotwin_root={cfg_root}')
    robotwin_root = (cli_root or cfg_root or default_robotwin_root()).resolve()
    validate_environment(manifest, robotwin_root)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    if args.suite_root:
        suite_root = _as_path(args.suite_root)
    else:
        suite_root = _as_path(str(DEFAULT_SUITE_PARENT / timestamp))

    custom_reasons: List[str] = []
    if Path(args.config).resolve() != DEFAULT_CONFIG.resolve():
        custom_reasons.append(f'config={args.config}')
    if len(cases) != len(manifest['tasks']) * len(manifest['conditions']):
        custom_reasons.append('task/condition subset')
    if episodes != protocol_episodes:
        custom_reasons.append(f'episodes={episodes}')
    if forwarded:
        custom_reasons.append('forwarded cfg-options: ' + ', '.join(forwarded))
    if args.ckpt_path:
        custom_reasons.append(f'checkpoint={args.ckpt_path}')
    if args.smoke:
        custom_reasons.append('smoke')

    return Plan(
        manifest=manifest,
        cases=cases,
        config=Path(args.config),
        python=str(args.python),
        episodes=episodes,
        suite_root=suite_root,
        ckpt_path=args.ckpt_path,
        forwarded_cfg_options=forwarded,
        protocol_label='custom protocol' if custom_reasons else 'official',
        custom_reasons=custom_reasons,
        robotwin_root=robotwin_root,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
        timestamp=timestamp,
    )


def format_plan(plan: Plan, show_commands: bool) -> str:
    protocol = plan.manifest['protocol']
    lines = [
        'PatchWAM RoboTwin 2.0 official benchmark',
        f'  protocol:  {plan.protocol_label}' + (
            f" ({'; '.join(plan.custom_reasons)})"
            if plan.custom_reasons else ''),
        f'  config:    {plan.config}',
        f'  manifest:  {DEFAULT_MANIFEST}',
        f'  robotwin:  {plan.robotwin_root}',
        f'  settings:  horizon={protocol["action_horizon"]} '
        f'replan={protocol["replan_steps"]} '
        f'denoise={protocol["num_inference_steps"]} '
        f'instruction_type={protocol["instruction_type"]} '
        f'expert_check={protocol["expert_check"]} seed={protocol["seed"]} '
        f'selection={protocol["instruction_selection"]}',
        f'  python:    {plan.python}',
        f'  cuda:      {plan.cuda_visible_devices}',
        f'  cases:     {len(plan.cases)}',
        f'  episodes:  {len(plan.cases) * plan.episodes} '
        f'({plan.episodes} per case)',
        f'  suite:     {plan.suite_root}',
    ]
    for index, case in enumerate(plan.cases, start=1):
        lines.append(f'  [{index:03d}/{len(plan.cases)}] {case.slug} '
                     f'-> {plan.case_dir(case)}')
        if show_commands:
            lines.append('        ' + ' '.join(plan.command_for(case)))
    return '\n'.join(lines)


def _case_record(case: Case, plan: Plan, status: str,
                 results: Optional[Dict[str, Any]] = None,
                 error: Optional[str] = None) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        'task': case.task,
        'condition': case.condition,
        'task_config': case.task_config,
        'slug': case.slug,
        'dir': str(plan.case_dir(case)),
        'status': status,
    }
    if results is not None:
        record.update({
            'requested_episodes': results.get('requested_episodes'),
            'completed_episodes': results.get('completed_episodes'),
            'successes': results.get('successes'),
            'success_rate': results.get('success_rate'),
        })
    if error is not None:
        record['error'] = error
    return record


def _read_case_results(case_dir: Path) -> Optional[Dict[str, Any]]:
    path = case_dir / 'results.json'
    if not path.is_file():
        return None
    with open(path, 'r', encoding='utf-8') as stream:
        return json.load(stream)


def _summarize(records: Sequence[Dict[str, Any]],
               status: str) -> Dict[str, Any]:
    completed = [record for record in records
                 if record['status'] == 'complete']
    return {
        'status': status,
        'cases_planned': len(records),
        'cases_complete': len(completed),
        'episodes_requested': sum(
            int(record.get('requested_episodes') or 0)
            for record in completed),
        'episodes_completed': sum(
            int(record.get('completed_episodes') or 0)
            for record in completed),
        'successes': sum(int(record.get('successes') or 0)
                         for record in completed),
    }


def _write_metadata(plan: Plan, records: Sequence[Dict[str, Any]],
                    status: str) -> None:
    totals = _summarize(records, status)
    totals['success_rate'] = (totals['successes'] /
                              totals['episodes_completed']
                              if totals['episodes_completed'] else None)
    protocol = plan.manifest['protocol']
    payload = {
        'benchmark': plan.manifest.get('benchmark', 'RoboTwin 2.0'),
        'suite': plan.manifest.get('suite', 'patchwam_robotwin_official'),
        'created': plan.timestamp,
        'protocol_label': plan.protocol_label,
        'custom_reasons': plan.custom_reasons,
        'protocol': {
            'tasks': len(plan.manifest['tasks']),
            'conditions': list(plan.manifest['conditions']),
            'episodes_per_task': plan.episodes,
            'expected_cases': len(plan.manifest['tasks']) *
            len(plan.manifest['conditions']),
            'expected_episodes': (
                len(plan.manifest['tasks']) *
                len(plan.manifest['conditions']) * plan.episodes),
            'seed': protocol['seed'],
            'action_horizon': protocol['action_horizon'],
            'replan_steps': protocol['replan_steps'],
            'num_inference_steps': protocol['num_inference_steps'],
            'instruction_type': protocol['instruction_type'],
            'instruction_selection': protocol['instruction_selection'],
            'expert_check': protocol['expert_check'],
        },
        'config': str(plan.config),
        'python': plan.python,
        'cuda_visible_devices': plan.cuda_visible_devices,
        'forwarded_cfg_options': plan.forwarded_cfg_options,
        'command_example': (plan.command_for(plan.cases[0])
                            if plan.cases else None),
        'cases': list(records),
        'totals': totals,
    }
    plan.suite_root.mkdir(parents=True, exist_ok=True)
    with open(plan.suite_root / 'suite_metadata.json', 'w',
              encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2)


def check_collisions(plan: Plan) -> None:
    existing = [str(plan.case_dir(case)) for case in plan.cases
                if plan.case_dir(case).exists()]
    if existing:
        preview = ', '.join(existing[:5])
        raise BenchmarkError(
            f'{len(existing)} case directory(ies) already exist under '
            f'{plan.suite_root} ({preview}); refusing to overwrite results. '
            'Choose a different --suite-root.')


def execute_plan(plan: Plan,
                 runner=subprocess.run,
                 log=print) -> int:
    """Run every case sequentially, aggregating results into the suite root."""
    if not plan.cases:
        raise BenchmarkError('no cases selected')
    check_collisions(plan)
    plan.suite_root.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    env = dict(os.environ)
    env.setdefault('CUDA_VISIBLE_DEVICES', plan.cuda_visible_devices)
    _write_metadata(plan, records, 'running')
    for index, case in enumerate(plan.cases, start=1):
        command = plan.command_for(case)
        log(f'[{index}/{len(plan.cases)}] {case.slug}')
        log('  ' + ' '.join(command))
        try:
            completed = runner(command, cwd=str(REPO_ROOT), env=env)
        except KeyboardInterrupt:
            records.append(_case_record(case, plan, 'interrupted'))
            _write_metadata(plan, records, 'interrupted')
            raise
        if completed.returncode != 0:
            records.append(_case_record(
                case, plan, 'failed',
                error='process exited with returncode '
                f'{completed.returncode}'))
            _write_metadata(plan, records, 'failed')
            log(f'case {case.slug} failed '
                f'(returncode {completed.returncode}); stopping the suite')
            return completed.returncode
        results = _read_case_results(plan.case_dir(case))
        if results is None:
            records.append(_case_record(
                case, plan, 'failed', error='results.json was not written'))
            _write_metadata(plan, records, 'failed')
            log(f'case {case.slug} produced no results.json; stopping')
            return 1
        requested = results.get('requested_episodes')
        completed_episodes = results.get('completed_episodes')
        if int(requested or 0) != plan.episodes \
                or int(completed_episodes or 0) != plan.episodes:
            records.append(_case_record(
                case, plan, 'incomplete', results=results,
                error=(f'expected {plan.episodes} episodes, results.json '
                       f'reports requested={requested} '
                       f'completed={completed_episodes}')))
            _write_metadata(plan, records, 'failed')
            log(f'case {case.slug} did not complete the protocol '
                f'(requested={requested} completed={completed_episodes} '
                f'expected={plan.episodes}); stopping the suite')
            return 1
        records.append(_case_record(case, plan, 'complete', results=results))
        _write_metadata(plan, records, 'running')
        log(f'  -> {results.get("successes")}/'
            f'{results.get("completed_episodes")} success ({case.slug})')
    _write_metadata(plan, records, 'complete')
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args, extras = parser.parse_known_args(argv)
    if extras:
        print(f'patchwam-official-benchmark: unexpected arguments: {extras}; '
              'pass eval overrides through --cfg-options (place it last)',
              file=sys.stderr)
        return 2
    try:
        manifest = load_manifest(_as_path(args.manifest))
        plan = build_plan(args, manifest)
        if args.list_only:
            for case in plan.cases:
                print(f'{case.task} {case.condition}')
            return 0
        if args.dry_run:
            print(format_plan(plan, show_commands=True))
            return 0
        return execute_plan(plan)
    except BenchmarkError as error:
        print(f'patchwam-official-benchmark: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
