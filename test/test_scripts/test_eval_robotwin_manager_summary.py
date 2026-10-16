"""Worker-summary selection when a retry reuses the output directory.

A retry shares OUTPUT_DIR with earlier attempts, so the manager must pick the
attempt that belongs to the checkpoint it is about to evaluate. These tests run
the real ``worker_summary_path`` helper (extracted verbatim from the manager)
against fake summary trees; no simulator or GPU is involved.

Re-summarizing that same OUTPUT_DIR has a second failure mode: results from an
earlier run under a different layout are still on disk. The summarizer CLI
tests build those trees and assert that the run's ``tasks.txt`` manifest, not
the leftover directories, decides what is reported.
"""

import csv
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANAGER = REPO_ROOT / 'scripts' / 'eval_robotwin_manager.sh'
SUMMARIZER = REPO_ROOT / 'tools' / 'summarize_robotwin_eval_results.py'


def _extract_function(name: str) -> str:
    lines = MANAGER.read_text(encoding='utf-8').splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith(f'{name}() {{'))
    end = next(i for i in range(start, len(lines)) if lines[i] == '}')
    return '\n'.join(lines[start:end + 1])


def _write_summary(path: Path, successes: int, episodes: int,
                   checkpoint: str = '') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            'ckpt': checkpoint,
            'task_results': {
                'rotate_qrcode': {
                    'status': 'COMPLETED' if episodes else 'MISSING',
                    'successes': successes,
                    'total_episodes': episodes,
                    'success_rate': (successes / episodes * 100
                                     if episodes else None),
                }}}),
        encoding='utf-8')


def _run_selector(root: Path, tag: str, task: str = 'rotate_qrcode') -> str:
    script = '\n'.join([
        'set -euo pipefail',
        'declare -A JOB_ROOT',
        f'JOB_ROOT[{task}]={shlex.quote(str(root))}',
        f'ckpt_tag={shlex.quote(tag)}',
        _extract_function('worker_summary_path'),
        f'worker_summary_path {shlex.quote(task)}',
    ])
    # The manager helper runs under the same PATH contract as the manager
    # itself: `python` must resolve to the configured interpreter.
    env = dict(os.environ)
    env['PATH'] = os.pathsep.join(
        [os.path.dirname(sys.executable), env.get('PATH', '')])
    result = subprocess.run(['bash', '-c', script], capture_output=True,
                            text=True, check=False, timeout=60, env=env)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture
def attempt_tree(tmp_path):
    """Old-attempt tree (0 episodes) that is *newer* by mtime than the new one."""
    root = tmp_path / 'run'
    old = root / 'workers' / 'rotate_qrcode' / 'eval_runs' / 'step_140000' \
        / 'EVAL-old-attempt' / 'summary.json'
    new = root / 'workers' / 'rotate_qrcode' / 'eval_runs' \
        / 'step_140000_native' / 'EVAL-new-attempt' / 'summary.json'
    _write_summary(old, 0, 0)
    _write_summary(new, 3, 10)
    os.utime(new, (1_700_000_000, 1_700_000_000))
    os.utime(old, (1_800_000_000, 1_800_000_000))
    return root, old, new


def test_selector_ignores_a_newer_summary_from_another_checkpoint(attempt_tree):
    root, old, new = attempt_tree
    selected = _run_selector(root, 'step_140000_native')
    assert Path(selected) == new
    assert 'step_140000/' not in selected


def test_selector_returns_nothing_when_no_attempt_matches_the_checkpoint(
        attempt_tree):
    root, old, _ = attempt_tree
    assert _run_selector(root, 'a_different_checkpoint') == ''


def test_selector_prefers_the_newest_attempt_of_the_same_checkpoint(
        attempt_tree):
    root, _, new = attempt_tree
    older = new.parent.parent / 'EVAL-earlier-attempt' / 'summary.json'
    _write_summary(older, 1, 10)
    os.utime(older, (1_600_000_000, 1_600_000_000))
    assert Path(_run_selector(root, 'step_140000_native')) == new


def _load_summarizer():
    spec = importlib.util.spec_from_file_location('summarize_robotwin_eval',
                                                  SUMMARIZER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aggregator_picks_the_newest_attempt(tmp_path):
    root = tmp_path / 'run'
    older = root / 'workers' / 'rotate_qrcode' / 'eval_runs' / 'ckpt' \
        / 'EVAL-earlier' / 'summary.json'
    newer = root / 'workers' / 'rotate_qrcode' / 'eval_runs' / 'ckpt' \
        / 'EVAL-later' / 'summary.json'
    _write_summary(older, 0, 0)
    _write_summary(newer, 3, 10)
    os.utime(older, (1_600_000_000, 1_600_000_000))
    os.utime(newer, (1_700_000_000, 1_700_000_000))
    summaries = _load_summarizer()._load_worker_summaries(root)
    stats = summaries['rotate_qrcode']['task_results']['rotate_qrcode']
    assert stats['successes'] == 3
    assert stats['total_episodes'] == 10


def test_aggregator_ignores_a_newer_different_checkpoint(tmp_path):
    root = tmp_path / 'run'
    workers = root / 'workers' / 'rotate_qrcode' / 'eval_runs'
    current = workers / 'current' / 'attempt' / 'summary.json'
    other = workers / 'other' / 'attempt' / 'summary.json'
    checkpoint = str(tmp_path / 'current.safetensors')
    _write_summary(current, 3, 10, checkpoint)
    _write_summary(other, 9, 10, str(tmp_path / 'other.safetensors'))
    os.utime(current, (1_600_000_000, 1_600_000_000))
    os.utime(other, (1_800_000_000, 1_800_000_000))
    summaries = _load_summarizer()._load_worker_summaries(root, checkpoint)
    stats = summaries['rotate_qrcode']['task_results']['rotate_qrcode']
    assert stats['successes'] == 3


CKPT_NAME = 'step_140000_native.safetensors'
SUITE_GROUPS = {'clean': 'Easy', 'random': 'Hard'}


def _write_worker_summary(workers_root: Path, task: str, successes: int,
                          episodes: int, condition: str = 'clean',
                          checkpoint: str = '', attempt: str = 'EVAL-attempt',
                          mtime: float = None) -> Path:
    """Write one worker summary the way ``RobotwinEvalRunner`` lays it out."""
    path = (workers_root / task / 'eval_runs' / Path(checkpoint).stem
            / attempt / 'summary.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            'run_id': f'{attempt}-{task}',
            'ckpt': checkpoint,
            'config': 'test_config',
            'task_suite_name': condition,
            'instruction_type': 'unseen',
            'seed': 0,
            'eval_chunk_size': 16,
            'max_episode_steps': None,
            'trials_per_task': 10,
            'task_list': [task],
            'duration_scope': 'task_wall_time_including_expert_validation',
            'group_stats': {
                SUITE_GROUPS[condition]: {
                    'total_successes': successes,
                    'total_trials': episodes,
                }
            },
            'task_results': {
                task: {
                    'status': 'COMPLETED' if episodes else 'MISSING',
                    'successes': successes,
                    'total_episodes': episodes,
                    'success_rate': (successes / episodes * 100
                                     if episodes else None),
                    'duration': 100.0,
                    'gpu_id': 0,
                }
            },
        }),
        encoding='utf-8')
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def run_summarizer():
    """Run the real summarizer CLI with no Feishu credentials in the env."""

    def run(run_dir: Path, output_dir: Path, ckpt: str, *extra: str,
            title: str = 'Results') -> subprocess.CompletedProcess:
        env = {
            key: value for key, value in os.environ.items()
            if key not in ('FEISHU_SHEET_URL', 'FEISHU_APP_ID',
                           'FEISHU_APP_SECRET', 'FEISHU_TIMEOUT', 'CKPT',
                           'CONFIG')
        }
        command = [
            sys.executable, str(SUMMARIZER),
            '--run-dir', str(run_dir),
            '--output-dir', str(output_dir),
            '--title', title,
            '--ckpt', ckpt,
            *extra,
        ]
        return subprocess.run(command, capture_output=True, text=True,
                              check=False, timeout=120, env=env)

    return run


def _summary_row(summary_csv: Path, metric: str) -> List[str]:
    rows = [row for row in csv.reader(summary_csv.read_text().splitlines())
            if row and row[0] == metric]
    assert len(rows) == 1, rows
    return rows[0][1:]


def _reused_multi_condition_run(tmp_path: Path):
    """A reused OUTPUT_DIR: stale flat workers plus the qualified manifest."""
    root = tmp_path / 'my_run'
    ckpt = str(tmp_path / CKPT_NAME)
    (root / 'clean').mkdir(parents=True)
    (root / 'random').mkdir(parents=True)
    (root / 'tasks.txt').write_text(
        'clean/beat_block_hammer\nrandom/beat_block_hammer\n',
        encoding='utf-8')
    (root / 'clean' / 'tasks.txt').write_text('beat_block_hammer\n',
                                              encoding='utf-8')
    (root / 'random' / 'tasks.txt').write_text('beat_block_hammer\n',
                                               encoding='utf-8')
    # An earlier single-condition turn left complete flat results behind.
    _write_worker_summary(root / 'workers', 'turn_switch', 7, 10,
                          checkpoint=ckpt, attempt='stale-turn_switch')
    _write_worker_summary(root / 'workers', 'rotate_qrcode', 8, 10,
                          checkpoint=ckpt, attempt='stale-rotate_qrcode')
    # So did an earlier selection of a different task.
    _write_worker_summary(root / 'clean' / 'workers', 'adjust_bottle', 10, 10,
                          checkpoint=ckpt, attempt='stale-adjust_bottle')
    _write_worker_summary(root / 'random' / 'workers', 'adjust_bottle', 8, 10,
                          condition='random', checkpoint=ckpt,
                          attempt='stale-adjust_bottle')
    # The current run's results cover exactly the manifest's jobs.
    _write_worker_summary(root / 'clean' / 'workers', 'beat_block_hammer', 10,
                          10, checkpoint=ckpt,
                          attempt='current-beat_block_hammer')
    _write_worker_summary(root / 'random' / 'workers', 'beat_block_hammer', 6,
                          10, condition='random', checkpoint=ckpt,
                          attempt='current-beat_block_hammer')
    # The earlier flat summarization also left its per-task table at the root.
    with open(root / 'task_success_rates.csv', 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Task', 'Successes', 'Episodes', 'Success Rate (%)'])
        writer.writerows([
            ['clean/beat_block_hammer', 0, 0, ''],
            ['random/beat_block_hammer', 0, 0, ''],
        ])
    return root, ckpt


def test_cli_reports_the_qualified_manifest_not_stale_flat_workers(
        tmp_path, run_summarizer):
    root, ckpt = _reused_multi_condition_run(tmp_path)
    result = run_summarizer(root, root, ckpt, title=CKPT_NAME)
    assert result.returncode == 0, result.stderr

    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    # Two condition rates, and no invented combined success rate.
    assert 'overall' not in summary
    easy = summary['conditions']['clean']
    hard = summary['conditions']['random']
    assert (easy['difficulty'], easy['successes'], easy['episodes'],
            easy['success_rate']) == ('Easy', 10, 10, 100.0)
    assert (hard['difficulty'], hard['successes'], hard['episodes'],
            hard['success_rate']) == ('Hard', 6, 10, 60.0)
    assert summary['group_stats'] == {
        'Easy': {'total_successes': 10, 'total_trials': 10},
        'Hard': {'total_successes': 6, 'total_trials': 10},
    }
    episodes = [int(value) for value in
                _summary_row(root / 'summary.csv', 'Episodes')]
    successes = [int(value) for value in
                 _summary_row(root / 'summary.csv', 'Successes')]
    assert sum(episodes) == 20
    assert sum(successes) == 16

    for condition, difficulty, expected in (('clean', 'Easy', 10),
                                            ('random', 'Hard', 6)):
        inner = json.loads((root / condition / 'summary.json')
                           .read_text(encoding='utf-8'))
        assert set(inner['task_results']) == {'beat_block_hammer'}
        stats = inner['task_results']['beat_block_hammer']
        assert stats['status'] == 'COMPLETED'
        assert stats['total_episodes'] == 10
        assert stats['successes'] == expected
        assert inner['group_stats'][difficulty] == {
            'total_successes': expected,
            'total_trials': 10,
        }

    reported = '\n'.join((root / name).read_text(encoding='utf-8')
                         for name in ('summary.json', 'summary.txt',
                                      'summary.csv'))
    for stale in ('turn_switch', 'rotate_qrcode', 'adjust_bottle'):
        assert stale not in reported

    with open(root / 'task_success_rates.csv', newline='') as handle:
        task_rows = list(csv.reader(handle))
    assert task_rows == [
        ['Task', 'Successes', 'Episodes', 'Success Rate (%)'],
        ['clean/beat_block_hammer', '10', '10', '100.00'],
        ['random/beat_block_hammer', '6', '10', '60.00'],
    ]
    assert 'clean/beat_block_hammer,10,10,100.00' in result.stdout


def test_cli_keeps_the_flat_layout_when_the_manifest_is_bare(
        tmp_path, run_summarizer):
    root = tmp_path / 'my_run'
    ckpt = str(tmp_path / CKPT_NAME)
    root.mkdir(parents=True)
    (root / 'tasks.txt').write_text('beat_block_hammer\n', encoding='utf-8')
    _write_worker_summary(root / 'workers', 'beat_block_hammer', 10, 10,
                          checkpoint=ckpt, attempt='current',
                          mtime=1_700_000_000)
    # Older nested trees from a previous multi-condition run, and newer by
    # mtime so only the manifest can keep them from winning.
    _write_worker_summary(root / 'clean' / 'workers', 'beat_block_hammer', 7,
                          10, checkpoint=ckpt, attempt='stale',
                          mtime=1_800_000_000)
    _write_worker_summary(root / 'random' / 'workers', 'beat_block_hammer', 3,
                          10, condition='random', checkpoint=ckpt,
                          attempt='stale', mtime=1_800_000_000)

    result = run_summarizer(root, root, ckpt, title=CKPT_NAME)
    assert result.returncode == 0, result.stderr

    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    assert 'conditions' not in summary
    stats = summary['task_results']['beat_block_hammer']
    assert (stats['successes'], stats['total_episodes']) == (10, 10)
    assert summary['overall']['success_rate'] == 100.0
    assert not (root / 'clean' / 'summary.json').exists()
    assert not (root / 'random' / 'summary.json').exists()
    assert 'Overall statistics' in (root / 'summary.txt').read_text(
        encoding='utf-8')


def test_cli_loads_a_lone_nested_condition_without_a_root_manifest(
        tmp_path, run_summarizer):
    root = tmp_path / 'my_run'
    ckpt = str(tmp_path / CKPT_NAME)
    (root / 'clean' / 'workers').mkdir(parents=True)
    (root / 'clean' / 'tasks.txt').write_text('rotate_qrcode\n',
                                              encoding='utf-8')
    _write_worker_summary(root / 'clean' / 'workers', 'rotate_qrcode', 3, 10,
                          checkpoint=ckpt, attempt='only')

    result = run_summarizer(root, root, ckpt, title=CKPT_NAME)
    assert result.returncode == 0, result.stderr

    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    assert 'conditions' not in summary
    assert summary['task_results']['rotate_qrcode']['successes'] == 3
    assert summary['task_results']['rotate_qrcode']['status'] == 'COMPLETED'
    assert summary['overall']['success_rate'] == 30.0
    assert summary['overall']['completed_tasks'] == 1

    with open(root / 'task_success_rates.csv', newline='') as handle:
        task_rows = list(csv.reader(handle))
    assert task_rows == [
        ['Task', 'Successes', 'Episodes', 'Success Rate (%)'],
        ['rotate_qrcode', '3', '10', '30.00'],
    ]


def test_cli_bare_tasks_apply_to_every_detected_condition(tmp_path,
                                                          run_summarizer):
    root, ckpt = _reused_multi_condition_run(tmp_path)
    result = run_summarizer(root, root, ckpt, '--tasks', 'beat_block_hammer',
                            title=CKPT_NAME)
    assert result.returncode == 0, result.stderr

    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    assert summary['conditions']['clean']['successes'] == 10
    assert summary['conditions']['random']['successes'] == 6


def test_cli_qualified_tasks_select_a_subset_of_the_manifest(tmp_path,
                                                             run_summarizer):
    root, ckpt = _reused_multi_condition_run(tmp_path)
    result = run_summarizer(root, root, ckpt, '--tasks',
                            'clean/beat_block_hammer', title=CKPT_NAME)
    assert result.returncode == 0, result.stderr

    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    assert 'conditions' not in summary
    assert set(summary['task_results']) == {'beat_block_hammer'}
    assert summary['overall']['success_rate'] == 100.0


def _write_manifest(run_dir: Path, manifest: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'tasks.txt').write_text(manifest, encoding='utf-8')


def test_collect_conditions_prefers_the_qualified_manifest(tmp_path):
    root = tmp_path / 'run'
    _write_manifest(root, 'clean/a\nrandom/b\nclean/c\n')
    (root / 'workers').mkdir()
    assert _load_summarizer()._collect_conditions(root) == ['clean', 'random']


def test_collect_conditions_keeps_the_flat_layout_for_a_bare_manifest(
        tmp_path):
    root = tmp_path / 'run'
    _write_manifest(root, 'beat_block_hammer\n')
    (root / 'clean' / 'workers').mkdir(parents=True)
    (root / 'random' / 'workers').mkdir(parents=True)
    assert _load_summarizer()._collect_conditions(root) == []


def test_collect_conditions_falls_back_to_the_directory_layout(tmp_path):
    module = _load_summarizer()
    flat = tmp_path / 'flat'
    (flat / 'workers').mkdir(parents=True)
    assert module._collect_conditions(flat) == []
    nested = tmp_path / 'nested'
    (nested / 'random' / 'workers').mkdir(parents=True)
    assert module._collect_conditions(nested) == ['random']
