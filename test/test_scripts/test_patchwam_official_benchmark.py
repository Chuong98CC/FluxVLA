# Copyright 2026 Limx Dynamics
"""Official PatchWAM benchmark launcher tests (no GPU, no simulator)."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / 'tools' / 'patchwam_official_benchmark.py'
MANIFEST_PATH = (REPO_ROOT / 'configs' / 'patchwam' /
                 'patchwam_robotwin_official_benchmark_tasks.json')


@pytest.fixture(scope='module')
def benchmark():
    spec = importlib.util.spec_from_file_location(
        'patchwam_official_benchmark', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def manifest(benchmark):
    return benchmark.load_manifest(MANIFEST_PATH)


@pytest.fixture()
def robotwin(tmp_path, manifest):
    """Minimal static checkout listing every module the manifest needs."""
    root = tmp_path / 'RoboTwin'
    (root / 'envs').mkdir(parents=True)
    (root / 'description' / 'task_instruction').mkdir(parents=True)
    (root / 'task_config').mkdir(parents=True)
    for task in manifest['tasks']:
        (root / 'envs' / f'{task}.py').write_text('')
        (root / 'description' / 'task_instruction' /
         f'{task}.json').write_text('{}')
    for spec in manifest['conditions'].values():
        config_name = spec['task_config']
        (root / 'task_config' / f'{config_name}.yml').write_text('{}\n')
    return root


def _plan(benchmark, manifest, robotwin, tmp_path, *extra):
    argv = ['--robotwin-root', str(robotwin), '--suite-root',
            str(tmp_path / 'suite'), *extra]
    return benchmark.build_plan(benchmark.build_parser().parse_args(argv),
                                manifest)


def test_manifest_matches_released_suite(manifest):
    assert len(manifest['tasks']) == 50
    assert len(set(manifest['tasks'])) == 50
    assert set(manifest['conditions']) == {'clean', 'random'}
    assert manifest['conditions']['clean']['task_config'] == 'demo_clean'
    assert manifest['conditions']['random']['task_config'] == 'demo_randomized'
    protocol = manifest['protocol']
    assert protocol['episodes_per_task'] == 100
    assert protocol['total_episodes'] == 10000
    assert protocol['seed'] == 0
    assert protocol['action_horizon'] == 16
    assert protocol['replan_steps'] == 16
    assert protocol['num_inference_steps'] == 10
    assert protocol['instruction_type'] == 'unseen'
    assert protocol['expert_check'] is True
    assert protocol['instruction_selection'] == 'official'


def test_plan_expands_to_100_cases_and_10000_episodes(benchmark, manifest,
                                                      robotwin, tmp_path):
    plan = _plan(benchmark, manifest, robotwin, tmp_path)
    assert len(plan.cases) == 100
    assert len(plan.cases) * plan.episodes == 10000
    assert plan.episodes == 100
    slugs = [case.slug for case in plan.cases]
    assert len(set(slugs)) == 100
    assert len({case.task for case in plan.cases}) == 50
    assert plan.protocol_label == 'official'


def test_case_command_carries_per_case_overrides(benchmark, manifest, robotwin,
                                                 tmp_path):
    plan = _plan(benchmark, manifest, robotwin, tmp_path)
    case = plan.cases[7]
    command = plan.command_for(case)
    assert command[command.index('--config') + 1].endswith(
        'patchwam_robotwin_official_benchmark.py')
    assert '--cfg-options' in command
    assert f'eval.task_name={case.task}' in command
    assert f'eval.task_config={case.task_config}' in command
    assert 'eval.episodes_per_task=100' in command
    assert f'eval.robotwin_root={robotwin.resolve()}' in command
    assert f'eval.run_id={case.slug}' in command
    assert f'eval.output_dir={plan.suite_root}' in command
    assert plan.case_dir(case) == plan.suite_root / case.slug


def test_cfg_options_task_name_filters_instead_of_repeating(benchmark,
                                                            manifest):
    parsed = benchmark.parse_cfg_options(
        ['eval.task_name=click_alarmclock',
         'eval.task_config=demo_randomized',
         'eval.num_inference_steps=8'], manifest)
    tasks, conditions, episodes, forwarded, root = parsed
    assert tasks == ['click_alarmclock']
    assert conditions == ['random']
    assert episodes is None
    assert root is None
    assert forwarded == ['eval.num_inference_steps=8']


def test_robotwin_root_is_forwarded_to_every_case(benchmark, manifest,
                                                  robotwin, tmp_path):
    plan = _plan(benchmark, manifest, robotwin, tmp_path)
    assert plan.robotwin_root == robotwin.resolve()
    for case in plan.cases:
        assert f'eval.robotwin_root={robotwin.resolve()}' in \
            plan.command_for(case)


def test_cfg_option_robotwin_root_is_used_and_forwarded(benchmark, manifest,
                                                        robotwin, tmp_path):
    plan = _plan(benchmark, manifest, robotwin, tmp_path, '--cfg-options',
                 f'eval.robotwin_root={robotwin}')
    assert plan.robotwin_root == robotwin.resolve()
    assert f'eval.robotwin_root={robotwin.resolve()}' in \
        plan.command_for(plan.cases[0])


def test_conflicting_robotwin_roots_are_rejected(benchmark, robotwin,
                                                 tmp_path, capsys):
    other = tmp_path / 'other-RoboTwin'
    other.mkdir()
    rc = benchmark.main([
        '--robotwin-root', str(robotwin), '--suite-root',
        str(tmp_path / 'suite'), '--dry-run', '--cfg-options',
        f'eval.robotwin_root={other}'])
    assert rc == 2
    assert 'conflicting RoboTwin roots' in capsys.readouterr().err


def test_cfg_options_owned_keys_are_rejected(benchmark, manifest):
    for token in ('eval.output_dir=/tmp/x', 'eval.run_id=foo'):
        with pytest.raises(benchmark.BenchmarkError):
            benchmark.parse_cfg_options([token], manifest)
    with pytest.raises(benchmark.BenchmarkError):
        benchmark.parse_cfg_options(['eval.task_name'], manifest)


def test_cfg_options_task_name_filter_selects_one_case(benchmark, manifest,
                                                       robotwin, tmp_path,
                                                       capsys):
    rc = benchmark.main([
        '--robotwin-root', str(robotwin), '--suite-root',
        str(tmp_path / 'suite'), '--dry-run', '--cfg-options',
        'eval.task_name=click_alarmclock', 'eval.task_config=demo_clean'])
    assert rc == 0
    plan_text = capsys.readouterr().out
    assert 'click_alarmclock__clean' in plan_text
    assert 'click_alarmclock__random' not in plan_text


def test_unknown_task_filter_fails_before_running(benchmark, robotwin,
                                                  tmp_path, capsys):
    rc = benchmark.main(['--robotwin-root', str(robotwin), '--task',
                         'not_a_task'])
    assert rc == 2
    assert 'unknown benchmark task' in capsys.readouterr().err


def test_unexpected_arguments_fail_fast(benchmark, capsys):
    assert benchmark.main(['--not-an-option']) == 2
    assert 'unexpected arguments' in capsys.readouterr().err


def test_dry_run_lists_every_case_and_writes_nothing(benchmark, robotwin,
                                                     tmp_path, capsys):
    suite = tmp_path / 'suite'
    rc = benchmark.main(['--robotwin-root', str(robotwin), '--suite-root',
                         str(suite), '--dry-run'])
    assert rc == 0
    out = capsys.readouterr().out
    case_lines = [
        line for line in out.splitlines() if line.strip().startswith('[')
    ]
    assert len(case_lines) == 100
    slugs = [line.split('] ', 1)[1].split(' ->', 1)[0] for line in case_lines]
    assert len(set(slugs)) == 100
    assert 'adjust_bottle__clean' in slugs
    assert 'turn_switch__random' in slugs
    assert not suite.exists()


def test_list_prints_one_line_per_case(benchmark, robotwin, capsys):
    rc = benchmark.main(['--robotwin-root', str(robotwin), '--list'])
    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 100
    assert lines.count('adjust_bottle clean') == 1
    assert lines.count('adjust_bottle random') == 1


def test_collisions_refuse_to_overwrite(benchmark, robotwin, tmp_path, capsys):
    suite = tmp_path / 'suite'
    (suite / 'adjust_bottle__clean').mkdir(parents=True)
    rc = benchmark.main(['--robotwin-root', str(robotwin), '--suite-root',
                         str(suite), '--task', 'adjust_bottle'])
    assert rc == 2
    assert 'refusing to overwrite' in capsys.readouterr().err


def test_custom_protocol_is_labelled(benchmark, manifest, robotwin, tmp_path):
    plan = _plan(benchmark, manifest, robotwin, tmp_path,
                 '--episodes', '3', '--task', 'adjust_bottle')
    assert plan.protocol_label == 'custom protocol'
    assert any('episodes=3' in reason for reason in plan.custom_reasons)
    assert any('subset' in reason for reason in plan.custom_reasons)


def test_episode_override_conflict_is_rejected(benchmark, robotwin, tmp_path,
                                               capsys):
    rc = benchmark.main(['--robotwin-root', str(robotwin), '--episodes', '4',
                         '--cfg-options', 'eval.episodes_per_task=5'])
    assert rc == 2
    assert 'conflicts' in capsys.readouterr().err


class _Result:
    def __init__(self, returncode):
        self.returncode = returncode


def test_failure_propagates_and_is_recorded(benchmark, manifest, robotwin,
                                            tmp_path):
    plan = _plan(benchmark, manifest, robotwin, tmp_path, '--task',
                 'adjust_bottle')
    calls = []

    def runner(command, cwd=None, env=None):
        calls.append(command)
        return _Result(7)

    rc = benchmark.execute_plan(plan, runner=runner, log=lambda *a, **k: None)
    assert rc == 7
    assert len(calls) == 1
    metadata_path = plan.suite_root / 'suite_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    assert metadata['totals']['status'] == 'failed'
    assert metadata['cases'][0]['status'] == 'failed'
    assert metadata['cases'][0]['task'] == 'adjust_bottle'


def test_success_aggregates_case_results(benchmark, manifest, robotwin,
                                         tmp_path):
    plan = _plan(benchmark, manifest, robotwin, tmp_path, '--task',
                 'adjust_bottle')

    def runner(command, cwd=None, env=None):
        run_id = next(token.split('=', 1)[1] for token in command
                      if token.startswith('eval.run_id='))
        case_dir = plan.suite_root / run_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / 'results.json').write_text(json.dumps({
            'requested_episodes': plan.episodes,
            'completed_episodes': plan.episodes,
            'successes': 1,
            'success_rate': 0.01,
        }))
        return _Result(0)

    rc = benchmark.execute_plan(plan, runner=runner, log=lambda *a, **k: None)
    assert rc == 0
    metadata_path = plan.suite_root / 'suite_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    assert metadata['totals']['status'] == 'complete'
    assert metadata['totals']['cases_complete'] == 2
    assert metadata['totals']['episodes_requested'] == 2 * plan.episodes
    assert metadata['totals']['episodes_completed'] == 2 * plan.episodes
    assert metadata['totals']['successes'] == 2
    assert metadata['cases'][0]['slug'] == 'adjust_bottle__clean'


def _write_results_runner(plan, requested, completed):
    """Fake runner that writes a results.json with the given counts."""

    def runner(command, cwd=None, env=None):
        run_id = next(token.split('=', 1)[1] for token in command
                      if token.startswith('eval.run_id='))
        case_dir = plan.suite_root / run_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / 'results.json').write_text(json.dumps({
            'requested_episodes': requested,
            'completed_episodes': completed,
            'successes': 0,
            'success_rate': 0.0,
        }))
        return _Result(0)

    return runner


@pytest.mark.parametrize('requested,completed', [(100, 99), (50, 50)])
def test_incomplete_case_fails_suite_with_preserved_metadata(
        benchmark, manifest, robotwin, tmp_path, requested, completed):
    plan = _plan(benchmark, manifest, robotwin, tmp_path, '--task',
                 'adjust_bottle')
    assert plan.episodes == 100
    runner = _write_results_runner(plan, requested, completed)
    rc = benchmark.execute_plan(plan, runner=runner, log=lambda *a, **k: None)
    assert rc == 1
    metadata_path = plan.suite_root / 'suite_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    assert metadata['totals']['status'] == 'failed'
    assert metadata['totals']['cases_complete'] == 0
    assert metadata['cases'][0]['status'] == 'incomplete'
    assert metadata['cases'][0]['requested_episodes'] == requested
    assert metadata['cases'][0]['completed_episodes'] == completed
    assert 'expected 100 episodes' in metadata['cases'][0]['error']


def test_complete_case_is_not_flagged_incomplete(benchmark, manifest, robotwin,
                                                 tmp_path):
    plan = _plan(benchmark, manifest, robotwin, tmp_path, '--task',
                 'adjust_bottle', '--condition', 'clean')
    runner = _write_results_runner(plan, plan.episodes, plan.episodes)
    rc = benchmark.execute_plan(plan, runner=runner, log=lambda *a, **k: None)
    assert rc == 0
    metadata = json.loads(
        (plan.suite_root / 'suite_metadata.json').read_text())
    assert metadata['totals']['status'] == 'complete'
    assert metadata['cases'][0]['status'] == 'complete'
