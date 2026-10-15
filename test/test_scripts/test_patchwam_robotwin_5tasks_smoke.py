"""Focused units for the five-task RoboTwin-Clean smoke helper."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_PATH = REPO_ROOT / 'tools' / 'patchwam_robotwin_5tasks_smoke.py'
TASKS = ['turn_switch', 'rotate_qrcode', 'blocks_ranking_size',
         'move_stapler_pad', 'hanging_mug']


def smoke_module():
    spec = importlib.util.spec_from_file_location('patchwam_5tasks_smoke', SMOKE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_root(parent, task, episodes=2, fps=15):
    """A tiny LeRobot v2.1 root: metadata only, no data/video payload."""
    root = parent / task
    (root / 'meta').mkdir(parents=True)
    info = {
        'codebase_version': 'v2.1',
        'total_episodes': episodes,
        'fps': fps,
        'features': {
            'observation.images.cam_high': {},
            'observation.images.cam_left_wrist': {},
            'observation.images.cam_right_wrist': {},
            'action': {'shape': [14]},
            'observation.state': {'shape': [14]},
        },
    }
    (root / 'meta/info.json').write_text(json.dumps(info))
    rows = [json.dumps({'episode_index': index, 'length': 3,
                        'tasks': [f'{task} instruction {index}']})
            for index in range(episodes)]
    (root / 'meta/episodes.jsonl').write_text('\n'.join(rows) + '\n')
    return root


def test_selection_reports_per_folder_counts_and_provenance(tmp_path):
    smoke = smoke_module()
    first = write_root(tmp_path, 'blocks_ranking_size')
    second = write_root(tmp_path, 'hanging_mug', episodes=3)
    missing = tmp_path / 'turn_switch'
    report = smoke.selection_report([first, second, missing])
    assert report['status'] == 'missing_folders'
    assert report['missing'] == [str(missing)]
    assert report['configured_roots'] == [str(first), str(second), str(missing)]
    folders = {folder['root']: folder for folder in report['folders']}
    assert folders[str(missing)]['exists'] is False
    assert folders[str(first)]['episodes'] == 2
    assert folders[str(first)]['fps'] == 15
    assert folders[str(first)]['codebase_version'] == 'v2.1'
    assert folders[str(first)]['cameras'] == [
        'observation.images.cam_high', 'observation.images.cam_left_wrist',
        'observation.images.cam_right_wrist']
    assert folders[str(first)]['first_instruction'] == 'blocks_ranking_size instruction 0'
    assert folders[str(first)]['last_instruction'] == 'blocks_ranking_size instruction 1'
    assert folders[str(second)]['episodes'] == 3


def test_missing_folders_exit_nonzero_naming_every_absent_root(tmp_path):
    smoke = smoke_module()
    present = write_root(tmp_path, 'hanging_mug')
    absent = [tmp_path / 'turn_switch', tmp_path / 'rotate_qrcode']
    with pytest.raises(SystemExit) as error:
        smoke.require_roots([present, *absent])
    message = str(error.value)
    assert 'Missing RoboTwin-Clean task roots' in message
    for root in absent:
        assert str(root) in message
    # A complete preflight is not an error.
    smoke.require_roots([present])


def test_real_config_lists_the_five_requested_roots():
    smoke = smoke_module()
    roots = smoke.dataset_roots(smoke._config())
    assert [root.name for root in roots] == TASKS
    assert {str(root.parent) for root in roots} == {'/data/RoboTwin/RoboTwin-Clean'}
