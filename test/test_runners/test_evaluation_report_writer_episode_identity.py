from fluxvla.engines.runners.serving import evaluation_report_writer


def _started_report_writer(tmp_path):
    report_writer = evaluation_report_writer.FluxVLAEvaluationReportWriter(
        result_root=tmp_path,
        config_path=tmp_path / 'config.py',
        ckpt_path=tmp_path / 'model.pt',
        eval_config={
            'model_family': 'pi05',
            'num_trials_per_task': 1,
            'task_suite_name': 'robodojo',
        },
        report_kind='robodojo',
        logger=lambda message: None,
    )
    task = {
        'task_id': 'pick',
        'task_index': 0,
        'description': 'Pick',
        'metadata': {
            'benchmark': 'robodojo',
        },
    }
    run_start = dict(
        schema_version='1.0',
        run_name='test-run',
        base_seed=7,
        episodes_per_task=1,
        max_episode_steps=10,
        execute_horizon=None,
        total_tasks=1,
        total_episodes=1,
        tasks=[task],
    )
    response = report_writer.process_event(
        'run_start',
        request_id='request-1',
        run_session_id='session',
        sequence=1,
        payload=run_start,
    )
    assert response['accepted'] is True
    return report_writer


def _episode_start_payload():
    return {
        'task_id': 'pick',
        'episode_index': 0,
        'policy_seed': 7,
        'environment_seed': None,
        'environment_case_index': None,
        'episode_id': 'session:1',
        'started_at': '2026-10-10T00:00:00+00:00',
    }


def test_report_writer_accepts_episode_identity(tmp_path):
    report_writer = _started_report_writer(tmp_path)

    response = report_writer.process_event(
        'episode_start',
        request_id='request-2',
        run_session_id='session',
        sequence=2,
        payload=_episode_start_payload(),
    )

    assert response['accepted'] is True
