"""CPU units for the advisory expected-global-batch check and LR auto-scaling.

``expected_global_batch_size`` is an opt-in recipe guard: when the configured
value differs from ``per_device_batch_size x world_size x
grad_accumulation_steps`` the runner warns once on rank zero and keeps running
with the effective size. ``auto_scale_lr`` (default on) additionally rescales
the configured optimizer LR by ``effective / expected`` before the optimizer
and scheduler are built. Only the heavyweight VLA/metric/collator builders and
the distributed rank probe are substituted.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch.nn as nn

import fluxvla.optimizers  # noqa: F401  (LR scheduler registration)
from fluxvla.engines import utils as engines_utils
from fluxvla.engines.runners import base_train_runner
from fluxvla.engines.runners.base_train_runner import (
    BaseTrainRunner, check_expected_global_batch_size)
from fluxvla.engines.utils import build_lr_scheduler_from_cfg


class _Warnings:
    def __init__(self):
        self.messages = []

    def __call__(self, message, *args, **kwargs):
        self.messages.append(str(message))


def _make_overwatch(*, rank_zero=True, world_size=4):
    warning_recorder = _Warnings()
    info_recorder = _Warnings()
    return SimpleNamespace(is_rank_zero=lambda: rank_zero,
                           local_rank=lambda: 0,
                           rank=lambda: 0 if rank_zero else 1,
                           world_size=lambda: world_size,
                           warning=warning_recorder,
                           info=info_recorder,
                           error=_Warnings(),
                           debug=_Warnings()), warning_recorder, info_recorder


class _StubVLA:
    all_module_keys = ['head']
    llm_backbone = None
    vlm_backbone = None


class _Runner(BaseTrainRunner):
    def save_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    def clip_grad_norm(self):
        raise NotImplementedError

    def _load_model_state(self, checkpoint_model_state):
        raise NotImplementedError

    def _load_optimizer_state(self, checkpoint_optimizer_state):
        return False


class _TinyVLA(nn.Module):
    """Two named modules so param-wise LR groups stay distinguishable."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)


_DEFAULT_OPTIMIZER = dict(type='AdamW', lr=1.0e-4)


def _build_runner(*, expected, per_device_batch_size, grad_accumulation_steps,
                  rank_zero=True, world_size=4, auto_scale_lr=True,
                  optimizer=None):
    """Initialize the real base runner with stubbed heavy dependencies."""
    overwatch, recorder, info_recorder = _make_overwatch(
        rank_zero=rank_zero, world_size=world_size)
    cfg = SimpleNamespace(
        train_dataloader=SimpleNamespace(
            per_device_batch_size=per_device_batch_size),
        model=SimpleNamespace(type='Stub'),
        filename='/tmp/recipe.py')
    with mock.patch.object(base_train_runner, 'overwatch', overwatch), \
            mock.patch.object(engines_utils.builder, 'build_vla_from_cfg',
                              return_value=_StubVLA()), \
            mock.patch.object(engines_utils.builder, 'build_metric_from_cfg',
                              return_value=SimpleNamespace()), \
            mock.patch.object(engines_utils.builder, 'build_collator_from_cfg',
                              return_value=lambda batch: batch):
        runner = _Runner(cfg=cfg, device_id=0,
                         collator=dict(type='DictCollator'),
                         sampler='distributed', metric=dict(type='VLAMetric'),
                         optimizer=(_DEFAULT_OPTIMIZER if optimizer is None
                                    else optimizer),
                         max_epochs=1,
                         grad_accumulation_steps=grad_accumulation_steps,
                         expected_global_batch_size=expected,
                         auto_scale_lr=auto_scale_lr,
                         enable_mixed_precision_training=False)
    return runner, recorder, info_recorder


def _optimizer_param_group_lrs(optimizer_cfg, vla):
    """Build param groups through the real policy and read their LRs."""
    policy = build_lr_scheduler_from_cfg({'type': 'constant'})
    runner = SimpleNamespace(optimizer_cfg=optimizer_cfg, vla=vla)
    optimizer = policy.build_optimizer(runner)
    return [group['lr'] for group in optimizer.param_groups]


def test_helper_warns_with_factors_and_both_numbers():
    recorder = _Warnings()
    warned = check_expected_global_batch_size(
        1024, 173, per_device_batch_size=1, world_size=173,
        grad_accumulation_steps=1, rank_zero=True, warning=recorder)
    assert warned is True
    assert len(recorder.messages) == 1
    message = recorder.messages[0]
    assert '1024' in message and '173' in message
    assert 'per_device_batch_size' in message
    assert 'world_size' in message
    assert 'grad_accumulation_steps' in message
    assert 'continuing' in message.lower() or 'continue' in message.lower()


@pytest.mark.parametrize('expected,actual', [(256, 256), (None, 173)])
def test_helper_stays_quiet_on_match_or_unspecified(expected, actual):
    recorder = _Warnings()
    warned = check_expected_global_batch_size(
        expected, actual, per_device_batch_size=1, world_size=actual,
        grad_accumulation_steps=1, rank_zero=True, warning=recorder)
    assert recorder.messages == []
    assert warned is (expected is not None and expected != actual)


def test_helper_stays_quiet_off_rank_zero():
    recorder = _Warnings()
    warned = check_expected_global_batch_size(
        8, 16, per_device_batch_size=2, world_size=4,
        grad_accumulation_steps=2, rank_zero=False, warning=recorder)
    assert recorder.messages == []
    assert warned is True


def test_runner_warns_and_keeps_running_on_mismatch():
    runner, recorder, _ = _build_runner(expected=8, per_device_batch_size=2,
                                       grad_accumulation_steps=4)
    assert runner.global_batch_size == 2 * 4 * 4
    assert runner.expected_global_batch_size == 8
    assert len(recorder.messages) == 1
    assert '8' in recorder.messages[0] and '32' in recorder.messages[0]


def test_runner_stays_quiet_when_the_expectation_matches():
    runner, recorder, _ = _build_runner(expected=32, per_device_batch_size=2,
                                       grad_accumulation_steps=4)
    assert runner.global_batch_size == 32
    assert recorder.messages == []


def test_runner_stays_quiet_without_an_expectation():
    runner, recorder, _ = _build_runner(expected=None, per_device_batch_size=1,
                                       grad_accumulation_steps=1)
    assert runner.expected_global_batch_size is None
    assert runner.global_batch_size == 4
    assert recorder.messages == []


def test_nonzero_ranks_do_not_warn_but_still_initialize():
    runner, recorder, _ = _build_runner(expected=8, per_device_batch_size=2,
                                       grad_accumulation_steps=4,
                                       rank_zero=False)
    assert runner.global_batch_size == 32
    assert recorder.messages == []


@pytest.mark.parametrize('expected,factor', [(8, 4.0), (64, 0.5)])
def test_auto_scale_lr_rescales_base_and_paramwise_lrs(expected, factor):
    """The effective / expected factor covers base LR and overrides alike."""
    optimizer = dict(type='AdamW', lr=1.0e-4, weight_decay=0.01,
                     paramwise_learning_rate={'head': 2.0e-4})
    runner, recorder, info_recorder = _build_runner(
        expected=expected, per_device_batch_size=2,
        grad_accumulation_steps=4, optimizer=optimizer)
    assert runner.global_batch_size == 32
    assert runner.lr_scale_factor == pytest.approx(factor)
    assert runner.optimizer_cfg['lr'] == pytest.approx(1.0e-4 * factor)
    assert runner.optimizer_cfg['paramwise_learning_rate'][
        'head'] == pytest.approx(2.0e-4 * factor)

    param_group_lrs = _optimizer_param_group_lrs(runner.optimizer_cfg,
                                                 _TinyVLA())
    assert sorted(set(param_group_lrs)) == pytest.approx(
        sorted({1.0e-4 * factor, 2.0e-4 * factor}))

    # The recipe's own numbers stay untouched and the warning still fires.
    assert optimizer['lr'] == 1.0e-4
    assert optimizer['paramwise_learning_rate'] == {'head': 2.0e-4}
    assert len(recorder.messages) == 1
    assert '32' in recorder.messages[0]
    # The applied factor is visible in the run log.
    assert len(info_recorder.messages) == 1
    assert f'{factor:g}' in info_recorder.messages[0]


def test_auto_scale_lr_disabled_keeps_the_configured_lrs():
    optimizer = dict(type='AdamW', lr=1.0e-4,
                     paramwise_learning_rate={'head': 2.0e-4})
    runner, recorder, info_recorder = _build_runner(
        expected=8, per_device_batch_size=2, grad_accumulation_steps=4,
        auto_scale_lr=False, optimizer=optimizer)
    assert runner.lr_scale_factor is None
    assert runner.optimizer_cfg['lr'] == 1.0e-4
    assert runner.optimizer_cfg['paramwise_learning_rate'] == {'head': 2.0e-4}
    # Warning-only behavior is exactly the pre-existing contract.
    assert len(recorder.messages) == 1
    assert info_recorder.messages == []


@pytest.mark.parametrize('expected', [32, None])
def test_auto_scale_lr_is_inert_without_a_mismatch(expected):
    runner, recorder, info_recorder = _build_runner(
        expected=expected, per_device_batch_size=2,
        grad_accumulation_steps=4)
    assert runner.lr_scale_factor is None
    assert runner.optimizer_cfg['lr'] == _DEFAULT_OPTIMIZER['lr']
    assert runner.optimizer_cfg['paramwise_learning_rate'] == {}
    assert recorder.messages == []
    assert info_recorder.messages == []


def test_auto_scale_lr_factor_stays_applied_once():
    """Rebuilding the optimizer reuses the one scaled config."""
    runner, _, _ = _build_runner(expected=8, per_device_batch_size=2,
                                 grad_accumulation_steps=4)
    first = _optimizer_param_group_lrs(runner.optimizer_cfg, _TinyVLA())
    second = _optimizer_param_group_lrs(runner.optimizer_cfg, _TinyVLA())
    assert first == second
    assert first == pytest.approx([4.0e-4])
    assert runner.optimizer_cfg['lr'] == pytest.approx(4.0e-4)


@pytest.mark.parametrize('expected', [0, -4, 1.5, 'big', True])
def test_invalid_expectations_are_config_errors(expected):
    with pytest.raises(ValueError, match='expected_global_batch_size'):
        _build_runner(expected=expected, per_device_batch_size=1,
                      grad_accumulation_steps=1)


def test_ddp_runner_accepts_the_setting_and_rejects_nothing_else():
    import inspect

    from fluxvla.engines.runners.ddp_train_runner import DDPTrainRunner

    signature = inspect.signature(DDPTrainRunner.__init__)
    assert 'expected_global_batch_size' in signature.parameters
    assert signature.parameters[
        'expected_global_batch_size'].default is None
    assert signature.parameters['auto_scale_lr'].default is True
