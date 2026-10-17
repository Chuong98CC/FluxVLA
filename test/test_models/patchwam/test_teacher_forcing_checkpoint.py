"""Checkpoint initialization and resume identity for teacher forcing."""

import copy

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from fluxvla.engines import build_vla_from_cfg
from fluxvla.models.heads import teacher_forcing_patchwam_head  # noqa: F401
from fluxvla.models.vlas import teacher_forcing_patchwam_vla  # noqa: F401
from fluxvla.models.backbones.visions.frozen_vision import SpatialFeatureOutput


SPEC = dict(
    name='radio',
    feature_dim=8,
    representation_id='fake-radio-checkpoint',
    checkpoint_sha256='1' * 64,
    stream='spatial_features',
    preprocessing='unit_range_once',
    input_size=(8, 12),
    native_grid=(2, 3),
)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_dim = 8
        self.weight = nn.Parameter(torch.ones(1))

    def feature_grid(self, height, width):
        return 2, 3

    def encode_features(self, images):
        batch = images.shape[0]
        return SpatialFeatureOutput(
            tokens=torch.zeros(batch, 6, 8), grid=(2, 3))


def teacher_config(**overrides):
    values = dict(
        encoders=[dict(SPEC)],
        predictor=dict(type='linear'),
    )
    values.update(overrides)
    return values


def teacher_vla(path=None, mode=None, config=None, encoder=None):
    values = dict(
        type='TeacherForcingPatchWAMVLA',
        action_horizon=16,
        num_inference_steps=2,
        vla_head=dict(
            type='TeacherForcingPatchWAMHead',
            denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128,
                          text_dim=32, width=32, heads=4, depth=2),
            action_dim=14, text_dim=32, proprio_dim=14, token_dim=128,
            teacher_forcing=config or teacher_config()),
    )
    if path is not None:
        values['pretrained_name_or_path'] = str(path)
        values['teacher_checkpoint_mode'] = mode
    if encoder is not None:
        values['teacher_extractors'] = {'radio': encoder}
        values['teacher_provenance'] = {
            'radio': dict(
                representation_id=SPEC['representation_id'],
                test_contract='explicit-unit-fixture')}
    return build_vla_from_cfg(values)


def baseline_vla():
    return build_vla_from_cfg(dict(
        type='PatchWAMVLA',
        action_horizon=16,
        num_inference_steps=2,
        vla_head=dict(
            type='PatchWAMHead',
            denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128,
                          text_dim=32, width=32, heads=4, depth=2),
            action_dim=14, text_dim=32, proprio_dim=14, token_dim=128),
    ))


def test_base_init_allows_only_new_teacher_branches(tmp_path):
    source = baseline_vla()
    source.vla_head.state_projection.weight.data.fill_(0.125)
    path = tmp_path / 'base.safetensors'
    save_file(source.state_dict(), str(path))
    target = teacher_vla(path, 'base_init')
    report = target.from_pretrained()
    assert report['mode'] == 'base_init'
    torch.testing.assert_close(
        target.vla_head.state_projection.weight,
        source.vla_head.state_projection.weight)
    assert torch.count_nonzero(
        target.vla_head.teacher_current_projectors['radio'].weight) == 0

    state = copy.deepcopy(source.state_dict())
    state.pop('vla_head.state_projection.weight')
    missing_path = tmp_path / 'missing_inherited.safetensors'
    save_file(state, str(missing_path))
    broken = teacher_vla(missing_path, 'base_init')
    before = broken.vla_head.state_projection.weight.detach().clone()
    with pytest.raises(ValueError, match='state_projection'):
        broken.from_pretrained()
    torch.testing.assert_close(
        broken.vla_head.state_projection.weight, before)


def test_resume_round_trip_and_missing_branch_rejection(tmp_path):
    source = teacher_vla()
    with torch.no_grad():
        source.vla_head.teacher_current_projectors['radio'].weight.normal_()
        source.vla_head.teacher_conditioners['radio'].weight.normal_()
        source.vla_head.teacher_predictors['radio'].weight.normal_()
        source.vla_head.state_projection.weight.fill_(0.375)
    path = tmp_path / 'teacher.safetensors'
    save_file(source.state_dict(), str(path))
    target = teacher_vla(path, 'resume')
    target.from_pretrained()
    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value,
                                   rtol=0, atol=0)

    missing = copy.deepcopy(source.state_dict())
    missing.pop('vla_head.teacher_current_projectors.radio.weight')
    missing_path = tmp_path / 'missing_branch.safetensors'
    save_file(missing, str(missing_path))
    broken = teacher_vla(missing_path, 'resume')
    with pytest.raises(ValueError, match='teacher_current_projectors'):
        broken.from_pretrained()


def test_strict_false_resume_still_checks_branch_identity():
    source = teacher_vla()
    missing = copy.deepcopy(source.state_dict())
    missing.pop('vla_head.teacher_conditioners.radio.weight')
    target = teacher_vla()
    target.teacher_checkpoint_mode = 'resume'
    with pytest.raises(ValueError, match='teacher_conditioners'):
        target.load_state_dict(missing, strict=False)


def test_representation_and_predictor_changes_reject_resume(tmp_path):
    source = teacher_vla()
    path = tmp_path / 'teacher.safetensors'
    save_file(source.state_dict(), str(path))

    changed_identity = dict(SPEC)
    changed_identity['checkpoint_sha256'] = '2' * 64
    identity_model = teacher_vla(
        path, 'resume', config=teacher_config(
            encoders=[changed_identity]))
    with pytest.raises(ValueError, match='contract mismatch'):
        identity_model.from_pretrained()

    predictor_model = teacher_vla(
        path, 'resume', config=teacher_config(
            predictor=dict(type='mlp2', hidden_dim=16)))
    with pytest.raises(
            ValueError, match='missing required tensors|contract mismatch'):
        predictor_model.from_pretrained()


def test_strict_false_resume_rejects_digest_mismatch():
    source = teacher_vla()
    state = copy.deepcopy(source.state_dict())
    state['vla_head.teacher_contract_digest'] = torch.zeros_like(
        state['vla_head.teacher_contract_digest'])
    target = teacher_vla()
    target.teacher_checkpoint_mode = 'resume'
    with pytest.raises(ValueError, match='contract mismatch'):
        target.load_state_dict(state, strict=False)


def test_load_patchwam_state_dict_strict_false_cannot_bypass_resume():
    source = teacher_vla()
    state = copy.deepcopy(source.state_dict())
    state.pop('vla_head.teacher_conditioners.radio.weight')
    target = teacher_vla()
    target.teacher_checkpoint_mode = 'resume'
    with pytest.raises(ValueError, match='teacher_conditioners'):
        target.load_patchwam_state_dict(
            state, strict=False,
            allow_missing=target.teacher_new_model_keys())


def test_direct_head_load_requires_a_complete_matching_teacher_state():
    source = teacher_vla()
    state = copy.deepcopy(source.vla_head.state_dict())
    target = teacher_vla()
    target.vla_head.load_state_dict(state, strict=True)
    missing = copy.deepcopy(state)
    missing.pop('teacher_conditioners.radio.weight')
    with pytest.raises(ValueError, match='teacher_conditioners'):
        target.vla_head.load_state_dict(missing, strict=False)


def test_direct_vla_load_validates_teacher_identity_and_supports_restore():
    source = teacher_vla()
    target = teacher_vla()
    target.load_state_dict(source.state_dict(), strict=False)
    torch.testing.assert_close(
        target.vla_head.teacher_current_projectors['radio'].weight,
        source.vla_head.teacher_current_projectors['radio'].weight)
    wrong = copy.deepcopy(source.state_dict())
    wrong['vla_head.teacher_contract_digest'] = torch.zeros_like(
        wrong['vla_head.teacher_contract_digest'])
    with pytest.raises(ValueError, match='contract mismatch'):
        target.load_state_dict(wrong, strict=False)

    baseline = baseline_vla()
    with pytest.raises(ValueError, match='missing all teacher tensors'):
        target.load_state_dict(baseline.state_dict(), strict=False)


def test_parent_recursion_cannot_bypass_teacher_preflight():
    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.vla = model

    source = teacher_vla()
    wrapped = Wrapper(teacher_vla())
    wrong = copy.deepcopy(source.state_dict())
    wrong['vla_head.teacher_contract_digest'] = torch.zeros_like(
        wrong['vla_head.teacher_contract_digest'])
    prefixed = {f'vla.{name}': value for name, value in wrong.items()}
    with pytest.raises(ValueError, match='contract mismatch'):
        wrapped.load_state_dict(prefixed, strict=False)


def test_base_init_does_not_mix_a_mismatched_teacher_branch(tmp_path):
    source = teacher_vla()
    state = copy.deepcopy(source.state_dict())
    state['vla_head.teacher_contract_digest'] = torch.zeros_like(
        state['vla_head.teacher_contract_digest'])
    path = tmp_path / 'mismatched_teacher.safetensors'
    save_file(state, str(path))
    target = teacher_vla(path, 'base_init')
    with pytest.raises(ValueError, match='contract mismatch'):
        target.from_pretrained()


def test_teacher_runtime_is_outside_policy_state():
    encoder = Encoder()
    model = teacher_vla(encoder=encoder)
    keys = set(model.state_dict())
    assert not any('teacher_extractors' in key for key in keys)
    assert not any(key.startswith('teacher_encoder') for key in keys)
    assert all('radio' not in key or key.startswith(
        ('vla_head.teacher_current_projectors',
         'vla_head.teacher_conditioners',
         'vla_head.teacher_predictors',
         'vla_head.teacher_feature_norms'))
        for key in keys if 'radio' in key)
    model.close_teacher_extractors()


def test_explicit_checkpoint_mode_is_required_for_pretrained_teacher_model(
        tmp_path):
    path = tmp_path / 'teacher.safetensors'
    save_file(teacher_vla().state_dict(), str(path))
    with pytest.raises(ValueError, match='teacher_checkpoint_mode'):
        teacher_vla(path, None)
