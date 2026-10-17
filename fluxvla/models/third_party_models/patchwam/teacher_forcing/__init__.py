"""Teacher-feature conditioning contracts and policy helpers."""

from .contracts import (RasterLayout, TeacherEncoderSpec, TeacherForcingConfig,
                        build_feature_predictor,
                        masked_teacher_cosine_loss,
                        prepare_teacher_fusion_input,
                        teacher_policy_digest,
                        teacher_representation_fingerprint)
from .policy import TeacherForcingPolicyMixin

__all__ = [
    'RasterLayout',
    'TeacherEncoderSpec',
    'TeacherForcingConfig',
    'TeacherForcingPolicyMixin',
    'build_feature_predictor',
    'masked_teacher_cosine_loss',
    'prepare_teacher_fusion_input',
    'teacher_policy_digest',
    'teacher_representation_fingerprint',
]
