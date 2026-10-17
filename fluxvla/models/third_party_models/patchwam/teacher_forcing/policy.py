"""Polymorphic teacher-forcing behavior layered over the PatchWAM policy."""

from __future__ import annotations

import math
from contextlib import contextmanager
from itertools import pairwise
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from ..conditioning import PreparedDenoiserPrefix
from .contracts import (RasterLayout, TeacherForcingConfig,
                        build_feature_predictor,
                        masked_teacher_cosine_loss,
                        prepare_teacher_fusion_input,
                        teacher_policy_digest,
                        teacher_representation_fingerprint)
from ..geometry import raster_coordinates


class TeacherForcingPolicyMixin:
    """Add observed current features and predicted future feedback to PatchWAM."""

    def _init_teacher_policy(self, config: TeacherForcingConfig) -> None:
        parameter = next(self.denoiser.parameters(), None)
        device = torch.device('cpu') if parameter is None else parameter.device
        dtype = torch.float32 if parameter is None else parameter.dtype
        self.teacher_config = config
        hidden = getattr(self.denoiser, 'representation_dim', None)
        if type(hidden) is not int or hidden < 1:
            raise ValueError(
                'The denoiser must expose a positive representation_dim')
        self.teacher_hidden_dim = hidden
        self.teacher_feature_norms = nn.ModuleDict()
        self.teacher_current_projectors = nn.ModuleDict()
        self.teacher_conditioners = nn.ModuleDict()
        self.teacher_predictors = nn.ModuleDict()
        for spec in config.encoders:
            if config.current_conditioning or config.future_feedback:
                self.teacher_feature_norms[spec.name] = nn.LayerNorm(
                    spec.feature_dim, elementwise_affine=False,
                    eps=config.norm_eps)
            if config.current_conditioning:
                self.teacher_current_projectors[spec.name] = nn.Linear(
                    spec.feature_dim, hidden, bias=False)
                self._initialize_teacher_projector(
                    self.teacher_current_projectors[spec.name],
                    config.current_projector_init,
                    config.current_projector_init_scale)
            if config.future_feedback:
                self.teacher_conditioners[spec.name] = nn.Linear(
                    spec.feature_dim, hidden, bias=False)
                self._initialize_teacher_projector(
                    self.teacher_conditioners[spec.name],
                    config.feedback_projector_init,
                    config.feedback_projector_init_scale)
            if config.future_prediction:
                self.teacher_predictors[spec.name] = build_feature_predictor(
                    config.predictor_config, input_dim=hidden,
                    output_dim=spec.feature_dim)
        self.teacher_feature_norms.to(device=device, dtype=dtype)
        self.teacher_current_projectors.to(device=device, dtype=dtype)
        self.teacher_conditioners.to(device=device, dtype=dtype)
        self.teacher_predictors.to(device=device, dtype=dtype)
        if self.state_projection is not None:
            self.state_projection.to(device=device, dtype=dtype)
        digest = teacher_policy_digest(config)
        self.register_buffer(
            'teacher_contract_digest',
            torch.tensor(list(digest), dtype=torch.uint8),
            persistent=True)

    @staticmethod
    def _initialize_teacher_projector(module: nn.Linear, mode: str,
                                      scale: float | None) -> None:
        if mode == 'zero':
            nn.init.zeros_(module.weight)
            return
        if mode != 'scaled_xavier_uniform' or scale is None:
            raise ValueError('Unsupported teacher projector initialization')
        nn.init.xavier_uniform_(module.weight)
        with torch.no_grad():
            module.weight.mul_(float(scale))

    def _teacher_new_state_keys_from_modules(self) -> list[str]:
        names = []
        for container in ('teacher_current_projectors',
                          'teacher_conditioners', 'teacher_predictors'):
            module = getattr(self, container, None)
            if module is None:
                continue
            for local_name, _ in module.named_parameters():
                names.append(f'{container}.{local_name}')
        names.append('teacher_contract_digest')
        return sorted(names)

    def teacher_new_parameter_names(self) -> list[str]:
        """Exact M/G/J parameters that a base checkpoint may omit."""
        return [
            name for name in self._teacher_new_state_keys_from_modules()
            if name != 'teacher_contract_digest'
        ]

    def teacher_new_state_keys(self) -> list[str]:
        return list(self._teacher_new_state_keys_from_modules())

    def teacher_checkpoint_contract(self) -> dict[str, Any]:
        return {
            **self.teacher_config.contract(),
            'digest': bytes(
                self.teacher_contract_digest.detach().cpu().tolist()).hex(),
            'true_feature_prob': self.teacher_config.true_feature_prob,
            'self_cond_drop_prob':
            self.teacher_config.self_cond_drop_prob,
            'current_projector_init':
            self.teacher_config.current_projector_init,
            'feedback_projector_init':
            self.teacher_config.feedback_projector_init,
        }

    def _teacher_validate_encoded(
        self, batch: Mapping[str, Any], *, target: bool,
        future_length: int | None = None,
    ) -> dict[str, Any]:
        reference = batch['reference_tokens']
        batch_size = int(reference.shape[0])
        specs = {spec.name: spec for spec in self.teacher_config.encoders}
        current = batch.get('teacher_current_tokens')
        future = batch.get('teacher_future_tokens')
        current_layout_raw = batch.get('current_raster_layout')
        future_layout_raw = batch.get('future_raster_layout')
        current_valid = batch.get('teacher_current_valid')
        future_valid_teacher = batch.get('teacher_future_valid')
        current_fingerprints = batch.get('teacher_current_fingerprints')
        future_fingerprints = batch.get('teacher_future_fingerprints')

        def role_mapping(value, key: str, required: bool):
            if value is None:
                if required:
                    raise ValueError(
                        f'Missing {key} for encoders {sorted(specs)}')
                return None
            if not isinstance(value, Mapping):
                raise TypeError(f'{key} must map encoder names to tensors')
            names = set(value)
            expected = set(specs)
            if names != expected:
                raise ValueError(
                    f'{key} names must match configured encoders exactly; '
                    f'missing={sorted(expected - names)}, '
                    f'extra={sorted(names - expected)}')
            return dict(value)

        need_current = self.teacher_config.current_conditioning
        need_future = target and self.teacher_config.future_prediction
        current = role_mapping(current, 'teacher_current_tokens',
                               need_current)
        future = (
            role_mapping(future, 'teacher_future_tokens', need_future)
            if target else None)
        current_valid = role_mapping(
            current_valid, 'teacher_current_valid',
            need_current)
        future_valid_teacher = (
            role_mapping(
                future_valid_teacher, 'teacher_future_valid', need_future)
            if target else None)
        current_fingerprints = role_mapping(
            current_fingerprints, 'teacher_current_fingerprints',
            need_current)
        future_fingerprints = (
            role_mapping(
                future_fingerprints, 'teacher_future_fingerprints',
                need_future)
            if target else None)
        if not target:
            # Future targets are training-only inputs. Ignoring them here keeps
            # every inference path independent of teacher target keys.
            future = future_valid_teacher = future_fingerprints = None
            future_layout_raw = None
        if current is not None and current_layout_raw is None:
            raise ValueError(
                'teacher_current_tokens require an explicit '
                'current_raster_layout')
        if future is not None and future_layout_raw is None:
            raise ValueError(
                'teacher_future_tokens require an explicit '
                'future_raster_layout')
        current_layout = (
            None if current_layout_raw is None else
            RasterLayout.from_mapping(current_layout_raw))
        future_layout = (
            None if future_layout_raw is None else
            RasterLayout.from_mapping(future_layout_raw))

        def validate_layout(layout, role):
            if layout is None:
                return
            for spec in self.teacher_config.encoders:
                if layout.video_size != spec.input_size:
                    raise ValueError(
                        f'{role}_raster_layout input {layout.video_size} '
                        f'does not match encoder {spec.name!r} input '
                        f'{spec.input_size}')
                if layout.grid != spec.native_grid:
                    raise ValueError(
                        f'{role}_raster_layout grid {layout.grid} does not '
                        f'match encoder {spec.name!r} native_grid '
                        f'{spec.native_grid}')

        validate_layout(current_layout, 'current')
        validate_layout(future_layout, 'future')
        if (current_layout is not None and future_layout is not None
                and current_layout != future_layout):
            raise ValueError(
                'current_raster_layout and future_raster_layout must match '
                'views, input size, native grid, and view order')

        lengths = []
        for mapping in (current, future):
            if mapping:
                lengths.extend(
                    int(tensor.shape[1]) for tensor in mapping.values())
        if lengths and len(set(lengths)) != 1:
            raise ValueError(
                'All teacher descriptor roles must share one token length')
        if lengths:
            token_length = lengths[0]
        else:
            token_length = None
            for key in ('current_raster_length', 'sampling_future_length'):
                value = batch.get(key)
                if value is not None:
                    token_length = int(value)
                    break
            if token_length is None and future_length is not None:
                token_length = int(future_length)
            if token_length is None:
                token_length = int(reference.shape[1])
        if future_length is not None and int(future_length) != token_length:
            raise ValueError(
                f'future_length={int(future_length)} does not match teacher '
                f'raster length {token_length}')
        if token_length < 1 or token_length > reference.shape[1]:
            raise ValueError(
                'Teacher descriptor length must be positive and no longer '
                'than the reference-token sequence')
        declared = batch.get('current_raster_length')
        if declared is not None and int(declared) != token_length:
            raise ValueError(
                'current_raster_length does not match teacher descriptors')
        if current_layout is not None:
            if current_layout.token_count != token_length:
                raise ValueError(
                    'current_raster_layout token count does not match teacher '
                    'descriptors')
        if future_layout is not None:
            if future_layout.token_count != token_length:
                raise ValueError(
                    'future_raster_layout token count does not match teacher '
                    'descriptors')
        reference_ids = batch.get('reference_ids')
        if (reference_ids is not None
                and reference_ids.shape != (*reference.shape[:2], 4)):
            raise ValueError('reference_ids must match reference_tokens')
        future_ids = batch.get('future_ids')
        if (target and future_ids is not None
                and future_ids.shape != (batch_size, token_length, 4)):
            raise ValueError('future_ids must match the teacher raster')
        if current_layout is not None and reference_ids is not None:
            expected = torch.cat([
                raster_coordinates(
                    1, current_layout.grid[0], current_layout.grid[1],
                    group=10 + view, device=reference_ids.device)
                for view in range(current_layout.views)
            ], 1)
            if not torch.equal(
                    reference_ids[:, :token_length],
                    expected.expand(batch_size, -1, -1)):
                raise ValueError(
                    'reference_ids do not match current_raster_layout view '
                    'and spatial order')
        if future_layout is not None and future_ids is not None:
            expected = torch.cat([
                raster_coordinates(
                    1, future_layout.grid[0], future_layout.grid[1],
                    group=view, device=future_ids.device)
                for view in range(future_layout.views)
            ], 1)
            if not torch.equal(
                    future_ids, expected.expand(batch_size, -1, -1)):
                raise ValueError(
                    'future_ids do not match future_raster_layout view and '
                    'spatial order')

        reference_valid = batch.get('reference_valid')
        if reference_valid is not None:
            if reference_valid.shape != reference.shape[:2]:
                raise ValueError('reference_valid must match reference_tokens')
            generated_valid = reference_valid[:, :token_length].to(
                device=reference.device, dtype=torch.bool)
        else:
            generated_valid = torch.ones(
                batch_size, token_length, dtype=torch.bool,
                device=reference.device)
        flow_future_valid = batch.get('future_valid')
        if flow_future_valid is not None:
            if flow_future_valid.shape != (batch_size, token_length):
                raise ValueError(
                    'future_valid must match the teacher raster length')
            flow_future_valid = flow_future_valid.to(
                device=reference.device, dtype=torch.bool)
        else:
            flow_future_valid = torch.ones(
                batch_size, token_length, dtype=torch.bool,
                device=reference.device)

        def validate_role(tokens, valid, fingerprints, role, base_valid):
            if tokens is None:
                return None, None
            token_result, valid_result = {}, {}
            for name, spec in specs.items():
                tensor = tokens[name]
                if (not torch.is_tensor(tensor) or tensor.ndim != 3
                        or tensor.shape[0] != batch_size
                        or tensor.shape[1] != token_length
                        or tensor.shape[-1] != spec.feature_dim):
                    raise ValueError(
                        f'{role} descriptors for {name!r} must be '
                        f'[B,{token_length},{spec.feature_dim}]')
                tensor = tensor.to(device=reference.device)
                descriptor_valid = (
                    valid.get(name) if valid is not None else None)
                if descriptor_valid is None or descriptor_valid.shape != (
                        batch_size, token_length):
                    raise ValueError(
                        f'{role} validity for {name!r} must be [B,S]')
                descriptor_valid = descriptor_valid.to(
                    device=reference.device, dtype=torch.bool)
                descriptor_valid = descriptor_valid & base_valid
                if descriptor_valid.any():
                    observed = tensor[descriptor_valid]
                    if not torch.isfinite(observed).all():
                        raise ValueError(
                            f'Valid {role} descriptors for {name!r} must be '
                            'finite')
                    if (observed.float().norm(dim=-1) <= 1e-6).any():
                        raise ValueError(
                            f'Valid {role} descriptors for {name!r} must be '
                            'nonzero')
                fingerprint = (
                    fingerprints.get(name)
                    if fingerprints is not None else None)
                if fingerprint is None:
                    raise ValueError(
                        f'Missing {role} fingerprint for {name!r}')
                if (not torch.is_tensor(fingerprint)
                        or fingerprint.shape != (batch_size, 32)):
                    raise ValueError(
                        f'{role} fingerprint for {name!r} must be [B,32]')
                expected = torch.tensor(
                    list(teacher_representation_fingerprint(spec)),
                    dtype=torch.uint8, device=reference.device)
                fingerprint = fingerprint.to(
                    device=reference.device, dtype=torch.uint8)
                if not torch.equal(
                        fingerprint,
                        expected[None].expand(batch_size, -1)):
                    raise ValueError(
                        f'{role} representation fingerprint mismatch for '
                        f'{name!r}; current descriptors must come from the '
                        'configured encoder and representation')
                token_result[name] = tensor.detach()
                valid_result[name] = descriptor_valid
            return token_result, valid_result

        current_tokens, current_masks = validate_role(
            current, current_valid, current_fingerprints, 'current',
            generated_valid)
        future_tokens, future_masks = validate_role(
            future, future_valid_teacher, future_fingerprints, 'future',
            flow_future_valid)
        if self.teacher_config.future_feedback and current_masks is None:
            current_masks = {
                spec.name: generated_valid.clone()
                for spec in self.teacher_config.encoders}
        return {
            'length': token_length,
            'generated_valid': generated_valid,
            'flow_future_valid': flow_future_valid,
            'current_tokens': current_tokens,
            'current_valid': current_masks,
            'future_tokens': future_tokens,
            'future_valid': future_masks,
        }

    def _teacher_current_residual(
        self, encoded: Mapping[str, Any], reference_length: int,
    ) -> Tensor | None:
        if not self.teacher_config.current_conditioning:
            return None
        tokens = encoded['current_tokens']
        valid = encoded['current_valid']
        residual = None
        for spec in self.teacher_config.encoders:
            projector = self.teacher_current_projectors[spec.name]
            tokens_spec = tokens[spec.name].to(
                device=projector.weight.device, dtype=projector.weight.dtype)
            normalized = prepare_teacher_fusion_input(
                tokens_spec, valid[spec.name],
                norm=self.teacher_feature_norms[spec.name])
            projected = projector(normalized)
            residual = projected if residual is None else residual + projected
        token_length = encoded['length']
        if reference_length > token_length:
            residual = torch.cat((
                residual,
                residual.new_zeros(
                    residual.shape[0], reference_length - token_length,
                    residual.shape[-1]),
            ), 1)
        return residual

    def _teacher_prepare_prefix(
        self, batch: Mapping[str, Any], context: Tensor,
        residual: Tensor | None,
    ) -> PreparedDenoiserPrefix:
        reference = batch['reference_tokens']
        valid = batch.get('reference_valid')
        return self.prepare_prefix(
            reference, context, current_hidden_residual=residual,
            reference_valid=valid)

    def _teacher_feedback_residual(
        self, tokens: Mapping[str, Tensor], valid: Mapping[str, Tensor], *,
        sample_drop: Tensor | None,
    ) -> Tensor | None:
        if not self.teacher_config.future_feedback:
            return None
        residual = None
        for spec in self.teacher_config.encoders:
            conditioner = self.teacher_conditioners[spec.name]
            tokens_spec = tokens[spec.name].to(
                device=conditioner.weight.device,
                dtype=conditioner.weight.dtype)
            normalized = prepare_teacher_fusion_input(
                tokens_spec, valid[spec.name],
                norm=self.teacher_feature_norms[spec.name],
                allow_zero=True)
            if sample_drop is not None:
                normalized = torch.where(
                    sample_drop[:, None, None], 0, normalized)
            projected = conditioner(normalized)
            residual = projected if residual is None else residual + projected
        return residual

    def _teacher_predictions(
        self, features: Tensor, token_length: int,
    ) -> dict[str, Tensor]:
        future_features = features[:, :token_length]
        return {
            spec.name: self.teacher_predictors[spec.name](future_features)
            for spec in self.teacher_config.encoders
        }

    def _teacher_feature_loss(
        self, predictions: Mapping[str, Tensor],
        encoded: Mapping[str, Any],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        weighted = predictions[list(predictions)[0]].new_zeros(
            (), dtype=torch.float32)
        raw = {}
        for spec in self.teacher_config.encoders:
            loss = masked_teacher_cosine_loss(
                predictions[spec.name],
                encoded['future_tokens'][spec.name],
                encoded['future_valid'][spec.name],
                eps=self.teacher_config.cosine_eps)
            raw[spec.name] = loss
            weighted = weighted + spec.loss_weight * loss
        return weighted, raw

    @contextmanager
    def _teacher_lookahead_mode(self):
        modules = [self.denoiser, *self.teacher_predictors.values(),
                   *self.teacher_conditioners.values()]
        descendants = [
            (module, module.training)
            for root in modules for module in root.modules()
        ]
        try:
            # Assign the flag directly on every descendant. Calling
            # ``Module.train(False)`` and later ``train(parent_state)`` would
            # overwrite intentional per-child eval/train modes.
            for module, _ in descendants:
                module.training = False
            yield
        finally:
            for module, state in descendants:
                module.training = state

    def _teacher_velocity(
        self, batch: Mapping[str, Any], x: Tensor, u: Tensor,
        sigma: Tensor, *, context: Tensor, text_valid: Tensor | None,
        prefix: PreparedDenoiserPrefix, future_residual: Tensor | None,
        capture: bool,
    ):
        options = dict(
            prepared_context=(context, text_valid),
            prepared_prefix=prefix,
            future_hidden_residual=future_residual,
        )
        if capture:
            options['representation_layer'] = (
                self.denoiser.representation_layers)
        return self._velocity(batch, x, u, sigma, **options)

    def forward(
        self, batch: Mapping[str, Any], *, generator=None,
        sigma: Tensor | None = None, future_noise: Tensor | None = None,
        action_noise: Tensor | None = None,
        condition_drop_mask: Tensor | None = None,
        second_sigma: Tensor | None = None,
        timestep_mask: Tensor | None = None,
        pseudo_label_mask: Tensor | None = None,
        structured_branches: Tensor | None = None,
        token_sigma: Tensor | None = None,
        representation_only: bool = False,
        representation_layer: int | None = None,
        teacher_sampling: bool = False, sampling_steps: int | None = None,
        horizon: int = 16, future_length: int | None = None,
        capture_layer: int | None = None, supervise_video: bool = True,
    ) -> dict[str, Tensor]:
        if teacher_sampling or representation_only:
            raise ValueError(
                'TeacherForcingPatchWAMHead does not support EMA '
                'representation_only/teacher_sampling; representation capture '
                'is owned by the teacher-forcing forward path')
        if token_sigma is not None:
            raise ValueError(
                'Teacher feature forcing supports the ordinary shared-sigma '
                'flow path; per-token token_sigma is unsupported')
        if capture_layer is not None:
            raise ValueError(
                'Teacher forcing owns representation capture; pass '
                'teacher_sampling=True for EMA-style calls')
        if self.self_flow_variant:
            raise ValueError(
                'Teacher feature forcing cannot be combined with Self-Flow')
        inputs = self._prepare_flow_training(
            batch, generator=generator, sigma=sigma,
            future_noise=future_noise, action_noise=action_noise,
            condition_drop_mask=condition_drop_mask,
            second_sigma=second_sigma, timestep_mask=timestep_mask,
            pseudo_label_mask=pseudo_label_mask,
            structured_branches=structured_branches,
            token_sigma=token_sigma)
        encoded = self._teacher_validate_encoded(
            batch, target=True, future_length=inputs.x.shape[1])
        context, text_valid = self._context(batch, inputs.drop_mask)
        current_residual = self._teacher_current_residual(
            encoded, batch['reference_tokens'].shape[1])
        prefix = self._teacher_prepare_prefix(
            batch, context, current_residual)
        token_length = encoded['length']
        batch_size = inputs.x.shape[0]
        predictions = None
        capture = self.teacher_config.future_prediction
        drop_predictions = None
        true_predictions = None
        if self.teacher_config.future_feedback:
            zeros = {
                spec.name: inputs.x.new_zeros(
                    batch_size, token_length, spec.feature_dim)
                for spec in self.teacher_config.encoders}
            zero_valid = {
                spec.name: torch.zeros(
                    batch_size, token_length, dtype=torch.bool,
                    device=inputs.x.device)
                for spec in self.teacher_config.encoders}
            with torch.no_grad(), self._teacher_lookahead_mode():
                zero_residual = self._teacher_feedback_residual(
                    zeros, zero_valid, sample_drop=None)
                lookahead = self._teacher_velocity(
                    batch, inputs.x, inputs.u, inputs.sigma,
                    context=context, text_valid=text_valid,
                    prefix=prefix, future_residual=zero_residual,
                    capture=True)
                lookahead_predictions = self._teacher_predictions(
                    lookahead[2], token_length)
            lookahead_predictions = {
                name: value.detach()
                for name, value in lookahead_predictions.items()}
            feedback_tokens = {
                name: value.clone()
                for name, value in lookahead_predictions.items()}
            feedback_valid = {
                name: mask.clone()
                for name, mask in encoded['current_valid'].items()}
            if self.training:
                true_mask = torch.rand(
                    batch_size, device=inputs.x.device,
                    generator=generator) < (
                        self.teacher_config.true_feature_prob)
                drop_mask = torch.rand(
                    batch_size, device=inputs.x.device,
                    generator=generator) < (
                        self.teacher_config.self_cond_drop_prob)
                if true_mask.any():
                    for spec in self.teacher_config.encoders:
                        target = encoded['future_tokens'][spec.name]
                        target_valid = encoded['future_valid'][spec.name]
                        feedback_tokens[spec.name] = torch.where(
                            true_mask[:, None, None],
                            torch.where(
                                target_valid[..., None], target,
                                lookahead_predictions[spec.name]),
                            feedback_tokens[spec.name])
                        feedback_valid[spec.name] = (
                            feedback_valid[spec.name]
                            & (~true_mask[:, None] | target_valid))
                drop_predictions = drop_mask
                true_predictions = true_mask
            else:
                drop_mask = None
            feedback_residual = self._teacher_feedback_residual(
                feedback_tokens, feedback_valid,
                sample_drop=drop_mask)
            trainable = self._teacher_velocity(
                batch, inputs.x, inputs.u, inputs.sigma,
                context=context, text_valid=text_valid,
                prefix=prefix, future_residual=feedback_residual,
                capture=True)
            features = trainable[2]
            predictions = self._teacher_predictions(
                features, token_length)
            prediction_x, prediction_u = trainable[:2]
        else:
            if capture:
                trainable = self._teacher_velocity(
                    batch, inputs.x, inputs.u, inputs.sigma,
                    context=context, text_valid=text_valid,
                    prefix=prefix, future_residual=None, capture=True)
                prediction_x, prediction_u, features = trainable
                predictions = self._teacher_predictions(
                    features, token_length)
            else:
                prediction_x, prediction_u = self._teacher_velocity(
                    batch, inputs.x, inputs.u, inputs.sigma,
                    context=context, text_valid=text_valid,
                    prefix=prefix, future_residual=None, capture=False)
        loss_video, loss_action = self._flow_losses(
            batch, prediction_x, prediction_u, inputs.velocity_x,
            inputs.velocity_u, inputs, supervise_video=supervise_video)
        result = {
            'loss': self.video_weight * loss_video
            + self.action_weight * loss_action,
            'loss_video': loss_video,
            'loss_action': loss_action,
            'sigma_mean': inputs.sigma.float().mean().detach(),
        }
        if predictions is not None:
            weighted_feature, raw_feature = self._teacher_feature_loss(
                predictions, encoded)
            result['loss_teacher'] = weighted_feature
            result['loss'] = result['loss'] + weighted_feature
            for name, value in raw_feature.items():
                result[f'loss_teacher_{name}'] = value
        if true_predictions is not None:
            result['teacher_true_fraction'] = (
                true_predictions.float().mean().detach())
        if drop_predictions is not None:
            result['teacher_feedback_drop_fraction'] = (
                drop_predictions.float().mean().detach())
        return result

    def _teacher_sample_velocity(
        self, batch: Mapping[str, Any], x: Tensor, u: Tensor,
        sigma: Tensor, *, context: Tensor, text_valid: Tensor | None,
        prefix: PreparedDenoiserPrefix,
        feedback: Mapping[str, Tensor] | None,
        valid: Mapping[str, Tensor], token_length: int,
    ):
        feedback_residual = None
        if self.teacher_config.future_feedback:
            feedback_residual = self._teacher_feedback_residual(
                feedback, valid, sample_drop=None)
        capture = self.teacher_config.future_feedback
        step = self._teacher_velocity(
            batch, x, u, sigma, context=context, text_valid=text_valid,
            prefix=prefix, future_residual=feedback_residual,
            capture=capture)
        if not capture:
            return step[0], step[1], None
        return step[0], step[1], self._teacher_predictions(
            step[2], token_length)

    @torch.no_grad()
    def sample_actions(
        self, batch: Mapping[str, Any], *, horizon: int = 16,
        future_length: int | None = None, steps: int = 20,
        generator=None, initial_future: Tensor | None = None,
        initial_action: Tensor | None = None,
        guidance_scale: float = 1,
        action_guidance_scale: float | None = None,
    ) -> dict[str, Tensor]:
        reference = batch['reference_tokens']
        if (not self.teacher_config.current_conditioning
                and not self.teacher_config.future_feedback):
            return super().sample_actions(
                batch, horizon=horizon, future_length=future_length,
                steps=steps, generator=generator,
                initial_future=initial_future,
                initial_action=initial_action,
                guidance_scale=guidance_scale,
                action_guidance_scale=action_guidance_scale)
        action_guidance_scale = (
            guidance_scale if action_guidance_scale is None else
            action_guidance_scale)
        if not all(math.isfinite(value) and value >= 0
                   for value in (guidance_scale, action_guidance_scale)):
            raise ValueError(
                'Guidance scales must be finite and nonnegative')
        if horizon <= 0:
            raise ValueError('horizon must be positive')
        if future_length is None:
            future_length = (
                batch['future_ids'].shape[1]
                if 'future_ids' in batch else reference.shape[1])
        future_length = int(future_length)
        if future_length <= 0:
            raise ValueError('future_length must be positive')
        b, _, width = reference.shape
        if width != self.visual_dim:
            raise ValueError(
                f'reference_tokens must be [B,R,{self.visual_dim}]')
        encoded = self._teacher_validate_encoded(
            batch, target=False, future_length=future_length)
        context, text_valid = self._context(batch)
        current_residual = self._teacher_current_residual(
            encoded, reference.shape[1])
        prefix = self._teacher_prepare_prefix(
            batch, context, current_residual)
        guided = (guidance_scale != 1
                  or action_guidance_scale != 1)
        if guided:
            uncond_context, uncond_valid = self._context(
                batch, torch.ones(
                    b, device=reference.device, dtype=torch.bool))
            uncond_prefix = self._teacher_prepare_prefix(
                batch, uncond_context, current_residual)
        else:
            uncond_context = uncond_valid = uncond_prefix = None
        token_length = encoded['length']
        feedback_valid = {
            name: mask.clone()
            for name, mask in encoded['current_valid'].items()}
        cond_feedback = None
        uncond_feedback = None
        if self.teacher_config.future_feedback:
            cond_feedback = {
                spec.name: reference.new_zeros(
                    b, token_length, spec.feature_dim)
                for spec in self.teacher_config.encoders}
            if guided:
                uncond_feedback = {
                    name: value.clone()
                    for name, value in cond_feedback.items()}
        x = initial_future
        if x is None:
            x = torch.randn(
                (b, future_length, self.visual_dim),
                device=reference.device, dtype=reference.dtype,
                generator=generator)
        u = initial_action
        if u is None:
            u = torch.randn(
                (b, horizon, self.codec.token_dim),
                device=reference.device, dtype=reference.dtype,
                generator=generator)
        if (x.shape != (b, future_length, self.visual_dim)
                or u.shape != (b, horizon, self.codec.token_dim)):
            raise ValueError(
                'Initial latent shapes differ from the requested output')
        schedule = self.flow.schedule(steps, device=reference.device)
        for start, end in pairwise(schedule):
            vx, vu, cond_feedback = self._teacher_sample_velocity(
                batch, x, u, start.expand(b), context=context,
                text_valid=text_valid, prefix=prefix,
                feedback=cond_feedback, valid=feedback_valid,
                token_length=token_length)
            if guided:
                uncond_x, uncond_u, uncond_feedback = (
                    self._teacher_sample_velocity(
                        batch, x, u, start.expand(b),
                        context=uncond_context, text_valid=uncond_valid,
                        prefix=uncond_prefix, feedback=uncond_feedback,
                        valid=feedback_valid, token_length=token_length))
                vx = uncond_x + guidance_scale * (vx - uncond_x)
                vu = (uncond_u
                      + action_guidance_scale * (vu - uncond_u))
            x, u = x + (end - start) * vx, u + (end - start) * vu
        return {'action': self.codec.decode(u), 'future_tokens': x}


__all__ = ['TeacherForcingPolicyMixin']

