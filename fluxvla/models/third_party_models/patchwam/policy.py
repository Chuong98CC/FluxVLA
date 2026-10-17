"""Joint latent flow objective and action-chunk sampling."""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .codec import RepeatedActionCodec
from .conditioning import PreparedDenoiserPrefix
from .flow import ShiftedFlow, masked_sample_mse, weighted_masked_sample_mse
from .geometry import joint_visibility, sequence_coordinates
from .tiny import SmallJointTransformer


@dataclass
class FlowTrainingInputs:
    """Prepared noisy state and masks shared by training evaluations."""

    future_valid: Tensor | None
    valid_dimensions: Tensor
    valid_steps: Tensor
    drop_mask: Tensor | None
    pseudo_mask: Tensor
    labeling_active: bool
    pseudo_mse: Tensor
    action_tokens: Tensor
    sigma: Tensor
    token_sigma: Tensor | None
    teacher_sigma: Tensor | None
    future_noise: Tensor
    action_noise: Tensor
    x: Tensor
    velocity_x: Tensor
    u: Tensor
    velocity_u: Tensor
    x_sigma: Tensor
    u_sigma: Tensor


class PatchFlowPolicy(nn.Module):
    """Joint flow on encoded observations and normalized actions.

    Inputs: reference_tokens [B,R,128], future_tokens [B,S,128], text_tokens
    [B,L,C], action [B,H,A], optional proprio [B,P] or [B,H,P]. Use raster IDs
    for real images.
    Padding True means invalid: dimensions [A], [B,A], or [B,H,A]; steps [B,H].
    """

    def __init__(
        self, denoiser: nn.Module, *, action_dim: int, text_dim: int,
        proprio_dim: int | None = None, token_dim: int = 128, action_scale: float = 1,
        visual_dim: int | None = None,
        shift: float = 5, video_weight: float = 0.5, action_weight: float = 1,
        isolate_actions: bool = False, condition_dropout: float = 0,
        self_flow_variant: int = 0, self_flow_mask_ratio: float = 0.25,
        self_flow_representation_weight: float = 0.8,
        self_flow_student_layer: int | None = None, self_flow_teacher_layer: int | None = None,
        self_flow_structure_ratio: float = 0.5, self_flow_edge_ratio: float = 0.5,
        self_flow_pseudo_label_fraction: float = 0.25, self_flow_sampling_steps: int = 8,
        self_flow_label_warmup: int = 10000, self_flow_pseudo_label_weight: float = 0.5,
        flow_normalization: str = "continuous",
        pack_proprio_after_text: bool = False,
        action_mode: str = "joint_only",
        joint_action_dim: int | None = None,
        end_effector_action_dim: int | None = None,
    ):
        super().__init__()
        if video_weight < 0 or action_weight < 0:
            raise ValueError("Loss weights must be nonnegative")
        if not 0 <= condition_dropout <= 1:
            raise ValueError("condition_dropout must be in [0,1]")
        if self_flow_variant not in (0, 1, 2, 3):
            raise ValueError("self_flow_variant must be 0, 1, 2, or 3")
        if not 0 <= self_flow_mask_ratio <= 0.5 or not 0 <= self_flow_pseudo_label_fraction <= 1:
            raise ValueError("Self-Flow mask ratio must be in [0,0.5] and pseudo-label fraction in [0,1]")
        if not 0 <= self_flow_structure_ratio <= 1 or not 0 <= self_flow_edge_ratio <= 1:
            raise ValueError("Self-Flow structure and edge ratios must be in [0,1]")
        if type(self_flow_label_warmup) is not int or self_flow_label_warmup < 0 or not math.isfinite(self_flow_pseudo_label_weight) or self_flow_pseudo_label_weight < 0:
            raise ValueError("Self-Flow label warmup and pseudo-label weight must be nonnegative")
        if not math.isfinite(self_flow_representation_weight) or self_flow_representation_weight < 0 or self_flow_sampling_steps < 1:
            raise ValueError("Self-Flow representation weight must be nonnegative and sampling steps positive")
        if visual_dim is not None and (type(visual_dim) is not int or visual_dim < 1):
            raise ValueError("visual_dim must be a positive integer when given")
        self.denoiser = denoiser
        self.codec = RepeatedActionCodec(
            action_dim, token_dim, action_scale, action_mode=action_mode,
            joint_action_dim=joint_action_dim,
            end_effector_action_dim=end_effector_action_dim)
        # The visual feature width is independent of the action codec width.
        # ``None`` keeps the original shared 128-dimensional token space.
        self.visual_dim = self.codec.token_dim if visual_dim is None else int(visual_dim)
        self.flow = ShiftedFlow(shift, normalization=flow_normalization)
        # Opt-in fused velocity transform (RADIO variant). ``None`` keeps the
        # original single-output-projection path bit-identical.
        self.fused_hidden_layer = None
        self.velocity_transform = None
        self.state_projection = None if proprio_dim is None else nn.Linear(proprio_dim, text_dim)
        self.text_dim = text_dim
        self.video_weight, self.action_weight = video_weight, action_weight
        self.isolate_actions = isolate_actions
        self.pack_proprio_after_text = bool(pack_proprio_after_text)
        self.condition_dropout = float(condition_dropout)
        self.self_flow_variant = self_flow_variant
        self.self_flow_mask_ratio = self_flow_mask_ratio
        self.self_flow_representation_weight = self_flow_representation_weight
        self.self_flow_pseudo_label_fraction = self_flow_pseudo_label_fraction
        self.self_flow_sampling_steps = self_flow_sampling_steps
        self.self_flow_structure_ratio, self.self_flow_edge_ratio = self_flow_structure_ratio, self_flow_edge_ratio
        self.self_flow_label_warmup, self.self_flow_pseudo_label_weight = self_flow_label_warmup, self_flow_pseudo_label_weight
        self._optimizer_updates = 0
        self.representation_projection = None
        if self_flow_variant:
            depth = getattr(denoiser, "representation_layers", 0)
            width = getattr(denoiser, "representation_dim", 0)
            student = self_flow_student_layer if self_flow_student_layer is not None else max(1, round(0.3 * depth))
            teacher = self_flow_teacher_layer if self_flow_teacher_layer is not None else min(depth, max(student + 1, round(0.7 * depth)))
            if not width or not 1 <= student < teacher <= depth:
                raise ValueError("Self-Flow requires a feature-capable denoiser and 1 <= student layer < teacher layer <= depth")
            self.self_flow_student_layer, self.self_flow_teacher_layer = student, teacher
            self.representation_projection = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        object.__setattr__(self, "_self_flow_teacher", None)

    @property
    def action_mode(self) -> str:
        """Selected action representation (see ``RepeatedActionCodec``)."""
        return self.codec.action_mode

    @property
    def visual_tokens_are_shared_width(self) -> bool:
        """Whether visual and action tokens share one token width."""
        return self.visual_dim == self.codec.token_dim

    def install_visual_width_check(self, fused: bool) -> None:
        """Reject the fused RADIO transform for independent visual widths.

        The fused velocity transform folds an adapter algebra into the
        action-width output projection, so it cannot describe tokens of a
        different visual width. The check is explicit instead of silently
        corrupting dimensions.
        """
        if fused and not self.visual_tokens_are_shared_width:
            raise ValueError(
                'The fused RADIO velocity transform requires the shared token '
                'width; visual_dim and token_dim must match for that variant')

    @property
    def joint_action_dim(self):
        """Joint-coordinate width of the packed action vector, if selected."""
        return self.codec.joint_action_dim

    @property
    def end_effector_action_dim(self):
        """End-effector width of the packed action vector, if selected."""
        return self.codec.end_effector_action_dim

    def set_optimizer_updates(self, count: int):
        """Set successful updates for pseudo-label warmup and resume."""
        if type(count) is not int or count < 0:
            raise ValueError("Successful optimizer update count must be a nonnegative integer")
        self._optimizer_updates = count

    def attach_self_flow_teacher(self, teacher: Callable):
        """Attach the engine's detached, checkpointed EMA evaluator."""
        if not callable(teacher):
            raise TypeError("Self-Flow teacher must be a callable EMA evaluator")
        # Keep the engine-owned teacher outside the policy's module tree.
        object.__setattr__(self, "_self_flow_teacher", teacher)

    def _condition_mask(self, batch, mask, generator):
        b = batch["reference_tokens"].shape[0]
        device = batch["reference_tokens"].device
        if mask is None and self.training and self.condition_dropout:
            mask = torch.rand(b, device=device, generator=generator) < self.condition_dropout
        if mask is not None:
            if mask.shape != (b,):
                raise ValueError("condition_drop_mask must be [B]")
            mask = mask.to(device=device, dtype=torch.bool)
        return mask

    def _context(self, batch: Mapping[str, Any], drop_mask: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        context = batch["text_tokens"]
        if context.ndim != 3 or context.shape[-1] != self.text_dim:
            raise ValueError(f"text_tokens must be [B,L,{self.text_dim}]")
        valid = batch.get("text_valid")
        if valid is not None and valid.shape != context.shape[:2]:
            raise ValueError("text_valid shape must match text_tokens")
        if drop_mask is not None:
            valid = torch.ones(context.shape[:2], device=context.device, dtype=torch.bool) if valid is None else valid.to(context.device).bool()
            valid = valid & ~drop_mask.to(context.device)[:, None]
        if valid is not None:
            valid = valid.to(device=context.device, dtype=torch.bool)
            # Masked NaN keys still contaminate SDPA.
            context = torch.where(valid[..., None], context, 0)
        if self.state_projection is not None:
            state = batch.get("proprio")
            if state is None:
                raise ValueError("The configured state projection requires proprio")
            if state.ndim == 3:
                state = state[:, 0]
            if state.ndim != 2 or state.shape[0] != context.shape[0]:
                raise ValueError("proprio must be [B,P] or [B,H,P]")
            projected = self.state_projection(state.to(context)).unsqueeze(1)
            if self.pack_proprio_after_text and valid is not None:
                # ImageWAM-lineage checkpoints (RobotWin/RoboDojo recipes) pack
                # the state token directly after the valid instruction prefix
                # and move the padding behind it. Positional text ids are then
                # built over the packed sequence, so this changes the state
                # token's rotation and must match the trained recipe.
                valid = valid.to(device=context.device, dtype=torch.bool)
                counts = valid.sum(1)
                lengths = context.shape[1]
                positions = torch.arange(lengths + 1, device=context.device)
                valid_rank = valid.cumsum(1) - 1
                invalid_rank = (~valid).cumsum(1) - 1
                target = torch.where(valid, valid_rank,
                                     counts[:, None] + 1 + invalid_rank)
                packed = context.new_zeros(context.shape[0], lengths + 1,
                                           context.shape[2])
                packed.scatter_(
                    1,
                    target[..., None].expand(-1, -1, context.shape[2]),
                    context)
                batch_index = torch.arange(context.shape[0],
                                           device=context.device)
                packed[batch_index, counts] = projected[:, 0]
                context = packed
                valid = positions[None, :] <= counts[:, None]
            else:
                context = torch.cat((context, projected), 1)
                if valid is not None:
                    state_valid = torch.ones(context.shape[0], 1, device=valid.device, dtype=torch.bool)
                    valid = torch.cat((valid.bool(), state_valid), 1)
        return context, valid

    def _padding(self, batch: Mapping[str, Any], actions: Tensor) -> tuple[Tensor, Tensor]:
        dimensions = batch.get("action_dim_is_pad")
        if dimensions is None:
            valid_dimensions = torch.ones_like(actions, dtype=torch.bool)
        else:
            dimensions = dimensions.to(device=actions.device, dtype=torch.bool)
            if dimensions.ndim == 2:
                dimensions = dimensions[:, None, :]
            try:
                valid_dimensions = ~torch.broadcast_to(dimensions.bool(), actions.shape)
            except RuntimeError as exc:
                raise ValueError("action_dim_is_pad cannot broadcast to [B,H,A]") from exc
        steps = batch.get("action_is_pad")
        if steps is None:
            valid_steps = torch.ones(actions.shape[:2], device=actions.device, dtype=torch.bool)
        else:
            if steps.shape != actions.shape[:2]:
                raise ValueError("action_is_pad must be [B,H]")
            valid_steps = ~steps.to(device=actions.device, dtype=torch.bool)
        return valid_dimensions & valid_steps[..., None], valid_steps

    def prepare_prefix(
        self, reference: Tensor, context: Tensor, *,
        current_hidden_residual: Tensor | None = None,
        reference_valid: Tensor | None = None,
    ) -> PreparedDenoiserPrefix:
        """Project clean reference/context embeddings for reuse.

        ``current_hidden_residual`` may cover only the current raster
        ``[B,S,D]``; zero residual rows are appended for any reference history.
        """
        if reference.ndim != 3 or reference.shape[-1] != self.visual_dim:
            raise ValueError(
                f'reference must be [B,R,{self.visual_dim}]')
        if context.ndim != 3 or context.shape[-1] != self.text_dim:
            raise ValueError(f'context must be [B,L,{self.text_dim}]')
        if reference.shape[0] != context.shape[0]:
            raise ValueError('reference and context batch sizes differ')
        if reference_valid is not None:
            if reference_valid.shape != reference.shape[:2]:
                raise ValueError('reference_valid must match reference')
            reference = torch.where(
                reference_valid.to(device=reference.device, dtype=torch.bool)[
                    ..., None],
                reference, torch.zeros_like(reference))
        prepare = getattr(self.denoiser, 'prepare_prefix', None)
        if not callable(prepare):
            raise ValueError(
                'The configured denoiser does not support prepared prefixes')
        residual = current_hidden_residual
        if residual is not None:
            if (not torch.is_tensor(residual) or residual.ndim != 3
                    or residual.shape[0] != reference.shape[0]
                    or residual.shape[1] > reference.shape[1]
                    or residual.shape[-1] != getattr(
                        self.denoiser, 'representation_dim', -1)):
                raise ValueError(
                    'current_hidden_residual must be [B,S,D] with S<=R and '
                    'D equal to the denoiser hidden width')
            if residual.device != reference.device or (
                    residual.dtype != reference.dtype):
                raise ValueError(
                    'current_hidden_residual must match reference device and '
                    'dtype')
            if residual.shape[1] < reference.shape[1]:
                padding = torch.zeros(
                    residual.shape[0],
                    reference.shape[1] - residual.shape[1],
                    residual.shape[2],
                    device=residual.device, dtype=residual.dtype)
                residual = torch.cat((residual, padding), 1)
        prefix = prepare(reference, context,
                         current_hidden_residual=residual)
        if not isinstance(prefix, PreparedDenoiserPrefix):
            raise TypeError(
                'denoiser.prepare_prefix must return PreparedDenoiserPrefix')
        return prefix

    def _velocity(
        self, batch: Mapping[str, Any], noisy_future: Tensor, noisy_action: Tensor, sigma: Tensor,
        *, condition_drop_mask=None, token_sigma=None, representation_layer=None,
        prepared_context=None, prepared_prefix: PreparedDenoiserPrefix | None = None,
        future_hidden_residual: Tensor | None = None,
    ):
        reference = batch["reference_tokens"]
        if prepared_context is None:
            context, text_valid = self._context(batch, condition_drop_mask)
        else:
            if (not isinstance(prepared_context, (tuple, list))
                    or len(prepared_context) != 2):
                raise ValueError(
                    'prepared_context must be (context, text_valid)')
            context, text_valid = prepared_context
        if not torch.is_tensor(context) or context.ndim != 3:
            raise ValueError('Prepared context must be [B,L,text_dim]')
        if text_valid is not None and text_valid.shape != context.shape[:2]:
            raise ValueError('Prepared text validity must match context')
        b, future_length = noisy_future.shape[:2]
        if reference.ndim != 3 or reference.shape[0] != b or reference.shape[-1] != self.visual_dim:
            raise ValueError(f"reference_tokens must be [B,R,{self.visual_dim}]")
        if noisy_future.shape[-1] != self.visual_dim:
            raise ValueError(f"future tokens must be [B,S,{self.visual_dim}]")
        if noisy_action.ndim != 3 or noisy_action.shape[-1] != self.codec.token_dim:
            raise ValueError(f"action tokens must be [B,H,{self.codec.token_dim}]")
        if context.shape[0] != b:
            raise ValueError("Text and visual batch sizes differ")
        device = reference.device
        reference_valid = batch.get("reference_valid")
        if reference_valid is not None:
            if reference_valid.shape != reference.shape[:2]:
                raise ValueError("reference_valid must match reference_tokens [B,R]")
            reference_valid = reference_valid.to(device=device, dtype=torch.bool)
            reference = torch.where(reference_valid[..., None], reference, 0)
        future_valid = batch.get("future_valid")
        if future_valid is not None:
            if future_valid.shape != (b, future_length):
                raise ValueError("future_valid must match future tokens [B,S]")
            future_valid = future_valid.to(device=device, dtype=torch.bool)
            # Invalid future slots cannot contribute attention keys or values.
            noisy_future = torch.where(future_valid[..., None], noisy_future,
                                       torch.zeros_like(noisy_future))
        # Real images supply raster IDs; synthetic examples use sequence IDs.
        ref_ids = batch.get("reference_ids")
        if ref_ids is None:
            ref_ids = sequence_coordinates(b, reference.shape[1], group=10, device=device)
        if reference_valid is not None:
            ref_ids = torch.where(reference_valid[..., None], ref_ids.to(device), 0)
        future_ids = batch.get("future_ids")
        if future_ids is None:
            future_ids = sequence_coordinates(b, future_length, group=0, device=device)
        action_ids = batch.get("action_ids")
        if action_ids is None:
            action_ids = sequence_coordinates(b, noisy_action.shape[1], group=20, device=device)
        else:
            # Optional validated spatial/time IDs for variants whose action
            # tokens carry a raster layout (group, row, column, timestep).
            # The absent-key path above is unchanged for the robot variants.
            if not torch.is_tensor(action_ids):
                raise TypeError("action_ids must be a tensor")
            action_ids = action_ids.to(device=device)
            if (action_ids.ndim != 3 or action_ids.shape[0] != b
                    or action_ids.shape[1] != noisy_action.shape[1]
                    or action_ids.shape[2] != 4):
                raise ValueError(
                    "action_ids must be [B,H,4] matching the action tokens")
        text_ids = sequence_coordinates(b, context.shape[1], device=device)
        visibility = joint_visibility(
            context.shape[1], reference.shape[1], future_length, noisy_action.shape[1],
            text_valid=text_valid, reference_valid=reference_valid,
            future_valid=future_valid,
            isolate_actions=self.isolate_actions, device=device,
        )
        options = {}
        if token_sigma is not None:
            options["token_sigma"] = token_sigma
        fused = (self.fused_hidden_layer is not None
                 or self.velocity_transform is not None)
        if fused:
            if representation_layer is not None:
                raise ValueError(
                    "A fused velocity transform cannot be combined with "
                    "representation_layer capture")
            options["fused_hidden_layer"] = self.fused_hidden_layer
            options["velocity_transform"] = self.velocity_transform
        elif representation_layer is not None:
            options["representation_layer"] = representation_layer
        if prepared_prefix is not None:
            if not isinstance(prepared_prefix, PreparedDenoiserPrefix):
                raise TypeError(
                    'prepared_prefix must be a PreparedDenoiserPrefix')
            if (prepared_prefix.reference_hidden.shape[:2]
                    != reference.shape[:2]):
                raise ValueError(
                    'Prepared reference prefix does not match reference_tokens')
            if prepared_prefix.context_hidden.shape[1] != context.shape[1]:
                raise ValueError(
                    'Prepared context prefix does not match text context')
            options["prepared_prefix"] = prepared_prefix
        if future_hidden_residual is not None:
            if (not torch.is_tensor(future_hidden_residual)
                    or future_hidden_residual.ndim != 3
                    or future_hidden_residual.shape[:2]
                    != (b, future_length)):
                raise ValueError(
                    'future_hidden_residual must be [B,S,D] matching future '
                    'tokens')
            options["future_hidden_residual"] = future_hidden_residual
        prediction = self.denoiser(
            reference, self._concatenate_tokens(noisy_future, noisy_action), sigma, context,
            reference_ids=ref_ids, noisy_ids=torch.cat((future_ids, action_ids), 1),
            context_ids=text_ids, visibility=visibility, **options,
        )
        features = None
        fused_context = None
        if fused:
            fused_context = {name: prediction[name]
                             for name in ("final_hidden", "selected_hidden")}
            prediction = prediction["velocity"]
        elif representation_layer is not None:
            prediction, features = prediction
        velocities = self._split_velocity(prediction, b, future_length,
                                          noisy_action.shape[1])
        if fused:
            return (*velocities, fused_context)
        return velocities if representation_layer is None else (*velocities, features)

    def _concatenate_tokens(self, visual: Tensor, action: Tensor):
        """Join tokens for the denoiser; a tuple when the widths differ."""
        if self.visual_tokens_are_shared_width:
            return torch.cat((visual, action), 1)
        return (visual, action)

    def _split_velocity(self, prediction, batch: int, future_length: int,
                        horizon: int):
        """Split denoiser output into future-visual and action velocities."""
        if isinstance(prediction, (tuple, list)):
            if len(prediction) != 2:
                raise ValueError(
                    "Denoiser must return two tensors for distinct visual and "
                    "action widths")
            visual, action = prediction
        else:
            if not self.visual_tokens_are_shared_width:
                raise ValueError(
                    "A denoiser with distinct visual and action widths must "
                    "return both tensors")
            if prediction.shape != (batch, future_length + horizon, self.codec.token_dim):
                raise ValueError("Denoiser returned an incompatible velocity shape")
            visual, action = (prediction[:, :future_length],
                              prediction[:, future_length:])
        if (visual.shape != (batch, future_length, self.visual_dim)
                or action.shape != (batch, horizon, self.codec.token_dim)):
            raise ValueError("Denoiser returned an incompatible velocity shape")
        return visual, action

    def _prepare_flow_training(
        self, batch: Mapping[str, Any], *, generator=None,
        sigma: Tensor | None = None, future_noise: Tensor | None = None,
        action_noise: Tensor | None = None,
        condition_drop_mask: Tensor | None = None,
        second_sigma: Tensor | None = None,
        timestep_mask: Tensor | None = None,
        pseudo_label_mask: Tensor | None = None,
        structured_branches: Tensor | None = None,
        token_sigma: Tensor | None = None,
        require_teacher: bool = True,
        representation_only: bool = False,
    ) -> FlowTrainingInputs:
        """Prepare the shared noisy states used by flow and feature losses."""
        future, actions = batch["future_tokens"], batch["action"]
        if future.ndim != 3 or future.shape[-1] != self.visual_dim:
            raise ValueError(f"future_tokens must be [B,S,{self.visual_dim}]")
        if actions.ndim != 3 or actions.shape[0] != future.shape[0]:
            raise ValueError("action must be [B,H,A] with matching batch size")
        valid_future = batch.get("future_valid")
        if valid_future is not None:
            if valid_future.shape != future.shape[:2]:
                raise ValueError("future_valid must match future_tokens [B,S]")
            valid_future = valid_future.to(
                device=future.device, dtype=torch.bool)[..., None]
            future = torch.where(valid_future, future,
                                 torch.zeros_like(future))
        valid_dimensions, valid_steps = self._padding(batch, actions)
        drop_mask = self._condition_mask(batch, condition_drop_mask, generator)
        teacher = self._self_flow_teacher
        if self.self_flow_variant and require_teacher and teacher is None:
            raise RuntimeError(
                "Self-Flow requires an attached checkpointed EMA teacher")
        pseudo_mask = torch.zeros(
            actions.shape[0], device=actions.device, dtype=torch.bool)
        labeling_active = (
            self._optimizer_updates >= self.self_flow_label_warmup)
        pseudo_mse = actions.new_zeros((), dtype=torch.float32)
        if self.self_flow_variant == 3 and not representation_only:
            if pseudo_label_mask is None:
                pseudo_mask = (torch.rand(
                    actions.shape[0], device=actions.device,
                    generator=generator)
                    < self.self_flow_pseudo_label_fraction)
            else:
                if pseudo_label_mask.shape != pseudo_mask.shape:
                    raise ValueError("pseudo_label_mask must be [B]")
                pseudo_mask = pseudo_label_mask.to(
                    device=actions.device, dtype=torch.bool)
            if labeling_active and pseudo_mask.any():
                inference_batch = {
                    key: value for key, value in batch.items()
                    if key not in ("action", "action_is_pad",
                                   "action_dim_is_pad")}
                indices = pseudo_mask.nonzero().flatten().tolist()
                inference_batch = {
                    key: value[pseudo_mask]
                    if torch.is_tensor(value) and value.ndim
                    and value.shape[0] == actions.shape[0]
                    else [value[index] for index in indices]
                    if isinstance(value, (list, tuple))
                    and len(value) == actions.shape[0]
                    else value
                    for key, value in inference_batch.items()
                }
                with torch.no_grad():
                    generated = teacher(
                        inference_batch, teacher_sampling=True,
                        sampling_steps=self.self_flow_sampling_steps,
                        horizon=actions.shape[1],
                        future_length=future.shape[1],
                        generator=generator)["action"].detach()
                if (generated.shape != actions[pseudo_mask].shape
                        or not torch.isfinite(generated).all()):
                    raise ValueError(
                        "EMA pseudo-label actions must be finite and match the "
                        "withheld [B,H,A] subset")
                with torch.no_grad():
                    pseudo_mse = (
                        generated.to(actions).float()
                        - actions.float()[pseudo_mask]).square().mean()
                actions = actions.clone()
                actions[pseudo_mask] = generated.to(actions)
            elif not labeling_active:
                actions = torch.where(
                    pseudo_mask[:, None, None], 0, actions)
        clean_actions = torch.where(valid_dimensions, actions, 0)
        action_tokens = self.codec.encode(clean_actions).to(future)
        teacher_sigma = None
        if self.self_flow_variant and not representation_only:
            sigma, token_sigma, teacher_sigma = self.flow.dual_timesteps(
                future.shape[0], future.shape[1], actions.shape[1],
                mask_ratio=self.self_flow_mask_ratio,
                structured=self.self_flow_variant >= 2,
                device=future.device, generator=generator, first=sigma,
                second=second_sigma, mask=timestep_mask,
                structure_ratio=self.self_flow_structure_ratio,
                edge_ratio=self.self_flow_edge_ratio,
                branches=structured_branches)
            if self.self_flow_variant == 3 and not labeling_active:
                token_sigma[pseudo_mask, future.shape[1]:] = 1
                teacher_sigma = token_sigma.min(1).values
        if sigma is None:
            sigma = self.flow.sample(
                future.shape[0], device=future.device,
                generator=generator)
        if (sigma.shape != (future.shape[0],)
                or not torch.isfinite(sigma).all()
                or ((sigma < 0) | (sigma > 1)).any()):
            raise ValueError(
                "sigma must be finite [B] values in [0,1]")
        sigma = sigma.to(device=future.device, dtype=torch.float32)
        if future_noise is None:
            future_noise = torch.randn(
                future.shape, device=future.device, dtype=future.dtype,
                generator=generator)
        if action_noise is None:
            action_noise = torch.randn(
                action_tokens.shape, device=future.device,
                dtype=future.dtype, generator=generator)
        if valid_future is not None:
            future_noise = torch.where(
                valid_future, future_noise, torch.zeros_like(future_noise))
        x_sigma = (sigma if token_sigma is None else
                   token_sigma[:, :future.shape[1]])
        u_sigma = (sigma if token_sigma is None else
                   token_sigma[:, future.shape[1]:])
        x, velocity_x = self.flow.perturb(future, future_noise, x_sigma)
        u, velocity_u = self.flow.perturb(action_tokens, action_noise, u_sigma)
        return FlowTrainingInputs(
            future_valid=valid_future,
            valid_dimensions=valid_dimensions,
            valid_steps=valid_steps,
            drop_mask=drop_mask,
            pseudo_mask=pseudo_mask,
            labeling_active=labeling_active,
            pseudo_mse=pseudo_mse,
            action_tokens=action_tokens,
            sigma=sigma,
            token_sigma=token_sigma,
            teacher_sigma=teacher_sigma,
            future_noise=future_noise,
            action_noise=action_noise,
            x=x,
            velocity_x=velocity_x,
            u=u,
            velocity_u=velocity_u,
            x_sigma=x_sigma,
            u_sigma=u_sigma,
        )

    def _flow_losses(
        self, batch: Mapping[str, Any], prediction_x: Tensor,
        prediction_u: Tensor, velocity_x: Tensor, velocity_u: Tensor,
        inputs: FlowTrainingInputs, *, supervise_video: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Compute the unchanged PatchWAM flow losses."""
        future = batch["future_tokens"]
        action_token_valid = batch.get("action_token_valid")
        if action_token_valid is None:
            valid_token_coordinates = (
                self.codec.coordinate_validity(inputs.valid_dimensions)
                & inputs.valid_steps[..., None]).to(future.device)
        else:
            if (not torch.is_tensor(action_token_valid)
                    or action_token_valid.dtype != torch.bool
                    or action_token_valid.shape != inputs.u.shape):
                raise ValueError(
                    "action_token_valid must be a boolean [B,H,token_dim] "
                    "tensor matching the action tokens")
            valid_token_coordinates = action_token_valid.to(future.device)
        if inputs.token_sigma is None:
            per_video = (
                masked_sample_mse(
                    prediction_x, velocity_x, inputs.future_valid)
                if supervise_video else None)
            per_action = masked_sample_mse(
                prediction_u, velocity_u, valid_token_coordinates)
            weight = self.flow.weight(inputs.sigma)
            loss_action = (weight * per_action).mean()
            loss_video = (
                (weight * per_video).mean() if supervise_video else
                torch.zeros((), device=prediction_u.device,
                            dtype=torch.float32))
        else:
            video_weights = self.flow.weight(inputs.x_sigma)
            action_weights = self.flow.weight(inputs.u_sigma)
            video_weights = torch.where(
                inputs.x_sigma == 0, 0, video_weights)
            action_weights = torch.where(
                inputs.u_sigma == 0, 0, action_weights)
            if self.self_flow_variant == 3:
                multiplier = (
                    self.self_flow_pseudo_label_weight
                    if inputs.labeling_active else 0)
                action_weights = torch.where(
                    inputs.pseudo_mask[:, None],
                    action_weights * multiplier, action_weights)
            loss_video = (
                weighted_masked_sample_mse(
                    prediction_x, velocity_x, video_weights,
                    inputs.future_valid).mean()
                if supervise_video else
                torch.zeros((), device=prediction_u.device,
                            dtype=torch.float32))
            loss_action = weighted_masked_sample_mse(
                prediction_u, velocity_u, action_weights,
                valid_token_coordinates).mean()
        return loss_video, loss_action

    def forward(
        self, batch: Mapping[str, Any], *, generator=None, sigma: Tensor | None = None,
        future_noise: Tensor | None = None, action_noise: Tensor | None = None,
        condition_drop_mask: Tensor | None = None, second_sigma: Tensor | None = None,
        timestep_mask: Tensor | None = None, pseudo_label_mask: Tensor | None = None,
        structured_branches: Tensor | None = None, token_sigma: Tensor | None = None,
        representation_only: bool = False, representation_layer: int | None = None,
        teacher_sampling: bool = False, sampling_steps: int | None = None,
        horizon: int = 16, future_length: int | None = None,
        capture_layer: int | None = None,
        supervise_video: bool = True,
    ) -> dict[str, Tensor]:
        # EMA calls bypass recursive teacher evaluation.
        if teacher_sampling:
            return self.sample_actions_from_future(
                batch, horizon=horizon, steps=self.self_flow_sampling_steps if sampling_steps is None else sampling_steps,
                generator=generator,
            )
        inputs = self._prepare_flow_training(
            batch, generator=generator, sigma=sigma,
            future_noise=future_noise, action_noise=action_noise,
            condition_drop_mask=condition_drop_mask,
            second_sigma=second_sigma, timestep_mask=timestep_mask,
            pseudo_label_mask=pseudo_label_mask,
            structured_branches=structured_branches,
            token_sigma=token_sigma,
            require_teacher=not representation_only,
            representation_only=representation_only)
        future = batch["future_tokens"]
        actions = batch["action"]
        valid_future = inputs.future_valid
        valid_dimensions = inputs.valid_dimensions
        valid_steps = inputs.valid_steps
        drop_mask = inputs.drop_mask
        pseudo_mask = inputs.pseudo_mask
        labeling_active = inputs.labeling_active
        pseudo_mse = inputs.pseudo_mse
        action_tokens = inputs.action_tokens
        sigma = inputs.sigma
        token_sigma = inputs.token_sigma
        teacher_sigma = inputs.teacher_sigma
        future_noise = inputs.future_noise
        action_noise = inputs.action_noise
        x_sigma = inputs.x_sigma
        u_sigma = inputs.u_sigma
        x, velocity_x = inputs.x, inputs.velocity_x
        u, velocity_u = inputs.u, inputs.velocity_u
        if representation_only:
            if representation_layer is None:
                raise ValueError("representation_only requires representation_layer")
            _, _, features = self._velocity(batch, x, u, sigma, condition_drop_mask=drop_mask,
                                             token_sigma=token_sigma, representation_layer=representation_layer)
            return {"features": features}
        fused_context = None
        if self.self_flow_variant:
            prediction_x, prediction_u, features = self._velocity(
                batch, x, u, sigma, condition_drop_mask=drop_mask, token_sigma=token_sigma,
                representation_layer=self.self_flow_student_layer,
            )
        else:
            if self.fused_hidden_layer is not None or self.velocity_transform is not None:
                prediction_x, prediction_u, fused_context = self._velocity(
                    batch, x, u, sigma, condition_drop_mask=drop_mask)
            elif capture_layer is None:
                prediction_x, prediction_u = self._velocity(batch, x, u, sigma, condition_drop_mask=drop_mask)
            else:
                prediction_x, prediction_u, features = self._velocity(
                    batch, x, u, sigma, condition_drop_mask=drop_mask,
                    representation_layer=capture_layer,
                )
        loss_video, loss_action = self._flow_losses(
            batch, prediction_x, prediction_u, velocity_x, velocity_u,
            inputs, supervise_video=supervise_video)
        result = {
            "loss": self.video_weight * loss_video + self.action_weight * loss_action,
            "loss_video": loss_video, "loss_action": loss_action,
            "sigma_mean": sigma.float().mean().detach(),
        }
        if not self.self_flow_variant and capture_layer is not None:
            # Selected transformer hidden states of the noisy future/action tokens,
            # captured during the same pass as the velocities.
            result["velocity_features"] = features
            # Predicted 128-wide future VELOCITY from the same pass. Opt-in
            # consumers (the RADIO auxiliary output branch) use this slice, not
            # the noisy input and not an estimated clean latent.
            result["velocity_future"] = prediction_x
        if fused_context is not None:
            # Everything the fused RADIO objective needs, from the same single
            # pass: selected/final conditioned hidden, fused future velocity,
            # the actual sanitized future noise, sigma and the future mask.
            fused_context.update(future_velocity=prediction_x,
                                 future_noise=future_noise, sigma=sigma,
                                 valid_future=valid_future)
            result["fused_context"] = fused_context
        if self.self_flow_variant:
            teacher_batch = dict(batch, action=clean_actions)
            teacher_times = None
            if self.self_flow_variant == 3 and not labeling_active:
                teacher_times = teacher_sigma[:, None].expand_as(token_sigma).clone()
                teacher_times[pseudo_mask, future.shape[1]:] = 1
            with torch.no_grad():
                target = teacher(teacher_batch, representation_only=True,
                                 representation_layer=self.self_flow_teacher_layer, sigma=teacher_sigma,
                                 future_noise=future_noise, action_noise=action_noise,
                                 condition_drop_mask=drop_mask, token_sigma=teacher_times)["features"].detach()
            projected = self.representation_projection(features)
            if target.shape != projected.shape or not torch.isfinite(target).all():
                raise ValueError("EMA representation shape or values are invalid")
            valid_features = torch.ones(projected.shape[:2], device=future.device, dtype=torch.bool)
            if self.self_flow_variant == 3 and not labeling_active:
                valid_features[pseudo_mask, future.shape[1]:] = False
            cosine = F.cosine_similarity(projected.float(), target.float(), dim=-1)
            loss_representation = 1 - torch.where(valid_features, cosine, 0).sum() / valid_features.sum().clamp_min(1)
            result["loss_representation"] = loss_representation
            result["loss"] = result["loss"] + self.self_flow_representation_weight * loss_representation
            result["pseudo_label_fraction"] = pseudo_mask.float().mean().detach()
            if self.self_flow_variant == 3:
                result["pseudo_mse"] = pseudo_mse.detach()
        return result

    def loss(self, batch: Mapping[str, Any], **kwargs) -> dict[str, Tensor]:
        return self(batch, **kwargs)

    @torch.no_grad()
    def sample_actions(
        self, batch: Mapping[str, Any], *, horizon: int = 16, future_length: int | None = None,
        steps: int = 20, generator=None, initial_future: Tensor | None = None, initial_action: Tensor | None = None,
        guidance_scale: float = 1, action_guidance_scale: float | None = None,
    ) -> dict[str, Tensor]:
        """Generate normalized actions and future latent tokens."""
        reference = batch["reference_tokens"]
        action_guidance_scale = guidance_scale if action_guidance_scale is None else action_guidance_scale
        if not all(math.isfinite(value) and value >= 0 for value in (guidance_scale, action_guidance_scale)):
            raise ValueError("Guidance scales must be finite and nonnegative")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if future_length is None:
            future_length = batch["future_ids"].shape[1] if "future_ids" in batch else reference.shape[1]
        if future_length <= 0:
            raise ValueError("future_length must be positive")
        b, _, visual_width = reference.shape
        if visual_width != self.visual_dim:
            raise ValueError(f"reference_tokens must be [B,R,{self.visual_dim}]")
        width = self.codec.token_dim
        x = initial_future
        if x is None:
            x = torch.randn((b, future_length, visual_width), device=reference.device, dtype=reference.dtype, generator=generator)
        u = initial_action
        if u is None:
            u = torch.randn((b, horizon, width), device=reference.device, dtype=reference.dtype, generator=generator)
        if x.shape != (b, future_length, visual_width) or u.shape != (b, horizon, width):
            raise ValueError("Initial latent shapes differ from the requested output")
        schedule = self.flow.schedule(steps, device=reference.device)
        for start, end in pairwise(schedule):
            step = self._velocity(batch, x, u, start.expand(b))
            vx, vu = step[0], step[1]
            if guidance_scale != 1 or action_guidance_scale != 1:
                uncond = self._velocity(
                    batch, x, u, start.expand(b),
                    condition_drop_mask=torch.ones(b, device=reference.device, dtype=torch.bool),
                )
                uncond_x, uncond_u = uncond[0], uncond[1]
                vx = uncond_x + guidance_scale * (vx - uncond_x)
                vu = uncond_u + action_guidance_scale * (vu - uncond_u)
            x, u = x + (end - start) * vx, u + (end - start) * vu
        return {"action": self.codec.decode(u), "future_tokens": x}

    @torch.no_grad()
    def sample_actions_from_future(
        self, batch: Mapping[str, Any], *, horizon: int = 16, steps: int = 20,
        generator=None, initial_action: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Inverse dynamics: clamp future tokens and generate actions."""
        if self.self_flow_variant < 2:
            raise ValueError("Conditional dynamics sampling requires a structured Self-Flow policy")
        future = batch["future_tokens"]
        if future.ndim != 3 or future.shape[-1] != self.visual_dim or horizon < 1:
            raise ValueError(f"Future tokens must be [B,S,{self.visual_dim}] and horizon positive")
        b, length, width = future.shape
        action_width = self.codec.token_dim
        actions = initial_action
        if actions is None:
            actions = torch.randn((b, horizon, action_width), device=future.device, dtype=future.dtype, generator=generator)
        if actions.shape != (b, horizon, action_width):
            raise ValueError("Initial action token shape differs from requested horizon")
        schedule = self.flow.schedule(steps, device=future.device)
        for start, end in pairwise(schedule):
            levels = torch.cat((torch.zeros(b, length, device=future.device), start.expand(b, horizon)), 1)
            step = self._velocity(batch, future, actions, start.expand(b), token_sigma=levels)
            velocity = step[1]
            actions = actions + (end - start) * velocity
        return {"action": self.codec.decode(actions), "future_tokens": future}

    @torch.no_grad()
    def sample_future_from_actions(
        self, batch: Mapping[str, Any], *, future_length: int | None = None,
        steps: int = 20, generator=None, initial_future: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward dynamics: clamp normalized actions and generate future tokens."""
        if self.self_flow_variant < 2:
            raise ValueError("Conditional dynamics sampling requires a structured Self-Flow policy")
        actions = batch["action"]
        valid, _ = self._padding(batch, actions)
        action_tokens = self.codec.encode(torch.where(valid, actions, 0)).to(batch["reference_tokens"])
        if future_length is None:
            future_length = batch["future_ids"].shape[1] if "future_ids" in batch else batch["reference_tokens"].shape[1]
        if future_length < 1:
            raise ValueError("future_length must be positive")
        b, horizon, _ = action_tokens.shape
        width = batch["reference_tokens"].shape[-1]
        if width != self.visual_dim:
            raise ValueError(f"reference_tokens must be [B,R,{self.visual_dim}]")
        future = initial_future
        if future is None:
            future = torch.randn((b, future_length, width), device=action_tokens.device, dtype=action_tokens.dtype, generator=generator)
        if future.shape != (b, future_length, width):
            raise ValueError("Initial future token shape differs from requested layout")
        schedule = self.flow.schedule(steps, device=future.device)
        for start, end in pairwise(schedule):
            levels = torch.cat((start.expand(b, future_length), torch.zeros(b, horizon, device=future.device)), 1)
            step = self._velocity(batch, future, action_tokens, start.expand(b), token_sigma=levels)
            velocity = step[0]
            future = future + (end - start) * velocity
        return {"action": actions, "future_tokens": future}


def make_tiny_policy(
    *, action_dim: int = 14, proprio_dim: int | None = 14, text_dim: int = 32,
    token_dim: int = 128, width: int = 64, heads: int = 4, depth: int = 2, **policy_options,
) -> PatchFlowPolicy:
    denoiser = SmallJointTransformer(token_dim=token_dim, text_dim=text_dim, width=width, heads=heads, depth=depth)
    return PatchFlowPolicy(
        denoiser, action_dim=action_dim, token_dim=token_dim, text_dim=text_dim, proprio_dim=proprio_dim, **policy_options,
    )
