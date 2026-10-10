"""Parameter-free action coordinates in the visual token space.

``action_mode`` selects which normalized coordinate set is packed into the
128-wide action tokens:

* ``joint_only`` (default): the legacy single vector, bit-identical to before.
* ``end_effector_only``: the end-effector vector supplied by the dataset.
* ``dual``: one packed vector ``[joint coordinates, end-effector coordinates]``.

No mode adds learned weights: the existing repeat-into-tokens layout, mask
expansion and group-mean decode are applied to the packed vector unchanged.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

ACTION_MODES = ('joint_only', 'end_effector_only', 'dual')


def normalize_action_mode(action_mode: str) -> str:
    """Validate a public action-mode name."""
    if not isinstance(action_mode, str) or action_mode not in ACTION_MODES:
        raise ValueError(
            f"action_mode must be one of {list(ACTION_MODES)}, got "
            f"{action_mode!r}")
    return action_mode


def _positive_dim(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class RepeatedActionCodec:
    """Repeat normalized actions into tokens; decode by group averaging.

    The data processor handles normalization; the codec has no learned weights.

    ``joint_action_dim`` / ``end_effector_action_dim`` describe the packed
    layout. In a single mode the selected width defaults to ``action_dim``, so
    ``joint_only`` with both left ``None`` is exactly the legacy codec.
    """

    action_dim: int
    token_dim: int = 128
    scale: float = 1.0
    action_mode: str = 'joint_only'
    joint_action_dim: Optional[int] = None
    end_effector_action_dim: Optional[int] = None

    def __post_init__(self):
        if not 0 < self.action_dim <= self.token_dim:
            raise ValueError("action_dim must be positive and no greater than token_dim")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")
        object.__setattr__(self, 'action_mode',
                           normalize_action_mode(self.action_mode))
        joint = self.joint_action_dim
        end_effector = self.end_effector_action_dim
        if self.action_mode == 'joint_only':
            if joint is None:
                joint = self.action_dim
            elif _positive_dim('joint_action_dim', joint) != self.action_dim:
                raise ValueError(
                    f"joint_action_dim={joint} must equal action_dim="
                    f"{self.action_dim} in joint_only mode")
        elif self.action_mode == 'end_effector_only':
            if end_effector is None:
                end_effector = self.action_dim
            elif _positive_dim('end_effector_action_dim',
                               end_effector) != self.action_dim:
                raise ValueError(
                    f"end_effector_action_dim={end_effector} must equal "
                    f"action_dim={self.action_dim} in end_effector_only mode")
        else:
            if joint is None or end_effector is None:
                raise ValueError(
                    "dual action mode requires both joint_action_dim and "
                    "end_effector_action_dim")
            joint = _positive_dim('joint_action_dim', joint)
            end_effector = _positive_dim('end_effector_action_dim',
                                         end_effector)
            if joint + end_effector != self.action_dim:
                raise ValueError(
                    "dual action mode requires action_dim == joint_action_dim "
                    f"+ end_effector_action_dim ({joint} + {end_effector} != "
                    f"{self.action_dim})")
        if joint is not None:
            joint = _positive_dim('joint_action_dim', joint)
        if end_effector is not None:
            end_effector = _positive_dim('end_effector_action_dim',
                                         end_effector)
        for name, width in (('joint_action_dim', joint),
                            ('end_effector_action_dim', end_effector)):
            if width is not None and width > self.token_dim:
                raise ValueError(
                    f"{name}={width} must not exceed token_dim={self.token_dim}")
        object.__setattr__(self, 'joint_action_dim', joint)
        object.__setattr__(self, 'end_effector_action_dim', end_effector)

    @property
    def coordinate_layout(self) -> dict:
        """Named coordinate ranges of the packed action vector."""
        if self.action_mode == 'joint_only':
            return {'joint': slice(0, self.action_dim)}
        if self.action_mode == 'end_effector_only':
            return {'end_effector': slice(0, self.action_dim)}
        return {'joint': slice(0, self.joint_action_dim),
                'end_effector': slice(self.joint_action_dim, self.action_dim)}

    def split(self, coordinates: Tensor) -> dict:
        """Separate packed coordinates by name (dual returns both sets)."""
        if not torch.is_tensor(coordinates):
            raise TypeError("split expects an action-coordinate tensor")
        if coordinates.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected {self.action_dim} action coordinates, got "
                f"{coordinates.shape[-1]}")
        return {name: coordinates[..., part]
                for name, part in self.coordinate_layout.items()}

    @property
    def copies(self) -> int:
        return self.token_dim // self.action_dim

    @property
    def occupied(self) -> int:
        return self.action_dim * self.copies

    def encode(self, actions: Tensor) -> Tensor:
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected {self.action_dim} action coordinates")
        expanded = actions.unsqueeze(-1).expand(*actions.shape, self.copies)
        result = actions.new_zeros(*actions.shape[:-1], self.token_dim)
        result[..., :self.occupied] = expanded.flatten(-2) * self.scale
        return result

    def decode(self, tokens: Tensor) -> Tensor:
        if tokens.shape[-1] != self.token_dim:
            raise ValueError(f"Expected {self.token_dim} token coordinates")
        groups = tokens[..., :self.occupied].unflatten(-1, (self.action_dim, self.copies))
        return groups.mean(-1) / self.scale

    def coordinate_validity(self, valid_dimensions: Tensor) -> Tensor:
        """Expand valid dimensions; keep fill coordinates supervised."""
        if valid_dimensions.shape[-1] != self.action_dim:
            raise ValueError("Dimension mask does not match action_dim")
        result = torch.ones(
            *valid_dimensions.shape[:-1], self.token_dim,
            device=valid_dimensions.device, dtype=torch.bool,
        )
        result[..., :self.occupied] = valid_dimensions.bool().unsqueeze(-1).expand(
            *valid_dimensions.shape, self.copies,
        ).flatten(-2)
        return result
