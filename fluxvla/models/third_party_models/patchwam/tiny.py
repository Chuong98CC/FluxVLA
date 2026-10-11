"""Randomly initialized transformer for CPU checks, not benchmark scores."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _four_axis_rotation(query: Tensor, key: Tensor, coordinates: Tensor) -> tuple[Tensor, Tensor]:
    per_axis = query.shape[-1] // 4
    frequencies = torch.exp(
        -math.log(2000) * torch.arange(0, per_axis, 2, device=query.device).float() / per_axis,
    )
    phase = (coordinates.float().unsqueeze(-1) * frequencies).flatten(-2).unsqueeze(1)
    cosine, sine = phase.cos(), phase.sin()

    def rotate(value):
        pairs = value.float().unflatten(-1, (-1, 2))
        a, b = pairs.unbind(-1)
        return torch.stack((a * cosine - b * sine, a * sine + b * cosine), -1).flatten(-2).to(value)

    return rotate(query), rotate(key)


class ConditionalTokenBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Linear(width, width * 3)
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.mix = nn.Linear(width, width, bias=False)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, states: Tensor, time: Tensor, coordinates: Tensor, visible: Tensor) -> Tensor:
        modulation = self.modulation(F.silu(time))
        offset, scale, gate = (modulation.unsqueeze(1) if time.ndim == 2 else modulation).chunk(3, -1)
        modulated = self.norm(states) * (1 + scale) + offset
        qkv = self.qkv(modulated).unflatten(-1, (3, self.heads, -1))
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query, key = _four_axis_rotation(query, key, coordinates)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=visible)
        attended = attended.transpose(1, 2).flatten(-2)
        states = states + torch.tanh(gate) * self.mix(attended)
        return states + self.ff(self.ff_norm(states))


class SmallJointTransformer(nn.Module):
    def __init__(self, *, token_dim: int = 128, text_dim: int = 32, width: int = 64, heads: int = 4, depth: int = 2,
                 visual_dim: int | None = None):
        super().__init__()
        if width % (heads * 8) != 0 or depth < 1:
            raise ValueError("width / heads must be divisible by eight, and depth positive")
        if visual_dim is not None and (type(visual_dim) is not int or visual_dim < 1):
            raise ValueError("visual_dim must be a positive integer when given")
        self.token_dim, self.text_dim = token_dim, text_dim
        self.visual_dim = token_dim if visual_dim is None else visual_dim
        self.representation_dim, self.representation_layers = width, depth
        self.token_input = nn.Linear(token_dim, width, bias=False)
        # Separate visual maps appear only when the widths differ, so the
        # shared-width default keeps its original parameter layout and keys.
        self.visual_in = None
        self.visual_out = None
        if self.visual_dim != token_dim:
            self.visual_in = nn.Linear(self.visual_dim, width, bias=False)
            self.visual_out = nn.Linear(width, self.visual_dim, bias=False)
        self.text_input = nn.Linear(text_dim, width, bias=False)
        self.time_input = nn.Sequential(nn.Linear(64, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(ConditionalTokenBlock(width, heads) for _ in range(depth))
        self.output_norm = nn.LayerNorm(width)
        self.token_output = nn.Linear(width, token_dim, bias=False)
        self.output_bias_enabled = False

    @property
    def final_output_name(self) -> str:
        """Module path of the final output projection."""
        return 'token_output'

    def final_output_affine(self):
        """Final ``(weight, bias)`` in float32 for fusion algebra."""
        linear = self.token_output
        bias = (torch.zeros(linear.out_features, device=linear.weight.device,
                            dtype=linear.weight.dtype)
                if linear.bias is None else linear.bias.detach())
        return linear.weight.detach().float(), bias.float()

    def enable_output_bias(self) -> bool:
        """Give the output projection a zero-initialized bias (fused export)."""
        linear = self.token_output
        if linear.bias is not None:
            self.output_bias_enabled = True
            return False
        replacement = nn.Linear(linear.in_features, linear.out_features,
                                bias=True, device=linear.weight.device,
                                dtype=linear.weight.dtype)
        with torch.no_grad():
            replacement.weight.copy_(linear.weight)
            replacement.bias.zero_()
        self.token_output = replacement
        self.output_bias_enabled = True
        return True

    def forward(
        self, reference: Tensor, noisy: Tensor | tuple[Tensor, Tensor], sigma: Tensor, context: Tensor,
        *, reference_ids: Tensor, noisy_ids: Tensor, context_ids: Tensor, visibility: Tensor,
        token_sigma: Tensor | None = None, representation_layer: int | None = None,
        fused_hidden_layer: int | None = None, velocity_transform=None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if representation_layer is not None and not 1 <= representation_layer <= self.representation_layers:
            raise ValueError("representation_layer must be a 1-based transformer layer")
        if fused_hidden_layer is not None and not 1 <= fused_hidden_layer <= self.representation_layers:
            raise ValueError("fused_hidden_layer must be a 1-based transformer layer")
        if fused_hidden_layer is not None and representation_layer is not None:
            raise ValueError("fused_hidden_layer and representation_layer are mutually exclusive")
        fused = fused_hidden_layer is not None or velocity_transform is not None
        if fused and fused_hidden_layer is None:
            raise ValueError("velocity_transform requires fused_hidden_layer")
        if fused and self.visual_dim != self.token_dim:
            raise ValueError(
                "The fused velocity transform requires the shared token width; "
                "visual and action widths must match for that variant")
        text_length, reference_length = context.shape[1], reference.shape[1]
        distinct = isinstance(noisy, (tuple, list))
        if distinct:
            noisy_visual, noisy_action = noisy
            if reference.shape[-1] != self.visual_dim or noisy_visual.shape[-1] != self.visual_dim:
                raise ValueError(f"Visual tokens must be {self.visual_dim}-wide")
            if noisy_action.shape[-1] != self.token_dim:
                raise ValueError(f"Action tokens must be {self.token_dim}-wide")
            states = torch.cat((
                self.text_input(context),
                self.visual_in(torch.cat((reference, noisy_visual), 1)),
                self.token_input(noisy_action),
            ), 1)
            noisy_shape = (noisy_visual.shape[0],
                           noisy_visual.shape[1] + noisy_action.shape[1])
        else:
            if reference.shape[-1] != self.visual_dim:
                raise ValueError(f"Visual tokens must be {self.visual_dim}-wide")
            states = torch.cat((self.text_input(context), self.token_input(torch.cat((reference, noisy), 1))), 1)
            noisy_shape = noisy.shape[:2]
        coordinates = torch.cat((context_ids, reference_ids, noisy_ids), 1)
        frequencies = torch.exp(-math.log(10000) * torch.arange(32, device=sigma.device).float() / 32)
        if token_sigma is not None:
            if token_sigma.shape != noisy_shape:
                raise ValueError("token_sigma must match the noisy token shape [B,N]")
            sigma = torch.cat((sigma[:, None].expand(-1, text_length + reference_length), token_sigma), 1)
        angles = sigma.float()[..., None] * 1000 * frequencies
        time = self.time_input(torch.cat((angles.cos(), angles.sin()), -1).to(states))
        features = None
        raw_selected = None
        for index, block in enumerate(self.blocks, 1):
            states = block(states, time, coordinates, visibility)
            if index == representation_layer:
                features = states[:, text_length + reference_length:]
            if index == fused_hidden_layer:
                raw_selected = states[:, text_length + reference_length:]
        generated = states[:, text_length + reference_length:]
        if fused:
            # H is the exact conditioned input to the output projection.
            conditioned = self.output_norm(generated)
            selected = (conditioned
                        if fused_hidden_layer == self.representation_layers
                        else self.output_norm(raw_selected))
            velocity = self.token_output(conditioned)
            if velocity_transform is not None:
                velocity = velocity_transform(velocity, selected)
            return {"velocity": velocity, "final_hidden": conditioned,
                    "selected_hidden": selected}
        if distinct:
            split = generated.shape[1] - noisy_action.shape[1]
            velocity = (self.visual_out(self.output_norm(generated[:, :split])),
                        self.token_output(self.output_norm(generated[:, split:])))
        else:
            velocity = self.token_output(self.output_norm(generated))
        return velocity if representation_layer is None else (velocity, features)

    def forward_prefix(
        self, context: Tensor, reference: Tensor, sigma: Tensor,
        *, context_ids: Tensor, reference_ids: Tensor, visibility: Tensor,
        reference_sigma: Tensor | None = None,
        capture_layers=(),
        capture=None,
    ) -> tuple[Tensor, Tensor]:
        """Prefix-only pass mirroring :meth:`forward`, without future/action.

        Kept numerically identical to the official adapter's streaming prefix
        contract so CPU fixtures exercise the same call shape and gradient
        routing without downloading FLUX.2 weights.
        """
        text_length, reference_length = context.shape[1], reference.shape[1]
        if visibility.dim() != 4 or visibility.shape[-1] != text_length + reference_length:
            raise ValueError(
                'prefix `visibility` must be [B, 1, L, L] over context + '
                'reference tokens')
        layers = tuple(int(layer) for layer in capture_layers)
        if any(layer < 1 or layer > self.representation_layers for layer in layers):
            raise ValueError(
                f'`capture_layers` must be within [1, {self.representation_layers}]')
        if tuple(sorted(set(layers))) != layers:
            raise ValueError('`capture_layers` must be strictly increasing')
        if layers and capture is None:
            raise ValueError('`capture_layers` requires a `capture` callback')
        visual_input = self.token_input if self.visual_in is None else self.visual_in
        states = torch.cat((self.text_input(context), visual_input(reference)), 1)
        coordinates = torch.cat((context_ids, reference_ids), 1)
        frequencies = torch.exp(-math.log(10000) * torch.arange(
            32, device=sigma.device).float() / 32)
        angles = sigma.float()[..., None] * 1000 * frequencies
        time = self.time_input(torch.cat((angles.cos(), angles.sin()), -1).to(states))
        capture_set = set(layers)
        for index, block in enumerate(self.blocks, 1):
            states = block(states, time, coordinates, visibility)
            if index in capture_set:
                capture(index, states[:, :text_length], states[:, text_length:])
        return states[:, :text_length], states[:, text_length:]
