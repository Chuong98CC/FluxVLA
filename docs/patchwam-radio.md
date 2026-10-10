# PatchWAM with fused RADIO feedback and configurable action modes

`PatchWAMRadioAuxVLA` keeps the original PatchWAM VAE state, noising, sampler
and action flow objective, and adds a linear velocity residual with RADIO
auxiliary supervision. The transformer keeps its original weights and
initialization and is fine-tuned.

## Fused linear head

Three biased linear adapters and one fixed projection matrix are added:

| tensor | shape | initialization |
| --- | --- | --- |
| `radio_a` | `token_dim -> radio_dim` | nonzero, bias zero |
| `radio_b` | `hidden -> radio_dim` | weight and bias zero |
| `radio_c` | `radio_dim -> token_dim` | weight and bias zero |
| `radio_noise_projection` (buffer `P`) | `radio_dim x token_dim` | seeded, unit-norm rows |

`H` is the conditioned hidden input to the original final output projection
(the adaLN-conditioned `final_layer` input for FLUX.2, `output_norm` for the
tiny transformer), not the raw block output. For every generated future **and**
action token the shared head computes

```
K  = B(H_selected)
Y' = A(Y) + K                 # Y = F(H) is the original velocity
Y'' = Y + C(Y')               # shared velocity used by losses and sampling
R  = A(Y'') + K               # training-only readout
```

Zero-initialized `B`/`C` weights and all-zero biases make `Y'' == Y` at
startup, so a pretrained model reproduces its exact original outputs; `R`
reuses `A`/`B` so `C` receives auxiliary gradients immediately.

## RADIO objective (shared-noise lifting)

The future VAE velocity MSE is **not computed** for this variant
(`video_weight` must be 0; nonzero values are rejected) and its reported
`loss_video`/`loss_vae` metrics are detached zeros. The RADIO target couples
the RADIO noise to the actual parent future noise `epsilon_z`:

```
epsilon_R = P @ epsilon_z            # float32, fixed seeded P
target_vR = epsilon_R - r0           # r0 = detached clean RADIO features
loss_R    = flow.weight(sigma) * masked_mse(R_future, target_vR)
loss      = radio_weight * loss_R + action_weight * loss_action
```

`P` is built from a private seeded generator (`radio_noise_seed`, default 0);
no global RNG draw is consumed and `P` is never learned. The target is a
rank-limited, correlated lifting of the 128-wide parent noise: this is an
experimental coupled auxiliary, **not** an exact `radio_dim`-wide RADIO flow.
The original 128-wide VAE-noised state, masks, weighting and descending-Euler
sampler are unchanged.

`radio_weight=0` is action-only training: no teacher, no readout, no targets
while the learned velocity residual still consumes the selected hidden state.

## Layer selection

`adapter_layer_idex` (zero-based, `-1` = last block, default) selects which
block's conditioned hidden feeds `B`. Blocks are ordered as all FLUX.2 double
blocks followed by all single blocks, so for `D` double and `S` single blocks
(`N = D + S`) `0` is the first double block, `D - 1` the last double block,
`D` the first single block and `N - 1` the last single block (identical to
`-1`). Values outside `[-1, N - 1]`, booleans and non-integers are rejected at
construction.

The default/final selection uses the exact same `H` as the original output
projection and can be folded into a single affine head. An earlier selection
cannot be fused and keeps the teacher-free runtime adapters; the fused export
fails clearly for it.

## Inference, export and checkpoints

* `export_inference_state_dict()` (also `fused_output_state_dict()`) folds
  the final-layer adapters into the output
  projection (`W = F + C A F + C B`, `b = f + C(A f + a + b) + c`, computed in
  float32) for a final-layer selection. The deployable model is the native
  PatchWAM stack with the bias-enabled output head
  (`denoiser.output_bias=True`, one added bias tensor, zero-initialized after
  the original strict asset load). Load this state with `strict=True` into the
  recipe's `inference_model`. Its 128 bias parameters are the only added
  inference parameters; the teacher, adapters and `P` are removed. Floating-point
  evaluation can differ slightly from the unfused multiplication order.
* Earlier-layer selection uses `export_unfused_inference_state_dict()` with
  the recipe's `inference_model_unfused`, retaining adapters and `P` while
  running without a teacher. Single-head export rejects earlier layers.
* `export_original_core_state_dict()` is for initialization/comparison only;
  it discards the residual and must not deploy a trained RADIO model.
* Native resume requires all six adapter tensors **and** the `P` buffer
  (presence, shape, finiteness) before mutation, even under `strict=False`.
  Legacy one-projection RADIO checkpoints are not silently reinterpreted.
  `initialize_from_legacy_state_dict()` is the explicit path that leaves only
  the declared new tensors fresh.
* The teacher stays frozen, lazy, local-only and outside the module tree; it
  never appears in checkpoints and never runs at inference.

## Action modes

`action_mode` selects the packed action representation on both heads and the
codec: `joint_only` (default, legacy), `end_effector_only`, or `dual`, which
packs `[joint coordinates, end-effector coordinates]` in that order into the
existing parameter-free repeat-into-128 layout. `joint_action_dim` /
`end_effector_action_dim` describe the layout; `codec.split()` separates the
two coordinate sets for dual. Decoding still returns `[B,H,A]` with
`A = joint / end-effector / joint + end-effector`. No mode adds learned weights
and no model/config keeps a second inference architecture.

Normalization stays in the data path. `PatchWAMParquetDataset` accepts
`action_mode`, `joint_action_key`, `end_effector_action_key` plus dimension and
statistics-key metadata, reads the configured source fields through the narrow
`ParquetDataset._get_action_value` hook and derives `stats['action']` for the
selected coordinates from the per-source statistics. End-effector targets are
opaque normalized coordinates supplied by the dataset: they are never
generated with FK/IK and never copied from observation states.

The RobotWin recipe is configured through environment variables:
`PATCHWAM_ACTION_MODE` (default `joint_only`), `PATCHWAM_JOINT_ACTION_KEY`
(default `action`), `PATCHWAM_EE_ACTION_KEY` (required for the end-effector and
dual modes), `PATCHWAM_JOINT_ACTION_DIM`/`PATCHWAM_EE_ACTION_DIM` (default 14)
and `PATCHWAM_*_ACTION_STATS_KEY` overrides. Selecting a mode without an
end-effector source fails at config construction.

## Scope and limitations

Full-size C-RADIO/FLUX.2 training and robot benchmarks are not covered here; the
CPU suites exercise reduced official FLUX.2 blocks, deterministic tiny models
and injected teachers. The shared-noise objective is lossy and couples RADIO
noise to the 128-wide parent noise, so it is not an exact RADIO flow; no
performance, control or cost improvements are claimed.
