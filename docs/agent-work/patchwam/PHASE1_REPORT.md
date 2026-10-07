# Phase 1 report — `patchwam-core` (native core and numerical contract)

- Status: **ready_for_review**
- Task ID: `patchwam-core`
- Feature workspace: `/home/chuong/workspace/FluxVLA`
- Source (read-only): `/home/chuong/workspace/PatchWAM` @
  `dea19d21182a26a8ee16efc2c4125ee7aecf44fb` (verified clean after the run)
- Interpreter: `/home/chuong/miniconda3/envs/fluxvla/bin/python`
  (Python 3.10.22, torch 2.8.0+cu128, transformers 5.3.0). CPU tests run with
  `CUDA_VISIBLE_DEVICES=''`.

## Changed paths and behavior

- `fluxvla/models/third_party_models/patchwam/` (new package)
  - `codec.py`, `geometry.py`, `flow.py`, `tiny.py`, `policy.py` copied verbatim
    from `PatchWAM/src/patchwam/models/`. Copied filenames are byte-identical to
    the source (`sha256` matches: codec `1eb10adf…`, geometry `f4c77b74…`,
    flow `3f68d1e9…`, tiny `d0cd82ef…`, policy `3a38428e…`).
  - `__init__.py` exports only the numerical core. The asset-dependent
    `flux.py` / `vision_language.py` are intentionally deferred to Phase 2 so
    the core imports without model assets.
  - `LICENSE` (Apache-2.0) and `NOTICE` copied from the source; `ATTRIBUTION.txt`
    records origin, revision, license, the relative-import preservation, and the
    deferred files.
- `fluxvla/models/heads/patchwam_head.py` (new)
  - `PatchWAMHead(PatchFlowPolicy)` registered in `HEADS`. Builds the denoiser
    from an `nn.Module` or a `type`-keyed config; `PatchWAMTinyDenoiser`
    (registered in `HEADS`) is the offline default and wraps the vendored
    `SmallJointTransformer`. Rejects nonzero `self_flow_variant` with a clear
    error (EMA/Self-Flow deferred).
- `fluxvla/models/vlas/patchwam_vla.py` (new)
  - `PatchWAMVLA(BaseVLA)` registered in `VLAS`; builds `vlm_backbone` (optional,
    Phase 2) and `vla_head` through the existing builders; sets
    `all_module_keys = ['vlm_backbone', 'vla_head']`.
  - `forward(...)` returns the native loss dictionary (`loss`, `loss_video`,
    `loss_action`, `sigma_mean`). `predict_action(...)` returns normalized
    `[B, H, A]` actions with a default of ten solver steps and per-call override.
  - `resolve_action_padding(...)` is the single explicit adapter that converts
    native `True`-means-valid masks to the source `action_is_pad` /
    `action_dim_is_pad` (True means padded) contract, preserving time and
    dimension validity separately.
- `fluxvla/models/heads/__init__.py`, `fluxvla/models/vlas/__init__.py`
  (modified, +1 line each): added `patchwam_head` / `patchwam_vla` entries to
  the existing lazy-import tables.
- `test/test_models/test_patchwam.py` (new): 13 CPU tests.
- `docs/agent-work/patchwam/PHASE1_REPORT.md` (this file).

No fake backbone is built and no full-asset coverage is claimed. The head/
backbone ownership boundary is already in place: `PatchWAMVLA` calls
`self.vlm_backbone.encode(request, target=...)` when a backbone exists and
otherwise consumes pre-encoded tensors, and the head owns the denoiser, proprio
projection, codec, and flow objective.

## Verification

| Command | Exit | Salient result |
| --- | --- | --- |
| initial `pytest --cpu-model-tests test/test_models/test_patchwam.py -q` | 1 | `ModuleNotFoundError: … third_party_models.patchwam` (recorded failing-first state) |
| `CUDA_VISIBLE_DEVICES='' python -m pytest --cpu-model-tests test/test_models/test_patchwam.py -q` | 0 | **13 passed** |
| `CUDA_VISIBLE_DEVICES='' python -m pytest --cpu-model-tests $(cat test/cpu-tests.txt) -q` | 1 | **51 passed, 2 failed** — both failures pre-existing/environmental |
| `python -m py_compile` on all 5 vendored + 2 wrapper + 1 test files | 0 | syntax OK |
| `git diff --check` (tracked) + `--no-index --check` scan (untracked) | 0 | no whitespace/conflict markers; all files end with newline |
| lazy-registration helper for `patchwam_head` / `patchwam_vla` | 0 | both symbols import and register (`HEADS`/`VLAS` lookups True) |
| `/tmp/patchwam_parity.py` three-way parity | 0 | **ALL PARITY CHECKS PASS** |

The two CPU-manifest failures are
`test/test_models/test_vla.py::test_tiny_pi_forward_backward[PI0FlowMatching]`
and `[PI05FlowMatching]`, both `RuntimeError: DNNL does not support bf16/f16
backward on the platform with avx2_vnni_2`. They reproduce when run in
isolation, and `test_vla.py` contains no `patchwam` references, so they are
environmental and unrelated to this change.

## Numerical parity evidence

`/tmp/patchwam_parity.py` loads the sibling source `models/` package under a
synthetic name, builds the tiny policy in source and vendored copy with the same
seed, and drives all three paths with identical tiny weights, sigma
`[0.3, 0.7]`, and Gaussian noise:

```
[PASS] copy parameters == source parameters
[PASS] source/copy loss                max_abs_delta=0.0
[PASS] source/copy gradients           max_abs_delta=0.0
[PASS] source/wrapper loss             max_abs_delta=0.0
[PASS] source/wrapper gradients        max_abs_delta=0.0
[PASS] source/wrapper seeded/initial-latent samples   max_abs_delta=0.0
```

The committed `test/test_models/test_patchwam.py` locks the copy↔wrapper half of
this contract (identical weights, `sigma`, and noise; loss, gradient, and
`sample_actions` with explicit initial latents) plus mask polarity/dimension
time separation, all-invalid/NaN behavior, native prediction shape and seed
determinism, an optimizer step, and a checkpoint round trip.

## Post-review correction: `resolve_action_padding` broadcast

Review reproduced a defect: when a coordinate mask `[B, H, A]` was combined with
a separable dimension mask `[B, A]`, the code OR-combined the two tensors
without a horizon axis. Reproduced before the fix with actions `[2, 3, 4]`,
`action_masks = ones[2, 3, 4]`, `action_dim_masks = ones[2, 4]`:

```
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at
non-singleton dimension 1
```

The same right-aligned broadcast silently cross-wired batch and horizon when
`B == H`. The fix expands a `[B, A]` dimension mask to `[B, 1, A]` only when a
coordinate mask is present, and keeps the separable `[B, A]` output otherwise
(`fluxvla/models/vlas/patchwam_vla.py`).

Regression coverage added in
`test/test_models/test_patchwam.py::test_dimension_mask_broadcast_has_no_cross_batch_contamination`:

- `B != H` (`2, 3, 4`): coordinate mask + distinct per-sample dimension mask
  combine to exactly `(~coordinate) | (~explicit)[:, None, :]`; sample 0 keeps a
  dimension that sample 1 pads, and vice versa.
- `B == H` (`3, 3, 2`): distinct per-sample invalid dimensions and temporal
  coordinate pads; asserts `dim_pad[:, 0, 1] == [True, False, False]`, i.e. no
  cross-batch contamination from the would-be right-aligned broadcast.
- No coordinate mask: the output stays `[B, A]`.

Evidence after the fix:

- `resolve_action_padding([2,3,4], ones[2,3,4], ones[2,4])` → `dim_pad` shape
  `(2, 3, 4)` (no error); with a `[B, H]` step mask only → `(2, 4)`.
- `CUDA_VISIBLE_DEVICES='' python -m pytest --cpu-model-tests
  test/test_models/test_patchwam.py -q` → exit 0, **13 passed**.
- Three-way parity script → exit 0, **ALL PARITY CHECKS PASS** (unchanged).

## Remaining limits

- No FLUX.2 denoiser, official checkpoints, Qwen encoder, real assets, GPU
  training, or benchmark parity were exercised. Everything here is CPU,
  randomly initialized tiny weights.
- `frame_masks` / `img_masks` are accepted for interface compatibility but do
  not constrain future latent tokens in Phase 1 (no raster). They apply once the
  Phase 2 `PatchWAMBackbone` exists.
- `PatchWAMHead.get_fsdp_wrapping_policy` is not yet layer-type aware; Phase 2
  refines it for the official denoiser.
- `flake8`/`isort` are not installed and the pinned `yapf==0.32.0` is not
  available in this env (0.43.0 differs from the repo's pinned style). Native
  files follow the repo's evident convention; the coordinator should run
  pre-commit once.
- The new test file is not yet added to `test/cpu-tests.txt` (out of the owned
  path list) — the coordinator may add `test/test_models/test_patchwam.py`
  there.

## Review dispositions

- Tiny denoiser in `HEADS`: **approved** by review.
- `sigma_mean` alongside the native `loss`/`loss_video`/`loss_action`: **approved**.
- Official denoiser: review accepts a registered `PatchWAMFluxDenoiser` wrapper
  in `HEADS` with real asset/shape validation, rather than passing tiny-only
  kwargs through. This is Phase 2 work.
- `frame_masks`/`img_masks` completion and layer-aware FSDP remain planned for
  Phase 2.

## Checkpoint (if this run is interrupted)

Implementation and CPU verification of `patchwam-core` are complete; the two
`../../` paths in the test table show recorded evidence. Resume by re-reading
this file, confirming `git status` still shows only the five new files plus the
two one-line registry edits, then re-running the patchwam CPU test and the
parity script. Phase 2 (`patchwam-integration`) has not started.
