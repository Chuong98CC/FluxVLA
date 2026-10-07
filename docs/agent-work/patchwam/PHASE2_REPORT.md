# Phase 2 report — `patchwam-integration`

- Status: **ready_for_review**
- Task ID: `patchwam-integration`
- Workspace: `/home/chuong/workspace/FluxVLA`
- Source (read-only): `/home/chuong/workspace/PatchWAM`
  `dea19d21182a26a8ee16efc2c4125ee7aecf44fb` (verified clean)
- Interpreter: `/home/chuong/miniconda3/envs/fluxvla/bin/python`
  (Python 3.10.22, torch 2.8.0+cu128, transformers 5.3.0, torchvision 0.23.0,
  torchcodec, pyav). CPU runs use `CUDA_VISIBLE_DEVICES=''`.

## What ships

- **Vendored core** `models/third_party_models/patchwam/`: `codec`, `geometry`,
  `flow`, `policy`, `tiny`, `encoding`, `flux`, `vision_language`, and
  `data/{augmentation,appearance,history,cameras}`. Copied modules are
  byte-identical to source except documented FluxVLA-local changes:
  `policy`/`geometry` gained optional `future_valid` masking (no-op when absent),
  `encoding.py` is a local extraction that also sanitizes invalid rasters before
  the autoencoder, and `vision_language.py` has two targeted Transformers-5.3
  compatibility adaptations (`get_image_features` return object and
  `get_rope_index(mm_token_type_ids, image_grid_thw=...)`). See `ATTRIBUTION.txt`.
- **Registered natives**: `PatchWAMVLA` (`VLAS`), `PatchWAMBackbone`
  (`VLM_BACKBONES`), `PatchWAMHead` / `PatchWAMTinyDenoiser` /
  `PatchWAMFluxDenoiser` (`HEADS`), `PatchWAMParquetDataset` /
  `PatchWAMLiberoEvalDataset` (`DATASETS`), and the PatchWAM transforms
  (`PatchWAMRawInputs`, `PatchWAMDecodeFrames`, `PatchWAMResizeCameras`,
  `PatchWAMSelectWindow`, `PatchWAMSelectHistory`, `PatchWAMEndpoints`,
  `PatchWAMEvalPack`, `PatchWAMCleanVL`, `PatchWAMWorldAugment`,
  `PatchWAMComposeWorld`, `PatchWAMWorldHistory`, `PatchWAMKeep`,
  `PatchWAMHistoryBuffer`).
- **Configs** under `configs/patchwam/` with a complete `DDPTrainRunner`
  (collator/optimizer/lr_scheduler/metric in the runner), independent train and
  validation dataset objects, `action_key='action'`, native `proprio` statistics
  naming, source padded-delta normalization, source ClipAugment recipes, and a
  LIBERO `eval` + `inference` (`LiberoEvalRunner`) block with `norm_stats_key` on
  the runner and a `PatchWAMLiberoEvalDataset` that keeps the raw instruction.
- **Docs** `docs/patchwam.md`.

## Verification (exact commands)

| Command | Exit | Result |
| --- | --- | --- |
| full PatchWAM suite (10 files, see report body) | 0 | **83 passed** |
| checkpoint regressions (bare/compact/native + missing/shape/unexpected) | 0 | **2 passed** |
| affected dataset regressions | 0 | **24 passed** |
| `/tmp/patchwam_parity.py` + `/tmp/patchwam_phase2_parity.py` | 0 | **ALL PASS** |
| `python -m py_compile` + `git diff --check` + whitespace scan | 0 | clean |
| prior manifest run (`cpu-tests.txt`, before this bundle) | 1 | **130 passed, 2 failed** — both pre-existing `test_vla.py::test_tiny_pi_forward_backward[PI0/PI05]` BF16 DNNL failures (unrelated). A fresh full-manifest rerun was declined by automatic approval review as outside the focused-correction scope; the added checkpoint file's 2 tests are covered by the focused suite above. |

Key genuine checks:

- **Shipped-config end-to-end** (`test_patchwam_pipeline.py`): a real on-disk
  LeRobot fixture (parquet rows + mp4 videos written with `av`, decoded with
  torchcodec) carrying native `episodes_stats.jsonl`. `DistributedRepeatingDataset`
  aggregates the statistics itself under `proprio`/`action`, and each of the
  three shipped configs' train and validation chains run through the configured
  transforms, `DictCollator`, and a tiny-backed real model with finite loss and
  an optimizer step on training. No state/action/decoded tensors are injected.
- **Native LIBERO inference path**: the shipped `inference`/`eval` block is a
  `LiberoEvalRunner` with `norm_stats_key` and `PatchWAMLiberoEvalDataset`; the
  test calls the real eval dataset on a raw LIBERO observation and asserts the
  raw instruction reaches the encoder inputs.
- **Real adapter stacks**: `test_patchwam_vision_language.py` runs a real tiny
  Qwen3VL forward/backward with the local Qwen3 tokenizer (subtask causality and
  LoRA gradients with frozen base/vision); `test_patchwam_flux_adapter.py` builds
  a small real official `Flux2` and runs the copied `OfficialFluxDenoiser`.
- **Checkpoint regressions restored** (`test_patchwam_checkpoint.py`): bare,
  compact `policy.*`+adapter, and native full checkpoints load through
  `from_pretrained`; missing, shape-mismatched, and unexpected keys raise before
  any parameter is mutated.
- **Native inference at shipped geometry**: the LIBERO eval test runs the real
  eval dataset at 224x448, normalizes with statistics aggregated by
  `DistributedRepeatingDataset`, calls `model.predict_action(**batch)`, applies
  the configured `DenormalizeLiberoAction`, and proves the instruction reaches
  the encoder (different prompts change the prediction).
- **Normalization/inverse-decoding parity** (`test_patchwam_parity.py` vs source
  `RobotFeatureCodec`/`FeatureScaler`) plus `test_patchwam_flux_adapter.py`'s
  real-projection width validation.

## Review corrections applied

1. Configs define `runner`; collator/statistics/actions/normalizer/decoder are in
   native locations; tests derive from the shipped configs and native statistics.
2. Stats use the native `proprio` key end-to-end; `inference` is defined for the
   CLI; `norm_stats_key` lives on `LiberoEvalRunner`; a narrow
   `PatchWAMLiberoEvalDataset` keeps raw prompts.
3. VL history is decoded directly from its own clock in `PatchWAMDecodeFrames`;
   stale `PatchWAMSelectHistory` dependencies were removed from the configs;
   world/VL history rows are episode/root-bounded and validated.
4. `PatchWAMParquetDataset(training=...)` is set from the config; the pipeline
   test asserts deterministic validation history across RNG seeds.
5. Source recipes: LIBERO/RobotWin mosaics, RobotWin current-image-only (no
   history), RoboDojo separate views + 20-slot history; per-camera resize to
   source geometry before augmentation.
6. `text_dim=7680`, validated against the real denoiser `txt_in`/`img_in`.
7. Targeted Transformers-5.3 adaptations in the copied encoder (documented).
8. Invalid current/future rasters are sanitized before the autoencoder and
   before perturbation; invalid future keys are hidden from joint attention.
9. Format-aware `from_pretrained()` for bare/compact/native checkpoints with
   strict required-tensor validation.
10. FSDP wraps only the head execution boundary; inner FLUX blocks are not
    auto-wrapped.
11. Online buffer validates range, world resize, and explicit 4-D layout, and
    emits train-compatible world and clean VL history.

## Remaining limits (explicit)

- Full-size Klein-4B weights, GPU/distributed FSDP, and real LIBERO/RobotWin/
  RoboDojo rollouts are **unverified** here.
- RobotWin/RoboDojo have no native benchmark inference runner in FluxVLA, so only
  the LIBERO config ships a working `inference` block; this is documented in
  `docs/patchwam.md`.
- RoboDojo's source recipe has no fixed update budget; the config keeps the
  recipe values and does not invent one.
- Online history remains a validated helper with a documented clock contract;
  wiring a specific benchmark's per-step reset/cadence stays an eval-loop
  responsibility and is documented, not claimed.
