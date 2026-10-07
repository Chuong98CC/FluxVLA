# PatchWAM integration acceptance

Accepted on 2026-10-07 for the joint world/action flow scope in DESIGN.md.
Implementation remains uncommitted in `/home/chuong/workspace/FluxVLA`.

## Verified result

- Native registered VLA/backbone/head wrappers, copied PatchWAM VLM/core and
  attribution, native Parquet dataset extension, decoded episode-bound windows
  and distinct world/VL history, source augmentation and camera geometry.
- Three source-based native DDP training configs with independent validation,
  native statistics/normalization/collation, and optimizer-step fixture tests.
- Native LIBERO inference config/dataset preserves raw instructions; prediction
  at the shipped 224x448 mosaic size is finite, and changing the instruction
  changes actions with identical sampling noise. The configured native decoder
  runs per action, matching LiberoEvalRunner.
- Bare source policy, compact policy+LoRA and full native checkpoints load through
  `from_pretrained`, with strict validation before parameter mutation. The native
  inference CLI consumes native checkpoints through its existing eval loader.
- FSDP wraps the called head boundary only. Inner FLUX blocks are manually
  executed by the copied adapter and are not safe auto-wrap boundaries.

## Test evidence

Interpreter: `/home/chuong/miniconda3/envs/fluxvla/bin/python`.

Worker: all ten PatchWAM test files **83 passed**, affected existing dataset
regressions **24 passed**, both source-parity scripts passed (numerical core
deltas zero), syntax and whitespace checks passed. Genuine tiny Qwen3-VL and
small official FLUX.2 blocks ran without downloaded pretrained weights.

Root's focused acceptance command (after final test strengthening):

```
CUDA_VISIBLE_DEVICES='' /home/chuong/miniconda3/envs/fluxvla/bin/python -m pytest --cpu-model-tests test/test_configs/test_patchwam_config.py test/test_datasets/test_patchwam_pipeline.py test/test_models/test_patchwam_checkpoint.py -q
```

Result: **14 passed**, including the shipped config-derived train/validation
fixtures, real raw-observation inference and actual checkpoint files. Root
retained the same sampling seed across prompts and switched the test to the
exact configured per-action decoder. Source code was unchanged by those final
test/doc integration edits.

The earlier broad CPU manifest had 130 passed and two known unrelated PI0/PI05
BF16 DNNL backward failures. A redundant final broad rerun was rejected by
automatic approval review as outside the focused correction scope; it is not a
required outstanding check. No workaround or extra permission request was used.

## Limits

- Full-size pretrained weights/GPU training, distributed FSDP and actual robot
  benchmark rollouts were not validated.
- RobotWin/RoboDojo native benchmark runners are absent; their configs cover
  training. Online history is a tested helper requiring explicit clock/reset
  signals, not a wired benchmark history eval loop.
- EMA and Self-Flow optimizer/teacher hooks remain explicitly unsupported.
- Pinned formatting tools were unavailable; syntax, whitespace and native style
  were checked without changing the environment.

## Workflow

Root remained user-authorized `gpt-6.1-sol`. One native Flash writer was reused
for both phases and corrections, then interrupted after completion. Host and
router records verified `deepseek/deepseek-v4.1-flash`, high effort, provider
`deepseek` and successful requests during implementation. No paid setup probe,
credentials, downloads, commits, pushes or deployments were introduced.
