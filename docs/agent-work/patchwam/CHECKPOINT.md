# PatchWAM integration checkpoint

- State: both phases accepted. Native joint-flow port implemented and focused
  acceptance checks pass. Final scope/evidence/limits are in ACCEPTANCE.md.
- Feature workspace: `/home/chuong/workspace/FluxVLA`.
- Original baselines and routing evidence: BASELINE.json.
- New coordinator artifacts: DESIGN.md, PLAN.md, CHECKPOINT.md, BASELINE.json.
- Current session: `01a117bd-84a5-7703-84dd-60e6815f4dca`.
- Active model verified from this session's `turn_context`: `gpt-6.1-sol`, xhigh.
  Doctor's observed default agrees. No root model or effort changes were made.
- Native worker `/root/patchwam_port`, session
  `01a11843-2215-7860-aff2-94b61a8806bb`: role `astra_flash_builder`,
  `deepseek/deepseek-v4.1-flash`, high effort confirmed in host metadata.
  Router usage records confirm the Flash route, provider `deepseek`, and HTTP
  status 200 during this useful implementation task. No paid probe was run.
- Root blocker resolved: user explicitly authorized `gpt-6.1-sol` and instructed
  the agent to ignore the skill's Astra-only root requirement.
- Evidence: 13 corrected core tests passed; worker's three-way source/copy/wrapper CPU
  parity has max absolute delta zero for losses, gradients, and sampling.
  Two existing PI0/PI05 BF16 CPU backward tests fail in this environment.
- Review finding resolved: [B,A] dimension masks acquire the horizon axis when
  combined with [B,H,A] coordinate masks. Exact unequal/equal B/H regressions
  pass and the original separable output shape remains unchanged.
- Architecture decisions: tiny and official denoiser config wrappers may use
  HEADS; source sigma_mean diagnostic is retained. Component ownership remains
  disjoint. FSDP wraps only the called head boundary; inner FLUX blocks are
  manually executed and must not be wrapped individually.
- Phase 2 worker evidence: 50 PatchWAM units and 24 affected dataset tests passed;
  source parity passed. Review found synthetic tests bypassed actual decoding,
  native runner construction, normalization, and source checkpoint loading.
- Findings and revised FSDP call-boundary decision: PHASE2_REVIEW.md.
- First correction evidence: 72 PatchWAM tests and 24 dataset regressions;
  genuine tiny Qwen3-VL and small official FLUX.2 forward/backward pass.
  Root reproduced shipped-config failures on real parquet/MP4 fixtures using
  native aggregate stats: LIBERO KeyError states; RobotWin/RoboDojo missing
  decoded_frames for VL history; all lack cfg.inference and val training mode.
- Resolved: shipped config-derived train/val chains now pass with actual
  parquet/MP4 decoding and native aggregate stats. Source camera geometry,
  clean VL ordering, current-only RobotWin, distinct causal RoboDojo histories,
  invalid-raster sanitization, and validation mode are wired and tested.
- Final worker evidence: 83 PatchWAM tests, 24 affected dataset regressions,
  source parity, compilation and whitespace passed. Real tiny Qwen3-VL and
  official small FLUX.2 forward/backward were exercised.
- Root acceptance: 14 focused config/pipeline/checkpoint tests passed after
  preserving identical noise across prompts and using the actual per-action
  native decoder. Native raw LIBERO observations reach prediction and decoding.
- Limits: full pretrained/GPU training, distributed FSDP, actual robot rollouts,
  RobotWin/RoboDojo benchmark runners and a benchmark history loop are unverified
  or absent. EMA/Self-Flow remains explicitly unsupported as designed.
- No required implementation work remains within the accepted scope. The worker
  was interrupted after completion; no child is executing. Changes are uncommitted.
