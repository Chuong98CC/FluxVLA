# Phase 2 acceptance review — changes required

Both specification and quality were reviewed together against the actual patch.
The numerical core and copied data utilities are useful, but Phase 2 is not yet
accepted: passing synthetic units bypassed missing native pipeline stages.

Root independently reproduced: `runner_present: False`, missing explicit
`action_key`, collator under the dataloader rather than runner,
`ValueError: decoded_frames is required to select a window`, and
`ValueError: Head frame must be CHW RGB` for a documented `[V,T,3,H,W]` online
input. All three configs also shallow-copy nested training/validation settings.

## Consolidated corrections

1. **Complete native training configs.** `scripts/train.py` builds `cfg.runner`;
   none of the new configs defines it. Use native `DDPTrainRunner` or a properly
   supported FSDP runner with its optimizer/scheduler/metric/collator fields in
   the native locations. Remove the unused generic `train` structure. Set child
   `statistic_name` to the wrapper's name, `action_key='action'`, explicit native
   name mappings, normalization, padded delta handling, and statistics inputs.
   Otherwise the child defaults to `private` statistics and `observation.state`
   as action targets. Add matching native action decoding and a working LIBERO
   eval/inference configuration and documented CLI invocation.

2. **Wire actual decoding and state processing.** The configs begin with
   `PatchWAMSelectWindow` but no native component emits `decoded_frames`. The
   claimed preceding `ProcessParquetInputs` and normalization steps are absent.
   Do not require an entire concatenated dataset's video in a tensor. Add a
   narrow registered decoder/adapter using existing video/path helpers to load
   only the requested episode-local current/future/history frames. Preserve raw
   fields needed by subsequent state/statistics transforms; produce normalized
   states/actions from actual raw parquet samples. A synthetic on-disk LeRobot
   fixture must exercise the real dataset constructor and configured chain,
   collator, native loss/optimizer step, save/load, and native inference inputs.
   Do not inject prebuilt actions/states/decoded tensors to bypass this gap.

3. **Bound visual selection as well as metadata.** `SelectWindow` ignores
   `patchwam_future_row` and slices `[row:row+horizon+1]` across episodes/roots,
   or raises at the dataset tail. Select/repeat valid endpoint frames inside the
   same episode, as the source does, with correct invalid masks. Historical
   absolute indices must likewise be checked against the actual episode/root.
   Test tail samples and adjacent episodes/roots with distinguishable pixels
   through the full transform chain, not just the metadata hook.

4. **Separate validation configuration and mode.** Replace shallow copies with
   independent native config objects. RobotWin/RoboDojo assignments currently
   replace training transforms through the shared nested dictionaries; LIBERO
   validation retains training augmentation. Give datasets an explicit training
   mode at construction; validation history must disable jitter/dropout and
   world augmentation without changing training. Assert distinct configs and
   deterministic validation samples across different RNG seeds.

5. **Restore source recipe behavior.** Compare all three configs with the named
   sibling YAML recipes. LIBERO and ordinary RobotWin encode mosaics; their
   `separate_views=True` currently causes the backbone to prefer separate-camera
   tokens and different positional groups. Resize each camera to its source
   geometry before augmentation. LIBERO's p=0.65/operation recipe, default
   `isolate_actions=False`, frozen current-image RobotWin VL behavior, and
   RoboDojo's 4B/20-slot/required-subtask contract must not become new experiments
   silently. Use the original camera order, normalization, and training values
   where the config represents that recipe. Clearly label any example-specific
   adjustment. Preserve subtask fields through transforms and meta collation.

6. **Use real encoder/denoiser dimensions and validate them.** All configs use
   text_dim=4096, whereas the copied encoders concatenate three hidden-state
   readouts; that width cannot be 4096 for the configured three-layer readout.
   Obtain the correct width from the chosen official Klein config and encoder
   assets/source; validate token/text widths against the actual transformer's
   input projections at construction. Validate injected modules too. Don't
   just make two incorrect config fields agree. Test real small official FLUX.2
   blocks and real tiny HF Qwen modules using existing local source/dependencies
   where available. No pretrained-weight download is needed for these tests.

7. **Test the actual copied VLM forward.** The present test instantiates LoRA on
   stub modules and never runs `_prepare_sample` or the real language/vision
   stack. Run a tiny actual Qwen3-VL forward in the installed Transformers
   environment, with processor/tokenizer fixtures, history, adapters, labels,
   and backward. Make only targeted version compatibility adaptations to the
   copied module if needed, retaining behavior and attribution. Assert that
   changing future targets/subtask labels cannot change conditioning features,
   and that LoRA receives gradients while base/vision/AE remain frozen. Retain
   CPU tests without falsely equating them with pretrained/GPU validation.

8. **Mask invalid image values before joint attention.** The `future_valid`
   extension only masks MSE; invalid/NaN `future_tokens` still enter perturbation
   and joint attention and can corrupt valid action predictions. Sanitize
   invalid future/current pixels/tokens before encoding/perturbation, and keep
   invalid future keys out of joint attention when masks are supplied. Preserve
   exact source numerical behavior when those optional masks are absent. Add
   tests with NaNs/extreme values in invalid rasters proving finite losses,
   finite gradients, and invariance of valid action losses/predictions.

9. **Fix actual checkpoint initialization.** The source's `policy_weight_state`
   and `load_policy_weights` save head-only unprefixed keys when there are no
   LoRA adapters, or `policy.*` plus only trainable `text_encoder.*` adapters
   when LoRA is enabled; frozen AE/Qwen assets are not saved. The new default
   mapping/report expects a full FluxAssetPolicy state, and the report helper is
   not called by `from_pretrained`. Default non-strict mapped loading may
   silently skip all bare policy weights; native full weights also should not
   be automatically mapped into nonexistent source prefixes. Implement
   format-aware initialization using the native loading entrypoint with strict
   validation of all required trainable tensors/adapter keys, shapes, and
   unexpected keys before mutation. Keep frozen assets loaded separately for
   compact source weights. Test actual safetensors files for bare policy,
   policy+LoRA, and native full checkpoints plus malformed/missing tensors.

10. **Use FSDP call boundaries that actually execute.** The copied
    `OfficialFluxDenoiser` accesses block parameters via `_paired_block` and
    `_unified_block` instead of invoking the block's `forward`. Auto-wrapping
    those blocks in FSDP bypasses their all-gather hooks. Wrap a called module
    such as the whole head/denoiser, or introduce a justified callable adapter
    before enabling inner block wrapping; do not advertise unsafe layer wrapping.
    Default recipes can use the existing DDP runner. Root authorizes revising
    the original layer-type wrapping contract to this correct execution boundary.
    Record GPU/distributed FSDP as unverified where hardware is unavailable.

11. **Complete online input handling.** `PatchWAMHistoryBuffer._current_frame`
    indexes documented `[V,T,3,H,W]` as `[head,0,0]`, dropping RGB. Fix and test
    all accepted layouts. Ensure buffered frame range/resizing and `vl_*` versus
    world history match training. Wire the normal no-history inference/eval path
    to native inputs. For the history example, use existing timestamp/reset
    signals with preprocessing every observation where supported, otherwise
    explicitly fail and document the required clock contract; a helper alone is
    not an implemented benchmark history-evaluation pipeline.

## Verification required after correction

- New regression tests must exercise all concrete defects above. Existing source
  parity, affected dataset regressions, and corrected PatchWAM tests must pass.
- Real on-disk sample-to-configured-training and native inference tests, with
  actual fixture decoding, are mandatory. Mock only unavailable pretrained
  assets with genuine small tensor modules, not missing pipeline stages.
- Config tests must validate `runner`, collator/statistics/actions/norm/decoder
  placement and source recipe values, as well as registration.
- Source normalization/inverse decoding parity needs actual fixtures; the initial
  report had no such test despite claiming native normalization ownership.
- Check actual native from_pretrained/load/resume APIs; state_dict cloning and a
  string-key report alone do not verify these paths.
- Run focused corrected tests plus the accepted affected dataset checks;
  don't repeat the unrelated known PI0/PI05 BF16 failures.
- Report exact interpreter/commands/results and remaining GPU/asset limits.

This is the single consolidated Phase 2 correction request. The additional
architecture checks are justified by concrete unusable training configs,
episode-boundary defects, checkpoint initialization gaps, and unsafe FSDP
execution boundaries. The same worker retains ownership for the correction.
