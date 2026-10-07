# Phase 2 correction review — native configs still fail

The corrected numerical/model work is promising, including genuine tiny Qwen
and official FLUX blocks. However, the reported correction does not satisfy
the already approved native pipeline contract. Root ran the **shipped configs**
against the worker's real LeRobot fixture, filling actual episodes statistics
and camera files so the native wrapper computes statistics itself:

```
patchwam_libero_full_finetune: KeyError 'states'
patchwam_robotwin_vlm_full_finetune: ValueError decoded_frames is required to gather history
patchwam_robodojo_vlm_history_example: ValueError decoded_frames is required to gather history
all three: inference section absent; validation dataset training flag absent
```

This further correction is justified by concrete unusable user-facing configs
and native inference entrypoints. It finishes the previous contract; it adds
no new feature scope. Preserve accepted core/VLM/checkpoint work.

1. **Use the shipped configs in end-to-end tests.** The current pipeline test
   handwrites another `_transforms()`/`_dataset()` and supplies `dataset_statistics`
   with a `states` key, hiding the shipped configs' `proprio` mapping. Derive the
   dataset/transform/collator/normalizer configuration from each actual config.
   Build fixtures with native metadata statistics, correct dimensions, cameras,
   and subtask labels; let `DistributedRepeatingDataset` aggregate stats. Override
   only fixture paths, available tiny assets/model widths, and execution device.
   Exercise all three train and validation chains plus native `model(**batch)`.
   Restore/save/load and native LIBERO inference must run through those configs.

2. **Fix normalization names and native inference schema.** All configs map
   observation state stats to `proprio`, while normalizers look up `states`.
   Align the stats key used by training and `LiberoProprioFromInputs` (keeping
   native `proprio` is straightforward). Define `inference` as required by
   `scripts/inference.py`; an `eval` block alone cannot run the documented CLI.
   Put `norm_stats_key` on `LiberoEvalRunner` itself, since it overwrites the
   dataset field with its default `libero_10_no_noops`, which disagrees with
   training's `libero_all_no_noops` key. Check the runner's actual signatures.
   Native `LiberoParquetEvalDataset` drops `task_description`; PatchWAM needs
   raw prompts. Add a narrow subclass or appropriate native extension retaining
   prompts/required PatchWAM fields, rather than only manually calling prediction
   on training tensors. Test the real eval dataset call and action denormalizer
   on raw observation fixtures; patch only the existing `.cuda()` device transfer
   if required for CPU testing. Assert the task instruction reaches the encoder.

3. **Finish direct history decoding.** `PatchWAMDecodeFrames` emits world/history
   frames, not `decoded_frames`, but history configs immediately call
   `PatchWAMSelectHistory` on VL absolute rows. Decode the distinct VL timestamps
   directly through the native decoder, zero invalid slots, and remove that
   stale transform dependency from these configs. Test world and VL history
   have their respective clocks/masks and never cross episodes/roots, with
   distinguishable fixture pixels. Validate history rows before accessing the
   native dataset; don't trust frame arithmetic without bounds/root/episode checks.

4. **Expose and configure dataset training mode.** Dataset constructor still
   unconditionally sets `self.training=True`; validation configs never call
   `eval()`. Add an explicit constructor flag, set it from `_dataset(training)`,
   and test actual validation sample values across RNG seeds with nonempty valid
   history. Distinct dicts and disabled image augmentation alone do not disable
   history jitter/dropout. Preserve stochastic training settings.

5. **Finish source image preprocessing/recipes.** RobotWin's source
   `robotwin_vlm.yaml` is frozen **current-image only**, so remove the added
   20-slot history; update tests to compare the actual source recipe rather than
   assert the newly invented setting. RoboDojo remains 20-slot/separate-view.
   Resize each decoded world camera to source shape **before** augmentation:
   LIBERO 224x224, RobotWin 240x320, RoboDojo 256x256. World history must be
   resized/augmented at the matching source geometry. Keep clean VL current/
   VL history at the copied source's intended pre-augmentation stage and size.
   Separate views currently retain raw video resolution because ComposeWorld
   only resizes the unused mosaic. Add geometry/augmentation fixture parity
   checks using nonuniform pixels. Label RoboDojo's unknown max-update budget
   explicitly as an example adjustment, or support a required native max_steps.

6. **Sanitize invalid rasters before encoding/perturbation.** The revised policy
   hides invalid future keys, but `encoding.py` still passes masked NaN current/
   future pixels into AE, and policy perturbs invalid future targets before
   sanitization. Complete the existing requested sanitization contract before
   AE and perturbation; when masks are absent retain exact source parity.
   Test raw invalid pixels with a genuine small AE that cannot safely consume
   NaNs, then finite losses/gradients and valid-action invariance. Include current
   and future camera masks, not only preencoded future tokens.

7. **Finish online range/resize contract.** The corrected buffer chooses RGB
   properly, but simply buffers raw input without range normalization or world
   resize; `_vl_history` also just resizes those values. For [0,1]/uint8 decoded
   frames this doesn't match training [-1,1], and the `auto` 4D layout always
   treats [V,3,H,W] as time. Specify/validate ambiguous layouts, input range,
   world size, and clean VL size; produce train-compatible values/masks. Keep
   unsupported benchmark history evaluation explicitly documented rather than
   claiming the helper alone wires it.

8. **Refresh the documentation/report to final facts.** PHASE2_REPORT retains
   old claims (byte-identical VLM, layer FSDP, absent decoder) contradicted by
   its appended correction. Rewrite the final report and attribution around
   final behavior and exact test commands/results. Document the working native
   inference CLI, remaining pretrained/GPU/rollout limitations, and source recipe
   adjustments. Do not claim deterministic validation until samples test it.

One focused end-to-end integration correction should address this complete
batch. Do not repeat unrelated BF16 failures or start downloads/paid probes.
