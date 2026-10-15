---
name: fluxvla-integrate-method
description: Use when implementing or adapting a new method in FluxVLA that changes model architecture, action prediction, training objectives, data flow, or training, inference, and evaluation engines. Always use together with astra-flash-orchestrator.
---

# Integrating methods into FluxVLA

Deliver the requested method as working FluxVLA code, registry wiring,
configuration, and focused verification. Preserve the method's mathematics and
the repository's interfaces.

## Required companion skill

The root agent must always load `$astra-flash-orchestrator` together with this
skill, before planning or execution. Reuse it if already loaded in the current
session. Its installed entrypoint is
`/home/chuong/.agents/skills/astra-flash-orchestrator/SKILL.md`; discover the
installed skill by name if that location changes. If it is unavailable, report
the missing dependency before starting the workflow.

Astra owns planning, architecture, review and integration. Follow the
orchestrator's task classification: substantial implementation goes to one
native `astra_flash_builder`; trivial edits remain with the root. A Flash
worker using these task instructions executes only its assigned bundle and
must not load the orchestration skill or delegate recursively.

## Establish the contract

Read the applicable `AGENTS.md`, existing changes, supplied method description,
and closest working recipe. Follow the repository's execution/delegation policy;
this skill supplies integration guidance, not a different execution policy.

Reuse an approved design. Otherwise state a compact integration contract:

- **Method:** objective, conditioning, architecture, training and sampling rules;
  distinguish source requirements from implementation choices.
- **Baseline:** existing VLA, backbone, head, recipe, checkpoints, and reuse.
- **Data:** inputs/targets, shapes, masks, normalization, temporal alignment,
  physical versus padded action dimensions, and train/inference availability.
- **Changes:** extension points, files, configuration keys, and dependencies.
- **Acceptance:** observable method behavior and integration checks.

Resolve missing information that determines the algorithm. Continue independent
discovery while awaiting an answer; do not replace an unspecified method with
a generic MLP, diffusion objective, or placeholder. Implementation authorization
already given by the user remains valid.

## Choose the smallest compatible extension

| Requested change | Starting point |
| --- | --- |
| Existing component combination | Configuration only |
| Action decoder or action objective | `HEADS`, `models/heads/` |
| Feature extraction or modality adapter | Relevant backbone registry or `PROJECTORS` |
| Changed model orchestration or additional objectives | Compatible VLA subclass or `BaseVLA`, `VLAS` |
| New targets, temporal windows, preprocessing | Dataset, transforms, collator, and model input path |
| Distributed execution or optimizer-step semantics | Training runner, `RUNNERS` |
| Robot observation/action loop or transport | Inference runner, `RUNNERS`; operator, `OPERATORS` |
| Simulation benchmark | Evaluation runner, `RUNNERS` |

Read [model contracts](references/models.md) for model/data changes,
[engine contracts](references/engines.md) for execution/platform changes, and
[verification](references/verification.md) before declaring completion.

## Implement and wire

Inspect the actual constructor and caller before writing the component. Register
the implementation, import it through the package's discovery path, and select
its exact registered name with `type` in a real recipe. Match constructor keys;
trace new inputs through dataset, transforms, collator, runner, and model.

Keep training `model` and optional `inference_model` compatible with the same
checkpoint. Reuse existing loading, freezing, loss-composition, and FSDP helpers.
Keep optional runtimes lazy where package discovery would otherwise require them.

Write new files without a copyright or license banner: `docs/CONTRIBUTING.md`
states that the repository-level `LICENSE` covers all contents and that new
files must not include license headers, so the
`# Copyright 2026 Limx Dynamics` / Apache License 2.0 block that some existing
`fluxvla/` files carry must not be copied into new or rewritten files.

Finish with changed files, method-to-code mapping, runnable configuration/commands,
checks actually executed, and remaining hardware/dependency limits. Separate
implementation verification from benchmark reproduction.

## Example request

“Use `$fluxvla-integrate-method` to integrate the method in my attached design
into the existing LlavaVLA/Eagle recipe. Implement training and action inference,
configuration and CPU contract tests. Preserve the current FSDP runner.”

## Tutorial sources

Use the upstream [model](https://fluxvla.limxdynamics.com/md_source/tutorials/private_model.html),
[module](https://fluxvla.limxdynamics.com/md_source/tutorials/private_module.html),
and [engine](https://fluxvla.limxdynamics.com/md_source/tutorials/private_engine.html)
tutorials for orientation. The references also capture contracts checked against
the local source; recheck them in the target checkout when applying this skill.
