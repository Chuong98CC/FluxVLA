# Verification and completion

Choose checks that expose integration failures for the method being implemented.
Use the target environment's Python interpreter. This checkout uses `test/`, not
`tests/`; inspect `test/conftest.py`, sibling tests, and the project's test commands.
Its `--cpu-model-tests` mode provides isolated imports for compatible model units;
checkpoint regressions are separately enabled and normally require CUDA.

## Minimum observable evidence

| Change | Evidence to obtain |
| --- | --- |
| Registered component | Normal package discovery resolves the exact `type`; the relevant builder constructs from representative config without manually importing the new leaf in the test |
| Recipe | Real config loader resolves the recipe; constructor keys and injected defaults match; referenced components exist |
| Action objective | Small forward/backward pass, finite scalar loss, gradients on intended trainable parameters, masked/padded and all-invalid cases |
| Inference | Expected batch, horizon and physical-action mapping; inference needs no training-only targets; sampler obeys requested settings |
| Backbone/projector | Consumer-compatible feature/mask shapes, input range, dtype/device behavior, freezing and gradient flow |
| Additional objective/inputs | New fields reach the loss; perturb valid targets and observe the intended effect; padded/invalid targets do not affect valid results; weighted total includes each term once |
| Checkpoint changes | Small save/load round trip and deliberate mapping/mismatch checks; train and inference configurations accept the intended weights |
| Training backend | Execution ownership and resume-state contract, with focused distributed checks when the backend changes |
| Robot/simulator adapter | Fake observations/environment and recording commands confirm order, dimensions, timing/reset semantics, and cleanup |

Use tiny local modules or narrow dependency substitutes where model loading would
otherwise require large checkpoints or accelerators. Exercise the actual new
logic and consumer boundary; a substitute returning the expected answer cannot
prove the method works. Keep at least one representative registry/config check
through normal discovery so import bugs are visible.

For stochastic methods, hold random seeds/noise fixed when testing invariance or
sampling behavior. Check the source's objective numerically on a small known
case when practical, including time/noise endpoints and loss reduction. A loss
decreasing once does not establish paper fidelity.

Run the focused suite with the repository's supported invocation, for example
`python3 -m pytest --cpu-model-tests test/test_models/test_pi0_flow_shapes.py -q`
for the existing PI0 shape contracts. Select tests for the changed method rather
than copying this example as its acceptance suite. Run `git diff --check` for
tracked changes and inspect new files too. Confirm that new and edited files
carry no `# Copyright 2026 Limx Dynamics` / Apache License 2.0 banner:
`docs/CONTRIBUTING.md` states that the repository-level `LICENSE` covers all
contents and that new files must not include license headers.

Keep CPU contract checks, CUDA/kernel checks, distributed execution, checkpoint
inference, hardware runs and benchmark reproduction as distinct evidence.
Execute available checks within the requested scope; state exactly which checks
were blocked by unavailable resources. Do not download large models, install
new system runtimes, or start long training/robot runs merely to fill a checklist.

## Completion report

Provide the implementation and configuration paths, how the method's main
equations or steps map to them, and commands to use the recipe. Include actual
test results and concise remaining limits or source deviations. If missing
method/data information leaves the objective unimplemented, identify that gap
instead of calling a stub complete. Keep performance and reproduction claims
tied to measurements.
