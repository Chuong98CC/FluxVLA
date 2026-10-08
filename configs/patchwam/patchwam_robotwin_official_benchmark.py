# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Official PatchWAM RoboTwin 2.0 C2R benchmark protocol.
#
# This config applies the released protocol published with
# ``QZWang/patchwam-robotwin-c2r-step140000`` (seed 0, action horizon 16,
# replan 16, 10 denoising steps, unseen instructions, expert seed validation,
# 100 episodes per task per condition) on top of the native single-task smoke
# config. It intentionally changes nothing about the model, assets, embodiment
# (``aloha-agilex``) or normalization contract.
#
# The 50 released task names and the clean/random conditions live in
# ``patchwam_robotwin_official_benchmark_tasks.json`` next to this file; that
# manifest is the source of truth consumed by
# ``tools/patchwam_official_benchmark.py``. Running the whole 50 x 2 x 100
# suite is the launcher's job, not this config's; the config only fixes the
# per-case protocol so a single case can also be launched by hand.

_base_ = './patchwam_robotwin_c2r_eval.py'

seed = 0

eval = dict(
    seed=seed,
    episodes_per_task=100,
    # The released implementation picks the instruction with a global
    # ``np.random.choice`` over the generated variants. The native smoke
    # default stays ``seeded`` (deterministic per episode seed); the benchmark
    # opts into the additive ``official`` selection.
    instruction_selection='official',
    instruction_pool_size=100,
    output_dir='runs/evaluation/patchwam_robotwin_official',
)
