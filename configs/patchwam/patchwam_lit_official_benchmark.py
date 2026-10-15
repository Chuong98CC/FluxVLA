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
# Official PatchWAM + LIT RoboTwin 2.0 benchmark protocol.
#
# Same released protocol as `patchwam_robotwin_official_benchmark.py` (seed 0,
# action horizon 16, 10 denoising steps, unseen instructions, 100 trials per
# task per condition, the 50 released task names), applied to the trained
# PatchWAM + LIT Stage-2 head. The LIT-specific difference is normalization:
# `norm_stats_path` points at the Stage-2 training statistics
# (`dataset_statistics.json`) so the trained weights are evaluated with the
# statistics they were trained under.
#
# Launch it through the shared RoboTwin manager so the runner contract,
# per-worker summaries and result aggregation use the common FluxVLA path:
#
#   CONFIG=configs/patchwam/patchwam_lit_official_benchmark.py \
#   CKPT=work_dirs/patchwam_lit_stage2_robotwin/checkpoints/latest-checkpoint.pt \
#     bash scripts/eval_robotwin_manager.sh

_base_ = './patchwam_lit_eval_robotwin.py'

seed = 0

# All 50 RoboTwin tasks, matching the upstream RoboTwin benchmark and
# `patchwam_robotwin_official_benchmark_tasks.json`.
_ROBOTWIN_TASK_LIST = [
    'adjust_bottle',
    'beat_block_hammer',
    'blocks_ranking_rgb',
    'blocks_ranking_size',
    'click_alarmclock',
    'click_bell',
    'dump_bin_bigbin',
    'grab_roller',
    'handover_block',
    'handover_mic',
    'hanging_mug',
    'lift_pot',
    'move_can_pot',
    'move_pillbottle_pad',
    'move_playingcard_away',
    'move_stapler_pad',
    'open_laptop',
    'open_microwave',
    'pick_diverse_bottles',
    'pick_dual_bottles',
    'place_a2b_left',
    'place_a2b_right',
    'place_bread_basket',
    'place_bread_skillet',
    'place_burger_fries',
    'place_can_basket',
    'place_cans_plasticbox',
    'place_container_plate',
    'place_dual_shoes',
    'place_empty_cup',
    'place_fan',
    'place_mouse_pad',
    'place_object_basket',
    'place_object_scale',
    'place_object_stand',
    'place_phone_stand',
    'place_shoe',
    'press_stapler',
    'put_bottles_dustbin',
    'put_object_cabinet',
    'rotate_qrcode',
    'scan_object',
    'shake_bottle',
    'shake_bottle_horizontally',
    'stack_blocks_three',
    'stack_blocks_two',
    'stack_bowls_three',
    'stack_bowls_two',
    'stamp_seal',
    'turn_switch',
]

eval = dict(
    seed=seed,
    task_suite_name=['clean', 'random'],
    task_list=_ROBOTWIN_TASK_LIST,
    num_trials_per_task=100,
    result_output_dir='runs/evaluation/patchwam_lit_robotwin_official',
)
