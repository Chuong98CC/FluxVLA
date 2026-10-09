| Dataset / benchmark | State family | Action family | Config families using it |
|---|---|---|---|
| LIBERO (all 4 suites) | EE pose (8D: x,y,z, axis-angle, gripper×2) | EE pose (7D delta, or 10D rot6d) | pi0, pi05, GR00T N1.5/N1.7, LLaVA-VLA, OpenVLA, SmolVLA, Cosmos3, Dit4Dit, FastWAM, DreamZero, OpenAI astra |
| Franka dual eepose (real) | EE pose (16D) | EE pose (16D) | pi05, GR00T N1.7 |
| ALOHA / AgileX | joint (14D) | joint (14D) | pi0, pi05, SmolVLA, Cosmos3, GR00T N1.5, FluxBisim |
| RoboTwin (aloha-agilex) | joint (14D) | joint (14D) | PatchWAM |
| RoboCasa GR1 | joint (29D) | joint (29D) | pi0, pi05, SmolVLA, Cosmos3, Dit4Dit, FastWAM, GR00T N1.5/N1.7, OpenAI astra |
| RoboDojo ARX-X5 | joint (14D) | joint (14D) | pi0, pi05, SmolVLA, GR00T N1.5, PatchWAM |
| UR3 Chem | joint (7D) | joint (7D) | pi0, pi05, GR00T N1.5 |
| Tron2 | joint (16D) | joint (16D) | pi05 |
| Franka dual qpos (real) | joint (16D) | joint (16D) | pi05, GR00T N1.7 |
| OLI (mobile manipulator) | joint + base pose (33D) | joint + base pose (42D) | GR00T N1.5 |
| ARM / SARM | state only | no action (`action_key=None`) | arm, sarm |

## Per-dimension layout

The family table above only classifies each vector as EE-pose or joint. The table
below expands each vector into its actual channels. Index ranges are half-open
(`[start, end)`). `abs` means an absolute value; `Δ` means state-relative
(`action − current state`), which the eval denormalizer converts back to an
absolute command by re-adding the state. The channel order matches the action
profiles in `fluxvla/datasets/utils/transformed_statistics.py`, the `_MODALITY_KEYS`
in the GR00T configs, and `fluxvla/transforms/robocasa_transforms.py`.

| Dataset / benchmark | Vector | Dim | Index → meaning | Form |
|---|---|---|---|---|
| LIBERO | state | 8 | `[0:3]` eef x,y,z · `[3:6]` eef axis-angle rx,ry,rz · `[6:8]` gripper finger qpos (2) | abs |
| LIBERO | action (7D) | 7 | `[0:3]` Δeef x,y,z · `[3:6]` Δeef axis-angle rx,ry,rz · `[6]` gripper command (1) | Δ pose, abs gripper |
| LIBERO (Cosmos3) | action (10D) | 10 | `[0:3]` Δeef x,y,z · `[3:9]` Δeef rot6d (6) · `[9]` gripper command (1) | Δ |
| Franka dual eepose | state = action | 16 | `[0:3]` left x,y,z · `[3:7]` left quat qx,qy,qz,qw · `[7]` left gripper width · `[8:11]` right x,y,z · `[11:15]` right quat · `[15]` right gripper width | abs (state and action are the same vector) |
| ALOHA / AgileX | state, action | 14 | `[0:6]` left arm 6 joints · `[6]` left gripper · `[7:13]` right arm 6 joints · `[13]` right gripper | state abs; action Δ joints, abs grippers |
| RoboTwin (aloha-agilex) | state, action | 14 | same layout as ALOHA / AgileX | same |
| RoboCasa GR1 | state, action | 29 | native order (FluxVLA / pi0 / pi05): `[0:7]` left arm · `[7:13]` left hand · `[13:20]` right arm · `[20:26]` right hand · `[26:29]` waist. N1.5 / N1.7 reorder to left arm, right arm, left hand, right hand, waist | state abs; action Δ/abs split varies |
| RoboDojo ARX-X5 | state, action | 14 | `[0:6]` left arm 6 joints · `[6]` left gripper · `[7:13]` right arm 6 joints · `[13]` right gripper | state abs; action Δ joints, abs grippers |
| UR3 Chem | state, action | 7 | `[0:6]` UR3 6 joints (shoulder pan, shoulder lift, elbow, wrist 1/2/3) · `[6]` gripper | state abs; action Δ joints, abs gripper |
| Tron2 | state, action | 16 | `[0:7]` left arm 7 joints · `[7]` left gripper · `[8:15]` right arm 7 joints · `[15]` right gripper | state abs; action Δ joints, abs grippers |
| Franka dual qpos | state, action | 16 | `[0:7]` left arm 7 joints · `[7]` left gripper · `[8:15]` right arm 7 joints · `[15]` right gripper | state abs; action Δ joints, abs grippers |
| OLI (mobile manipulator) | state | 33 | `[0:31]` whole-body 31 joints (left leg 6, right leg 6, waist 3, head 2, left arm 7, right arm 7) · `[31:33]` left/right hand-closed flags | abs |
| OLI (mobile manipulator) | action | 42 | `[0:31]` whole-body joint position targets · `[31:34]` base x,y,z · `[34:40]` base rot6d (6) · `[40:42]` left/right hand-closed flags | abs |
| ARM / SARM | state only | — | no action; reward model | — |

Notes:

- RoboCasa GR1 arm joint order is shoulder pitch, shoulder roll, shoulder yaw,
  elbow pitch, wrist yaw, wrist roll, wrist pitch (7). Waist order is yaw, pitch,
  roll (3). Each Fourier hand contributes 6 channels.
- The RoboCasa absolute/delta split is model-family-specific: pi0/pi05 make the
  arms and waist state-relative and keep the hands absolute
  (`_ROBOCASA_JOINT_DELTA_MASK`), while GR00T N1.7 marks the arms and hands
  relative and the waist absolute.
- LIBERO state carries 2 gripper finger positions while the LIBERO action gripper
  is a single command channel, which is why the state is 8D but the action is 7D.
- RoboDojo and ALOHA share the same 6 + 1 + 6 + 1 arm layout; Tron2 and Franka
  qpos share the same 7 + 1 + 7 + 1 arm layout.
