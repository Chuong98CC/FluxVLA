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