# Moving cube to static bowl — task contract

Status: environment contract, PPO pipeline smoke, and the closed-loop scripted-expert physics acceptance are complete; no learned policy has been accepted yet.

> **2026-10-22 interface decision:** the 49-D state teacher below is retained as a privileged simulation
> baseline only. The deployable policy no longer requires object, conveyor, joint, or TCP velocity. The current
> speed-free GAP-style observation contract, distillation results, mistakes, and JAKA migration checklist are
> recorded in [moving_cube_speed_free_gap_jaka_migration_log_zh-CN.md](moving_cube_speed_free_gap_jaka_migration_log_zh-CN.md).
> Statements below about a future real cell estimating explicit speed are historical and are superseded by that
> document.

## Deployment camera and policy-input decision (2026-10-22)

The minimum and default real-world sensor configuration is **one wrist-mounted RGB-D camera**. Depth is used by
the perception stack to produce a point cloud, online grasp proposals, and a target-region centre. RGB may be
retained for visualization, debugging, or optional segmentation, but neither raw RGB nor raw depth is passed to
the RL actor. A second fixed/overview camera is optional redundancy for occlusion recovery, target reacquisition,
debugging, or safety monitoring; it is not a deployment requirement.

At each control step, perception converts its results into the AG95 TCP frame and supplies fixed-shape
`4 x 16 x 3` grasp-candidate points. The actor additionally receives TCP position and orientation, ZU5 joint
positions, AG95 position feedback, the previous action, and a 10-step history containing the detected target-region
centre relative to TCP, gripper position, and previous action. Camera calibration parameters are used outside the
actor for coordinate conversion. Object or conveyor velocity, exact object 6D pose, joint/TCP velocity, contact,
force, tactile, motor-current, and raw image inputs are not required.

If a second camera is introduced later, its observations must be fused in the perception layer into the same
grasp-candidate contract. It must not change the actor input shape or make an accepted policy depend on that camera.
Geometry-only RL training therefore keeps cameras disabled; future visual sim-to-real validation starts with a
single wrist RGB-D render and adds a second view only if controlled occlusion tests demonstrate a material benefit.

This task is independent from `FoodRobot-Cell-v0` and is registered as:

```text
FoodRobot-MovingCubeStaticBowl-v0
```

The Franka and its 7-D end-effector-delta action remain unchanged. A 4 cm cube starts on the driven conveyor pallet and moves along `+x`; the destination bowl is kinematic and fixed beside the belt at `(0.45, -0.10, 0.0)`. The new task uses a `0.10 m/s` nominal belt speed. Its placement window is 4 s, so the reachable zone remains the original 0.40 m instead of growing with speed.

The state-teacher observation is deliberately 49-D:

| Group | Contents | Dimensions |
|---|---|---:|
| `proprio` | same Franka state and previous 7-D action as the accepted task | 34 |
| `belt` | static destination-bowl position, sampled belt speed | 4 |
| `privileged` | cube position, quaternion, linear velocity, grasp flag | 11 |

Explicit velocity and belt speed are included because moving-object interception is not Markov from a single object position alone. They also define what a future real cell must estimate from a belt encoder or temporal vision.

Task outcomes are `success`, `cube_exited_zone`, `cube_off_belt`, and `time_out`. Full success means the cube is released and settled in the static bowl and, when enabled by the training config, the TCP has returned within 5 cm of Home. The two conveyor failures apply while the cube is near the conveyor: a lifted cube may intentionally travel sideways, and the static destination bowl's footprint is exempt from lateral departure. Falling below the conveyor surface remains a failure everywhere. Without these distinctions, correct transport or descent into the off-belt bowl would be falsely terminated.

The first full-task configuration is `pipeline/0_state_teacher/config_moving_cube_v1.yaml`, and its reward set is `pickplace/reward_sets/moving_cube_v1.yaml`. Future curriculum configs and all outputs must use `artifacts/moving_cube_static_bowl/`; existing `teacher_repro_*` and `artifacts/mujoco/` results are read-only baselines.

Completed foundation checks:

- 4-environment scene smoke: over 0.20 s each cube moved approximately 0.0200 m at 0.10 m/s, while every fixed bowl drifted 0 m;
- 49-D observation and task-specific outcome contracts passed;
- 64-environment PPO smoke: one iteration, 1,024 frames, checkpoint and manifest written under `artifacts/moving_cube_static_bowl/smoke_v1/`;
- all 212 simulator-free unit tests passed.
- closed-loop current-state expert at `0.02 m/s`, fixed reset: 2/2 successes;
- closed-loop current-state expert at the target `0.10 m/s`, fixed reset: 8/8 successes in at most 4.74 simulated seconds;
- closed-loop current-state expert at `0.10 m/s` with cube/pallet/robot reset randomization and ±5% speed noise: 20/20 successes in at most 5.52 simulated seconds. Every episode physically grasped the cube, placed it within 6.7 mm of the bowl centre, released it, returned Home, and triggered the environment's `success` termination.

The expert is implemented in `pickplace/experts/moving_cube.py` and run by `scripts/validate_moving_cube_expert.py`. It only reads the current state observation; the short intercept lead is computed from the observed current cube velocity, not from a future simulator trajectory. Reproduce the target-speed acceptance with:

```bash
XDG_CACHE_HOME="$PWD/.cache" OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/validate_moving_cube_expert.py \
  speed=0.10 num_envs=20 randomize=true seed=7 \
  out=artifacts/moving_cube_static_bowl/validation/expert_0.10_random_20.json
```

Next acceptance gates:

1. ~~scripted expert validates the complete motion chain at a slow speed~~;
2. ~~scripted expert validates randomized interception at the 0.10 m/s target speed (20/20)~~;
3. curriculum training at increasing speed;
4. randomized state-teacher evaluation at `0.10 m/s`;
5. single-wrist-RGB-D perception validation and MuJoCo closed-loop transfer.

## State-teacher curriculum

The local launcher is `scripts/run_moving_cube_curriculum_local.sh`. All runs are rooted at `artifacts/moving_cube_static_bowl/teachers/`, never under the accepted old task's `artifacts/teachers/` directory.

| Stage | Belt distribution | Success boundary | Initialization |
|---|---|---|---|
| C1 | 0.04 m/s, reduced cube reset range | released and settled in bowl | scratch |
| C2 | 0.08 m/s ±3%, full cube reset range | released and settled in bowl | best C1 checkpoint |
| C3-H | 0.10 m/s ±5%, full cube reset range | settled in bowl and TCP back Home | best C2 checkpoint |

C1 and C2 disable only the `return_home` shaping term and Home success requirement; grasp, lift, transport, placement, failure conditions, network inputs, and action space are identical across all three stages. Thus every checkpoint has the same 49-D input/7-D output architecture and can be warm-started strictly into the next stage. C3-H restores the complete task contract.
