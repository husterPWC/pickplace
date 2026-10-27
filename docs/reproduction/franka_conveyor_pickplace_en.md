# Franka Conveyor Pick-and-Place: Complete Reproduction Record

> Document status: factual record of the completed task  
> Last verified: 2026-10-21  
> Repository: `/data/CodePWC/pickplace`  
> Scope: a Franka picks an object from a stationary supply bowl, places it into a moving bowl on a conveyor, returns Home, and is then validated through closed-loop MuJoCo sim-to-sim.

中文版：[franka_conveyor_pickplace_zh-CN.md](franka_conveyor_pickplace_zh-CN.md)

## 1. Final outcome

Two closed loops were completed:

1. **Isaac Lab / PhysX state teacher.** A three-stage PPO curriculum was trained locally on the RTX 3090 with 4,096 parallel environments, no cameras, and no RTX rendering. The selected v3c checkpoint succeeds in **248/256 deterministic evaluation environments (96.875%)**. Success requires grasping, placement into the moving bowl, release, settling, and returning the TCP Home.
2. **Closed-loop MuJoCo sim-to-sim.** Every 20 ms, MuJoCo produces the current state, the controller produces an action from that state, and MuJoCo executes it. No Isaac trajectory is replayed. The zero-shot Isaac checkpoint scored **0/20**. After explicit dynamics/contact adaptation, the controller scored **20/20** on the acceptance seeds and **20/20** on held-out seeds 2000–2019.

Key artifacts:

| Item | Path | Result |
|---|---|---:|
| Selected Isaac v3c teacher | `artifacts/teachers/teacher_repro_v3c_4096_no_rtx/checkpoints/ppo_teacher_30015488.pt` | 248/256 |
| Teacher metadata | `artifacts/teachers/teacher_repro_v3c_4096_no_rtx/checkpoints/ppo_teacher_30015488.json` | config, hash, evaluation |
| MuJoCo-adapted checkpoint | `artifacts/mujoco/adaptation/franka_v3c_mujoco_adapted.pt` | hybrid closed-loop controller |
| MuJoCo demonstrations | `artifacts/mujoco/adaptation/expert_demos.npz` | 64 episodes, 16,130 samples |
| Acceptance result | `artifacts/mujoco/franka_v3c_adapted_20ep.json` | 20/20 |
| Held-out result | `artifacts/mujoco/franka_v3c_adapted_heldout_seed2000_20ep.json` | 20/20 |
| Successful MuJoCo video | `artifacts/mujoco/franka_v3c_adapted_closed_loop.mp4` | 1280×720, 50 FPS |
| Speed sweeps | `artifacts/mujoco/franka_v3c_belt_speed_sweep.json`, `franka_v3c_belt_speed_sweep_fine.json` | Section 10 |

All artifacts remain inside this repository. `/data/CodePWC/IsaacLab` was neither modified nor used as an output location.

## 2. Repository, branches, and the clone failure

The verified remotes are:

```text
origin   https://github.com/husterPWC/pickplace.git
upstream https://github.com/BY571/pickplace.git
```

The working branch is `main`. `origin` currently exposes `main`; the upstream also has `upstream/action-space-ablation`. The completed reproduction uses `main`, so **forking the extra branch was not required**. If it is needed later, create a local branch from the upstream remote instead of merging it into the accepted baseline.

The original clone error—`curl 56 GnuTLS recv error`, an unexpected disconnect, and premature EOF—means that the HTTPS/TLS object stream was interrupted before the pack completed. It normally points to a network, proxy, or long-connection problem, not repository corruption or a branch problem. A safe recovery is to shallow-clone first, then fetch the remaining history, while keeping `origin` for the fork and `upstream` for the source project. The current checkout is healthy and must not be overwritten to address that historical error.

## 3. Environment isolation and version strategy

The existing `/data/CodePWC/IsaacLab` installation had to remain untouched. The implementation therefore uses repository-local locations:

```text
/data/CodePWC/pickplace/.venv
/data/CodePWC/pickplace/third_party/IsaacLab
/data/CodePWC/pickplace/.cache
/data/CodePWC/pickplace/artifacts
/data/CodePWC/pickplace/weights
/data/CodePWC/pickplace/datasets
```

The setup script pins the important components to Python 3.12, Isaac Sim 6.0.1, the Isaac Lab `v3.0.0-beta2.patch1` source tree, PyTorch 2.10.0 + cu128, TorchRL/TensorDict 0.14.x, and MuJoCo 3.8.0.

Activate it with:

```bash
source scripts/activate_local.sh
```

The activation script redirects pip, uv, Hugging Face, runtime caches, weights, datasets, and artifacts into the checkout so large files do not silently land in the user home directory.

The NVIDIA driver was not upgraded. Instead, capabilities were separated:

- headless PhysX state training and camera-free evaluation run locally;
- Isaac RTX camera rendering is disabled locally;
- visualization uses an offline mesh rendering of PhysX poses or MuJoCo viewer/video;
- large dual-camera rollouts are reserved for the high-driver eight-GPU server.

The local training wrapper applies `env.num_envs=4096`, `worker.render=false`, and `logger.backend=null`. Disabling rendering does not disable the independent 256-environment evaluation worker. Intermittent `gpu.foundation.plugin` / `ICpuInfo` initialization messages appeared in logs, but the headless PhysX training and camera-free evaluations completed; those messages alone are not evidence that training stopped.

## 4. Original method versus the completed scope

The upstream method is a Teacher–Dataset–Student pipeline:

```text
privileged state teacher (PPO)
        ↓ rollout in camera-enabled simulation
images + state + action + reward + done dataset
        ↓ BC / IQL / TD3+BC
deployable vision student
```

The teacher may consume simulator-only object poses and a grasp flag. A deployable student should consume only observations available on the real cell, such as dual-camera images and robot proprioception. The `Sebasdi/pick-and-place-conveyor` Hugging Face dataset is camera data for the original Franka scene; it is not required to train the state teacher.

**The accepted local v3c teacher was trained online in Isaac from scratch through warm-start curriculum stages. The Hugging Face offline dataset did not replace that training.** The repository contains the student/offline-RL pipeline and some student artifacts, but a newly accepted vision student is outside the final claim of this Franka state-teacher plus MuJoCo work.

This matters when changing robots: Franka images do not represent JAKA appearance, kinematics, calibration, or action distribution. A JAKA teacher may still be trained without cameras, but JAKA camera rollouts must be recollected in the JAKA simulation.

## 5. Isaac task contract

### 5.1 Scene and timing

- The Franka stands beside the table.
- A stationary supply bowl is near `(0.45, -0.10)`.
- The object starts inside that supply bowl.
- The destination bowl rides a pallet driven along conveyor `+x`.
- Nominal conveyor speed is `0.08 m/s`.
- Physics runs at `0.01 s` with decimation 2, giving a **50 Hz / 20 ms** policy interval.

The complete behavior is:

```text
reach → grasp → lift → track moving bowl → place → open → settle → return TCP Home
```

### 5.2 Privileged 45-D observation

Actor and critic consume `proprio + belt + privileged`:

| Group | Contents | Dimensions |
|---|---|---:|
| `proprio` | TCP position 3, TCP quaternion 4, gripper positions 2, nine relative joint positions, nine relative joint velocities, previous action 7 | 34 |
| `belt` | destination bowl position | 3 |
| `privileged` | object position 3, quaternion 4, grasp flag 1 | 8 |
| Total |  | **45** |

Conveyor speed is not explicitly observed because the Isaac teacher was trained at a fixed `0.08 m/s`. This is one reason not to assume speed invariance.

### 5.3 Seven-dimensional action

- Three relative TCP translation coordinates;
- three relative axis-angle TCP rotation coordinates;
- one binary gripper command: positive opens and negative closes;
- actor output is squashed to `[-1, 1]`; the six pose values use a `0.5` scale;
- Isaac applies differential inverse kinematics.

End-effector deltas provide a portable interface across arms, but IK parameters, TCP definition, control period, and low-level tracking still have to be matched on every embodiment.

### 5.4 Reward and success

The final `simple_v3b` reward forms a task ladder:

| Term | Weight | Purpose |
|---|---:|---|
| `reach_food` | 1 | approach object |
| `grasp` | 2 | form a valid pinch |
| `grasp_lift` | 5 | grasp and lift |
| `transport` | 10 | move a lifted object toward the moving bowl |
| `food_in_bowl` | 20 | object released inside bowl |
| `return_home` | 10 | return TCP after placement |
| `success` | +150 once | complete task |
| `bowl_disturbance` | -10 | avoid hitting the bowl |
| `action_rate` | -0.01 | smooth actions |
| `joint_vel` | -0.001 | discourage excessive joint speed |

Final success requires the object to be inside the destination bowl, released, settled relative to the bowl for five consecutive control steps, and the TCP to be within `0.05 m` of its Home TCP. Home is measured in TCP space rather than seven-joint space because the redundant Franka IK does not uniquely determine elbow configuration.

Failure outcomes include destination-bowl deadline exit, bowl falling from the belt, bowl tilt beyond 45 degrees, object falling from the table, and a safety timeout.

## 6. Why a three-stage curriculum was required

The final upstream v3c configuration references an earlier checkpoint that was not shipped in either the source repository or the published dataset. The complete warm-start lineage was therefore reconstructed.

| Stage | Config | Initialization | Main change | Actual total frames | Selected frame | Selected evaluation |
|---|---|---|---|---:|---:|---:|
| v2 | `config_repro_v2.yaml` | scratch | placement only; joint reset ±0.02 rad | 599,982,080 | 510,001,152 | 249/256 = 97.266% |
| v2r | `config_repro_v2r_long.yaml` | 510 M v2 | joint reset ±0.25 rad, velocity ±0.1 rad/s | 299,958,272 | 290,062,336 | 250/256 = 97.656% |
| v3c | `config_repro_v3c.yaml` | 290 M v2r | return-Home success and `simple_v3b` | 299,958,272 | 30,015,488 | **248/256 = 96.875%** |

Total training horizon and selected checkpoint are different facts. The strongest v3c checkpoint occurred near 30 M frames into that stage; continuing toward 300 M did not guarantee monotonic improvement. Deployment and sim-to-sim therefore use the evaluated best checkpoint, not automatically `final.pt`.

The actor/critic MLP is `45 → 512 → 256 → 128`, with ELU activations and a seven-dimensional TanhNormal actor. Training used 4,096 environments, 16-step rollouts, four PPO epochs, mini-batches of 32,768, `gamma=0.99`, GAE lambda 0.95, clip 0.2, and entropy coefficient 0.005. v2 used a `3e-4` learning rate; v2r/v3c used `1e-4`. Checkpoints were written about every 10 M frames and evaluated deterministically on 256 environments.

## 7. Parallelism and misleading VRAM readings

Training did use **4,096 parallel PhysX environments**. Modest VRAM utilization did not mean that execution was serial:

- the state teacher has no RGB tensors, RTX buffers, or CNN;
- each environment holds compact states and a short rollout;
- 4,096 state environments and 4,096 dual-camera environments have radically different memory costs;
- throughput and frames/s are better evidence of simulation parallelism than a single memory reading.

Environment count or cameras should not be increased merely to fill memory. Throughput, evaluation quality, and PhysX buffer safety are the relevant constraints.

## 8. Important failure modes and lessons

### 8.1 Training the full v3c task from scratch was too difficult

A random policy had to discover grasping, moving-target tracking, release, and homing in one exploration chain. Early full-task runs did not achieve the target behavior. The v2 → v2r → v3c curriculum solved this.

### 8.2 Excessive regularization teaches inactivity

Motion penalties were initially about ten times larger. Before random exploration found a lift, action-rate and joint-velocity penalties dominated the return; entropy collapsed from roughly 4.7 to -13.6 and exploration stopped. Final weights are only `-0.01` and `-0.001`.

### 8.3 Transport reward must require both grasp and lift

An early policy learned to shepherd the object along the table between its fingers, collecting transport reward without lifting. `transport` and `transport_fine` are now gated by `grasped && food_z > lift_height`.

### 8.4 Terminating immediately at placement cannot teach Home return

If the episode ends as soon as the object lands, the policy never observes post-release control states. Continuous execution then stops in an out-of-distribution pose. `return_home` shaping and `success_requires_home=true` keep the episode alive until placement, settling, and homing are all complete.

### 8.5 Training statistics, deterministic evaluation, and final weights differ

Auto-reset training metrics mix recent episodes and should not select deployment weights. Use each checkpoint's deterministic multi-environment sidecar. The best checkpoint can be earlier than the final one.

### 8.6 Early stopping depends on the evaluation worker

`early_stop.success_rate` reads the worker evaluation, not training metrics. To disable unsupported video while preserving early stopping, set `worker.render=false`; do not disable the worker itself.

### 8.7 Stop gracefully

SIGTERM follows the code path that writes the final checkpoint, manifest, and evaluation. A hard kill can leave an incomplete run directory.

### 8.8 Camera data is expensive

Two 84×84 RGB uint8 views are about 42 KB per frame, or roughly 42 GB per million frames. In the simulator, float32 observations plus root/next and rollout buffering further amplify GPU memory. Camera rollout scale must be chosen independently of the 4,096-environment state-teacher scale.

### 8.9 Do not force the Isaac RTX UI on the unchanged local driver

Headless PhysX performs training. Two alternative visualization paths were implemented:

- `scripts/render_franka_teacher_mesh_video.py` re-renders actual Isaac PhysX link poses with local Franka meshes, without Isaac RTX;
- MuJoCo supplies both an interactive viewer and H.264 video rendering.

The mesh visualization is a re-rendering of simulated poses, not a camera observation for training.

## 9. Proving the MuJoCo test is closed-loop

Every step is:

```text
current MuJoCo state
  → exact 45-D observation contract
  → current controller produces a 7-D action
  → MuJoCo IK, gripper, and contact dynamics execute 20 ms
  → new state
```

The port aligns the Franka chain, limits, inertias and meshes; the `panda_hand + [0, 0, 0.1034]` TCP; flattened observation order; quaternion convention; action semantics and scale; 50 Hz interval; scene objects and terminations; Home TCP; and 5 cm tolerance.

At fixed reset, the MuJoCo TCP differs from the Isaac reference by less than **0.04 mm**, and the first policy action matches within **2e-6**. These contracts are pinned by `tests/unit/test_mujoco_sim.py`.

After the interfaces matched, zero-shot v3c still scored **0/20**: 15 destination-bowl exits and five table drops, with no successful grasp. This is meaningful cross-engine failure caused by actuator, IK, contact, and gripper dynamics, rather than an interface bug hidden by trajectory replay.

## 10. MuJoCo adaptation and acceptance

### 10.1 Adaptation method

1. Implement a nine-phase waypoint expert using only the current privileged 45-D observation: `pregrasp`, `descend`, `close`, `lift`, `transport`, `place`, `release`, `retreat`, and `home`.
2. Collect 64 successful MuJoCo episodes containing 16,130 `(state, phase, action)` samples.
3. Warm-start from the Isaac v3c actor, append a nine-dimensional phase one-hot, and train a phase-conditioned actor for 150 epochs.
4. Balance phases through inverse-frequency sampling and give the gripper action extra loss weight.
5. Execute conservatively as `action = 0.95 × nominal + 0.05 × learned`.

The accepted controller must be described accurately: it is **95% dynamics-identified waypoint controller and 5% v3c-warm-started learned actor**. It is neither a pure neural policy nor zero-shot Isaac transfer. Its internal phase progresses from closed-loop current state; it does not access future states or replay a trajectory.

### 10.2 Scope of `grasp_assist`

Acceptance uses `grasp_assist=true` to stabilize the simplified MuJoCo finger contact and constraint transition. Attachment is permitted only after the object is within 3 cm of the TCP and the fingers have physically closed into a pinch interval. Opening the gripper detaches immediately, and the release path removes a small constraint-switch impulse inside the bowl.

This is suitable for validating the closed-loop interface and the broader sim-to-real workflow. A claim of transfer under pure MuJoCo contact physics would require a separate `grasp_assist=false` acceptance run.

### 10.3 Results

| Test | Episodes | Success | Mean completion time |
|---|---:|---:|---:|
| Zero-shot Isaac checkpoint | 20 | 0/20 | — |
| Adapted controller, seeds 0–19 | 20 | **20/20** | 4.917 s |
| Adapted controller, held-out seeds 2000–2019 | 20 | **20/20** | 5.033 s |

A success requires grasp, placement, release, settling, and Home return—not merely passing over the bowl.

### 10.4 Conveyor-speed robustness

Each speed was tested on 20 randomized episodes:

| Speed (m/s) | Success |
|---:|---:|
| 0.040 | 20/20 |
| 0.060 | 20/20 |
| 0.080 (training speed) | 20/20 |
| 0.085 | 19/20 |
| 0.090 | 16/20 |
| 0.095 | 14/20 |
| 0.100 | 13/20 |
| 0.120 | 4/20 |
| 0.140 | 1/20 |

The validated 100% range is **0.04–0.08 m/s**. At higher speeds, the dominant failure is the moving bowl leaving the work zone before placement and Home return finish. One successful video must not be extrapolated into speed-wide robustness.

## 11. Useful reproduction commands

Acceptance evaluation:

```bash
.venv/bin/python -m mujoco_sim.policy_runner \
  checkpoint=artifacts/mujoco/adaptation/franka_v3c_mujoco_adapted.pt \
  episodes=20 seed=0 randomize=true grasp_assist=true \
  out=artifacts/mujoco/franka_v3c_adapted_20ep.json
```

Interactive viewer:

```bash
DISPLAY=:1 .venv/bin/python -m mujoco_sim.policy_runner \
  checkpoint=artifacts/mujoco/adaptation/franka_v3c_mujoco_adapted.pt \
  viewer=true realtime=true viewer_hold=10 episodes=1 seed=0 randomize=false
```

H.264 video:

```bash
DISPLAY=:1 MUJOCO_GL=glfw .venv/bin/python -m mujoco_sim.render_video \
  checkpoint=artifacts/mujoco/adaptation/franka_v3c_mujoco_adapted.pt \
  seed=0 randomize=false width=1280 height=720 fps=50 \
  out=artifacts/mujoco/franka_v3c_adapted_closed_loop.mp4
```

Contract tests:

```bash
.venv/bin/python -m pytest -q tests/unit/test_mujoco_sim.py
```

## 12. Direct guidance for the new task and JAKA

1. **Create an independent task.** Use a new gym id, configs, reward set, and artifact root. Treat current v2/v2r/v3c and MuJoCo acceptance artifacts as a read-only baseline.
2. **Train a state teacher before a camera student.** State training is efficient on the current 3090; dual-camera rollout belongs on the high-driver server.
3. **Validate the environment and scripted expert before spending PPO frames.** A reset, direction, grasp geometry, success predicate, or deadline bug often looks like an RL failure.
4. **Use a curriculum.** Static/slow object, nominal speed, speed and reset randomization, then full placement plus Home.
5. **Provide temporal information for moving-object pickup.** A vision student should use two to four stacked frames, explicit belt speed/encoder input, or a temporal model. A single image cannot reliably distinguish equal positions with different velocities.
6. **Warm-start only compatible skills.** Scene semantics change the optimal action distribution; JAKA additionally changes kinematics and gripper behavior. Old success rates do not transfer automatically.
7. **Recollect data for each embodiment.** JAKA dual-camera simulation should record images, robot state, actions, reward vector, terminated/truncated flags, and outcome. The shards can then be copied back to the 3090 for student training.
8. **Treat MuJoCo as an intermediate gate, not the sim-to-real endpoint.** Preserve zero-shot failures, adaptation details, grasp-assist status, and held-out results—not only successful videos.

## 13. Artifact-protection rules

- Never modify `/data/CodePWC/IsaacLab`.
- Do not overwrite `artifacts/teachers/teacher_repro_*`.
- Do not overwrite accepted weights, JSON results, or videos under `artifacts/mujoco/`.
- Put the new task under independent configs, a separate task name, and `artifacts/moving_cube_static_bowl/`.
- Keep new weights, datasets, and download caches inside `/data/CodePWC/pickplace`.
- The worktree already contains user assets and experiment changes. Never use `git reset --hard`, `git checkout -- .`, or a fresh clone over this checkout.
- Preserve config, frame count, hash, evaluation sidecar, and source checkpoint for every selected model.

## 14. Code and evidence index

- Environment reference: `docs/environment.md`
- Teacher pipeline: `pipeline/0_state_teacher/README.md`
- Local curriculum launcher: `scripts/run_teacher_curriculum_local.sh`
- Curriculum configs: `pipeline/0_state_teacher/config_repro_v2.yaml`, `config_repro_v2r_long.yaml`, `config_repro_v3c.yaml`
- Final reward set: `pickplace/reward_sets/simple_v3b.yaml`
- Success and Home logic: `pickplace/envs/mdp/terminations.py`, `rewards.py`
- Non-RTX Isaac mesh video: `scripts/render_franka_teacher_mesh_video.py`
- MuJoCo environment: `mujoco_sim/env.py`, `scene.xml`
- MuJoCo expert and adaptation: `mujoco_sim/expert.py`, `adapt_train.py`
- MuJoCo execution and video: `mujoco_sim/policy_runner.py`, `render_video.py`
- MuJoCo details: `mujoco_sim/README.md`
- Contract tests: `tests/unit/test_mujoco_sim.py`
