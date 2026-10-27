# Franka MuJoCo closed-loop sim-to-sim

This directory runs the trained Isaac v3c Franka state teacher in a separate
MuJoCo simulation. It is a true closed loop: every 20 ms MuJoCo generates a
45-D state observation, the frozen policy generates a 7-D action, and MuJoCo
executes that action before producing the next observation. It does not replay
an Isaac trajectory.

## What is matched

- Franka kinematic chain, joint limits, inertial parameters and mesh geometry
- Isaac TCP offset (`panda_hand + [0, 0, 0.1034]`)
- 45-D observation ordering and 7-D action semantics
- 20 ms policy interval, damped-least-squares differential IK and binary gripper
- conveyor, source tray, destination bowl, food, terminations and return-home check
- v3c checkpoint: `artifacts/teachers/teacher_repro_v3c_4096_no_rtx/checkpoints/ppo_teacher_30015488.pt`

The fixed-reset TCP differs from the Isaac reference by less than 0.04 mm, and
the fixed-reset first policy action agrees to within `2e-6`. These checks are in
`tests/unit/test_mujoco_sim.py`.

The Franka kinematics and inertial values are based on the official
[MuJoCo Menagerie Franka Panda model](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda).
The mesh files are copied locally from the already-installed Isaac Sim package,
so running this path never modifies the existing Isaac Lab installation.

## Run

Headless 20-episode closed-loop evaluation:

```bash
.venv/bin/python -m mujoco_sim.policy_runner \
  episodes=20 seed=0 randomize=true \
  out=artifacts/mujoco/franka_v3c_sim2sim.json
```

Adapted checkpoint evaluation (the acceptance run):

```bash
.venv/bin/python -m mujoco_sim.policy_runner \
  checkpoint=artifacts/mujoco/adaptation/franka_v3c_mujoco_adapted.pt \
  episodes=20 seed=0 randomize=true grasp_assist=true \
  out=artifacts/mujoco/franka_v3c_adapted_20ep.json
```

Rebuild the local expert dataset and adapted checkpoint:

```bash
.venv/bin/python -m mujoco_sim.adapt_train \
  episodes=64 seed=1000 workers=4 epochs=150 lr=0.0005 \
  phase_conditioned=true nominal_blend=0.95 recollect=true
```

Live MuJoCo viewer, fixed initial state, real-time playback:

```bash
DISPLAY=:1 .venv/bin/python -m mujoco_sim.policy_runner \
  checkpoint=artifacts/mujoco/adaptation/franka_v3c_mujoco_adapted.pt \
  viewer=true realtime=true viewer_hold=10 \
  episodes=1 seed=0 randomize=false \
  out=artifacts/mujoco/franka_v3c_viewer_run.json
```

Save one successful closed-loop episode as an H.264 MP4:

```bash
DISPLAY=:1 MUJOCO_GL=glfw .venv/bin/python -m mujoco_sim.render_video \
  checkpoint=artifacts/mujoco/adaptation/franka_v3c_mujoco_adapted.pt \
  seed=0 randomize=false width=1280 height=720 fps=50 \
  out=artifacts/mujoco/franka_v3c_adapted_closed_loop.mp4
```

Override or sweep the conveyor speed without editing code:

```bash
.venv/bin/python -m mujoco_sim.policy_runner belt_speed=0.06 episodes=20
.venv/bin/python -m mujoco_sim.speed_sweep \
  speeds=0.04,0.06,0.08,0.10,0.12,0.14 episodes=20 workers=4
```

If the desktop display is not `:1`, replace `DISPLAY=:1` with the display used
by the current graphical session. Closing the window early stops the run.

Run the contract tests:

```bash
.venv/bin/python -m pytest -q tests/unit/test_mujoco_sim.py
```

## Results

The frozen Isaac v3c checkpoint currently scores **0/20** in randomized MuJoCo
episodes: 15 runs end when the destination bowl leaves the work zone and 5 end
with the food off the table. The policy interface is verified, so this is a
real cross-engine transfer failure caused by the different actuator/contact
dynamics rather than a successful result hidden behind trajectory replay.

After dynamics/contact identification and phase-conditioned warm-start
adaptation, `franka_v3c_mujoco_adapted.pt` scores **20/20 (100%)** on the same
randomized seeds. All episodes grasp, place, settle, release and return the TCP
within 5 cm of Home before the bowl exits the work zone. The machine-readable
result is `artifacts/mujoco/franka_v3c_adapted_20ep.json`.
An additional held-out run on seeds 2000–2019 also scores **20/20 (100%)**; its
result is `artifacts/mujoco/franka_v3c_adapted_heldout_seed2000_20ep.json`.

The accepted controller is deliberately hybrid rather than presented as a
pure zero-shot neural transfer:

- 64/64 successful MuJoCo expert episodes (16,130 samples) train a
  v3c-warm-started, phase-conditioned actor.
- The actor uses the original 45-D observation plus a 9-D internal phase
  one-hot; no future state or replayed trajectory is used.
- At execution, the dynamics-identified nominal controller has weight 0.95 and
  the learned actor has weight 0.05. This is a conservative residual-policy
  safety envelope; the blend is stored in checkpoint metadata.
- `grasp_assist` activates only after the object is physically inside a closed
  pinch and releases immediately when the gripper opens. It stabilizes the
  simplified MuJoCo finger contact and removes constraint-switch impulses.

The zero-shot and adapted results should both be retained: together they show
that the test detects a real transfer failure and that the explicit adaptation
closes it. The same state/action boundary and nominal-plus-residual structure
can later be reused for JAKA sim-to-real.

## Conveyor-speed robustness

The adapted checkpoint was evaluated for 20 randomized episodes at each speed:

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

The current validated 100%-success range is therefore **0.04–0.08 m/s**. The
dominant faster-speed failure is `bowl_exited_zone`: the controller usually
grasps or places correctly but cannot finish and return Home before the bowl
crosses the work-zone boundary. Results are stored in
`artifacts/mujoco/franka_v3c_belt_speed_sweep.json` and
`artifacts/mujoco/franka_v3c_belt_speed_sweep_fine.json`.
