#!/usr/bin/env python3
"""PhysX validation and non-RTX video replay for the combined ZU5 + AG95 asset.

The simulation itself runs in Isaac Lab/PhysX.  Since the workstation's current
driver cannot start the RTX renderer, recorded rigid-body states are rendered
afterwards with matplotlib.  The MP4 is therefore a replay of measured physics
states, not a scripted animation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "zu5_ag95_validation"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

from pickplace.app import launch_app

simulation_app = launch_app(headless=True, enable_cameras=False, device="cuda:0")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab_physx.sim.schemas import ArticulationRootPropertiesCfg, RigidBodyPropertiesCfg  # noqa: E402


DT = 0.005
ROBOT_USD = ROOT / "assets" / "models" / "robots" / "Zu5Ag95" / "usd" / "zu5_ag95" / "zu5_ag95.usda"
TRAJECTORY = OUT / "trajectory.npz"
RESULT = OUT / "result.json"
VIDEO = OUT / "zu5_ag95_validation.mp4"

GRIPPER_JOINTS = [
    "left_inner_knuckle_joint",
    "left_outer_knuckle_joint",
    "right_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "left_finger_joint",
    "right_finger_joint",
]
FOLLOWER_JOINTS = [name for name in GRIPPER_JOINTS if name != "left_outer_knuckle_joint"]
# Center of the two pad meshes expressed in either finger-link frame.
PAD_POINT_F = torch.tensor([-0.022173, 0.00875, 0.030544], dtype=torch.float32)


def quat_apply_xyzw(quat: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """Rotate point(s) using this Isaac Lab build's scalar-last quaternion."""
    qv = quat[..., :3]
    qw = quat[..., 3:]
    point = point.expand_as(qv)
    return point + 2.0 * torch.cross(qv, torch.cross(qv, point, dim=-1) + qw * point, dim=-1)


def as_torch(value):
    return value.torch if hasattr(value, "torch") else value


def run_physics() -> dict:
    print("CHECKPOINT create_sim", flush=True)
    sim = SimulationContext(
        SimulationCfg(
            dt=DT,
            device="cuda:0",
            gravity=(0.0, 0.0, 0.0),
            render_interval=1000,
        )
    )
    print("CHECKPOINT create_robot_cfg", flush=True)

    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROBOT_USD),
            articulation_props=ArticulationRootPropertiesCfg(
                # The converted USD already contains Physics/root_joint.  Asking
                # the spawner to add a second world joint aborts this Isaac build.
                fix_root_link=None,
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
            ),
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            joint_pos={"joint_[1-6]": 0.0, ".*(knuckle|finger)_joint": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                stiffness=400.0,
                damping=40.0,
                effort_limit_sim=150.0,
                velocity_limit_sim=3.14,
            ),
            "ag95_master": ImplicitActuatorCfg(
                joint_names_expr=["left_outer_knuckle_joint"],
                stiffness=40.0,
                damping=4.0,
                effort_limit_sim=100.0,
                velocity_limit_sim=2.1,
            ),
            # The imported mimic constraints move these links.  They must be
            # passive: leaving the converter's position drives at target zero
            # makes them fight the master constraint.
            "ag95_passive_followers": ImplicitActuatorCfg(
                joint_names_expr=FOLLOWER_JOINTS,
                stiffness=0.0,
                damping=0.2,
                effort_limit_sim=100.0,
                velocity_limit_sim=2.1,
            ),
        },
    )
    print("CHECKPOINT create_robot", flush=True)
    robot = Articulation(robot_cfg)

    print("CHECKPOINT create_cube_cfg", flush=True)
    cube_cfg = RigidObjectCfg(
        prim_path="/World/TestCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.030, 0.030, 0.040),
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.0015,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.050),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5,
                dynamic_friction=1.2,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.75, 0.25)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -2.0)),
    )
    print("CHECKPOINT create_cube", flush=True)
    cube = RigidObject(cube_cfg)
    print("CHECKPOINT create_contact", flush=True)
    contact = ContactSensor(
        ContactSensorCfg(
            prim_path="/World/TestCube",
            update_period=0.0,
            history_length=1,
            track_pose=True,
            debug_vis=False,
        )
    )

    print("CHECKPOINT sim_reset", flush=True)
    sim.reset()
    print("CHECKPOINT asset_reset", flush=True)
    robot.reset()
    cube.reset()

    print("RUNTIME_JOINTS", robot.joint_names, flush=True)
    print("RUNTIME_BODIES", robot.body_names, flush=True)

    master_ids, master_names = robot.find_joints("left_outer_knuckle_joint")
    if len(master_ids) != 1:
        raise RuntimeError(f"Expected one AG95 master joint, found {master_names}")
    master_id = master_ids[0]
    left_id = robot.find_bodies("left_finger")[0][0]
    right_id = robot.find_bodies("right_finger")[0][0]

    arm_ids, _ = robot.find_joints("joint_[1-6]", preserve_order=True)
    arm_target = torch.zeros((1, len(arm_ids)), device=sim.device)
    master_target = torch.zeros((1, 1), device=sim.device)

    records: dict[str, list] = {
        "time": [],
        "phase": [],
        "body_pos": [],
        "body_quat": [],
        "joint_pos": [],
        "pad_left": [],
        "pad_right": [],
        "cube_pos": [],
        "cube_quat": [],
        "contact_force": [],
        "master_target": [],
    }
    placement = None

    phases = [
        ("open", 80, 0.0, 0.0),
        ("free_close", 190, 0.0, 0.943),
        ("reopen", 190, 0.943, 0.0),
        # A position target just inside the 30 mm object width supplies grip
        # force without the unrealistic impulse caused by commanding hard-stop.
        ("grasp_close", 240, 0.0, 0.760),
        ("hold", 120, 0.760, 0.760),
        ("loaded_hold", 160, 0.760, 0.760),
        ("release", 190, 0.760, 0.0),
    ]

    step_count = 0
    calibration_midpoints = []
    calibration_q = []
    cube_start = None
    cube_loaded_start = None
    cube_loaded_end = None

    for phase, count, q0, q1 in phases:
        if phase == "grasp_close":
            # Put the object where the pads met during the free-close calibration.
            idx = int(np.argmin(np.abs(np.asarray(calibration_q) - 0.68)))
            placement = np.asarray(calibration_midpoints[idx], dtype=np.float32)
            pose = torch.tensor(
                [[placement[0], placement[1], placement[2], 0.0, 0.0, 0.0, 1.0]],
                device=sim.device,
            )
            cube.write_root_pose_to_sim(pose)
            cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
            cube.reset()
            cube_start = placement.copy()

        if phase == "loaded_hold":
            cube_loaded_start = as_torch(cube.data.root_pos_w)[0].detach().cpu().numpy().copy()
            force = torch.tensor([[[0.0, 0.0, -1.0]]], device=sim.device)
            torque = torch.zeros_like(force)
            cube.permanent_wrench_composer.set_forces_and_torques(force, torque, is_global=True)
        elif phase == "release":
            cube_loaded_end = as_torch(cube.data.root_pos_w)[0].detach().cpu().numpy().copy()
            cube.permanent_wrench_composer.reset()

        for k in range(count):
            alpha = 1.0 if count <= 1 else k / (count - 1)
            target_q = q0 + alpha * (q1 - q0)
            master_target.fill_(target_q)
            robot.set_joint_position_target_index(target=arm_target, joint_ids=arm_ids)
            robot.set_joint_position_target_index(target=master_target, joint_ids=master_ids)
            robot.write_data_to_sim()
            cube.write_data_to_sim()
            sim.step(render=False)
            robot.update(DT)
            cube.update(DT)
            contact.update(DT, force_recompute=True)

            body_pos = as_torch(robot.data.body_link_pos_w)[0]
            body_quat = as_torch(robot.data.body_link_quat_w)[0]
            pad_local = PAD_POINT_F.to(device=sim.device)
            pad_left = body_pos[left_id] + quat_apply_xyzw(body_quat[left_id], pad_local)
            pad_right = body_pos[right_id] + quat_apply_xyzw(body_quat[right_id], pad_local)
            actual_q = float(as_torch(robot.data.joint_pos)[0, master_id].item())

            if phase == "free_close":
                calibration_midpoints.append((0.5 * (pad_left + pad_right)).detach().cpu().numpy())
                calibration_q.append(actual_q)

            if step_count % 5 == 0:
                net_force = as_torch(contact.data.net_forces_w)[0]
                records["time"].append(step_count * DT)
                records["phase"].append(phase)
                records["body_pos"].append(body_pos.detach().cpu().numpy())
                records["body_quat"].append(body_quat.detach().cpu().numpy())
                records["joint_pos"].append(as_torch(robot.data.joint_pos)[0].detach().cpu().numpy())
                records["pad_left"].append(pad_left.detach().cpu().numpy())
                records["pad_right"].append(pad_right.detach().cpu().numpy())
                records["cube_pos"].append(as_torch(cube.data.root_pos_w)[0].detach().cpu().numpy())
                records["cube_quat"].append(as_torch(cube.data.root_quat_w)[0].detach().cpu().numpy())
                records["contact_force"].append(float(torch.linalg.vector_norm(net_force).item()))
                records["master_target"].append(target_q)
            step_count += 1

    if cube_loaded_end is None:
        cube_loaded_end = as_torch(cube.data.root_pos_w)[0].detach().cpu().numpy().copy()

    arrays = {key: np.asarray(value) for key, value in records.items()}
    arrays["phase"] = arrays["phase"].astype("U16")
    arrays["body_names"] = np.asarray(robot.body_names, dtype="U64")
    arrays["joint_names"] = np.asarray(robot.joint_names, dtype="U64")
    np.savez_compressed(TRAJECTORY, **arrays)

    phases_np = arrays["phase"]
    free = phases_np == "free_close"
    grasp = np.isin(phases_np, ["grasp_close", "hold", "loaded_hold"])
    hold = phases_np == "hold"
    loaded = phases_np == "loaded_hold"
    gaps = np.linalg.norm(arrays["pad_left"] - arrays["pad_right"], axis=1)
    contact_force = arrays["contact_force"]
    q = arrays["joint_pos"][:, master_id]
    hold_position_error = float(np.mean(arrays["master_target"][hold] - q[hold]))
    runtime_followers = [
        name
        for name in (
            "left_inner_knuckle_joint",
            "left_finger_joint",
            "right_inner_knuckle_joint",
            "right_outer_knuckle_joint",
            "right_finger_joint",
        )
        if name in robot.joint_names
    ]
    empty = np.isin(phases_np, ["free_close", "reopen"])
    if runtime_followers:
        follower_ids = [robot.joint_names.index(name) for name in runtime_followers]
        mimic_error = float(np.max(np.abs(arrays["joint_pos"][empty][:, follower_ids] - q[empty, None])))
    else:
        # NewtonMimic joints are reduced out of the runtime DOF list.  Their body
        # motion and pad gap below are the runtime validation in that case.
        mimic_error = None

    load_displacement = float(np.linalg.norm(cube_loaded_end - cube_loaded_start))
    cube_from_place = np.linalg.norm(arrays["cube_pos"] - np.asarray(placement), axis=1)
    result = {
        "backend": "Isaac Lab PhysX CUDA",
        "dt_seconds": DT,
        "robot_usd": str(ROBOT_USD),
        "runtime_joint_names": list(robot.joint_names),
        "runtime_body_names": list(robot.body_names),
        "mimic_runtime_mode": "passive_mimic_followers" if runtime_followers else "reduced_mimic_dofs",
        "gripper_command_dimensions": 1,
        "runtime_follower_joints": runtime_followers,
        "mimic_max_abs_error_rad": mimic_error,
        "open_pad_gap_m": float(gaps[phases_np == "open"][-1]),
        "free_closed_pad_gap_m": float(np.min(gaps[free])),
        "grasp_min_pad_gap_m": float(np.min(gaps[grasp])),
        "master_q_free_close_rad": float(np.max(q[free])),
        "master_q_grasp_rad": float(np.max(q[grasp])),
        # This is the resultant on the cube; the two opposing normal forces
        # mostly cancel and are therefore larger individually than this value.
        "peak_cube_net_contact_force_n": float(np.max(contact_force[grasp])),
        "mean_hold_net_contact_force_n": float(np.mean(contact_force[hold])),
        "mean_loaded_net_contact_force_n": float(np.mean(contact_force[loaded])),
        "mean_hold_position_error_rad": hold_position_error,
        "load_force_n": 1.0,
        "cube_load_displacement_m": load_displacement,
        "cube_max_displacement_from_placement_m": float(np.max(cube_from_place[grasp])),
        "placement_world_m": np.asarray(placement).tolist(),
        "trajectory": str(TRAJECTORY),
        "video": str(VIDEO),
    }
    result["checks"] = {
        "finite": bool(all(np.isfinite(v).all() for k, v in arrays.items() if k not in ("phase", "body_names", "joint_names"))),
        "open_close_motion": bool(result["open_pad_gap_m"] - result["free_closed_pad_gap_m"] > 0.06),
        "contact_detected": bool(result["peak_cube_net_contact_force_n"] > 0.02 and hold_position_error > 0.03),
        "object_blocks_closure": bool(result["grasp_min_pad_gap_m"] > result["free_closed_pad_gap_m"] + 0.005),
        "loaded_object_retained": bool(load_displacement < 0.03),
        "mimic_coupling": bool(mimic_error is None or mimic_error < 0.02),
    }
    result["passed"] = bool(all(result["checks"].values()))
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("VALIDATION_RESULT", json.dumps(result, ensure_ascii=False), flush=True)
    return result


def cube_edges(center: np.ndarray, size=(0.030, 0.030, 0.040)):
    half = 0.5 * np.asarray(size)
    corners = np.asarray(
        [[x, y, z] for x in (-half[0], half[0]) for y in (-half[1], half[1]) for z in (-half[2], half[2])]
    ) + center
    pairs = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(np.abs(corners[i] - corners[j]) > 1e-8) == 1:
                pairs.append((corners[i], corners[j]))
    return pairs


def render_video(result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    data = np.load(TRAJECTORY)
    body_names = list(data["body_names"])
    body_index = {name: body_names.index(name) for name in body_names}
    arm_names = [name for name in ["Link_0", "Link_1", "Link_2", "Link_3", "Link_4", "Link_5", "Link_6", "ag95_base_link"] if name in body_index]
    grip_chains = [
        ["ag95_base_link", "left_outer_knuckle", "left_finger"],
        ["ag95_base_link", "right_outer_knuckle", "right_finger"],
        ["ag95_base_link", "left_inner_knuckle"],
        ["ag95_base_link", "right_inner_knuckle"],
    ]
    frame_ids = np.arange(len(data["time"]))

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#11151b")
    ax.set_facecolor("#11151b")

    phase_labels = {
        "open": "OPEN",
        "free_close": "FREE CLOSE",
        "reopen": "REOPEN",
        "grasp_close": "CONTACT GRASP",
        "hold": "HOLD",
        "loaded_hold": "1 N DOWNWARD LOAD",
        "release": "RELEASE",
    }

    def draw(frame_no):
        ax.clear()
        ax.set_facecolor("#11151b")
        pos = data["body_pos"][frame_no]
        arm = np.asarray([pos[body_index[name]] for name in arm_names])
        ax.plot(arm[:, 0], arm[:, 1], arm[:, 2], "o-", color="#d4d7dc", linewidth=7, markersize=5)
        for chain in grip_chains:
            chain = [name for name in chain if name in body_index]
            xyz = np.asarray([pos[body_index[name]] for name in chain])
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "o-", color="#f2b84b", linewidth=4, markersize=4)

        left, right = data["pad_left"][frame_no], data["pad_right"][frame_no]
        ax.scatter(*left, s=180, marker="s", color="#ff6262", label="left pad")
        ax.scatter(*right, s=180, marker="s", color="#61a8ff", label="right pad")

        cube = data["cube_pos"][frame_no]
        if cube[2] > 0.0:
            for a, b in cube_edges(cube):
                ax.plot(*zip(a, b), color="#47e36f", linewidth=3)

        phase = str(data["phase"][frame_no])
        gap = np.linalg.norm(left - right) * 1000.0
        q = data["joint_pos"][frame_no, list(data["joint_names"]).index("left_outer_knuckle_joint")]
        force = data["contact_force"][frame_no]
        ax.text2D(
            0.025,
            0.94,
            "ZU5 + AG95  |  Isaac PhysX trajectory replay",
            transform=ax.transAxes,
            color="white",
            fontsize=17,
            weight="bold",
        )
        ax.text2D(
            0.025,
            0.865,
            f"{phase_labels.get(phase, phase)}    master={q:.3f} rad    pad gap={gap:.1f} mm    contact={force:.2f} N",
            transform=ax.transAxes,
            color="#e6e6e6",
            fontsize=12,
        )
        ax.text2D(
            0.025,
            0.815,
            "Grey: ZU5   Yellow: AG95 linkage   Red/blue: pads   Green: 30 mm test cube",
            transform=ax.transAxes,
            color="#b8c1cc",
            fontsize=10,
        )
        ax.set_xlim(-0.05, 0.92)
        ax.set_ylim(-0.33, 0.14)
        ax.set_zlim(0.40, 0.78)
        ax.set_box_aspect((0.97, 0.47, 0.38))
        ax.view_init(elev=24, azim=118)
        ax.set_xlabel("X / m", color="white")
        ax.set_ylabel("Y / m", color="white")
        ax.set_zlabel("Z / m", color="white")
        ax.tick_params(colors="#c6cbd1")
        ax.grid(True, color="#35404c", alpha=0.5)
        return []

    ani = animation.FuncAnimation(fig, draw, frames=frame_ids, interval=50, blit=False)
    writer = animation.FFMpegWriter(fps=20, bitrate=3200, metadata={"title": "ZU5 AG95 PhysX validation"})
    ani.save(VIDEO, writer=writer)
    plt.close(fig)
    print(f"VIDEO_DONE {VIDEO} ({VIDEO.stat().st_size} bytes)", flush=True)


try:
    validation = run_physics()
    render_video(validation)
finally:
    simulation_app.close()

print(f"DONE passed={validation['passed']} result={RESULT} video={VIDEO}", flush=True)
