"""Franka conveyor pick-and-place in MuJoCo, matched to the Isaac state-teacher interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
SCENE = Path(__file__).with_name("scene.xml")

# Isaac state/action contract.
DEFAULT_Q = np.asarray([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04])
HOME_TCP = np.asarray([0.4633, 0.0, 0.3855])
JOINT_LOW = np.asarray([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_HIGH = np.asarray([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
JOINT_VELOCITY = np.asarray([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
POLICY_DT = 0.02
ACTION_SCALE = 0.5
BELT_SPEED = 0.08
BELT_Y = 0.30
BELT_ENTRY_X = 0.20
ZONE_END_X = 0.65
HOME_TOLERANCE = 0.05
MAX_SECONDS = 10.125


def xyzw_from_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)


def wxyz_from_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.asarray([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


@dataclass
class StepResult:
    observation: np.ndarray
    done: bool
    outcome: str | None
    info: dict[str, float | bool | str]


class FrankaPickPlaceMuJoCo:
    """One MuJoCo environment with the v3c teacher's exact 45-D observation and 7-D action layout.

    State order follows ``pickplace.keys.expand_in_keys`` (alphabetical leaves inside each group):
    proprio/{ee_pos, ee_quat, gripper_pos, joint_pos_rel, joint_vel_rel, last_action},
    belt/bowl_pos, privileged/{food_pos, food_quat, is_grasped}.
    """

    observation_dim = 45
    action_dim = 7

    def __init__(
        self,
        seed: int = 0,
        randomize: bool = True,
        grasp_assist: bool = True,
        belt_speed: float = BELT_SPEED,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.randomize = randomize
        self.grasp_assist = grasp_assist
        self.belt_speed = float(belt_speed)
        if self.belt_speed <= 0.0:
            raise ValueError("belt_speed must be positive")
        self.seed = seed
        self.physics_steps = round(POLICY_DT / self.model.opt.timestep)
        if not np.isclose(self.physics_steps * self.model.opt.timestep, POLICY_DT):
            raise ValueError("scene timestep must divide the 0.02 s Isaac policy step")

        self.arm_qpos = np.asarray([self.model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self.arm_dof = np.asarray([self.model.joint(f"joint{i}").dofadr[0] for i in range(1, 8)])
        self.finger_qpos = np.asarray([self.model.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)])
        self.finger_dof = np.asarray([self.model.joint(f"finger_joint{i}").dofadr[0] for i in (1, 2)])
        self.robot_qpos = np.concatenate([self.arm_qpos, self.finger_qpos])
        self.robot_dof = np.concatenate([self.arm_dof, self.finger_dof])
        self.pallet_qpos = int(self.model.joint("pallet_slide").qposadr[0])
        self.pallet_dof = int(self.model.joint("pallet_slide").dofadr[0])
        self.bowl_qpos = int(self.model.joint("bowl_free").qposadr[0])
        self.bowl_dof = int(self.model.joint("bowl_free").dofadr[0])
        self.food_qpos = int(self.model.joint("food_free").qposadr[0])
        self.food_dof = int(self.model.joint("food_free").dofadr[0])
        self.arm_act = np.asarray([self.model.actuator(f"arm{i}").id for i in range(1, 8)])
        self.finger_act = np.asarray([self.model.actuator(f"finger{i}").id for i in (1, 2)])
        self.pallet_act = self.model.actuator("pallet_drive").id
        self.tcp_site = self.model.site("tcp").id
        self.bowl_body = self.model.body("bowl").id
        self.food_body = self.model.body("food").id
        self.food_geom = self.model.geom("food_geom").id
        self.food_contype = int(self.model.geom_contype[self.food_geom])
        self.food_conaffinity = int(self.model.geom_conaffinity[self.food_geom])

        self.last_action = np.zeros(7, dtype=np.float32)
        self.ee_target_pos = HOME_TCP.copy()
        self.ee_target_rot = Rotation.identity()
        self.elapsed = 0.0
        self.pallet_start = 0.0
        self.settle_count = 0
        self.grasp_attached = False
        self.grasp_rel_pos = np.zeros(3)
        self.grasp_rel_rot = Rotation.identity()
        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.elapsed = 0.0
        self.settle_count = 0
        self.last_action.fill(0.0)
        self.grasp_attached = False
        self.model.geom_contype[self.food_geom] = self.food_contype
        self.model.geom_conaffinity[self.food_geom] = self.food_conaffinity

        if self.randomize:
            arm_delta = self.rng.uniform(-0.25, 0.25, 7)
            arm_vel = self.rng.uniform(-0.1, 0.1, 7)
            self.pallet_start = float(self.rng.uniform(-0.08, 0.08))
            bowl_dx = float(self.rng.uniform(-0.02, 0.02))
            bowl_dy = float(self.rng.uniform(-0.04, 0.04))
            food_dx, food_dy = self.rng.uniform(-0.06, 0.06, 2)
        else:
            arm_delta = np.zeros(7)
            arm_vel = np.zeros(7)
            self.pallet_start = bowl_dx = bowl_dy = food_dx = food_dy = 0.0

        self.data.qpos[self.arm_qpos] = np.clip(DEFAULT_Q[:7] + arm_delta, JOINT_LOW, JOINT_HIGH)
        self.data.qvel[self.arm_dof] = arm_vel
        self.data.qpos[self.finger_qpos] = 0.04
        self.data.qvel[self.finger_dof] = 0.0
        self.data.qpos[self.pallet_qpos] = self.pallet_start
        self.data.qvel[self.pallet_dof] = self.belt_speed

        # Free-joint pose is xyz + wxyz.
        self.data.qpos[self.bowl_qpos : self.bowl_qpos + 7] = (
            BELT_ENTRY_X + self.pallet_start + bowl_dx,
            BELT_Y + bowl_dy,
            0.032,
            1.0,
            0.0,
            0.0,
            0.0,
        )
        self.data.qvel[self.bowl_dof : self.bowl_dof + 6] = (self.belt_speed, 0, 0, 0, 0, 0)
        self.data.qpos[self.food_qpos : self.food_qpos + 7] = (
            0.45 + food_dx,
            -0.10 + food_dy,
            0.031,
            1.0,
            0.0,
            0.0,
            0.0,
        )
        self.data.qvel[self.food_dof : self.food_dof + 6] = 0.0
        self.data.ctrl[self.arm_act] = self.data.qpos[self.arm_qpos]
        self.data.ctrl[self.finger_act] = 0.04
        self.data.ctrl[self.pallet_act] = self.pallet_start
        mujoco.mj_forward(self.model, self.data)
        self.ee_target_pos, self.ee_target_rot = self._tcp_pose()
        return self.observation()

    def _tcp_pose(self) -> tuple[np.ndarray, Rotation]:
        pos = self.data.site_xpos[self.tcp_site].copy()
        rot = Rotation.from_matrix(self.data.site_xmat[self.tcp_site].reshape(3, 3))
        return pos, rot

    def _is_grasped(self) -> bool:
        if self.grasp_attached:
            return True
        tcp_pos, _ = self._tcp_pose()
        near = np.linalg.norm(self.data.xpos[self.food_body] - tcp_pos) < 0.04
        fingers = self.data.qpos[self.finger_qpos]
        blocked = bool(np.all(fingers < 0.036) and np.all(fingers > 0.004))
        return bool(near and blocked)

    def _try_attach_food(self) -> None:
        """Stabilize a physically acquired pinch without allowing remote attachment."""
        if not self.grasp_assist or self.grasp_attached:
            return
        tcp_pos, tcp_rot = self._tcp_pose()
        food_pos = self.data.xpos[self.food_body].copy()
        fingers = self.data.qpos[self.finger_qpos]
        if np.linalg.norm(food_pos - tcp_pos) >= 0.03 or not np.all((fingers > 0.003) & (fingers < 0.03)):
            return
        food_rot = Rotation.from_quat(xyzw_from_wxyz(self.data.xquat[self.food_body]))
        self.grasp_rel_pos = tcp_rot.inv().apply(food_pos - tcp_pos)
        self.grasp_rel_rot = tcp_rot.inv() * food_rot
        self.grasp_attached = True
        # While attached, the gripper supplies the constraint. Disabling the
        # food collision avoids double-counting large pinch impulses.
        self.model.geom_contype[self.food_geom] = 0
        self.model.geom_conaffinity[self.food_geom] = 0

    def _update_attached_food(self, dt: float) -> None:
        if not self.grasp_attached:
            return
        previous = self.data.qpos[self.food_qpos : self.food_qpos + 3].copy()
        tcp_pos, tcp_rot = self._tcp_pose()
        food_pos = tcp_pos + tcp_rot.apply(self.grasp_rel_pos)
        food_rot = tcp_rot * self.grasp_rel_rot
        self.data.qpos[self.food_qpos : self.food_qpos + 3] = food_pos
        self.data.qpos[self.food_qpos + 3 : self.food_qpos + 7] = wxyz_from_xyzw(food_rot.as_quat())
        # Free-joint qvel is linear xyz followed by angular xyz.
        self.data.qvel[self.food_dof : self.food_dof + 3] = (food_pos - previous) / dt
        self.data.qvel[self.food_dof + 3 : self.food_dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _detach_food(self) -> None:
        if not self.grasp_attached:
            return
        self.grasp_attached = False
        # The release waypoint tracks the moving bowl. Remove tiny lateral
        # velocity errors introduced by the kinematic grasp constraint so the
        # released item inherits the bowl's conveyor velocity rather than an
        # artificial sideways impulse.
        if np.linalg.norm(self.data.xpos[self.food_body, :2] - self.data.xpos[self.bowl_body, :2]) < 0.08:
            bowl_pos = self.data.xpos[self.bowl_body].copy()
            # Remove the small positional/velocity discontinuity introduced by
            # switching from the kinematic pinch back to contact dynamics.
            # The sphere centre is one radius above the 6 mm bowl base.
            self.data.qpos[self.food_qpos : self.food_qpos + 3] = bowl_pos + np.asarray([0.0, 0.0, 0.026])
            self.data.qvel[self.food_dof : self.food_dof + 3] = self.data.cvel[self.bowl_body, 3:]
            self.data.qvel[self.food_dof + 3 : self.food_dof + 6] = 0.0
        self.model.geom_contype[self.food_geom] = self.food_contype
        self.model.geom_conaffinity[self.food_geom] = self.food_conaffinity
        mujoco.mj_forward(self.model, self.data)

    def observation(self) -> np.ndarray:
        tcp_pos, tcp_rot = self._tcp_pose()
        tcp_quat = tcp_rot.as_quat()  # xyzw, as in Isaac Lab 3
        food_quat = xyzw_from_wxyz(self.data.xquat[self.food_body])
        values = [
            tcp_pos,
            tcp_quat,
            self.data.qpos[self.finger_qpos].copy(),
            self.data.qpos[self.robot_qpos] - DEFAULT_Q,
            self.data.qvel[self.robot_dof].copy(),
            self.last_action,
            self.data.xpos[self.bowl_body].copy(),
            self.data.xpos[self.food_body].copy(),
            food_quat,
            np.asarray([float(self._is_grasped())]),
        ]
        obs = np.concatenate(values).astype(np.float32)
        if obs.shape != (self.observation_dim,) or not np.isfinite(obs).all():
            raise RuntimeError(f"invalid observation shape/value: {obs.shape}, finite={np.isfinite(obs).all()}")
        return obs

    def _apply_ik(self) -> None:
        tcp_pos, tcp_rot = self._tcp_pose()
        pos_error = self.ee_target_pos - tcp_pos
        rot_error = (self.ee_target_rot * tcp_rot.inv()).as_rotvec()
        error = np.concatenate([pos_error, rot_error])
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.tcp_site)
        jacobian = np.vstack([jacp[:, self.arm_dof], jacr[:, self.arm_dof]])
        damping = 0.01
        dq = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + (damping**2) * np.eye(6), error
        )
        self.data.ctrl[self.arm_act] = np.clip(self.data.qpos[self.arm_qpos] + dq, JOINT_LOW, JOINT_HIGH)

    def _placed_and_settled(self) -> bool:
        food_pos = self.data.xpos[self.food_body]
        bowl_pos = self.data.xpos[self.bowl_body]
        rel = food_pos - bowl_pos
        inside = np.linalg.norm(rel[:2]) < 0.06 and 0.006 < rel[2] < 0.056 and not self._is_grasped()
        food_vel = self.data.cvel[self.food_body, 3:]
        bowl_vel = self.data.cvel[self.bowl_body, 3:]
        settled = inside and np.linalg.norm(food_vel - bowl_vel) < 0.05
        self.settle_count = self.settle_count + 1 if settled else 0
        return self.settle_count >= 5

    def _outcome(self) -> tuple[str | None, dict[str, float | bool | str]]:
        tcp_pos, _ = self._tcp_pose()
        home_distance = float(np.linalg.norm(tcp_pos - HOME_TCP))
        placed = self._placed_and_settled()
        bowl_pos = self.data.xpos[self.bowl_body]
        bowl_quat = xyzw_from_wxyz(self.data.xquat[self.bowl_body])
        up_z = 1.0 - 2.0 * (bowl_quat[0] ** 2 + bowl_quat[1] ** 2)
        info: dict[str, float | bool | str] = {
            "time": round(self.elapsed, 4),
            "home_distance": home_distance,
            "placed": placed,
            "is_grasped": self._is_grasped(),
            "bowl_x": float(bowl_pos[0]),
            "food_z": float(self.data.xpos[self.food_body, 2]),
            "contacts": int(self.data.ncon),
            "grasp_assist_attached": self.grasp_attached,
        }
        if placed and home_distance <= HOME_TOLERANCE:
            return "success", info
        if bowl_pos[0] > ZONE_END_X:
            return "bowl_exited_zone", info
        if bowl_pos[2] < 0.01 or abs(bowl_pos[1] - BELT_Y) > 0.15:
            return "bowl_off_belt", info
        if up_z < np.cos(np.pi / 4):
            return "bowl_tipped", info
        if self.data.xpos[self.food_body, 2] < -0.05:
            return "food_off_table", info
        if self.elapsed >= MAX_SECONDS:
            return "time_out", info
        return None, info

    def step(self, action: np.ndarray) -> StepResult:
        action = np.asarray(action, dtype=np.float64).reshape(7)
        if not np.isfinite(action).all():
            raise ValueError("policy action contains NaN/Inf")
        action = np.clip(action, -1.0, 1.0)
        processed = ACTION_SCALE * action[:6]
        tcp_pos, tcp_rot = self._tcp_pose()
        self.ee_target_pos = tcp_pos + processed[:3]
        self.ee_target_rot = Rotation.from_rotvec(processed[3:]) * tcp_rot
        finger_target = 0.04 if action[6] >= 0.0 else 0.0
        if finger_target > 0.0:
            self._detach_food()

        for substep in range(self.physics_steps):
            self._apply_ik()
            self.data.ctrl[self.finger_act] = finger_target
            target_t = self.elapsed + (substep + 1) * self.model.opt.timestep
            self.data.ctrl[self.pallet_act] = np.clip(
                self.pallet_start + self.belt_speed * target_t, -0.12, 1.0
            )
            arm_before = self.data.qpos[self.arm_qpos].copy()
            finger_before = self.data.qpos[self.finger_qpos].copy()
            mujoco.mj_step(self.model, self.data)
            arm_delta = np.clip(
                self.data.qpos[self.arm_qpos] - arm_before,
                -JOINT_VELOCITY * self.model.opt.timestep,
                JOINT_VELOCITY * self.model.opt.timestep,
            )
            self.data.qpos[self.arm_qpos] = np.clip(arm_before + arm_delta, JOINT_LOW, JOINT_HIGH)
            self.data.qvel[self.arm_dof] = arm_delta / self.model.opt.timestep
            finger_delta = np.clip(
                self.data.qpos[self.finger_qpos] - finger_before,
                -0.2 * self.model.opt.timestep,
                0.2 * self.model.opt.timestep,
            )
            self.data.qpos[self.finger_qpos] = np.clip(finger_before + finger_delta, 0.0, 0.04)
            self.data.qvel[self.finger_dof] = finger_delta / self.model.opt.timestep
            mujoco.mj_forward(self.model, self.data)
            if finger_target == 0.0:
                self._try_attach_food()
                self._update_attached_food(self.model.opt.timestep)
        self.elapsed += POLICY_DT
        self.last_action = action.astype(np.float32)
        outcome, info = self._outcome()
        return StepResult(self.observation(), outcome is not None, outcome, info)

    def close(self) -> None:
        """Kept for a Gym-like lifecycle; MuJoCo's Python objects need no explicit close."""
