"""Privileged waypoint expert used to adapt the Isaac teacher to MuJoCo dynamics."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_sim.env import ACTION_SCALE, HOME_TCP


class MuJoCoPickPlaceExpert:
    """Deterministic state-machine expert operating on the 45-D policy observation.

    The expert uses only fields already present in the privileged state teacher
    observation.  Its demonstrations can therefore supervise the same 45 -> 7
    actor architecture as the Isaac checkpoint.
    """

    phases = (
        "pregrasp",
        "descend",
        "close",
        "lift",
        "transport",
        "place",
        "release",
        "retreat",
        "home",
    )
    def __init__(
        self,
        position_gain: float = 0.9,
        transport_action_limit: float = 0.20,
        lift_action_limit: float = 0.20,
    ):
        self.position_gain = position_gain
        self.transport_action_limit = transport_action_limit
        self.lift_action_limit = lift_action_limit
        self.reset()

    def reset(self) -> None:
        self.phase = "pregrasp"
        self.phase_steps = 0
        self.grasp_xy: np.ndarray | None = None
        self.hold_quat: np.ndarray | None = None

    @staticmethod
    def _state(observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        tcp = observation[0:3].astype(np.float64)
        bowl = observation[34:37].astype(np.float64)
        food = observation[37:40].astype(np.float64)
        grasped = bool(observation[44] > 0.5)
        return tcp, bowl, food, grasped

    def _advance(self, phase: str) -> None:
        self.phase = phase
        self.phase_steps = 0

    def act(self, observation: np.ndarray) -> np.ndarray:
        tcp, bowl, food, grasped = self._state(observation)
        self.phase_steps += 1
        gripper = 1.0

        if self.phase == "pregrasp":
            target = np.asarray([food[0], food[1], 0.11])
            if np.linalg.norm(tcp - target) < 0.022 or self.phase_steps >= 55:
                self.grasp_xy = food[:2].copy()
                self._advance("descend")
        elif self.phase == "descend":
            target = np.asarray([food[0], food[1], food[2] + 0.004])
            if np.linalg.norm(tcp - target) < 0.014 or self.phase_steps >= 42:
                self.grasp_xy = food[:2].copy()
                self._advance("close")
        elif self.phase == "close":
            assert self.grasp_xy is not None
            target = np.asarray([self.grasp_xy[0], self.grasp_xy[1], food[2] + 0.004])
            gripper = -1.0
            if grasped and self.hold_quat is None:
                self.hold_quat = observation[3:7].astype(np.float64).copy()
            if (grasped and self.phase_steps >= 24) or self.phase_steps >= 38:
                self._advance("lift")
        elif self.phase == "lift":
            assert self.grasp_xy is not None
            target = np.asarray([self.grasp_xy[0], self.grasp_xy[1], 0.19])
            gripper = -1.0
            if tcp[2] > 0.165 or self.phase_steps >= 42:
                self._advance("transport")
        elif self.phase == "transport":
            # Lead the 0.08 m/s conveyor slightly and compensate for the food's
            # measured +x offset from the TCP while pinched between the pads.
            target = np.asarray([bowl[0] + 0.002, bowl[1], 0.19])
            gripper = -1.0
            if np.linalg.norm(tcp[:2] - target[:2]) < 0.055 or self.phase_steps >= 75:
                self._advance("place")
        elif self.phase == "place":
            target = np.asarray([bowl[0] + 0.002, bowl[1], bowl[2] + 0.078])
            gripper = -1.0
            if np.linalg.norm(tcp - target) < 0.018 or self.phase_steps >= 38:
                self._advance("release")
        elif self.phase == "release":
            target = np.asarray([bowl[0] + 0.002, bowl[1], bowl[2] + 0.090])
            if self.phase_steps >= 10:
                # A direct diagonal return can make the lagging arm dip through
                # the lightweight bowl. When time permits, retreat vertically
                # above its rim first. Near the zone end, go home immediately.
                self._advance("retreat" if bowl[0] < 0.50 else "home")
        elif self.phase == "retreat":
            target = np.asarray([tcp[0], tcp[1], 0.22])
            if tcp[2] > 0.19 or self.phase_steps >= 28:
                self._advance("home")
        else:
            target = HOME_TCP

        position_gain = {
            "lift": 0.70,
            "transport": 0.70,
            "place": 0.70,
            "release": 0.55,
            "retreat": 0.75,
            "home": 1.20,
        }.get(self.phase, self.position_gain)
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(
            position_gain * (target - tcp) / ACTION_SCALE,
            -1.0,
            1.0,
        )
        if self.phase in ("lift", "transport", "place"):
            # A sudden 40 cm lateral command twists the wrist faster than the
            # contact solver can maintain a pinch. Limit the Cartesian command
            # norm while keeping its direction.
            command_norm = float(np.linalg.norm(action[:3]))
            if self.phase == "transport":
                limit = self.transport_action_limit
            elif self.phase == "lift":
                limit = self.lift_action_limit
            else:
                limit = 0.20
            if command_norm > limit:
                action[:3] *= limit / command_norm
        if self.hold_quat is not None:
            current_rot = Rotation.from_quat(observation[3:7])
            desired_rot = Rotation.from_quat(self.hold_quat)
            rot_error = (desired_rot * current_rot.inv()).as_rotvec()
            action[3:6] = np.clip(0.45 * rot_error / ACTION_SCALE, -1.0, 1.0)
        action[6] = gripper
        return action
