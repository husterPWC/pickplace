"""Fast contract tests for the Franka MuJoCo sim-to-sim environment."""

from pathlib import Path

import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")

from mujoco_sim.env import (  # noqa: E402
    BELT_SPEED,
    DEFAULT_Q,
    HOME_TCP,
    POLICY_DT,
    FrankaPickPlaceMuJoCo,
)
from mujoco_sim.policy_runner import DEFAULT_CHECKPOINT, TeacherActor, load_actor  # noqa: E402


ADAPTED_CHECKPOINT = Path(
    "artifacts/mujoco/adaptation/franka_v3c_mujoco_adapted.pt"
)


def test_fixed_reset_matches_isaac_home_contract():
    env = FrankaPickPlaceMuJoCo(seed=0, randomize=False)
    observation = env.reset(seed=0)

    assert observation.shape == (45,)
    assert np.isfinite(observation).all()
    np.testing.assert_allclose(observation[:3], HOME_TCP, atol=5e-5)
    np.testing.assert_allclose(observation[9:18], np.zeros(9), atol=1e-7)
    np.testing.assert_allclose(env.data.qpos[env.robot_qpos], DEFAULT_Q, atol=1e-12)


def test_actor_first_action_matches_isaac_contract():
    if not Path(DEFAULT_CHECKPOINT).is_file():
        pytest.skip("local v3c checkpoint is not available")

    env = FrankaPickPlaceMuJoCo(seed=0, randomize=False)
    actor = TeacherActor(DEFAULT_CHECKPOINT)
    action = actor.act(env.reset(seed=0))

    # Captured from the same checkpoint in Isaac Lab with randomization disabled.
    expected = np.asarray(
        [0.296548, -0.204822, -0.513076, 0.398786, -0.448787, -0.171456, 0.302269]
    )
    np.testing.assert_allclose(action, expected, atol=2e-6)


def test_closed_loop_step_is_finite_and_advances_belt():
    env = FrankaPickPlaceMuJoCo(seed=0, randomize=False)
    observation = env.reset(seed=0)
    bowl_x_before = float(env.data.xpos[env.bowl_body, 0])

    result = env.step(np.zeros(7))

    assert result.observation.shape == observation.shape
    assert np.isfinite(result.observation).all()
    assert env.elapsed == pytest.approx(POLICY_DT)
    assert float(env.data.xpos[env.bowl_body, 0]) > bowl_x_before
    assert env.data.ctrl[env.pallet_act] == pytest.approx(BELT_SPEED * POLICY_DT)


def test_belt_speed_is_runtime_configurable():
    env = FrankaPickPlaceMuJoCo(seed=0, randomize=False, belt_speed=0.12)
    env.reset(seed=0)
    env.step(np.zeros(7))
    assert env.data.ctrl[env.pallet_act] == pytest.approx(0.12 * POLICY_DT)


def test_phase_conditioned_adapted_checkpoint_loads():
    if not ADAPTED_CHECKPOINT.is_file():
        pytest.skip("local MuJoCo-adapted checkpoint is not available")
    env = FrankaPickPlaceMuJoCo(seed=0, randomize=False)
    actor = load_actor(ADAPTED_CHECKPOINT.resolve())
    actor.reset()
    action = actor.act(env.reset(seed=0))
    assert action.shape == (7,)
    assert np.isfinite(action).all()
    assert np.max(np.abs(action)) <= 1.0
