"""Contract tests for the independent moving-sphere MuJoCo bridge."""

from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("mujoco")

from mujoco_sim.moving_sphere import (  # noqa: E402
    FrankaMovingSphereMuJoCo,
    GraspPointGRUActor,
    moving_sphere_outcome,
    moving_sphere_place_outcome,
)
from mujoco_sim.jaka_moving_sphere import JakaMovingSphereMuJoCo  # noqa: E402


CHECKPOINT = Path(
    "artifacts/moving_sphere_static_bowl/teachers/"
    "moving_sphere_g0_s0d_canonical_coarse_1024_no_rtx/checkpoints/ppo_teacher_262144.pt"
)
STATIC_MOVING_BOWL_CHECKPOINT = Path(
    "artifacts/jaka_zu5_ag95_static_sphere_moving_bowl/teachers/"
    "jaka_static_sphere_moving_bowl_m0_final_geometry_smoke16/checkpoints/"
    "ppo_teacher_final.pt"
)


def test_speed_free_observation_shapes_and_closed_loop_step():
    env = FrankaMovingSphereMuJoCo(seed=0)
    observation = env.reset(0)
    assert observation["points_ee"].shape == (40, 20, 3)
    assert observation["best_pose_delta"].shape == (6,)
    assert observation["position_action"].shape == (10, 12)
    result = env.step(np.zeros(7, dtype=np.float32))
    assert np.isfinite(result.observation["points_ee"]).all()
    assert result.info["sphere_x"] > 0.20
    assert result.info["ever_left_contact"] is False
    assert result.info["ever_right_contact"] is False
    assert result.info["ever_dual_contact"] is False
    assert np.isfinite(result.info["min_finger_gap"])


def test_checkpoint_actor_loads_and_is_finite():
    if not CHECKPOINT.is_file():
        pytest.skip("local G0-S0d checkpoint unavailable")
    env = FrankaMovingSphereMuJoCo(seed=0)
    actor = GraspPointGRUActor(CHECKPOINT)
    action = actor.act(env.reset(0))
    assert action.shape == (7,)
    assert np.isfinite(action).all()
    assert np.max(np.abs(action)) <= 1.0


def test_residual_baseline_controller_owns_coarse_motion_and_fingers():
    env = FrankaMovingSphereMuJoCo(seed=0, controller="residual_baseline")
    observation = env.reset(0)
    result = env.step(np.ones(7, dtype=np.float32))
    # The fixed interception baseline drives toward +Y from the Franka home
    # pose; policy translation is only a bounded correction.  Fingers stay
    # open even though the policy asks for full closure.
    assert result.info["controller"] == "residual_baseline"
    assert env.last_action[1] > 0.0
    assert env.last_action[-1] == pytest.approx(1.0)


def test_long_belt_variant_moves_only_the_configured_spawn_and_visual_edge():
    baseline = FrankaMovingSphereMuJoCo(seed=0)
    baseline_observation = baseline.reset(0)
    variant = FrankaMovingSphereMuJoCo(
        seed=0,
        sphere_start_x=0.05,
        belt_upstream_x=-0.25,
        pallet_start_x=0.05,
        lift_timeout_steps=1200,
        max_seconds=60.0,
        controller="residual_baseline",
    )
    variant_observation = variant.reset(0)
    belt = variant.model.geom("belt_visual").id
    assert variant.data.xpos[variant.food_body, 0] == pytest.approx(0.05)
    assert variant.data.xpos[variant.model.body("pallet").id, 0] == pytest.approx(0.05)
    assert variant.model.geom_pos[belt, 0] - variant.model.geom_size[belt, 0] == pytest.approx(-0.25)
    assert variant.model.geom_pos[belt, 0] + variant.model.geom_size[belt, 0] == pytest.approx(1.0)
    # Actor contract is unchanged; only the initial target geometry differs.
    assert variant_observation.keys() == baseline_observation.keys()
    for key in variant_observation:
        assert variant_observation[key].shape == baseline_observation[key].shape
    assert variant.option.baseline.config.lift_timeout_steps == 1200
    assert variant.max_seconds == pytest.approx(60.0)


def test_g0_s2c_alignment_and_close_following_reach_mujoco_controller():
    env = FrankaMovingSphereMuJoCo(
        seed=0,
        controller="residual_baseline",
        align_target_xy_feedforward=True,
        close_requires_raw_alignment=True,
        max_close_hold_xy_step=0.008,
    )
    config = env.option.baseline.config
    assert config.align_target_xy_feedforward is True
    assert config.close_requires_raw_alignment is True
    assert config.max_close_hold_xy_step == pytest.approx(0.008)


def test_attached_sphere_can_finish_lift_after_crossing_interception_deadline():
    assert moving_sphere_outcome(
        grasp_attached=False, lift=0.0, sphere_x=0.66, sphere_z=0.05, elapsed=9.0
    ) == "sphere_exited_zone"
    assert moving_sphere_outcome(
        grasp_attached=True, lift=0.001, sphere_x=0.66, sphere_z=0.051, elapsed=9.0
    ) is None
    assert moving_sphere_outcome(
        grasp_attached=True, lift=0.031, sphere_x=0.68, sphere_z=0.081, elapsed=10.0
    ) == "strict_lift_success"


def test_full_place_outcome_requires_settled_release_and_preserves_grasped_transport():
    assert moving_sphere_place_outcome(
        settled_in_bowl=True,
        grasp_ever_secured=True,
        sphere_x=0.45,
        sphere_z=0.026,
        elapsed=15.0,
    ) == "strict_place_success"
    assert moving_sphere_place_outcome(
        settled_in_bowl=False,
        grasp_ever_secured=False,
        sphere_x=0.66,
        sphere_z=0.05,
        elapsed=9.0,
    ) == "sphere_exited_zone"
    assert moving_sphere_place_outcome(
        settled_in_bowl=False,
        grasp_ever_secured=True,
        sphere_x=0.66,
        sphere_z=0.15,
        elapsed=9.0,
    ) is None


def test_g1_pick_place_controller_moves_fixed_destination_bowl_and_owns_handoff():
    env = FrankaMovingSphereMuJoCo(
        seed=0,
        controller="residual_baseline",
        pick_place=True,
        destination_bowl_pos=(0.45, 0.0, 0.0),
        translation_residual_limit=0.05,
        rotation_residual_limit=0.025,
    )
    assert np.allclose(env.model.body_pos[env.destination_bowl_body], (0.45, 0.0, 0.0))
    assert env.option.grasp.translation_residual_limit == pytest.approx(0.05)
    assert env.option.grasp.rotation_residual_limit == pytest.approx(0.025)
    result = env.step(np.zeros(7, dtype=np.float32))
    assert result.info["pick_place"] is True
    assert result.info["option_phase"] == "wait_grasp"
    assert result.info["settled_in_bowl"] is False


def test_sphere_spawn_y_is_configurable_for_position_randomization():
    env = FrankaMovingSphereMuJoCo(seed=0, sphere_start_y=0.307)
    env.reset(0)
    assert env.data.xpos[env.food_body, 1] == pytest.approx(0.307)


def test_jaka_j1_uses_high_speed_grasp_and_place_handoff_contract():
    env = JakaMovingSphereMuJoCo(
        seed=0,
        controller_profile="j1",
        destination_bowl_pos=(0.46, -0.01, 0.0),
    )
    assert env.pick_place is True
    assert np.allclose(env.model.body_pos[env.destination_bowl_body], (0.46, -0.01, 0.0))
    assert env.option.grasp.translation_residual_limit == pytest.approx(0.05)
    assert env.option.grasp.rotation_residual_limit == pytest.approx(0.025)
    assert env.option.grasp.baseline.config.intercept_lead_distance == pytest.approx(0.16)
    result = env.step(np.zeros(7, dtype=np.float32))
    assert result.info["pick_place"] is True
    assert result.info["option_phase"] == "wait_grasp"
    assert result.info["settled_in_bowl"] is False


def test_jaka_repeated_policy_call_spawns_next_ball_without_world_reset():
    env = JakaMovingSphereMuJoCo(
        seed=0,
        belt_speed=0.08,
        controller_profile="j1",
        destination_bowl_pos=(0.45, 0.0, 0.0),
        num_spheres=3,
    )
    env.reset(0)
    first_qpos = int(env.food_qpos_all[0])
    env.data.qpos[first_qpos : first_qpos + 3] = (0.45, 0.0, 0.026)
    arm_before = env.data.qpos[env.arm_qpos].copy()
    env.prepare_sphere_call(1, sphere_start_x=0.203, sphere_start_y=0.297)

    assert env.active_sphere_index == 1
    assert np.allclose(env.data.qpos[first_qpos : first_qpos + 3], (0.45, 0.0, 0.026))
    assert np.allclose(env.data.qpos[env.arm_qpos], arm_before)
    assert np.allclose(env.data.xpos[env.food_body], (0.203, 0.297, 0.05))
    assert env.model.geom_contype[env.food_geom_all[0]] != 0
    assert env.model.geom_contype[env.food_geom_all[1]] != 0
    assert env.model.geom_contype[env.food_geom_all[2]] == 0
    assert env.history_initialized is True
    assert env.option.phase is None or int(env.option.phase[0]) == env.option.WAIT_GRASP


def test_jaka_visible_queue_keeps_all_balls_physical_and_switches_without_respawn():
    env = JakaMovingSphereMuJoCo(
        seed=0,
        belt_speed=0.08,
        controller_profile="j1",
        destination_bowl_pos=(0.45, 0.0, 0.0),
        num_spheres=3,
        sphere_queue_spacing=0.71,
    )
    env.reset(0)
    expected_x = np.asarray((0.20, -0.51, -1.22))
    assert np.all(env.model.geom_contype[env.food_geom_all] != 0)
    assert np.allclose(env.data.xpos[env.food_body_all, 0], expected_x)
    assert np.allclose(env.data.qpos[env.pallet_qpos_all], (0.0, -0.71, -1.42))

    first_before = env.data.xpos[env.food_body_all[0]].copy()
    second_before = env.data.xpos[env.food_body_all[1]].copy()
    world_time_before = float(env.data.time)
    observation = env.prepare_sphere_call(1)
    assert env.active_sphere_index == 1
    assert float(env.data.time) == pytest.approx(world_time_before)
    assert np.allclose(env.data.xpos[env.food_body_all[0]], first_before)
    assert np.allclose(env.data.xpos[env.food_body_all[1]], second_before)
    assert observation["points_ee"].shape == (40, 20, 3)

    result = env.step(np.zeros(7, dtype=np.float32))
    assert result.info["sphere_x"] > second_before[0]
    assert env.data.xpos[env.food_body_all[0], 0] > first_before[0]
    assert np.all(np.diff(env.data.qpos[env.pallet_qpos_all]) < 0.0)


def test_jaka_static_ball_and_moving_bowl_queue_switches_existing_pairs():
    env = JakaMovingSphereMuJoCo(
        seed=0,
        belt_speed=0.10,
        controller_profile="static_pick",
        pick_place=True,
        task_variant="static_sphere_moving_bowl",
        num_spheres=3,
        moving_bowl_queue_spacing=0.75,
    )
    env.reset(0)
    assert np.all(env.model.geom_contype[env.food_geom_all] != 0)
    assert np.allclose(
        env.data.xpos[env.food_body_all],
        ((0.45, -0.06, 0.05), (0.395, -0.06, 0.05), (0.505, -0.06, 0.05)),
    )
    assert np.allclose(env.data.xpos[env.moving_bowl_body_all, 0], (0.08, -0.67, -1.42))
    assert np.allclose(env.data.qpos[env.pallet_qpos_all], (-0.12, -0.87, -1.62))

    first_sphere_before = env.data.xpos[env.food_body_all[0]].copy()
    second_sphere_before = env.data.xpos[env.food_body_all[1]].copy()
    bowls_before = env.data.xpos[env.moving_bowl_body_all].copy()
    world_time_before = float(env.data.time)
    observation = env.prepare_static_sphere_moving_bowl_call(1)
    assert env.active_sphere_index == 1
    assert env.destination_bowl_body == env.moving_bowl_body_all[1]
    assert float(env.data.time) == pytest.approx(world_time_before)
    assert np.allclose(env.data.xpos[env.food_body_all[0]], first_sphere_before)
    assert np.allclose(env.data.xpos[env.food_body_all[1]], second_sphere_before)
    assert observation["points_ee"].shape == (40, 20, 3)

    env.step(np.zeros(7, dtype=np.float32))
    assert np.all(env.data.xpos[env.moving_bowl_body_all, 0] > bowls_before[:, 0])
    assert np.allclose(env.data.xpos[env.food_body_all[0]], first_sphere_before, atol=1.0e-3)


def test_static_moving_bowl_frozen_checkpoint_produces_real_nonzero_residual():
    if not STATIC_MOVING_BOWL_CHECKPOINT.is_file():
        pytest.skip("local M0 checkpoint unavailable")
    actor = GraspPointGRUActor(STATIC_MOVING_BOWL_CHECKPOINT)
    env = JakaMovingSphereMuJoCo(
        seed=0,
        belt_speed=0.10,
        controller_profile="static_pick",
        pick_place=True,
        task_variant="static_sphere_moving_bowl",
        translation_residual_limit=0.03,
        rotation_residual_limit=0.015,
    )
    action = actor.act(env.reset(0))
    assert action.shape == (7,)
    assert np.isfinite(action).all()
    assert np.max(np.abs(action)) > 1.0e-4
    assert env.option.grasp.translation_residual_limit == pytest.approx(0.03)
    assert env.option.grasp.rotation_residual_limit == pytest.approx(0.015)
