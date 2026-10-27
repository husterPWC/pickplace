import pytest

from pickplace import rewards as R

# simple_v3b is the final reward set used by the published teacher and datasets.  simple_v2 is retained because
# the published policy was warm-started through it and cannot be reproduced from scratch with v3b reliably.
SIMPLE_V3B = {
    "reach_food": 1.0, "grasp": 2.0, "grasp_lift": 5.0, "transport": 10.0, "transport_fine": 0.0,
    "bowl_disturbance": -10.0, "action_rate": -0.01, "joint_vel": -0.001, "food_in_bowl": 20.0,
    "return_home": 10.0,
    "success": 150.0, "bowl_failure": 0.0, "food_dropped": 0.0,
}


def test_vocabulary_is_dense_then_events_without_duplicates():
    assert R.REWARD_TERMS == R.DENSE_TERMS + R.EVENT_TERMS
    assert len(set(R.REWARD_TERMS)) == len(R.REWARD_TERMS)
    assert set(R.EVENT_SOURCES) == set(R.EVENT_TERMS)
    assert R.EVENT_SOURCES["bowl_failure"] == ("bowl_off_belt", "bowl_tipped")


def test_simple_v3b_has_todays_weights():
    assert R.available_reward_sets() == [
        "moving_cube_bridge_v1",
        "moving_cube_close_v1",
        "moving_cube_lift_v1",
        "moving_cube_v1",
        "moving_sphere_lift_v1",
        "moving_sphere_v1",
        "simple_v2",
        "simple_v3b",
        "sphere_a0_lift_v1",
        "sphere_gap_lift_v1",
        "sphere_residual_lift_v1",
    ]
    weights = R.load_reward_set("simple_v3b")
    assert weights == SIMPLE_V3B
    assert set(weights) == set(R.REWARD_TERMS)
    active = {t for t, w in weights.items() if w != 0.0}
    assert active == set(R.REWARD_TERMS) - {"transport_fine", "bowl_failure", "food_dropped"}


def test_simple_v2_is_the_penalty_free_placement_curriculum():
    expected = {
        "reach_food": 1.0,
        "grasp": 2.0,
        "grasp_lift": 5.0,
        "transport": 10.0,
        "food_in_bowl": 20.0,
        "success": 100.0,
    }
    weights = R.load_reward_set("simple_v2")
    assert {term: weight for term, weight in weights.items() if weight != 0.0} == expected
    assert weights["return_home"] == 0.0


def test_moving_cube_reward_disables_fixed_bowl_failure_terms():
    weights = R.load_reward_set("moving_cube_v1")
    assert weights["success"] == 150.0
    assert weights["return_home"] == 10.0
    assert weights["bowl_disturbance"] == 0.0
    assert weights["bowl_failure"] == 0.0


def test_moving_cube_lift_curriculum_is_dense_and_has_no_placement_terms():
    weights = R.load_reward_set("moving_cube_lift_v1")
    assert weights["grasp_lift"] == 100.0
    assert weights["success"] == 250.0
    assert weights["transport"] == 0.0
    assert weights["food_in_bowl"] == 0.0
    assert weights["return_home"] == 0.0


def test_food_in_bowl_then_return_home_are_the_last_dense_terms():
    # appended in this order, so older vectors are a prefix of the dense block
    assert R.DENSE_TERMS[-2:] == ("food_in_bowl", "return_home")
    assert len(R.DENSE_TERMS) == 10 and len(R.REWARD_TERMS) == 13


def test_env_options_home_defaults_and_unknown_keys_raise():
    from pickplace.config import DEFAULT_ENV, build_cell_env_cfg

    assert DEFAULT_ENV["success_requires_home"] is False  # every existing config keeps today's success
    assert DEFAULT_ENV["home_tolerance"] == 0.05  # [m], TCP distance to its home position
    with pytest.raises(ValueError, match="succes_requires_home"):
        build_cell_env_cfg({"succes_requires_home": True})


def test_default_robot_reset_matches_todays_reset_joints_offset():
    from pickplace.config import DEFAULT_ENV

    # every existing config keeps today's tight reset (reset_joints_by_offset's params before robot_reset existed)
    assert DEFAULT_ENV["robot_reset"] == {"position_range": [-0.02, 0.02], "velocity_range": [0.0, 0.0]}


def test_grasp_stability_is_opt_in_and_preserves_existing_tasks():
    from pickplace.config import DEFAULT_ENV

    assert DEFAULT_ENV["grasp_stability"] == {
        "hold_steps": 0,
        "hold_reward": 3.0,
        "slip_penalty": 0.0,
        "prelift_max_height": None,
        "premature_lift_penalty": 0.0,
        "require_stable_success": False,
    }
    assert DEFAULT_ENV["grasp_hold_gate"] == {
        "hold_steps": 0, "max_upward_dz": 0.0, "force_close": True,
        "follow_object_xy": False, "follow_gain": 0.5, "max_follow_speed": 0.15,
        "freeze_rotation": False,
    }
    assert DEFAULT_ENV["geometric_grasp_stability"]["stable_steps"] == 0
    assert DEFAULT_ENV["geometric_grasp_stability"]["relative_speed_threshold"] == 0.035
    assert DEFAULT_ENV["geometric_lift_velocity"] == {"weight": 0.0, "std": 0.05}
    assert DEFAULT_ENV["geometric_close_curriculum"] == {
        "success_steps": 0,
        "distance_threshold": 0.045,
        "min_close_fraction": 0.15,
        "max_close_fraction": 0.90,
        "far_close_penalty": 0.0,
    }
    assert DEFAULT_ENV["geometric_close_lift_bridge"]["enabled"] is False
    assert DEFAULT_ENV["geometric_close_lift_bridge"]["target_lift"] == 0.005


def test_unlisted_terms_default_to_zero(tmp_path):
    p = tmp_path / "sparse.yaml"
    p.write_text("success: 1.0\n")
    weights = R.load_reward_set(str(p))
    assert weights["success"] == 1.0
    assert all(weights[t] == 0.0 for t in R.REWARD_TERMS if t != "success")


def test_unknown_term_in_a_set_is_an_error(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("reach_fod: 1.0\n")
    with pytest.raises(KeyError, match="reach_fod"):
        R.load_reward_set(str(p))


def test_unknown_set_name_lists_available():
    with pytest.raises(FileNotFoundError, match="simple_v3b"):
        R.load_reward_set("does_not_exist")


def test_resolve_defaults_to_simple_v3b():
    assert R.resolve_reward_weights({}) == SIMPLE_V3B


def test_resolve_precedence_set_then_legacy_then_reward_weights():
    cfg = {
        "reward_set": "simple_v3b",
        "rewards": {"reach_food": 2.0},
        "success_bonus": 100.0,
        "bowl_failure_penalty": None,
        "food_drop_penalty": 10.0,
        "reward_weights": {"reach_food": 3.0, "grasp": 0.0},
    }
    w = R.resolve_reward_weights(cfg)
    assert w["reach_food"] == 3.0          # reward_weights beats legacy `rewards`
    assert w["grasp"] == 0.0
    assert w["success"] == 100.0           # legacy one-shot field, positive bonus
    assert w["food_dropped"] == -10.0      # legacy penalty is stored as a negative weight
    assert w["bowl_failure"] == 0.0        # None leaves the set's value (simple_v3b zeroes this term)


@pytest.mark.parametrize("bad", [{"rewards": {"place_success": 5.0}}, {"rewards": {"bogus": 1.0}}])
def test_legacy_rewards_accepts_dense_terms_only(bad):
    with pytest.raises(KeyError):
        R.resolve_reward_weights(bad)


def test_reward_weights_rejects_unknown_names():
    with pytest.raises(KeyError, match="nope"):
        R.resolve_reward_weights({"reward_weights": {"nope": 1.0}})


def test_weight_vector_order_and_isaac_weight():
    w = R.load_reward_set("simple_v3b")
    assert R.weight_vector(w) == [SIMPLE_V3B[t] for t in R.REWARD_TERMS]
    assert R.isaac_weight(0.0) == 1.0 and R.isaac_weight(-1e-4) == -1e-4
