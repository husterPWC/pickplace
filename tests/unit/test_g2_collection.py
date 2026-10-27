from pickplace.g2_collection import (
    CALIBRATED_SCENE_KEYS,
    DEPLOYABLE_PROPRIO_KEYS,
    FORBIDDEN_STUDENT_SIGNALS,
    estimated_shard_bytes,
    profile,
    validate_g1_teacher_payload,
    validate_jaka_visual_teacher_payload,
)


def _payload():
    return {
        "frames": 262144,
        "config": {
            "env": {
                "task": "FoodRobot-MovingSphereStaticBowl-v0",
                "speed_free_observations": True,
                "stable_sphere_pick_place_skill": {"enabled": True},
            },
            "network": {
                "actor_in_keys": [
                    ["proprio", "ee_pos"],
                    ["proprio", "gripper_pos"],
                    ["grasp_guidance", "points_ee"],
                    ["position_history", "position_action"],
                ]
            },
        },
    }


def test_g2_contract_separates_deployable_inputs_from_teacher_labels():
    contract = validate_g1_teacher_payload(_payload())
    runtime = {tuple(key) for key in contract["student_runtime_keys"]}
    assert set(DEPLOYABLE_PROPRIO_KEYS) <= runtime
    assert set(CALIBRATED_SCENE_KEYS) <= runtime
    assert contract["teacher_only_groups"] == ["grasp_guidance", "position_history"]
    assert contract["action_semantics"] == "teacher_residual_before_environment_g1_controller"
    assert not any(signal in {leaf for key in runtime for leaf in key} for signal in FORBIDDEN_STUDENT_SIGNALS)


def test_g2_profiles_and_storage_estimate_are_monotonic():
    assert profile("smoke")["frames"] < profile("pilot")["frames"] < profile("full")["frames"]
    assert estimated_shard_bytes(1_000, 84) < estimated_shard_bytes(1_000, 128)
    assert estimated_shard_bytes(1_000, 84) * 10 == estimated_shard_bytes(10_000, 84)


def _jaka_payload(task, *, pick_place=False, home=False):
    payload = _payload()
    payload["config"]["env"].update(
        {
            "task": task,
            "arm": "zu5_ag95",
            "success_requires_home": home,
            "stable_sphere_pick_place_skill": {"enabled": pick_place},
        }
    )
    return payload


def test_jaka_moving_sphere_uses_frozen_lift_actor_with_full_place_environment():
    actor = _jaka_payload("FoodRobot-MovingSphereResidualGapLift-v0")
    env = _jaka_payload("FoodRobot-MovingSphereStaticBowl-v0", pick_place=True)
    contract = validate_jaka_visual_teacher_payload(
        actor, env, "moving_sphere_static_bowl"
    )
    assert contract["robot"] == "jaka_zu5_ag95"
    assert contract["actor_source_task"] == "FoodRobot-MovingSphereResidualGapLift-v0"
    assert contract["effective_task"] == "FoodRobot-MovingSphereStaticBowl-v0"
    assert contract["success_requires_home"] is False


def test_jaka_static_sphere_moving_bowl_requires_home():
    payload = _jaka_payload(
        "FoodRobot-StaticSphereMovingBowl-v0", pick_place=True, home=True
    )
    contract = validate_jaka_visual_teacher_payload(
        payload, payload, "static_sphere_moving_bowl"
    )
    assert contract["success_requires_home"] is True


def test_jaka_visual_contract_rejects_wrong_robot():
    actor = _jaka_payload("FoodRobot-MovingSphereResidualGapLift-v0")
    env = _jaka_payload("FoodRobot-MovingSphereStaticBowl-v0", pick_place=True)
    env["config"]["env"]["arm"] = "franka"
    try:
        validate_jaka_visual_teacher_payload(actor, env, "moving_sphere_static_bowl")
    except ValueError as error:
        assert "arm=zu5_ag95" in str(error)
    else:
        raise AssertionError("wrong robot must be rejected")
