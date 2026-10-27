from pickplace.failure_diagnostics import classify_lift_episode, summarise_episodes


def episode(**overrides):
    value = {
        "success": False,
        "outcome": "time_out",
        "first_near_step": 100,
        "first_close_step": 100,
        "first_grasp_step": 105,
        "nearest_step": 102,
        "max_held_lift_m": 0.02,
        "max_cube_lift_m": 0.02,
        "min_ee_cube_distance_m": 0.02,
        "grasp_lost": False,
        "opened_after_grasp": False,
        "episode_return": 1.0,
    }
    value.update(overrides)
    return value


def test_classifies_pregrasp_failures_by_earliest_stage():
    assert classify_lift_episode(episode(first_near_step=-1, first_grasp_step=-1)) == "missed_intercept"
    assert classify_lift_episode(episode(first_close_step=-1, first_grasp_step=-1)) == "never_closed_near_cube"
    assert classify_lift_episode(episode(first_close_step=80, first_grasp_step=-1)) == "closed_before_intercept"
    assert classify_lift_episode(episode(first_close_step=120, first_grasp_step=-1)) == "closed_after_intercept"
    assert classify_lift_episode(episode(first_grasp_step=-1, outcome="cube_off_belt")) == "contact_pushed_cube_off_belt"
    assert classify_lift_episode(episode(first_grasp_step=-1)) == "grasp_contact_failed"


def test_classifies_postgrasp_failures_and_success():
    assert classify_lift_episode(episode(success=True)) == "success"
    assert classify_lift_episode(episode(grasp_lost=True, max_held_lift_m=0.001)) == "lost_grasp_before_lift"
    assert classify_lift_episode(episode(grasp_lost=True, max_held_lift_m=0.01)) == "lost_grasp_during_early_lift"
    assert classify_lift_episode(episode(grasp_lost=True, max_held_lift_m=0.025)) == "lost_grasp_near_target"
    assert classify_lift_episode(episode(grasp_lost=True, opened_after_grasp=True)) == "released_during_lift"
    assert classify_lift_episode(episode(max_held_lift_m=0.001)) == "grasped_but_no_lift"
    assert classify_lift_episode(episode(max_held_lift_m=0.02)) == "insufficient_lift"
    assert classify_lift_episode(episode(max_held_lift_m=0.04)) == "lifted_but_not_credited"


def test_summary_counts_and_excludes_success_from_failed_metrics():
    rows = [episode(success=True, outcome="success"), episode()]
    for row in rows:
        row["failure_stage"] = classify_lift_episode(row)
    result = summarise_episodes(rows)
    assert result["episodes"] == 2
    assert result["successes"] == 1
    assert result["success_rate"] == 0.5
    assert result["failure_stages"] == {"insufficient_lift": 1, "success": 1}
    assert result["failed_episode_metrics"]["episode_return"]["mean"] == 1.0
