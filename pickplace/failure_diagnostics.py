"""Pure helpers for classifying and summarising moving-cube lift episodes.

This module deliberately has no Isaac Lab imports so that the classification
contract can be unit-tested without launching the simulator.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from statistics import mean, median


SUCCESS = "success"


def classify_lift_episode(episode: Mapping, *, target_lift: float = 0.03) -> str:
    """Assign the earliest failed stage of a moving-cube interception episode.

    Step fields use ``-1`` for an event that never happened.  The thresholds
    match the environment's 4 cm grasp proximity test; a five-control-step
    margin avoids labelling harmless one-frame command jitter as early/late.
    """
    if bool(episode.get("success")) or episode.get("outcome") == SUCCESS:
        return SUCCESS

    first_near = int(episode.get("first_near_step", -1))
    first_close = int(episode.get("first_close_step", -1))
    first_grasp = int(episode.get("first_grasp_step", -1))
    nearest = int(episode.get("nearest_step", -1))

    if first_near < 0:
        return "missed_intercept"
    if first_close < 0:
        return "never_closed_near_cube"
    if first_grasp < 0:
        if first_close < first_near - 5:
            return "closed_before_intercept"
        if nearest >= 0 and first_close > nearest + 5:
            return "closed_after_intercept"
        if episode.get("outcome") == "cube_off_belt":
            return "contact_pushed_cube_off_belt"
        return "grasp_contact_failed"

    max_held_lift = float(episode.get("max_held_lift_m", 0.0))
    if bool(episode.get("grasp_lost")):
        if bool(episode.get("opened_after_grasp")):
            return "released_during_lift"
        if max_held_lift < 0.005:
            return "lost_grasp_before_lift"
        if max_held_lift < 0.02:
            return "lost_grasp_during_early_lift"
        return "lost_grasp_near_target"
    if max_held_lift < 0.005:
        return "grasped_but_no_lift"
    if max_held_lift < target_lift:
        return "insufficient_lift"
    return "lifted_but_not_credited"


def _distribution(values: Iterable[float]) -> dict[str, float | None]:
    values = [float(v) for v in values]
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def summarise_episodes(episodes: list[Mapping]) -> dict:
    """Aggregate outcomes/stages and a few interpretable physical metrics."""
    n = len(episodes)
    outcomes = Counter(str(e["outcome"]) for e in episodes)
    stages = Counter(str(e["failure_stage"]) for e in episodes)
    failures = [e for e in episodes if e["failure_stage"] != SUCCESS]
    return {
        "episodes": n,
        "successes": n - len(failures),
        "failures": len(failures),
        "success_rate": (n - len(failures)) / n if n else 0.0,
        "outcomes": dict(sorted(outcomes.items())),
        "failure_stages": dict(sorted(stages.items())),
        "failed_episode_metrics": {
            "min_ee_cube_distance_m": _distribution(e["min_ee_cube_distance_m"] for e in failures),
            "max_cube_lift_m": _distribution(e["max_cube_lift_m"] for e in failures),
            "max_held_lift_m": _distribution(e["max_held_lift_m"] for e in failures),
            "episode_return": _distribution(e["episode_return"] for e in failures),
        },
    }
