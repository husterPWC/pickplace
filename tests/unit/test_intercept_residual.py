import torch

from pickplace.grasp_skills import (
    SphereGraspSkillConfig,
    StableSphereResidualSkill,
    StableTargetConfig,
)
from pickplace.intercept import (
    interception_objective,
    moving_intercept_retreat_comparable,
    multiscale_position_quality,
    outside_workspace,
    progressive_intercept_hover_delta,
    progress_augmented_step_reward,
    update_retreat_count,
)


def test_progressive_intercept_descends_before_final_alignment_without_velocity():
    tcp = torch.tensor([[0.45, 0.30, 0.13]]).repeat(3, 1)
    # Grasp Z is 0.05 m.  Object X moves from the 16 cm lead line to the
    # 4 cm final-alignment line while TCP stays at the fixed x=0.45 line.
    target = torch.tensor(
        [[0.29, 0.30, 0.05], [0.35, 0.30, 0.05], [0.41, 0.30, 0.05]]
    )
    delta, progress = progressive_intercept_hover_delta(
        tcp, target - tcp, 0.45, 0.16, 0.04, 0.08, 0.02
    )
    assert torch.allclose(progress, torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(delta[:, 0], torch.zeros(3))
    assert torch.allclose(delta[:, 2], torch.tensor([0.0, -0.03, -0.06]), atol=1.0e-6)


def test_progressive_interception_objective_switches_only_at_final_align_line():
    tcp = torch.tensor([[0.45, 0.30, 0.13], [0.45, 0.30, 0.13]])
    target = torch.tensor([[0.35, 0.30, 0.05], [0.415, 0.30, 0.05]])
    _, near, hover = interception_objective(
        tcp,
        torch.cat((target - tcp, torch.zeros(2, 3)), dim=-1),
        0.45,
        0.16,
        0.08,
        0.04,
        0.02,
    )
    assert near.tolist() == [False, True]
    assert hover[0, 2].item() < 0.0


def test_far_intercept_objective_is_position_only_not_orientation():
    tcp = torch.tensor([[0.20, 0.10, 0.30], [0.20, 0.10, 0.30]])
    pose = torch.tensor(
        [
            [0.10, 0.20, -0.25, 0.0, 0.0, 0.0],
            [0.10, 0.20, -0.25, 2.5, -1.7, 0.9],
        ]
    )
    distance, near, hover = interception_objective(tcp, pose, 0.45, 0.08, 0.08)
    assert not near.any()
    assert torch.allclose(distance[0], distance[1])
    assert torch.allclose(hover[0], hover[1])


def test_toward_step_cumulative_reward_is_strictly_above_away_step():
    """Acceptance gate requested before any new short training is launched."""
    dt, gain = 0.02, 20.0
    start = torch.tensor([0.30])
    toward_distances = (torch.tensor([0.29]), torch.tensor([0.28]))
    away_distances = (torch.tensor([0.31]), torch.tensor([0.32]))

    def cumulative(path):
        previous = start
        total = torch.zeros(1)
        for current in path:
            quality = multiscale_position_quality(current)
            raw = progress_augmented_step_reward(quality, previous, current, dt, gain)
            total += raw * dt  # Isaac's dense-reward integration.
            previous = current
        return total

    toward = cumulative(toward_distances)
    away = cumulative(away_distances)
    assert toward.item() > 0.0
    assert away.item() < 0.0
    assert toward.item() > away.item()


def test_workspace_detects_each_axis_escape():
    positions = torch.tensor(
        [
            [0.45, 0.30, 0.30],
            [0.09, 0.30, 0.30],
            [0.45, -0.21, 0.30],
            [0.45, 0.30, 0.76],
        ]
    )
    result = outside_workspace(positions, (0.10, -0.20, 0.02), (0.75, 0.48, 0.75))
    assert result.tolist() == [False, True, True, True]


def test_retreat_counter_requires_consecutive_away_motion():
    count = torch.zeros(2, dtype=torch.long)
    previous = torch.tensor([0.30, 0.30])
    comparable = torch.tensor([True, True])
    count = update_retreat_count(count, previous, torch.tensor([0.301, 0.299]), comparable, 0.0005)
    assert count.tolist() == [1, 0]
    count = update_retreat_count(count, torch.tensor([0.301, 0.299]), torch.tensor([0.302, 0.300]), comparable, 0.0005)
    assert count.tolist() == [2, 1]
    # Moving toward resets immediately; changing objective stage is not
    # compared and also resets rather than producing a false retreat.
    count = update_retreat_count(count, torch.tensor([0.302, 0.300]), torch.tensor([0.301, 0.500]), torch.tensor([True, False]), 0.0005)
    assert count.tolist() == [0, 0]


def test_progressive_intercept_suspends_ambiguous_retreat_after_descent_starts():
    initialized = torch.tensor([True, True, True])
    previous_near = torch.tensor([False, False, True])
    current_near = torch.tensor([False, False, True])
    target_x = torch.tensor([0.28, 0.30, 0.43])
    comparable = moving_intercept_retreat_comparable(
        initialized, previous_near, current_near, target_x, descent_start_x=0.29
    )
    assert comparable.tolist() == [True, False, False]
    # Legacy fixed-hover behavior remains unchanged.
    legacy = moving_intercept_retreat_comparable(
        initialized, previous_near, current_near, target_x, descent_start_x=None
    )
    assert legacy.tolist() == [True, True, True]


def test_policy_is_only_a_bounded_residual_and_never_controls_fingers():
    controller = StableSphereResidualSkill(
        SphereGraspSkillConfig(intercept_x=0.45, intercept_lead_distance=0.08),
        StableTargetConfig(stable_steps=4),
        translation_residual_limit=0.10,
        rotation_residual_limit=0.05,
    )
    policy = torch.ones(2, 7)
    ee = torch.tensor([[0.20, 0.30, 0.30], [0.20, 0.30, 0.30]])
    fingers = torch.full((2, 2), 0.04)
    candidates = torch.zeros(2, 40, 6)
    candidates[..., 0] = 0.10
    candidates[..., 2] = -0.25
    executed, phase, _, nominal = controller(policy, candidates, ee, fingers)
    assert phase.eq(controller.baseline.HOVER).all()
    delta = executed - nominal
    assert torch.all(delta[:, :3].abs() <= 0.10 + 1.0e-7)
    assert torch.all(delta[:, 3:6].abs() <= 0.05 + 1.0e-7)
    assert torch.equal(executed[:, -1], nominal[:, -1])
