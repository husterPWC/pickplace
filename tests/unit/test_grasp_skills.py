import pytest
import torch

from pickplace.grasp_skills import (
    SphereCloseHoldLiftOptionConfig,
    SphereGraspSkillConfig,
    StableGraspTarget,
    StableSphereCloseHoldLiftOption,
    StableSphereGraspSkill,
    StableTargetConfig,
)


def test_stable_target_filters_absolute_target_not_robot_motion():
    tracker = StableGraspTarget(
        StableTargetConfig(translation_alpha=1.0, rotation_alpha=1.0, stable_steps=2)
    )
    ee = torch.tensor([[0.0, 0.0, 0.1]])
    target = torch.tensor([[0.4, 0.3, 0.05]])
    pose = torch.cat((target - ee, torch.zeros(1, 3)), dim=-1)
    tracker.update(pose, ee)
    ee2 = ee + torch.tensor([[0.01, 0.0, 0.0]])
    pose2 = torch.cat((target - ee2, torch.zeros(1, 3)), dim=-1)
    filtered, ready, continuous = tracker.update(pose2, ee2)
    assert continuous.item()
    assert ready.item()
    assert torch.allclose(filtered[:, :3], target - ee2)


def test_target_jump_resets_readiness():
    tracker = StableGraspTarget(StableTargetConfig(stable_steps=1, max_translation_jump=0.01))
    ee = torch.zeros(1, 3)
    tracker.update(torch.zeros(1, 6), ee)
    _, ready, continuous = tracker.update(torch.tensor([[0.03, 0, 0, 0, 0, 0.0]]), ee)
    assert not continuous.item()
    assert not ready.item()


def test_candidate_identity_hysteresis_prevents_symmetric_target_chatter():
    tracker = StableGraspTarget(
        StableTargetConfig(
            translation_alpha=1.0,
            rotation_alpha=1.0,
            stable_steps=1,
            switch_margin=0.02,
        )
    )
    ee = torch.zeros(1, 3)
    candidates = torch.tensor([[[0, 0, 0, 0, 0, 0.05], [0, 0, 0, 0, 0, 0.06]]], dtype=torch.float32)
    tracker.update(candidates, ee)
    assert tracker.selected_index.item() == 0
    # Candidate 1 becomes only marginally better; keep candidate identity.
    candidates[0, 0, 5] = 0.06
    candidates[0, 1, 5] = 0.05
    tracker.update(candidates, ee)
    assert tracker.selected_index.item() == 0


def test_skill_requires_stable_alignment_then_closes_holds_and_lifts():
    skill = StableSphereGraspSkill(
        SphereGraspSkillConfig(
            hover_height=0.08,
            hover_translation_threshold=0.01,
            grasp_translation_threshold=0.005,
            alignment_steps=2,
            close_steps=2,
            hold_steps=2,
            lift_steps=4,
            lift_timeout_steps=12,
            lift_height=0.04,
        ),
        StableTargetConfig(stable_steps=1),
    )
    action = torch.zeros(1, 7)
    fingers = torch.full((1, 2), 0.04)
    # Already at the hover target: raw grasp target is 8 cm below TCP.
    ee = torch.tensor([[0.4, 0.3, 0.13]])
    pose = torch.tensor([[0.0, 0.0, -0.08, 0.0, 0.0, 0.0]])
    for _ in range(3):
        _, phase, _ = skill(action, pose, ee, fingers)
    assert phase.item() == skill.ALIGN

    # Align at the grasp target and wait for the explicit dwell.
    pose.zero_()
    for _ in range(3):
        out, phase, _ = skill(action, pose, ee, fingers)
    assert phase.item() == skill.CLOSE
    assert out[0, -1].item() == torch.tensor(-0.15).item()

    visited = {phase.item()}
    positive_lift = False
    for _ in range(20):
        out, phase, _ = skill(action, pose, ee, fingers)
        visited.add(phase.item())
        positive_lift |= out[0, 2].item() > 0.0
        # Idealized plant response for the simulator-free state-machine test.
        ee[:, 2] += out[:, 2] * 0.02
    assert skill.HOLD in visited
    assert skill.LIFT in visited
    assert skill.DONE in visited
    assert positive_lift


def test_skill_never_uses_finger_opening_as_success():
    skill = StableSphereGraspSkill()
    action = torch.zeros(1, 7)
    ee = torch.zeros(1, 3)
    far = torch.tensor([[0.2, 0.0, -0.08, 0.0, 0.0, 0.0]])
    # A physically plausible blocked aperture cannot advance a geometrically
    # unaligned skill.  Success remains an external physical lift criterion.
    for _ in range(20):
        _, phase, _ = skill(action, far, ee, torch.full((1, 2), 0.02))
    assert phase.item() == skill.HOVER


def test_align_uses_coarse_motion_only_until_near_contact():
    skill = StableSphereGraspSkill(
        SphereGraspSkillConfig(
            alignment_steps=1,
            align_fine_distance=0.02,
            max_align_position_step=0.02,
            max_align_fine_position_step=0.008,
            max_position_step=0.004,
        ),
        StableTargetConfig(stable_steps=1),
    )
    action = torch.zeros(1, 7)
    fingers = torch.full((1, 2), 0.04)
    ee = torch.tensor([[0.4, 0.3, 0.13]])

    hover_pose = torch.tensor([[0.0, 0.0, -0.08, 0.0, 0.0, 0.0]])
    skill(action, hover_pose, ee, fingers)
    _, phase, _ = skill(action, hover_pose, ee, fingers)
    assert phase.item() == skill.ALIGN

    far_pose = torch.tensor([[0.0, 0.0, -0.06, 0.0, 0.0, 0.0]])
    far_action, _, _ = skill(action, far_pose, ee, fingers)
    assert abs(far_action[0, 2].item()) > 0.9

    near_pose = torch.tensor([[0.0, 0.0, -0.01, 0.0, 0.0, 0.0]])
    near_action, _, _ = skill(action, near_pose, ee, fingers)
    assert abs(near_action[0, 2].item()) <= 0.4 + 1.0e-6


def test_fixed_intercept_waits_for_sphere_without_velocity_input():
    skill = StableSphereGraspSkill(
        SphereGraspSkillConfig(
            intercept_x=0.45,
            intercept_lead_distance=0.05,
            alignment_steps=1,
        ),
        StableTargetConfig(stable_steps=1),
    )
    action = torch.zeros(1, 7)
    fingers = torch.full((1, 2), 0.04)
    ee = torch.tensor([[0.45, 0.30, 0.112]])

    # The hand is at the fixed hover line, while the sphere is still upstream.
    upstream = torch.tensor([[0.35 - 0.45, 0.0, -0.08, 0.0, 0.0, 0.0]])
    for _ in range(4):
        out, phase, _ = skill(action, upstream, ee, fingers)
    assert phase.item() == skill.HOVER
    assert abs(out[0, 0].item()) < 1.0e-6

    # Current position crossing the lead line is sufficient; no speed is read.
    arriving = torch.tensor([[0.405 - 0.45, 0.0, -0.08, 0.0, 0.0, 0.0]])
    for _ in range(4):
        _, phase, _ = skill(action, arriving, ee, fingers)
    assert phase.item() == skill.ALIGN


def test_progressive_intercept_starts_descent_before_align_and_keeps_gripper_open():
    skill = StableSphereGraspSkill(
        SphereGraspSkillConfig(
            intercept_x=0.45,
            intercept_lead_distance=0.16,
            intercept_align_lead_distance=0.04,
            pre_descent_height=0.02,
            alignment_steps=1,
        ),
        StableTargetConfig(stable_steps=1),
    )
    action = torch.zeros(1, 7)
    fingers = torch.full((1, 2), 0.04)
    ee = torch.tensor([[0.45, 0.30, 0.13]])
    # Halfway through the spatial pre-descent window: the old controller
    # would wait at dz=0, while the new controller commands downward motion.
    pose = torch.tensor([[0.35 - 0.45, 0.0, -0.08, 0.0, 0.0, 0.0]])
    for _ in range(3):
        out, phase, _ = skill(action, pose, ee, fingers)
    assert phase.item() == skill.HOVER
    assert out[0, 2].item() < 0.0
    assert out[0, -1].item() == 1.0


def test_progressive_intercept_meets_sphere_before_wrist_fully_settles():
    progressive = StableSphereGraspSkill(
        SphereGraspSkillConfig(
            intercept_x=0.45,
            intercept_lead_distance=0.16,
            intercept_align_lead_distance=0.04,
            pre_descent_height=0.02,
            alignment_steps=1,
            rotation_threshold=0.15,
        ),
        StableTargetConfig(stable_steps=1),
    )
    legacy = StableSphereGraspSkill(
        SphereGraspSkillConfig(
            intercept_x=0.45,
            intercept_lead_distance=0.04,
            alignment_steps=1,
            rotation_threshold=0.15,
        ),
        StableTargetConfig(stable_steps=1),
    )
    action = torch.zeros(1, 7)
    fingers = torch.full((1, 2), 0.04)
    ee = torch.tensor([[0.45, 0.30, 0.072]])
    # The sphere is at the final lead line and the hand is at the scheduled
    # 2 cm clearance, but the wrist still has 0.5 rad left to converge.
    pose = torch.tensor([[-0.04, 0.0, -0.02, 0.0, 0.0, 0.5]])

    _, progressive_phase, _ = progressive(action, pose, ee, fingers)
    _, legacy_phase, _ = legacy(action, pose, ee, fingers)

    assert progressive_phase.item() == progressive.ALIGN
    assert legacy_phase.item() == legacy.HOVER


def test_lift_geometry_servo_does_not_outrun_observed_grasp_target():
    skill = StableSphereGraspSkill(
        SphereGraspSkillConfig(
            alignment_steps=1,
            close_steps=1,
            hold_steps=1,
            lift_steps=10,
            lift_timeout_steps=20,
            lift_height=0.05,
            lift_target_slack=0.005,
        ),
        StableTargetConfig(stable_steps=1),
    )
    action = torch.zeros(1, 7)
    fingers = torch.full((1, 2), 0.04)
    ee = torch.tensor([[0.4, 0.3, 0.13]])
    hover_pose = torch.tensor([[0.0, 0.0, -0.08, 0.0, 0.0, 0.0]])
    skill(action, hover_pose, ee, fingers)
    grasp_pose = torch.zeros(1, 6)
    # Advance through ALIGN, CLOSE, and HOLD into LIFT.
    for _ in range(5):
        out, phase, _ = skill(action, grasp_pose, ee, fingers)
    assert phase.item() == skill.LIFT

    # The observed grasp target has not risen, while TCP is already 8 mm
    # above it.  The servo must stop/undo vertical lead rather than blindly
    # continuing the open-loop lift trajectory.
    ee[:, 2] += 0.008
    lagging_pose = torch.tensor([[0.0, 0.0, -0.008, 0.0, 0.0, 0.0]])
    out, phase, _ = skill(action, lagging_pose, ee, fingers)
    assert phase.item() == skill.LIFT
    assert out[0, 2].item() <= 0.0


def test_lift_follower_activates_only_after_belt_detach_height():
    cfg = SphereGraspSkillConfig(
        lift_steps=10,
        lift_timeout_steps=20,
        lift_height=0.05,
        lift_target_slack=0.002,
        lift_detach_height=0.012,
    )
    skill = StableSphereGraspSkill(
        cfg,
        StableTargetConfig(
            translation_alpha=1.0,
            max_translation_jump=1.0,
            stable_steps=1,
        ),
    )
    action = torch.zeros(1, 7)
    fingers = torch.full((1, 2), 0.04)
    skill._ensure(action)
    skill.phase.fill_(skill.LIFT)
    skill.phase_step.fill_(5)
    skill.lift_anchor_z.fill_(0.13)

    # Before the TCP clears 12 mm, the nominal peel trajectory remains in
    # charge even though the observed sphere target is lagging.
    ee = torch.tensor([[0.4, 0.3, 0.13]])
    lagging = torch.tensor([[0.0, 0.0, -0.008, 0.0, 0.0, 0.0]])
    out, _, _ = skill(action, lagging, ee, fingers)
    assert out[0, 2].item() > 0.0

    # Once detached, the same lag activates the follower ceiling and stops
    # further upward lead.
    skill.target.reset()
    ee[:, 2] = 0.142
    out, _, _ = skill(action, lagging, ee, fingers)
    assert out[0, 2].item() <= 0.0


def test_lift_can_preserve_captured_relative_xy_geometry():
    cfg = SphereGraspSkillConfig(
        hold_steps=1,
        preserve_lift_relative_xy=True,
        lift_xy_gain=1.0,
        max_lift_xy_step=0.004,
    )
    skill = StableSphereGraspSkill(
        cfg,
        StableTargetConfig(
            translation_alpha=1.0,
            max_translation_jump=1.0,
            stable_steps=1,
        ),
    )
    action = torch.zeros(1, 7)
    fingers = torch.full((1, 2), 0.04)
    ee = torch.tensor([[0.4, 0.3, 0.052]])
    held_pose = torch.tensor([[0.006, -0.002, 0.0, 0.0, 0.0, 0.0]])
    skill._ensure(action)
    skill.phase.fill_(skill.HOLD)
    skill.phase_step.fill_(cfg.hold_steps - 1)

    skill(action, held_pose, ee, fingers)
    assert torch.allclose(skill.lift_relative_xy, held_pose[:, :2])

    # The unchanged captured offset is the desired grasp geometry, so the
    # lift controller must command no lateral motion.
    out, phase, _ = skill(action, held_pose, ee, fingers)
    assert phase.item() == skill.LIFT
    assert torch.allclose(out[0, :2], torch.zeros(2), atol=1e-7)


def test_post_close_target_displacement_is_fed_forward_without_speed_input():
    cfg = SphereGraspSkillConfig(
        hold_steps=1,
        preserve_lift_relative_xy=True,
        lift_xy_gain=1.0,
        max_lift_xy_step=0.004,
        target_xy_feedforward_gain=1.0,
        max_target_xy_feedforward_step=0.002,
    )
    skill = StableSphereGraspSkill(
        cfg,
        StableTargetConfig(
            translation_alpha=1.0,
            max_translation_jump=1.0,
            stable_steps=1,
        ),
    )
    action = torch.zeros(1, 7)
    fingers = torch.full((1, 2), 0.04)
    ee = torch.tensor([[0.4, 0.3, 0.052]])
    held_pose = torch.tensor([[0.006, 0.0, 0.0, 0.0, 0.0, 0.0]])
    skill(action, held_pose, ee, fingers)
    skill.phase.fill_(skill.LIFT)
    skill.lift_anchor_z.fill_(ee[0, 2])
    skill.lift_relative_xy.copy_(held_pose[:, :2])

    # The grasp target moves 1 mm in X while its relative geometry remains
    # exactly at the captured offset.  A position-only servo would output
    # zero; the geometry-difference feed-forward must command that 1 mm.
    moved_pose = held_pose.clone()
    moved_ee = ee.clone()
    moved_ee[:, 0] += 0.001
    out, phase, _ = skill(action, moved_pose, moved_ee, fingers)
    assert phase.item() == skill.LIFT
    assert out[0, 0].item() == pytest.approx(0.001 / cfg.action_scale, abs=1.0e-6)


def test_latched_lift_feedforward_ignores_lateral_transient_and_decays():
    cfg = SphereGraspSkillConfig(
        lift_steps=20,
        lift_timeout_steps=30,
        preserve_lift_relative_xy=True,
        lift_xy_gain=1.0,
        max_lift_xy_step=0.004,
        target_xy_feedforward_gain=1.0,
        max_target_xy_feedforward_step=0.004,
        latch_lift_xy_feedforward=True,
        lift_feedforward_x_only=True,
        lift_feedforward_decay_steps=4,
    )
    skill = StableSphereGraspSkill(
        cfg,
        StableTargetConfig(
            translation_alpha=1.0,
            max_translation_jump=1.0,
            stable_steps=1,
        ),
    )
    action = torch.zeros(1, 7)
    fingers = torch.full((1, 2), 0.04)
    ee = torch.tensor([[0.4, 0.3, 0.052]])
    pose = torch.zeros(1, 6)
    skill(action, pose, ee, fingers)
    skill.phase.fill_(skill.HOLD)
    skill.phase_step.fill_(cfg.hold_steps - 1)
    # Emulate the observed conveyor displacement plus a lateral contact
    # transient captured on the HOLD -> LIFT transition.
    skill.target.position = ee + torch.tensor([[0.001, -0.002, 0.0]])
    skill(action, pose, ee, fingers)
    assert skill.phase.item() == skill.LIFT
    assert abs(skill.lift_xy_feedforward[0, 0].item()) > 0.0
    assert skill.lift_xy_feedforward[0, 1].item() == 0.0

    skill.lift_relative_xy.zero_()
    skill.lift_xy_feedforward[0] = torch.tensor([0.004, 0.0])
    skill.phase_step.fill_(0)
    out, _, _ = skill(action, pose, ee, fingers)
    assert out[0, 0].item() == pytest.approx(0.004 / cfg.action_scale)
    skill.phase_step.fill_(4)
    out, _, _ = skill(action, pose, ee, fingers)
    assert out[0, 0].item() == pytest.approx(0.0, abs=1.0e-7)


def test_align_can_feed_forward_observed_target_displacement():
    cfg = SphereGraspSkillConfig(
        align_target_xy_feedforward=True,
        target_xy_feedforward_gain=1.0,
        max_target_xy_feedforward_step=0.008,
    )
    skill = StableSphereGraspSkill(
        cfg,
        StableTargetConfig(
            translation_alpha=1.0,
            rotation_alpha=1.0,
            max_translation_jump=1.0,
            max_rotation_jump=1.0,
            stable_steps=1,
        ),
    )
    action = torch.zeros(1, 7)
    fingers = torch.full((1, 2), 0.04)
    ee = torch.tensor([[0.4, 0.3, 0.052]])
    # Keep rotation deliberately outside the close gate so ALIGN remains
    # active while the target and TCP translate together.
    pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.2]])
    skill(action, pose, ee, fingers)
    skill.phase.fill_(skill.ALIGN)
    ee[:, 0] += 0.001
    out, phase, _ = skill(action, pose, ee, fingers)
    assert phase.item() == skill.ALIGN
    assert out[0, 0].item() == pytest.approx(0.001 / cfg.action_scale, abs=1.0e-6)


def test_close_gate_can_require_current_raw_alignment_as_well_as_filtered():
    cfg = SphereGraspSkillConfig(
        close_requires_raw_alignment=True,
        alignment_steps=1,
        grasp_translation_threshold=0.007,
    )
    skill = StableSphereGraspSkill(
        cfg,
        StableTargetConfig(
            translation_alpha=0.1,
            rotation_alpha=1.0,
            max_translation_jump=1.0,
            stable_translation=1.0,
            stable_steps=1,
        ),
    )
    action = torch.zeros(1, 7)
    fingers = torch.full((1, 2), 0.04)
    ee = torch.tensor([[0.4, 0.3, 0.052]])
    skill.target.update(torch.zeros(1, 6), ee)
    skill._ensure(action)
    skill.phase.fill_(skill.ALIGN)

    # The low-pass target is only 0.8 mm away, but the current target is 8 mm
    # away.  The old filtered-only gate would close here.
    raw_far = torch.tensor([[0.008, 0.0, 0.0, 0.0, 0.0, 0.0]])
    _, phase, _ = skill(action, raw_far, ee, fingers)
    assert phase.item() == skill.ALIGN

    raw_near = torch.tensor([[0.006, 0.0, 0.0, 0.0, 0.0, 0.0]])
    _, phase, _ = skill(action, raw_near, ee, fingers)
    assert phase.item() == skill.CLOSE


def test_close_following_can_use_a_larger_phase_specific_xy_limit():
    cfg = SphereGraspSkillConfig(
        target_xy_feedforward_gain=1.0,
        max_target_xy_feedforward_step=0.008,
        max_position_step=0.004,
        max_close_hold_xy_step=0.008,
    )
    skill = StableSphereGraspSkill(
        cfg,
        StableTargetConfig(
            translation_alpha=1.0,
            max_translation_jump=1.0,
            stable_steps=1,
        ),
    )
    action = torch.zeros(1, 7)
    fingers = torch.full((1, 2), 0.04)
    ee = torch.tensor([[0.4, 0.3, 0.052]])
    pose = torch.zeros(1, 6)
    skill(action, pose, ee, fingers)
    skill.phase.fill_(skill.CLOSE)

    # Target and TCP advance together by 5 mm, leaving zero position error.
    # The command therefore comes entirely from geometry-history feed-forward
    # and must not be clipped by the old generic 4 mm close limit.
    ee[:, 0] += 0.005
    out, phase, _ = skill(action, pose, ee, fingers)
    assert phase.item() == skill.CLOSE
    assert out[0, 0].item() == pytest.approx(0.005 / cfg.action_scale)


def test_target_discontinuity_is_not_fed_forward_post_close():
    cfg = SphereGraspSkillConfig(
        target_xy_feedforward_gain=1.0,
        max_target_xy_feedforward_step=0.002,
    )
    skill = StableSphereGraspSkill(
        cfg,
        StableTargetConfig(
            translation_alpha=1.0,
            max_translation_jump=0.01,
            stable_steps=1,
        ),
    )
    action = torch.zeros(1, 7)
    fingers = torch.full((1, 2), 0.04)
    ee = torch.tensor([[0.4, 0.3, 0.052]])
    pose = torch.zeros(1, 6)
    skill(action, pose, ee, fingers)
    skill.phase.fill_(skill.HOLD)

    jumped = pose.clone()
    jumped[:, 0] = 0.03
    out, _, _ = skill(action, jumped, ee, fingers)
    # The normal position servo remains active; only the 30 mm discontinuity
    # is excluded from the feed-forward branch.
    expected = min(cfg.position_gain * 0.03, cfg.max_position_step) / cfg.action_scale
    assert out[0, 0].item() == pytest.approx(expected)


def test_lift_geometry_is_relatched_once_after_initial_peel():
    cfg = SphereGraspSkillConfig(
        lift_geometry_relatch_height=0.012,
        preserve_lift_relative_xy=True,
    )
    skill = StableSphereGraspSkill(
        cfg,
        StableTargetConfig(
            translation_alpha=1.0,
            max_translation_jump=1.0,
            stable_steps=1,
        ),
    )
    action = torch.zeros(1, 7)
    fingers = torch.full((1, 2), 0.04)
    ee = torch.tensor([[0.4, 0.3, 0.052]])
    skill._ensure(action)
    skill.phase.fill_(skill.LIFT)
    skill.lift_anchor_z.fill_(0.052)
    skill.lift_relative_xy.copy_(torch.tensor([[0.006, 0.0]]))
    skill.lift_relative_z.fill_(0.0)

    loaded_pose = torch.tensor([[-0.002, 0.001, -0.006, 0.0, 0.0, 0.0]])
    ee[:, 2] = 0.064
    skill(action, loaded_pose, ee, fingers)
    assert skill.lift_geometry_relatched.item()
    assert torch.allclose(skill.lift_relative_xy, loaded_pose[:, :2])
    assert skill.lift_relative_z.item() == pytest.approx(-0.006)

    later_pose = torch.tensor([[-0.004, 0.002, -0.008, 0.0, 0.0, 0.0]])
    skill(action, later_pose, ee, fingers)
    assert torch.allclose(skill.lift_relative_xy, loaded_pose[:, :2])
    assert skill.lift_relative_z.item() == pytest.approx(-0.006)


def test_lift_can_latch_pregrasp_target_motion_instead_of_following_roll():
    cfg = SphereGraspSkillConfig(
        hold_steps=1,
        target_xy_feedforward_gain=1.0,
        max_target_xy_feedforward_step=0.002,
        latch_lift_xy_feedforward=True,
        preserve_lift_relative_xy=True,
        lift_xy_gain=1.0,
        max_lift_xy_step=0.004,
    )
    skill = StableSphereGraspSkill(
        cfg,
        StableTargetConfig(
            translation_alpha=1.0,
            max_translation_jump=1.0,
            stable_steps=1,
        ),
    )
    action = torch.zeros(1, 7)
    fingers = torch.full((1, 2), 0.04)
    ee = torch.tensor([[0.4, 0.3, 0.052]])
    pose = torch.tensor([[0.006, 0.0, 0.0, 0.0, 0.0, 0.0]])
    skill(action, pose, ee, fingers)
    skill.phase.fill_(skill.HOLD)
    skill.phase_step.fill_(cfg.hold_steps - 1)

    # During the final hold sample, both target and TCP advance by 1 mm.
    ee[:, 0] += 0.001
    _, phase, _ = skill(action, pose, ee, fingers)
    assert phase.item() == skill.LIFT
    assert skill.lift_xy_feedforward[0, 0].item() == pytest.approx(0.001, abs=1.0e-7)

    # The observed sphere then rolls 0.5 mm backwards relative to the hand.
    # The LIFT feed-forward must retain the pre-roll +1 mm direction.
    rolled_pose = pose.clone()
    rolled_pose[:, 0] -= 0.0005
    out, phase, _ = skill(action, rolled_pose, ee, fingers)
    assert phase.item() == skill.LIFT
    assert out[0, 0].item() > 0.0

def test_lift_relative_z_feedback_slows_but_does_not_stop_nominal_lift():
    def make_skill(gain):
        skill = StableSphereGraspSkill(
            SphereGraspSkillConfig(
                lift_steps=10,
                lift_timeout_steps=20,
                lift_height=0.05,
                lift_relative_z_gain=gain,
            ),
            StableTargetConfig(
                translation_alpha=1.0,
                max_translation_jump=1.0,
                stable_steps=1,
            ),
        )
        action = torch.zeros(1, 7)
        skill._ensure(action)
        skill.phase.fill_(skill.LIFT)
        skill.phase_step.fill_(5)
        skill.lift_anchor_z.fill_(0.13)
        skill.lift_relative_z.zero_()
        return skill, action

    baseline, action = make_skill(None)
    corrected, _ = make_skill(0.1)
    ee = torch.tensor([[0.4, 0.3, 0.13]])
    fingers = torch.full((1, 2), 0.04)
    lagging = torch.tensor([[0.0, 0.0, -0.008, 0.0, 0.0, 0.0]])

    baseline_out, _, _ = baseline(action, lagging, ee, fingers)
    corrected_out, _, _ = corrected(action, lagging, ee, fingers)
    assert 0.0 < corrected_out[0, 2].item() < baseline_out[0, 2].item()


def test_policy_option_preserves_arm_until_stable_trigger_then_owns_lift():
    option = StableSphereCloseHoldLiftOption(
        SphereCloseHoldLiftOptionConfig(
            alignment_steps=1,
            close_steps=2,
            hold_steps=2,
            lift_steps=4,
            lift_timeout_steps=12,
        ),
        StableTargetConfig(stable_steps=1),
    )
    policy = torch.tensor([[0.3, -0.2, 0.1, 0.0, 0.0, 0.0, -1.0]])
    ee = torch.tensor([[0.4, 0.3, 0.052]])
    far = torch.tensor([[[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    out, phase, active = option(policy, far, ee)
    assert phase.item() == option.IDLE and not active.item()
    assert torch.allclose(out[0, :6], policy[0, :6])
    assert out[0, -1].item() == 1.0

    aligned = torch.zeros(1, 1, 6)
    for _ in range(3):
        out, phase, active = option(policy, aligned, ee)
    assert active.item() and phase.item() >= option.CLOSE
    assert out[0, -1].item() == pytest.approx(-0.15)

    visited = {phase.item()}
    for _ in range(20):
        out, phase, _ = option(policy, aligned, ee)
        visited.add(phase.item())
        ee[:, 2] += out[:, 2] * 0.02
    assert option.HOLD in visited
    assert option.LIFT in visited
    assert option.DONE in visited


def test_pick_place_handoff_clears_vertically_then_moves_to_bowl_and_releases():
    from pickplace.grasp_skills import (
        SphereTransportPlaceConfig,
        StableSpherePickPlaceSkill,
        StableSphereResidualSkill,
    )

    grasp = StableSphereResidualSkill()
    skill = StableSpherePickPlaceSkill(
        grasp,
        SphereTransportPlaceConfig(
            clear_height=0.20,
            bowl_hover_height=0.20,
            release_height=0.11,
            stable_steps=1,
            release_steps=2,
        ),
    )
    action = torch.zeros(1, 7)
    candidates = torch.zeros(1, 1, 6)
    ee = torch.tensor([[0.45, 0.30, 0.10]])
    fingers = torch.full((1, 2), 0.02)
    bowl = torch.tensor([[0.45, -0.10, 0.0]])
    grasp.baseline._ensure(action)
    grasp.baseline.phase.fill_(grasp.baseline.DONE)

    out, _, phase = skill(action, candidates, ee, fingers, bowl)
    assert phase.item() == skill.CLEAR
    assert out[0, 2].item() > 0.0
    assert out[0, 1].item() == pytest.approx(0.0)
    assert out[0, -1].item() == pytest.approx(-0.15)

    ee[:] = torch.tensor([[0.45, 0.30, 0.20]])
    skill(action, candidates, ee, fingers, bowl)
    assert skill.phase.item() == skill.BOWL_HOVER
    out, _, _ = skill(action, candidates, ee, fingers, bowl)
    assert out[0, 1].item() < 0.0
    assert out[0, 2].item() == pytest.approx(0.0)

    ee[:] = torch.tensor([[0.45, -0.10, 0.20]])
    skill(action, candidates, ee, fingers, bowl)
    assert skill.phase.item() == skill.LOWER
    out, _, _ = skill(action, candidates, ee, fingers, bowl)
    assert out[0, 2].item() < 0.0
    assert out[0, -1].item() == pytest.approx(-0.15)

    ee[:] = torch.tensor([[0.45, -0.10, 0.11]])
    skill(action, candidates, ee, fingers, bowl)
    assert skill.phase.item() == skill.RELEASE
    out, _, _ = skill(action, candidates, ee, fingers, bowl)
    assert out[0, -1].item() == pytest.approx(1.0)
    out, _, phase = skill(action, candidates, ee, fingers, bowl)
    assert phase.item() == skill.DONE
    assert out[0, -1].item() == pytest.approx(1.0)


def test_pick_place_handoff_resets_only_selected_rows():
    from pickplace.grasp_skills import StableSpherePickPlaceSkill, StableSphereResidualSkill

    skill = StableSpherePickPlaceSkill(StableSphereResidualSkill())
    action = torch.zeros(2, 7)
    skill._ensure(action)
    skill.phase[:] = torch.tensor([skill.RELEASE, skill.BOWL_HOVER])
    skill.phase_step[:] = torch.tensor([9, 4])
    skill.translation_action[:] = 0.5
    skill.reset(torch.tensor([[True], [False]]))
    assert skill.phase.tolist() == [skill.WAIT_GRASP, skill.BOWL_HOVER]
    assert skill.phase_step.tolist() == [0, 4]
    assert torch.all(skill.translation_action[0] == 0.0)
    assert torch.all(skill.translation_action[1] == 0.5)


def test_pick_place_optional_tail_retreats_then_returns_home():
    from pickplace.grasp_skills import (
        SphereTransportPlaceConfig,
        StableSpherePickPlaceSkill,
        StableSphereResidualSkill,
    )

    skill = StableSpherePickPlaceSkill(
        StableSphereResidualSkill(),
        SphereTransportPlaceConfig(
            release_steps=1,
            home_position=(0.45, 0.30, 0.134),
            retreat_height=0.20,
            retreat_tolerance=0.015,
            home_tolerance=0.025,
        ),
    )
    action = torch.zeros(1, 7)
    candidates = torch.zeros(1, 1, 6)
    fingers = torch.zeros(1, 2)
    bowl = torch.tensor([[0.50, 0.30, 0.03]])
    ee = torch.tensor([[0.50, 0.30, 0.14]])
    skill._ensure(action)
    skill.phase.fill_(skill.RELEASE)

    out, _, phase = skill(action, candidates, ee, fingers, bowl)
    assert phase.item() == skill.RETREAT
    assert out[0, 2].item() > 0.0
    assert out[0, -1].item() == pytest.approx(1.0)

    ee[:] = torch.tensor([[0.50, 0.30, 0.20]])
    skill(action, candidates, ee, fingers, bowl)
    assert skill.phase.item() == skill.HOME
    out, _, _ = skill(action, candidates, ee, fingers, bowl)
    assert out[0, 0].item() < 0.0
    assert out[0, 2].item() < 0.0

    ee[:] = torch.tensor([[0.45, 0.30, 0.134]])
    _, _, phase = skill(action, candidates, ee, fingers, bowl)
    assert phase.item() == skill.DONE


def test_pick_place_transport_translation_command_is_slew_limited():
    from pickplace.grasp_skills import (
        SphereTransportPlaceConfig,
        StableSpherePickPlaceSkill,
        StableSphereResidualSkill,
    )

    skill = StableSpherePickPlaceSkill(
        StableSphereResidualSkill(),
        SphereTransportPlaceConfig(max_translation_action_delta=0.04),
    )
    action = torch.zeros(1, 7)
    candidates = torch.zeros(1, 1, 6)
    ee = torch.tensor([[0.45, 0.30, 0.20]])
    fingers = torch.full((1, 2), 0.02)
    bowl = torch.tensor([[0.45, -0.10, 0.0]])
    skill._ensure(action)
    skill.phase.fill_(skill.BOWL_HOVER)

    previous = torch.zeros(3)
    for _ in range(6):
        out, _, _ = skill(action, candidates, ee, fingers, bowl)
        assert torch.all(torch.abs(out[0, :3] - previous) <= 0.040001)
        previous = out[0, :3].clone()


def test_pick_place_waypoint_transition_stops_old_lateral_command():
    from pickplace.grasp_skills import StableSpherePickPlaceSkill, StableSphereResidualSkill

    skill = StableSpherePickPlaceSkill(StableSphereResidualSkill())
    action = torch.zeros(1, 7)
    candidates = torch.zeros(1, 1, 6)
    fingers = torch.full((1, 2), 0.02)
    bowl = torch.tensor([[0.45, 0.0, 0.0]])
    ee = torch.tensor([[0.45, 0.0, 0.20]])
    skill._ensure(action)
    skill.phase.fill_(skill.BOWL_HOVER)
    skill.stable_count.fill_(skill.config.stable_steps - 1)
    skill.translation_action[:] = torch.tensor([[0.0, -0.8, 0.0]])

    skill(action, candidates, ee, fingers, bowl)
    assert skill.phase.item() == skill.LOWER
    assert torch.all(skill.translation_action == 0.0)

    out, _, _ = skill(action, candidates, ee, fingers, bowl)
    assert out[0, 1].item() == pytest.approx(0.0)
    assert out[0, 2].item() < 0.0
