"""TorchRL entry point: the official IsaacLabWrapper plus generic bookkeeping transforms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pickplace.metrics import outcome_terms


def _unwrapped(env):
    base = env.base_env if hasattr(env, "base_env") else env
    return base._env.unwrapped


def _make_episode_outcome(terms: Sequence[str]):
    import torch
    from torchrl.data import Binary, Composite
    from torchrl.envs import Transform

    class EpisodeOutcome(Transform):
        """Writes ``("next", "outcome", <term>)`` bools: True on a done row whose episode ended by <term>.

        Reads Isaac Lab's termination manager right after the step. Under native auto-reset the manager's
        per-term flags still describe the step that just ended (the reset does not clear them), whereas the
        next observations are already NaN / reset. Reset tensordicts carry all-False flags.
        """

        def __init__(self):
            super().__init__(in_keys=[], out_keys=[("outcome", t) for t in terms])
            self.terms = tuple(terms)

        def _step(self, tensordict, next_tensordict):
            manager = _unwrapped(self.parent).termination_manager
            done = next_tensordict.get("done")
            for term in self.terms:
                flag = manager.get_term(term).reshape(done.shape).to(done.device)
                next_tensordict.set(("outcome", term), flag & done)
            return next_tensordict

        def _reset(self, tensordict, tensordict_reset):
            shape = (*tensordict_reset.batch_size, 1)
            for term in self.terms:
                tensordict_reset.set(
                    ("outcome", term), torch.zeros(shape, dtype=torch.bool, device=tensordict_reset.device)
                )
            return tensordict_reset

        def _reset_on_native_autoreset(self, tensordict, tensordict_reset):
            # Keep the flags of the episode that just ended on the done row (the default; stated explicitly
            # because RewardSum/StepCounter reset here instead).
            return tensordict_reset

        def transform_observation_spec(self, observation_spec):
            batch = observation_spec.shape
            # step_mdp_static=True: the generic native-autoreset "invalidate next observation on done
            # rows" pass (torchrl.envs.common.EnvBase._native_autoreset_set_invalid_next_observation)
            # sweeps every plain-Tensor leaf under observation_spec and zeroes/NaNs it on done rows,
            # which would erase the exact flags this transform means to keep. Marking the whole
            # "outcome" sub-composite step_mdp_static makes step_mdp copy it atomically instead of
            # descending into its leaves, so that pass skips it (it only touches `torch.Tensor` leaves).
            observation_spec["outcome"] = Composite(
                {
                    term: Binary(n=1, shape=(*batch, 1), dtype=torch.bool, device=observation_spec.device)
                    for term in self.terms
                },
                shape=batch,
                device=observation_spec.device,
                step_mdp_static=True,
            )
            return observation_spec

    return EpisodeOutcome()


def _make_reward_terms_vector(event_sources):
    import torch
    from torchrl.data import Unbounded
    from torchrl.envs import Transform

    from pickplace.rewards import DENSE_TERMS, EVENT_TERMS, REWARD_TERMS

    class RewardTermsVector(Transform):
        """Writes ``("next", "reward_terms")``: every reward term, unweighted, in ``REWARD_TERMS`` order.

        Dense components are ``value x dt`` recovered from Isaac Lab's reward manager (which stores
        ``value x weight`` per term; weights are kept non-zero by ``build_cell_env_cfg``). Event components are
        0/1 from the ``("next", "outcome", ...)`` flags written by ``EpisodeOutcome`` earlier in the Compose.
        """

        def __init__(self):
            super().__init__(in_keys=[], out_keys=["reward_terms"])

        def _step(self, tensordict, next_tensordict):
            u = _unwrapped(self.parent)
            rm = u.reward_manager
            names = list(rm.active_terms)
            step_reward = rm._step_reward
            n = step_reward.shape[0]
            # float32 by construction, matching transform_reward_spec below.
            out = torch.zeros(n, len(REWARD_TERMS), device=step_reward.device, dtype=torch.float32)
            for k, term in enumerate(DENSE_TERMS):
                i = names.index(term)
                out[:, k] = step_reward[:, i] / rm.get_term_cfg(term).weight * u.step_dt
            offset = len(DENSE_TERMS)
            for j, event in enumerate(EVENT_TERMS):
                fired = torch.zeros(n, dtype=torch.bool, device=out.device)
                for source in event_sources[event]:
                    fired |= next_tensordict.get(("outcome", source)).reshape(n).to(out.device)
                out[:, offset + j] = fired.to(out.dtype)
            next_tensordict.set("reward_terms", out.to(next_tensordict.device))
            return next_tensordict

        def transform_reward_spec(self, reward_spec):
            batch = reward_spec.shape
            reward_spec["reward_terms"] = Unbounded(
                shape=(*batch, len(REWARD_TERMS)), device=reward_spec.device, dtype=torch.float32
            )
            return reward_spec

    return RewardTermsVector()


def reward_weights(env_cfg: Mapping) -> dict[str, float]:
    from pickplace.config import DEFAULT_ENV
    from pickplace.rewards import resolve_reward_weights

    return resolve_reward_weights({**DEFAULT_ENV, **env_cfg})


def make_env(env_cfg: Mapping):
    """Create the food-cell env as a TorchRL ``TransformedEnv``. Launch the Isaac app before calling.

    Observations stay raw (uint8-valued float images, unnormalized states); preprocessing belongs to
    each algorithm. No running-statistics transforms here: terminal next-observations are NaN
    under native auto-reset. ``("next", "outcome", <term>)`` marks how each finished episode ended.

    ``("next", "reward_terms")`` holds every reward term unweighted (``pickplace.rewards.REWARD_TERMS``
    order); the training reward ``("next", "reward")`` is ``LineariseRewards`` over it with the weights of
    ``env_cfg["reward_set"]`` plus overrides (see ``reward_weights``). ``episode_reward`` is the running sum
    of ``reward`` and ``episode_reward_terms`` the per-term running sum.
    """
    import gymnasium as gym
    import torch
    from torchrl.envs import Compose, LineariseRewards, RewardSum, StepCounter, TransformedEnv
    from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

    import pickplace.envs  # noqa: F401  (gym registration)
    from pickplace.action_adapters import make_joint_action_adapter
    from pickplace.action_gates import (
        make_close_retract_option,
        make_grasp_pose_close_retract_option,
        make_stable_sphere_grasp_option,
        make_stable_sphere_residual_skill,
        make_stable_sphere_pick_place_skill,
        make_geometric_micro_lift_option,
        make_grasp_hold_action_gate,
    )
    from pickplace.config import DEFAULT_ENV, build_cell_env_cfg
    from pickplace.rewards import event_sources, weight_vector

    cfg = build_cell_env_cfg(env_cfg)
    adapter = env_cfg.get("action_adapter", DEFAULT_ENV["action_adapter"])
    gate = {**DEFAULT_ENV["grasp_hold_gate"], **(env_cfg.get("grasp_hold_gate") or {})}
    option = {**DEFAULT_ENV["close_retract_option"], **(env_cfg.get("close_retract_option") or {})}
    micro_lift = {
        **DEFAULT_ENV["geometric_micro_lift_option"],
        **(env_cfg.get("geometric_micro_lift_option") or {}),
    }
    pose_option = {
        **DEFAULT_ENV["grasp_pose_close_retract_option"],
        **(env_cfg.get("grasp_pose_close_retract_option") or {}),
    }
    sphere_option = {
        **DEFAULT_ENV["stable_sphere_grasp_option"],
        **(env_cfg.get("stable_sphere_grasp_option") or {}),
    }
    sphere_residual = {
        **DEFAULT_ENV["stable_sphere_residual_skill"],
        **(env_cfg.get("stable_sphere_residual_skill") or {}),
    }
    sphere_pick_place = {
        **DEFAULT_ENV["stable_sphere_pick_place_skill"],
        **(env_cfg.get("stable_sphere_pick_place_skill") or {}),
    }
    if bool(sphere_residual["enabled"]) and hasattr(cfg.terminations, "tcp_escape"):
        # Keep shaping and retreat detection on the exact same geometric
        # interception schedule as the controller for lift curricula.  The
        # full G1 placement task has generic reach/termination terms with a
        # different call signature, so it must not receive lift-only params.
        intercept_schedule = {
            "intercept_lead_distance": float(sphere_residual["intercept_lead_distance"]),
            "intercept_align_lead_distance": float(
                sphere_residual["intercept_align_lead_distance"]
            ),
            "pre_descent_height": (
                None
                if sphere_residual["pre_descent_height"] is None
                else float(sphere_residual["pre_descent_height"])
            ),
        }
        cfg.rewards.reach_food.params.update(intercept_schedule)
        cfg.terminations.tcp_escape.params.update(intercept_schedule)
        if bool(sphere_residual["settled_coupling_reference"]):
            cfg.rewards.grasp_lift.params["settled_reference"] = True
            cfg.terminations.success.params["settled_reference"] = True
    task = env_cfg.get("task", DEFAULT_ENV["task"])
    outcomes = outcome_terms(task)
    device = torch.device(env_cfg.get("device", DEFAULT_ENV["device"]))
    weights = torch.tensor(weight_vector(reward_weights(env_cfg)), device=device)
    base = IsaacLabWrapper(gym.make(task, cfg=cfg), native_autoreset=True, device=device)
    transforms = [
        # StepCounter first: it is the transform that can *add* a done (max_steps truncation), and the
        # two transforms below read `done` to decide which rows ended an episode.
        StepCounter(),
        _make_episode_outcome(outcomes),
        _make_reward_terms_vector(event_sources(task)),
        LineariseRewards(in_keys=["reward_terms"], out_keys=["reward"], weights=weights),
        RewardSum(in_keys=["reward"], out_keys=["episode_reward"]),
        RewardSum(in_keys=["reward_terms"], out_keys=["episode_reward_terms"]),
    ]
    if adapter != "native":
        # Last: Compose runs _inv_call in reverse, so the adapter is the first to see the action a policy
        # sampled in the adapted space and the last thing before the env gets its native one. Everything
        # above reads observations and rewards, which the adapter never touches.
        transforms.append(make_joint_action_adapter(adapter))
    if int(gate["hold_steps"]) > 0:
        # Last inverse transform = first controller-boundary operation. The
        # collector therefore keeps the policy action while Isaac receives the
        # gated command.
        transforms.append(
            make_grasp_hold_action_gate(
                hold_steps=int(gate["hold_steps"]),
                max_upward_dz=float(gate["max_upward_dz"]),
                force_close=bool(gate["force_close"]),
                follow_object_xy=bool(gate["follow_object_xy"]),
                control_dt=float(cfg.sim.dt * cfg.decimation),
                action_scale=float(cfg.arm.ik_action_scale),
                follow_gain=float(gate["follow_gain"]),
                max_follow_speed=float(gate["max_follow_speed"]),
                freeze_rotation=bool(gate["freeze_rotation"]),
            )
        )
    if int(option["steps"]) > 0:
        transforms.append(
            make_close_retract_option(
                steps=int(option["steps"]),
                retract_height=float(option["retract_height"]),
                trigger_distance=float(option["trigger_distance"]),
                trigger_gripper=float(option["trigger_gripper"]),
                control_dt=float(cfg.sim.dt * cfg.decimation),
                action_scale=float(cfg.arm.ik_action_scale),
                follow_gain=float(option["follow_gain"]),
                max_follow_speed=float(option["max_follow_speed"]),
                xy_blend=float(option["xy_blend"]),
            )
        )
    if bool(micro_lift["enabled"]):
        transforms.append(
            make_geometric_micro_lift_option(
                close_steps=int(micro_lift["close_steps"]),
                lift_steps=int(micro_lift["lift_steps"]),
                hold_steps=int(micro_lift["hold_steps"]),
                lift_height=float(micro_lift["lift_height"]),
                distance_threshold=float(micro_lift["distance_threshold"]),
                min_close_fraction=float(micro_lift["min_close_fraction"]),
                max_close_fraction=float(micro_lift["max_close_fraction"]),
                open_pos=float(cfg.arm.gripper_open),
                closed_pos=float(cfg.arm.gripper_closed),
                action_scale=float(cfg.arm.ik_action_scale),
                xy_gain=float(micro_lift["xy_gain"]),
                z_gain=float(micro_lift["z_gain"]),
                max_xy_step=float(micro_lift["max_xy_step"]),
                max_z_step=float(micro_lift["max_z_step"]),
                max_finger_delta=float(micro_lift["max_finger_delta"]),
                force_close=bool(micro_lift["force_close"]),
                freeze_rotation=bool(micro_lift["freeze_rotation"]),
            )
        )
    if bool(pose_option["enabled"]):
        transforms.append(
            make_grasp_pose_close_retract_option(
                translation_threshold=float(pose_option["translation_threshold"]),
                rotation_threshold=float(pose_option["rotation_threshold"]),
                approach_steps=int(pose_option["approach_steps"]),
                close_steps=int(pose_option["close_steps"]),
                retract_steps=int(pose_option["retract_steps"]),
                retract_height=float(pose_option["retract_height"]),
                action_scale=float(cfg.arm.ik_action_scale),
                close_action=float(pose_option["close_action"]),
                position_gain=float(pose_option["position_gain"]),
                rotation_gain=float(pose_option["rotation_gain"]),
                max_position_step=float(pose_option["max_position_step"]),
                max_rotation_step=float(pose_option["max_rotation_step"]),
            )
        )
    if bool(sphere_option["enabled"]):
        transforms.append(
            make_stable_sphere_grasp_option(
                **{
                    key: value
                    for key, value in sphere_option.items()
                    if key != "enabled"
                },
                action_scale=float(cfg.arm.ik_action_scale),
            )
        )
    if bool(sphere_pick_place["enabled"]):
        transforms.append(
            make_stable_sphere_pick_place_skill(
                transport_config={
                    key: value
                    for key, value in sphere_pick_place.items()
                    if key != "enabled"
                },
                **{
                    key: value
                    for key, value in sphere_residual.items()
                    if key != "enabled"
                },
                action_scale=float(cfg.arm.ik_action_scale),
            )
        )
    elif bool(sphere_residual["enabled"]):
        transforms.append(
            make_stable_sphere_residual_skill(
                **{
                    key: value
                    for key, value in sphere_residual.items()
                    if key != "enabled"
                },
                action_scale=float(cfg.arm.ik_action_scale),
            )
        )
    return TransformedEnv(base, Compose(*transforms))


def _log_stats(env, prefix: str) -> dict[str, float]:
    log = _unwrapped(env).extras.get("log", {})
    return {k[len(prefix):]: float(v) for k, v in log.items() if k.startswith(prefix)}


def termination_stats(env) -> dict[str, float]:
    """Fraction of sub-envs whose most recent finished episode ended by each termination term."""
    return _log_stats(env, "Episode_Termination/")


def reward_term_stats(env) -> dict[str, float]:
    """Per reward term: mean episodic sum per second over the envs reset in the latest step (Isaac Lab log)."""
    return _log_stats(env, "Episode_Reward/")
