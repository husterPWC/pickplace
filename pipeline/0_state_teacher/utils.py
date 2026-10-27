"""State-teacher models and checkpoints. Loaded by file path from this folder's scripts (see train.py)."""

from __future__ import annotations

import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torch import nn
from torchrl.envs import ExplorationType
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator

from pickplace.artifacts import git_commit, read_json, sha256_file, write_json
from pickplace.keys import expand_in_keys


class StateNet(nn.Module):
    """Flattens and concatenates vector observations -> MLP (ELU, orthogonal init)."""

    def __init__(self, shapes, hidden, out_dim: int, out_gain: float):
        super().__init__()
        self.shapes = shapes
        self.mlp = MLP(
            in_features=sum(math.prod(s) for s in shapes),
            out_features=out_dim,
            num_cells=list(hidden),
            activation_class=nn.ELU,
        )
        linears = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)]
        for layer in linears:
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(linears[-1].weight, out_gain)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        flat = [x.float().reshape(*x.shape[: x.dim() - len(s)], -1) for x, s in zip(xs, self.shapes)]
        return self.mlp(torch.cat(flat, dim=-1))


class ActorNet(nn.Module):
    def __init__(self, body: nn.Module, action_dim: int, log_std_init: float = 0.0):
        super().__init__()
        self.body = body
        self.scale = AddStateIndependentNormalScale(
            action_dim, scale_lb=1e-4, init_value=float(log_std_init)
        )

    def forward(self, *xs: torch.Tensor):
        return self.scale(self.body(*xs))


def action_l1_penalty(action: torch.Tensor) -> torch.Tensor:
    """Mean absolute policy command, used to regularize residual actions."""
    if action.numel() == 0:
        raise ValueError("action tensor must be non-empty")
    return action.abs().mean()


class TemporalResidualStateNet(nn.Module):
    """Warm-startable MLP plus a zero-initialized GRU residual.

    The base branch has exactly the original teacher architecture.  The GRU
    sees the oldest-to-newest motion history and initially contributes zero,
    so importing C0a starts from its sensor-compatible behaviour instead of
    relearning interception from scratch.
    """

    def __init__(
        self, base_shapes, history_shape, hidden, temporal_hidden: int,
        out_dim: int, out_gain: float, residual_limit: float | None = None,
    ):
        super().__init__()
        if len(history_shape) != 2:
            raise ValueError(f"temporal history must have shape (steps, features), got {history_shape}")
        self.base = StateNet(base_shapes, hidden, out_dim, out_gain)
        self.gru = nn.GRU(input_size=history_shape[-1], hidden_size=temporal_hidden, batch_first=True)
        self.temporal_head = nn.Linear(temporal_hidden, out_dim)
        self.residual_limit = residual_limit
        nn.init.zeros_(self.temporal_head.weight)
        nn.init.zeros_(self.temporal_head.bias)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        *base_xs, history = xs
        base = self.base(*base_xs)
        leading = history.shape[:-2]
        sequence = history.float().reshape(-1, *history.shape[-2:])
        encoded, _ = self.gru(sequence)
        residual = self.temporal_head(encoded[:, -1])
        if self.residual_limit is not None:
            residual = self.residual_limit * torch.tanh(residual)
        residual = residual.reshape(*leading, -1)
        return base + residual


class GraspPointGRUStateNet(nn.Module):
    """Encode grasp candidates and position history without explicit speed.

    Candidate points are encoded point-wise and pooled first within each grasp
    and then across grasps.  The recurrent branch sees only position, finger,
    and previous-action history.  Remaining vector observations form the robot
    state branch.
    """

    def __init__(
        self,
        shapes,
        grasp_index: int,
        history_index: int,
        hidden,
        point_hidden: int,
        temporal_hidden: int,
        out_dim: int,
        out_gain: float,
    ):
        super().__init__()
        grasp_shape, history_shape = shapes[grasp_index], shapes[history_index]
        if len(grasp_shape) != 3 or grasp_shape[-1] != 3:
            raise ValueError(f"grasp points must have shape (grasps, points, 3), got {grasp_shape}")
        if len(history_shape) != 2:
            raise ValueError(f"position history must have shape (steps, features), got {history_shape}")
        self.shapes = shapes
        self.grasp_index = grasp_index
        self.history_index = history_index
        self.point_encoder = nn.Sequential(
            nn.Linear(3, point_hidden), nn.ELU(), nn.Linear(point_hidden, point_hidden), nn.ELU()
        )
        self.grasp_encoder = nn.Sequential(nn.Linear(point_hidden, point_hidden), nn.ELU())
        self.gru = nn.GRU(history_shape[-1], temporal_hidden, batch_first=True)
        vector_dim = sum(math.prod(shape) for i, shape in enumerate(shapes) if i not in (grasp_index, history_index))
        self.fusion = MLP(
            in_features=vector_dim + point_hidden + temporal_hidden,
            out_features=out_dim,
            num_cells=list(hidden),
            activation_class=nn.ELU,
        )
        linears = [module for module in self.modules() if isinstance(module, nn.Linear)]
        for layer in linears:
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
        fusion_linears = [module for module in self.fusion.modules() if isinstance(module, nn.Linear)]
        nn.init.orthogonal_(fusion_linears[-1].weight, out_gain)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        points, history = xs[self.grasp_index], xs[self.history_index]
        leading = points.shape[:-3]
        point_features = self.point_encoder(points.float())
        grasp_features = point_features.max(dim=-2).values
        grasp_features = self.grasp_encoder(grasp_features).max(dim=-2).values
        sequence = history.float().reshape(-1, *history.shape[-2:])
        temporal_features, _ = self.gru(sequence)
        temporal_features = temporal_features[:, -1].reshape(*leading, -1)
        vectors = [
            value.float().reshape(*leading, -1)
            for index, value in enumerate(xs)
            if index not in (self.grasp_index, self.history_index)
        ]
        return self.fusion(torch.cat((*vectors, grasp_features, temporal_features), dim=-1))


def make_teacher_models(env, network_cfg, device: torch.device):
    """Separate MLP actor (TanhNormal on [-1, 1], state-independent scale) and MLP critic."""
    spec = env.observation_spec
    ndim = len(env.batch_size)
    action_dim = env.action_spec.shape[-1]

    def shapes(keys):
        return [tuple(spec[k].shape[ndim:]) for k in keys]

    architecture = str(getattr(network_cfg, "architecture", "mlp"))
    actor_keys = expand_in_keys(spec, list(network_cfg.actor_in_keys))
    critic_keys = expand_in_keys(spec, list(network_cfg.critic_in_keys))
    temporal_key = None
    grasp_key = None
    if architecture == "temporal_residual_gru":
        temporal_key = tuple(getattr(network_cfg, "temporal_in_key", ["temporal", "motion"]))
        if temporal_key not in actor_keys or temporal_key not in critic_keys:
            raise ValueError("temporal_residual_gru requires temporal_in_key in both actor_in_keys and critic_in_keys")
    elif architecture == "grasp_point_gru":
        temporal_key = tuple(getattr(network_cfg, "temporal_in_key", ["position_history", "position_action"]))
        grasp_key = tuple(getattr(network_cfg, "grasp_in_key", ["grasp_guidance", "points_ee"]))
        for name, keys in (("actor", actor_keys), ("critic", critic_keys)):
            if temporal_key not in keys or grasp_key not in keys:
                raise ValueError(
                    f"grasp_point_gru requires temporal_in_key and grasp_in_key in {name}_in_keys"
                )
    elif architecture != "mlp":
        raise ValueError(f"Unknown teacher network architecture {architecture!r}")
    for key in actor_keys + critic_keys:
        rank = len(spec[key].shape[ndim:])
        if rank > 1 and key not in (temporal_key, grasp_key):
            raise ValueError(f"The state teacher takes vector observations only; {key} has shape {spec[key].shape}.")

    def make_body(keys, out_dim, out_gain, residual_limit=None):
        if architecture == "mlp":
            return StateNet(shapes(keys), network_cfg.hidden, out_dim, out_gain)
        if architecture == "grasp_point_gru":
            return GraspPointGRUStateNet(
                shapes(keys),
                keys.index(grasp_key),
                keys.index(temporal_key),
                network_cfg.hidden,
                int(getattr(network_cfg, "point_hidden", 64)),
                int(getattr(network_cfg, "temporal_hidden", 64)),
                out_dim,
                out_gain,
            )
        base_keys = [key for key in keys if key != temporal_key]
        history_shape = tuple(spec[temporal_key].shape[ndim:])
        return TemporalResidualStateNet(
            shapes(base_keys), history_shape, network_cfg.hidden,
            int(getattr(network_cfg, "temporal_hidden", 64)), out_dim, out_gain, residual_limit,
        )

    actor = ProbabilisticActor(
        TensorDictModule(
            ActorNet(
                make_body(
                    actor_keys,
                    action_dim,
                    out_gain=float(getattr(network_cfg, "actor_out_gain", 0.01)),
                    residual_limit=getattr(network_cfg, "temporal_actor_residual_limit", None),
                ),
                action_dim,
                log_std_init=float(getattr(network_cfg, "log_std_init", 0.0)),
            ),
            in_keys=actor_keys,
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": -torch.ones(action_dim, device=device),
            "high": torch.ones(action_dim, device=device),
            "tanh_loc": False,
        },
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )
    critic = ValueOperator(make_body(critic_keys, 1, out_gain=1.0), in_keys=critic_keys)
    return actor.to(device), critic.to(device)


def _load_temporal_base(module, old_state: dict[str, torch.Tensor], kind: str) -> int:
    """Copy an old MLP into a temporal residual model, dropping trailing removed inputs."""
    new_state = module.state_dict()
    copied = 0
    for new_name, new_value in new_state.items():
        if kind == "actor":
            old_name = new_name.replace(".body.base.", ".body.")
        else:
            old_name = new_name.replace("module.base.", "module.")
        old_value = old_state.get(old_name)
        if old_value is None:
            continue
        if old_value.shape == new_value.shape:
            new_state[new_name] = old_value
            copied += 1
        elif old_value.ndim == 2 and old_value.shape[0] == new_value.shape[0] and old_value.shape[1] > new_value.shape[1]:
            # C0m removes the lexicographically-last privileged/is_grasped
            # scalar.  All preceding input columns retain their exact order.
            new_state[new_name] = old_value[:, : new_value.shape[1]]
            copied += 1
    module.load_state_dict(new_state)
    return copied


def load_teacher_weights(checkpoint_path, actor, critic, device: torch.device) -> None:
    """Warm-start freshly built actor/critic from an existing checkpoint (fine-tuning; fresh optimizer).

    Raises (via ``load_state_dict``'s strict mode) if the checkpoint's network shapes don't match this run's,
    naming the checkpoint so the mismatch is easy to place.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    try:
        actor.load_state_dict(checkpoint["actor"])
        critic.load_state_dict(checkpoint["critic"])
        return
    except RuntimeError as error:
        temporal = any(".body.base." in name for name in actor.state_dict())
        if not temporal:
            raise RuntimeError(
                f"init_checkpoint {checkpoint_path} does not match this run's network config: {error}"
            ) from error
    actor_count = _load_temporal_base(actor, checkpoint["actor"], "actor")
    critic_count = _load_temporal_base(critic, checkpoint["critic"], "critic")
    if actor_count < 2 or critic_count < 2:
        raise RuntimeError(
            f"init_checkpoint {checkpoint_path} did not contain a compatible MLP base "
            f"(copied actor={actor_count}, critic={critic_count} tensors)"
        )


def load_teacher_actor(checkpoint_path, env, device: torch.device):
    """Rebuild the actor from a teacher checkpoint (its own network config and weights)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor, _ = make_teacher_models(env, OmegaConf.create(checkpoint["config"]["network"]), device)
    actor.load_state_dict(checkpoint["actor"])
    return actor


def gaussian_reference_kl(
    loc: torch.Tensor,
    scale: torch.Tensor,
    reference_loc: torch.Tensor,
    reference_scale: torch.Tensor,
    action_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``KL(current || reference)`` and its mean-shift contribution.

    The teacher uses the same invertible tanh transform for both policies, so
    the KL of their pre-tanh diagonal Gaussians is also the KL of the action
    distributions.  The second result isolates the normalized mean drift; it
    is useful for explicitly keeping a fine-tuned deterministic policy close
    to the checkpoint that it started from.
    """
    reference_scale = reference_scale.clamp_min(torch.finfo(reference_scale.dtype).tiny)
    scale = scale.clamp_min(torch.finfo(scale.dtype).tiny)
    normalized_mean_sq = ((loc - reference_loc) / reference_scale).square()
    per_dim_full = (
        torch.log(reference_scale / scale)
        + 0.5 * ((scale / reference_scale).square() + normalized_mean_sq - 1.0)
    )
    if action_weights is not None:
        action_weights = torch.as_tensor(action_weights, dtype=loc.dtype, device=loc.device)
        if action_weights.ndim != 1 or action_weights.numel() != loc.shape[-1]:
            raise ValueError(
                f"reference action weights have shape {tuple(action_weights.shape)}; "
                f"expected ({loc.shape[-1]},)"
            )
        if bool((action_weights < 0).any()) or not bool((action_weights > 0).any()):
            raise ValueError("reference action weights must be non-negative with at least one positive value")
        normalized_mean_sq = normalized_mean_sq * action_weights
        per_dim_full = per_dim_full * action_weights
    mean_kl = 0.5 * normalized_mean_sq.sum(dim=-1).mean()
    full_kl = per_dim_full.sum(dim=-1).mean()
    return full_kl, mean_kl


def reference_policy_kl(actor, reference_actor, batch, action_weights=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the current/reference analytic KL on a PPO observation batch."""
    # Write distribution parameters to fresh TensorDicts: PPO batches contain
    # the rollout policy's ``loc``/``scale`` and those must remain unchanged.
    current = actor.get_dist(batch, tensordict_out=batch.empty())
    with torch.no_grad():
        reference = reference_actor.get_dist(batch, tensordict_out=batch.empty())
    return gaussian_reference_kl(
        current.loc,
        current.scale,
        reference.loc.detach(),
        reference.scale.detach(),
        action_weights=action_weights,
    )


def save_teacher_checkpoint(run_dir, name: str, actor, critic, optim, cfg, frames: int, iteration: int) -> Path:
    """``<run_dir>/checkpoints/<name>.pt`` plus the sidecar manifest ``<name>.json``."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{name}.pt"
    config = OmegaConf.to_container(cfg, resolve=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "optim": optim.state_dict(),
            "frames": frames,
            "iteration": iteration,
            "config": config,
        },
        tmp,
    )
    tmp.replace(path)
    write_json(
        path.with_suffix(".json"),
        {
            "checkpoint": path.name,
            "frames": frames,
            "iteration": iteration,
            "git_commit": git_commit(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sha256": sha256_file(path),
            "config": config,
            "eval": None,
            "video": None,
        },
    )
    return path


HERE = Path(__file__).resolve().parent


def frames_of(path) -> float:
    """Sort key for ``ppo_teacher_<frames>.pt``; ``ppo_teacher_final`` sorts last."""
    stem = Path(path).stem
    return float("inf") if stem.endswith("_final") else int(stem.rsplit("_", 1)[-1])


class CheckpointWorker:
    """Evaluates and renders a run's checkpoints in a background process while training continues.

    One pass = ``evaluate.py run=<run_dir>`` then ``render.py all=<run_dir>``; both skip checkpoints whose
    manifest already has an ``eval`` / ``video``, so a pass started later catches up on everything missed while
    the previous pass was busy. Output is appended to ``<run_dir>/worker.log``.
    """

    GRACE_S = 30.0  # between SIGTERM and SIGKILL when a pass overruns its timeout

    def __init__(self, run_dir, worker_cfg):
        self.run_dir = Path(run_dir)
        self.cfg = worker_cfg
        self.proc: subprocess.Popen | None = None
        self._reported: set[tuple[str, str]] = set()
        self.failed_passes = 0
        self._pass_checked = True  # nothing to check until a pass has been launched

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def check_failure(self) -> None:
        """Report a finished pass's non-zero exit code loudly, once.

        Without this a worker crash (e.g. "No CUDA GPUs are available" in a subprocess) is silent: no
        exception, training just never gains an eval/video for any checkpoint again.
        """
        if self.proc is None or self.running() or self._pass_checked:
            return
        self._pass_checked = True
        if self.proc.returncode == 0:
            return
        self.failed_passes += 1
        tail = []
        try:
            tail = (self.run_dir / "worker.log").read_text().splitlines()[-15:]
        except OSError:
            pass
        print(
            "WORKER_FAILED "
            + json.dumps(
                {
                    "exit_code": self.proc.returncode,
                    "failed_passes_total": self.failed_passes,
                    "unfinished": self.unfinished(),
                    "log_tail": tail,
                }
            ),
            flush=True,
        )

    def launch(self) -> bool:
        """Start a pass unless one is running; returns whether a pass was started."""
        self.check_failure()
        if self.running():
            return False
        python = shlex.quote(sys.executable)
        run = shlex.quote(str(self.run_dir))
        commands = [f"{python} {shlex.quote(str(HERE / 'evaluate.py'))} run={run} num_envs={int(self.cfg.eval_num_envs)}"]
        if self.cfg.render:
            commands.append(
                f"{python} {shlex.quote(str(HERE / 'render.py'))} all={run} "
                f"seconds={float(self.cfg.render_seconds)} image={int(self.cfg.render_image)}"
            )
        log = open(self.run_dir / "worker.log", "a")  # noqa: SIM115 - handed to the child process
        # start_new_session: `bash -c "a; b"` does not exec, so the Isaac Sim children are only reachable
        # through the process group -- see _signal_group(), used when a pass overruns its timeout.
        self.proc = subprocess.Popen(
            ["bash", "-c", "; ".join(commands)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        self._pass_checked = False
        log.close()
        return True

    def unfinished(self) -> list[str]:
        """``<checkpoint>(eval,video)`` for every checkpoint still missing a result, oldest first."""
        kinds = ("eval", "video") if self.cfg.render else ("eval",)
        out = []
        for path in sorted((self.run_dir / "checkpoints").glob("ppo_teacher_*.json"), key=frames_of):
            manifest = read_json(path)
            missing = [kind for kind in kinds if manifest.get(kind) is None]
            if missing:
                out.append(f"{manifest['checkpoint']}({','.join(missing)})")
        return out

    def _signal_group(self, sig) -> None:
        """Signal the whole pass (bash plus its Isaac Sim children); never raises."""
        try:
            os.killpg(os.getpgid(self.proc.pid), sig)
        except OSError:  # already gone
            pass

    def wait(self, timeout_s: float) -> None:
        """Wait for the running pass; terminate it if it overruns. Never raises.

        An overrunning pass must not be left behind: an orphaned Isaac Sim process outlives the trainer and
        collides with the next run (a stale ``/dev/shm`` carb semaphore alone can wedge ``launch_app()`` for
        tens of minutes). Whatever it did not finish is listed so it can be caught up by hand.
        """
        if not self.running():
            return
        try:
            self.proc.wait(timeout=timeout_s)
            self.check_failure()
            return
        except subprocess.TimeoutExpired:
            pass
        print(
            f"[worker] still running after {timeout_s:.0f} s; terminating it. Not finished: "
            f"{', '.join(self.unfinished()) or 'nothing'}. Catch up with: "
            f"evaluate.py run={self.run_dir} && render.py all={self.run_dir}",
            flush=True,
        )
        self._signal_group(signal.SIGTERM)
        try:
            self.proc.wait(timeout=self.GRACE_S)
        except subprocess.TimeoutExpired:
            self._signal_group(signal.SIGKILL)
            try:
                self.proc.wait(timeout=self.GRACE_S)
            except subprocess.TimeoutExpired:
                print("[worker] the pass survived SIGKILL; leaving it", flush=True)

    def new_results(self) -> list[tuple[str, dict]]:
        """(``"eval"`` | ``"video"``, checkpoint manifest) for results not reported before, oldest first."""
        results = []
        manifests = [read_json(p) for p in (self.run_dir / "checkpoints").glob("ppo_teacher_*.json")]
        for m in sorted(manifests, key=lambda m: m["frames"]):
            for kind in ("eval", "video"):
                if m.get(kind) is not None and (m["checkpoint"], kind) not in self._reported:
                    self._reported.add((m["checkpoint"], kind))
                    results.append((kind, m))
        return results
