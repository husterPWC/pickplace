"""Stage 1: record an offline-RL dataset shard by running a state-teacher checkpoint with cameras on.

The teacher acts deterministically (its mean action); ``noise_sigma > 0`` adds Gaussian noise to the executed
action (clipped to the action bounds) to produce a lower-quality tier. Per step the shard stores the newest
frame of each camera as uint8, the vector observations, the executed action, the teacher's ``loc``/``scale``,
the unweighted ``("next", "reward_terms")`` vector and the scalar reward, the done flags, the episode-outcome
flags, the trajectory id and the step count.

A shard is ``<out>/storage`` (a ``LazyMemmapStorage``-backed ``TensorDictReplayBuffer``, load it with
``pickplace.datasets.load_shard``) plus ``<out>/manifest.json``. Rows are time-major: row ``i`` and row
``i + num_envs`` are consecutive steps of the same sub-env.

Usage:
    python pipeline/1_collect_data/collect.py checkpoint=<path.pt> [frames=1000000] [noise_sigma=0.0]
        [seed=0] [num_envs=512] [image_size=84] [rollout_steps=16] [out=<dir>] [name=<shard name>]

``out`` defaults to ``$FOOD_ROBOT_ARTIFACTS/shards/<name>``. The env config is the checkpoint's own, with
cameras on and ``image_size``/``num_envs``/``seed`` overridden. ``frames`` is rounded up to a whole number of
collector batches (``num_envs x rollout_steps``).
"""

import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
TEACHER_DIR = os.path.join(os.path.dirname(HERE), "0_state_teacher")

cli = OmegaConf.from_dotlist(sys.argv[1:])
if not cli.get("checkpoint"):
    raise SystemExit("Pass checkpoint=<path.pt>.")
checkpoint = Path(cli.checkpoint)
frames = int(cli.get("frames", 1_000_000))
noise_sigma = float(cli.get("noise_sigma", 0.0))
seed = int(cli.get("seed", 0))
num_envs = int(cli.get("num_envs", 512))
image_size = int(cli.get("image_size", 84))
rollout_steps = int(cli.get("rollout_steps", 16))
include_depth = bool(cli.get("include_depth", False))
store_teacher_inputs = bool(cli.get("store_teacher_inputs", False))
jaka_visual_task = cli.get("jaka_visual_task")
env_checkpoint = Path(cli.get("env_checkpoint", checkpoint))

from pickplace.system import memory_used_gb  # noqa: E402

baseline_gb = memory_used_gb()

from pickplace.app import launch_app  # noqa: E402

app = launch_app(headless=True, enable_cameras=True)

import torch  # noqa: E402
from tensordict.nn import TensorDictModule, TensorDictSequential  # noqa: E402
from torchrl.collectors import Collector  # noqa: E402
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer  # noqa: E402
from torchrl.envs import ExplorationType  # noqa: E402

from pickplace.artifacts import artifacts_root, git_commit, read_json, sha256_file, write_json  # noqa: E402
from pickplace.datasets import STORAGE_DIR  # noqa: E402
from pickplace.g2_collection import validate_jaka_visual_teacher_payload  # noqa: E402
from pickplace.metrics import outcome_terms  # noqa: E402
from pickplace.rewards import REWARD_TERMS  # noqa: E402
from pickplace.torchrl_env import make_env, reward_weights  # noqa: E402

# Loaded by file path: with cameras enabled, Isaac Sim's bundled cv2/utils shadows `import utils`.
_spec = importlib.util.spec_from_file_location("teacher_utils", os.path.join(TEACHER_DIR, "utils.py"))
tu = importlib.util.module_from_spec(_spec)
sys.modules["teacher_utils"] = tu
_spec.loader.exec_module(tu)

CAMERAS = ("overview_rgb", "wrist_rgb")
DEPTH_CAMERAS = ("overview_depth", "wrist_depth")
DEPLOYABLE_PROPRIO_KEYS = (
    ("proprio", "joint_pos_rel"),
    ("proprio", "gripper_pos"),
    ("proprio", "ee_pos"),
    ("proprio", "ee_quat"),
    ("proprio", "last_action"),
)
STORE_KEYS = [
    *[("pixels", cam) for cam in CAMERAS],
    *([("pixels", cam) for cam in DEPTH_CAMERAS] if include_depth else []),
    *(DEPLOYABLE_PROPRIO_KEYS if store_teacher_inputs else ("proprio",)),
    "belt",
    "action",
    "loc",
    "scale",
    "step_count",
    ("collector", "traj_ids"),
    ("next", "reward"),
    ("next", "reward_terms"),
    ("next", "terminated"),
    ("next", "truncated"),
    ("next", "done"),
    ("next", "outcome"),
]
OPTIONAL_STORE_KEYS = ["privileged"]
TEACHER_INPUT_KEYS = ["grasp_guidance", "position_history"]


class Stats:
    """Running totals over every episode that finishes during the collection."""

    def __init__(self, outcomes):
        self.episodes = 0
        self.returns = 0.0
        self.lengths = 0.0
        self.terms = torch.zeros(len(REWARD_TERMS))
        self.outcomes = dict.fromkeys(outcomes, 0)

    def update(self, data) -> None:
        # Under native auto-reset the running sums reset on the done row, so a finished episode's totals are
        # the pre-step running sum plus this step's value (as in pickplace.training).
        done = data["next", "done"]
        if not bool(done.any()):
            return
        self.episodes += int(done.sum())
        self.returns += float((data["episode_reward"] + data["next", "reward"])[done].sum())
        self.lengths += float((data["step_count"] + 1)[done].sum())
        self.terms += (data["episode_reward_terms"] + data["next", "reward_terms"])[done.squeeze(-1)].sum(0).cpu()
        for term in self.outcomes:
            self.outcomes[term] += int((data["next", "outcome", term] & done).sum())

    def summary(self) -> dict:
        n = self.episodes
        if not n:
            return {"episodes": 0}
        return {
            "episodes": n,
            "success_rate": self.outcomes["success"] / n,
            "episode_return": self.returns / n,
            "episode_length": self.lengths / n,
            **{f"{term}_rate": count / n for term, count in self.outcomes.items()},
            **{f"terms/{name}": float(self.terms[k]) / n for k, name in enumerate(REWARD_TERMS)},
        }


def rows(data):
    """Select what the shard stores and flatten (num_envs, T) time-major, images cast to uint8."""
    optional = [key for key in OPTIONAL_STORE_KEYS if data.get(key, None) is not None]
    if store_teacher_inputs:
        missing = [key for key in TEACHER_INPUT_KEYS if data.get(key, None) is None]
        if missing:
            raise KeyError(f"G2 teacher input groups missing from rollout: {missing}")
        optional.extend(TEACHER_INPUT_KEYS)
    out = data.select(*STORE_KEYS, *optional, strict=True)
    for cam in CAMERAS:
        image = out["pixels", cam]
        # [..., -3:]: the newest frame when the env stacks frames. Values are integral floats in [0, 255].
        out.set(("pixels", cam), image[..., -3:].round().clamp(0, 255).to(torch.uint8))
    for cam in DEPTH_CAMERAS if include_depth else ():
        depth = out["pixels", cam]
        # Newest depth channel, finite metres. float16 keeps the RGB-D shard
        # near 70 KB/frame while retaining sub-millimetre precision near 1 m.
        out.set(("pixels", cam), depth[..., -1:].clamp(0, 3).to(torch.float16))
    if store_teacher_inputs:
        out.set(("teacher", "residual_action"), out["action"].clone())
    return out.permute(1, 0).reshape(-1)


name = cli.get("name") or f"shard_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
out_dir = Path(cli.out) if cli.get("out") else artifacts_root() / "shards" / name
out_dir.mkdir(parents=True, exist_ok=True)

frames_per_batch = num_envs * rollout_steps
batches = max(1, math.ceil(frames / frames_per_batch))
frames = batches * frames_per_batch

actor_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
env_payload = torch.load(env_checkpoint, map_location="cpu", weights_only=False)
env_cfg = env_payload["config"]["env"]
visual_contract = None
if jaka_visual_task:
    if not include_depth or not store_teacher_inputs:
        raise ValueError(
            "JAKA visual collection requires include_depth=true and "
            "store_teacher_inputs=true"
        )
    visual_contract = validate_jaka_visual_teacher_payload(
        actor_payload, env_payload, str(jaka_visual_task)
    )
env_cfg = {
    **env_cfg,
    "num_envs": num_envs,
    "cameras": True,
    "camera_depth": include_depth,
    "image_size": [image_size, image_size],
    "seed": seed,
}
if store_teacher_inputs:
    # G2 has a single-frame RGB-D contract.  Never silently inherit a legacy
    # camera stack from a checkpoint and then discard all but its newest frame.
    env_cfg["frame_stack"] = 1
torch.manual_seed(seed)
env = make_env(env_cfg)
device = env.device
actor = tu.load_teacher_actor(checkpoint, env, device)

policy = actor
if noise_sigma > 0.0:
    # Explicit noise instead of torchrl's AdditiveGaussianModule: that module only perturbs the action under
    # ExplorationType.RANDOM, which would also make the teacher sample from its distribution instead of
    # taking the mean action this collection is built on. Bounds are the teacher's TanhNormal support
    # [-1, 1] (the env's own action spec is unbounded).
    def add_noise(action: torch.Tensor) -> torch.Tensor:
        return (action + noise_sigma * torch.randn_like(action)).clamp(-1.0, 1.0)

    policy = TensorDictSequential(actor, TensorDictModule(add_noise, in_keys=["action"], out_keys=["action"]))

collector = Collector(
    env,
    policy,
    frames_per_batch=frames_per_batch,
    total_frames=-1,
    device=device,
    no_cuda_sync=True,
    trust_policy=True,
    exploration_type=ExplorationType.DETERMINISTIC,
)
buffer = TensorDictReplayBuffer(storage=LazyMemmapStorage(frames, scratch_dir=out_dir / STORAGE_DIR))

print(
    "COLLECT_INFO "
    + json.dumps(
        {
            "out": str(out_dir),
            "checkpoint": str(checkpoint),
            "env_checkpoint": str(env_checkpoint),
            "jaka_visual_task": str(jaka_visual_task) if jaka_visual_task else None,
            "frames": frames,
            "batches": batches,
            "num_envs": num_envs,
            "rollout_steps": rollout_steps,
            "image_size": image_size,
            "noise_sigma": noise_sigma,
            "include_depth": include_depth,
            "store_teacher_inputs": store_teacher_inputs,
            "seed": seed,
        }
    ),
    flush=True,
)

stats = Stats(outcome_terms(env_cfg["task"]))
stored, start = 0, time.monotonic()
for batch, data in enumerate(collector):
    stats.update(data)
    buffer.extend(rows(data))
    stored += data.numel()
    elapsed = time.monotonic() - start
    print(
        "PROGRESS "
        + json.dumps(
            {
                "frames": stored,
                "fraction": round(stored / frames, 4),
                "frames_per_hour": round(stored / elapsed * 3600.0),
                "eta_min": round((frames - stored) / (stored / elapsed) / 60.0, 1),
                "memory_used_gb": round(memory_used_gb(), 2),
                **({"success_rate": round(stats.summary()["success_rate"], 4)} if stats.episodes else {}),
            }
        ),
        flush=True,
    )
    if stored >= frames:
        break
collector.shutdown()

seconds = time.monotonic() - start
size_bytes = sum(p.stat().st_size for p in (out_dir / STORAGE_DIR).rglob("*") if p.is_file())
manifest = {
    "kind": "dataset_shard",
    "name": out_dir.name,
    "frames": stored,
    "num_envs": num_envs,
    "rollout_steps": rollout_steps,
    "successor_stride": num_envs,  # row i's successor step (same sub-env) is row i + num_envs
    "image_size": image_size,
    "cameras": list(CAMERAS),
    "depth_cameras": list(DEPTH_CAMERAS) if include_depth else [],
    "depth_unit": "metres" if include_depth else None,
    "depth_dtype": "float16" if include_depth else None,
    "teacher_inputs_stored": store_teacher_inputs,
    "g2_contract": (
        visual_contract
        if visual_contract is not None
        else {
            "student_runtime_keys": [
                ["pixels", "overview_rgb"],
                ["pixels", "wrist_rgb"],
                ["pixels", "overview_depth"],
                ["pixels", "wrist_depth"],
                ["proprio", "joint_pos_rel"],
                ["proprio", "gripper_pos"],
                ["proprio", "ee_pos"],
                ["proprio", "ee_quat"],
                ["proprio", "last_action"],
                ["belt", "bowl_pos"],
            ],
            "teacher_only_groups": ["grasp_guidance", "position_history"],
            "forbidden_student_signals": [
                "object_velocity", "belt_speed", "contact", "force",
                "tactile", "motor_current", "torque",
            ],
        }
        if store_teacher_inputs
        else None
    ),
    "action_semantics": (
        visual_contract["action_semantics"]
        if visual_contract is not None
        else (
            "teacher_residual_before_environment_g1_controller"
            if bool(env_cfg.get("stable_sphere_pick_place_skill", {}).get("enabled", False))
            else "policy_action"
        )
    ),
    "visual_task_profile": str(jaka_visual_task) if jaka_visual_task else None,
    "noise_sigma": noise_sigma,
    "seed": seed,
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": sha256_file(checkpoint),
    "env_checkpoint": str(env_checkpoint),
    "env_checkpoint_sha256": sha256_file(env_checkpoint),
    "checkpoint_eval": read_json(checkpoint.with_suffix(".json")).get("eval") if checkpoint.with_suffix(".json").exists() else None,
    "env": env_cfg,
    "reward_set": env_cfg.get("reward_set"),
    "reward_weights": reward_weights(env_cfg),
    "git_commit": git_commit(),
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "seconds": round(seconds, 1),
    "frames_per_hour": round(stored / seconds * 3600.0),
    "size_bytes": size_bytes,
    "size_gb": round(size_bytes / 1024**3, 2),
    "memory_used_gb": round(memory_used_gb(), 2),
    "stats": stats.summary(),
}
write_json(out_dir / "manifest.json", manifest)
print("COLLECT " + json.dumps({k: v for k, v in manifest.items() if k != "env"}), flush=True)
print(f"COLLECT_DONE frames={stored} out={out_dir}", flush=True)
os._exit(0)  # Isaac Sim shutdown can hang
