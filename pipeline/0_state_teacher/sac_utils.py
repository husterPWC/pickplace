"""Small, dependency-free SAC components for the sphere_gap curriculum."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn


class GapObservationEncoder(nn.Module):
    """GAP-style per-point/per-grasp pooling fused with robot vectors."""

    def __init__(self, shapes, grasp_index: int, hidden=(256, 256), point_hidden=64, latent_dim=256):
        super().__init__()
        self.shapes = list(shapes)
        self.grasp_index = int(grasp_index)
        grasp_shape = self.shapes[self.grasp_index]
        if tuple(grasp_shape) != (40, 20, 3):
            raise ValueError(f"sphere_gap expects grasp points (40,20,3), got {grasp_shape}")
        self.point_encoder = nn.Sequential(
            nn.Linear(3, point_hidden), nn.ReLU(),
            nn.Linear(point_hidden, point_hidden), nn.ReLU(),
        )
        self.grasp_encoder = nn.Sequential(nn.Linear(point_hidden, point_hidden), nn.ReLU())
        vector_dim = sum(math.prod(shape) for i, shape in enumerate(self.shapes) if i != self.grasp_index)
        layers: list[nn.Module] = []
        in_dim = vector_dim + point_hidden
        for width in hidden:
            layers.extend((nn.Linear(in_dim, int(width)), nn.ReLU()))
            in_dim = int(width)
        layers.append(nn.Linear(in_dim, latent_dim))
        layers.append(nn.ReLU())
        self.fusion = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, math.sqrt(2.0))
            nn.init.zeros_(module.bias)

    def forward(self, *values: torch.Tensor) -> torch.Tensor:
        points = values[self.grasp_index].float()
        point_features = self.point_encoder(points).amax(dim=-2)
        grasp_features = self.grasp_encoder(point_features).amax(dim=-2)
        leading = points.shape[:-3]
        vectors = [
            value.float().reshape(*leading, -1)
            for index, value in enumerate(values)
            if index != self.grasp_index
        ]
        return self.fusion(torch.cat((*vectors, grasp_features), dim=-1))


class SquashedGaussianActor(nn.Module):
    def __init__(
        self, encoder: nn.Module, latent_dim: int, action_dim: int,
        log_std_init: float = -3.67, mean_action_init=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.mean = nn.Linear(latent_dim, action_dim)
        self.log_std = nn.Linear(latent_dim, action_dim)
        nn.init.orthogonal_(self.mean.weight, 0.01)
        nn.init.zeros_(self.mean.bias)
        if mean_action_init is not None:
            initial = torch.as_tensor(mean_action_init, dtype=self.mean.bias.dtype)
            if initial.shape != (action_dim,) or bool((initial.abs() >= 1.0).any()):
                raise ValueError(f"mean_action_init must contain {action_dim} values strictly inside (-1,1)")
            with torch.no_grad():
                self.mean.bias.copy_(torch.atanh(initial))
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, float(log_std_init))

    def forward(self, *obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(*obs)
        return self.mean(latent), self.log_std(latent).clamp(-5.0, 1.0)

    def sample(self, *obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(*obs)
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = mean + log_std.exp() * torch.randn_like(mean)
        action = torch.tanh(pre_tanh)
        if deterministic:
            log_prob = torch.zeros((*action.shape[:-1], 1), dtype=action.dtype, device=action.device)
        else:
            normal_log_prob = -0.5 * (
                ((pre_tanh - mean) / log_std.exp()).square() + 2.0 * log_std + math.log(2.0 * math.pi)
            )
            log_prob = (normal_log_prob - torch.log(1.0 - action.square() + 1.0e-6)).sum(-1, keepdim=True)
        return action, log_prob


class SoftQNetwork(nn.Module):
    def __init__(self, encoder: nn.Module, latent_dim: int, action_dim: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1),
        )
        self.head.apply(GapObservationEncoder._init)
        nn.init.orthogonal_(self.head[-1].weight, 1.0)

    def forward(self, *obs_and_action: torch.Tensor) -> torch.Tensor:
        *obs, action = obs_and_action
        return self.head(torch.cat((self.encoder(*obs), action.float()), dim=-1))


def make_sac_models(shapes, grasp_index, action_dim, cfg, device):
    def encoder():
        return GapObservationEncoder(
            shapes,
            grasp_index,
            hidden=tuple(cfg.hidden),
            point_hidden=int(cfg.point_hidden),
            latent_dim=int(cfg.latent_dim),
        )

    actor = SquashedGaussianActor(
        encoder(), int(cfg.latent_dim), action_dim,
        log_std_init=float(cfg.log_std_init),
        mean_action_init=getattr(cfg, "mean_action_init", None),
    ).to(device)
    q1 = SoftQNetwork(encoder(), int(cfg.latent_dim), action_dim).to(device)
    q2 = SoftQNetwork(encoder(), int(cfg.latent_dim), action_dim).to(device)
    target_q1, target_q2 = copy.deepcopy(q1).requires_grad_(False), copy.deepcopy(q2).requires_grad_(False)
    return actor, q1, q2, target_q1, target_q2


class ReplayBuffer:
    """CPU ring buffer; large grasp point tensors are retained as float16."""

    def __init__(self, capacity: int, obs_shapes, grasp_index: int):
        self.capacity = int(capacity)
        self.grasp_index = int(grasp_index)
        self.obs = [
            torch.empty((capacity, *shape), dtype=torch.float16 if i == grasp_index else torch.float32)
            for i, shape in enumerate(obs_shapes)
        ]
        self.next_obs = [torch.empty_like(value) for value in self.obs]
        self.action = None
        self.reward = torch.empty(capacity, 1, dtype=torch.float32)
        self.done = torch.empty(capacity, 1, dtype=torch.float32)
        self.position = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        batch = action.shape[0]
        if self.action is None:
            self.action = torch.empty(self.capacity, action.shape[-1], dtype=torch.float32)
        if batch > self.capacity:
            obs, next_obs = [v[-self.capacity:] for v in obs], [v[-self.capacity:] for v in next_obs]
            action, reward, done = action[-self.capacity:], reward[-self.capacity:], done[-self.capacity:]
            batch = self.capacity
        indices = (torch.arange(batch) + self.position) % self.capacity
        for target, value in zip(self.obs, obs):
            target[indices] = value.detach().to(device="cpu", dtype=target.dtype)
        for target, value in zip(self.next_obs, next_obs):
            target[indices] = value.detach().to(device="cpu", dtype=target.dtype)
        self.action[indices] = action.detach().to("cpu", dtype=torch.float32)
        self.reward[indices] = reward.detach().reshape(batch, 1).to("cpu", dtype=torch.float32)
        self.done[indices] = done.detach().reshape(batch, 1).to("cpu", dtype=torch.float32)
        self.position = int((self.position + batch) % self.capacity)
        self.size = min(self.capacity, self.size + batch)

    def sample(self, batch_size: int, device):
        indices = torch.randint(self.size, (int(batch_size),))
        move = lambda value: value[indices].to(device=device, dtype=torch.float32, non_blocking=True)
        return (
            [move(value) for value in self.obs],
            move(self.action),
            move(self.reward),
            [move(value) for value in self.next_obs],
            move(self.done),
        )


@torch.no_grad()
def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for src, dst in zip(source.parameters(), target.parameters()):
        dst.lerp_(src, float(tau))
