"""
Transformer-based MAPPO-style agent for temperature correction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from .. import config


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class TransformerBackbone(nn.Module):
    def __init__(self, seq_len: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Create positional embedding dynamically in forward pass
        self.register_buffer('pos_embedding', self._create_pos_embedding(seq_len, d_model), persistent=False)

    def _create_pos_embedding(self, seq_len: int, d_model: int) -> torch.Tensor:
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        if d_model % 2 == 1:
            pe[0, :, 1::2] = torch.cos(pos * div_term[:-1])
        else:
            pe[0, :, 1::2] = torch.cos(pos * div_term)
        return pe

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs shape: (batch, seq_len)
        batch_size, actual_seq_len = obs.shape
        x = obs.unsqueeze(-1)  # (batch, seq_len, 1)
        x = self.input_proj(x)
        
        # Resize positional embedding if needed
        if actual_seq_len != self.pos_embedding.size(1):
            pos_emb = self._create_pos_embedding(actual_seq_len, self.d_model).to(obs.device)
        else:
            pos_emb = self.pos_embedding
        
        x = x + pos_emb
        x = self.encoder(x)
        return x.mean(dim=1)


class Actor(nn.Module):
    def __init__(self, seq_len: int, action_dim: int):
        super().__init__()
        self.backbone = TransformerBackbone(seq_len)
        self.mean_head = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.apply(_init_weights)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        mean = torch.tanh(self.mean_head(features))
        std = torch.exp(self.log_std).clamp(min=1e-3, max=2.0)
        std = std.unsqueeze(0).expand_as(mean)
        return mean, std


class Critic(nn.Module):
    def __init__(self, seq_len: int):
        super().__init__()
        self.backbone = TransformerBackbone(seq_len)
        self.value_head = nn.Linear(64, 1)
        self.apply(_init_weights)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(obs)
        return self.value_head(features)


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class MAPPOTransformerAgent:
    """
    Lightweight MAPPO implementation with shared transformer policies.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.actor = Actor(observation_dim, action_dim).to(self.device)
        self.critic = Critic(observation_dim).to(self.device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=config.RL_LR)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=config.RL_LR)

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.actor(obs_t)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            value = self.critic(obs_t)
        return (
            action.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_prob.cpu().numpy()[0]),
            float(value.cpu().numpy()[0]),
        )

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, _ = self.actor(obs_t)
        return mean.squeeze(0).cpu().numpy().astype(np.float32)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.actor(obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        values = self.critic(obs)
        return log_probs, entropy, values

    def _compute_returns_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        gamma: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        returns = rewards + gamma * np.concatenate([values[1:], values[-1:]], axis=0)
        advantages = returns - values
        return returns, advantages

    def update(self, batch: RolloutBatch) -> Dict[str, float]:
        obs = batch.observations.to(self.device)
        actions = batch.actions.to(self.device)
        old_log_probs = batch.log_probs.to(self.device)
        returns = batch.returns.to(self.device)
        advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
        advantages = advantages.to(self.device)

        log_probs, entropy, values = self.evaluate(obs, actions)
        ratios = torch.exp(log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - config.RL_CLIP_RANGE, 1.0 + config.RL_CLIP_RANGE) * advantages
        actor_loss = -(torch.min(surr1, surr2)).mean()

        value_loss = nn.functional.mse_loss(values, returns)

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.optimizer_critic.step()

        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy.mean().detach().cpu()),
        }


def collect_rollout(
    env,
    agent: MAPPOTransformerAgent,
    batch_size: int,
) -> RolloutBatch:
    obs_list = []
    actions = []
    log_probs = []
    rewards = []
    values = []

    for _ in range(batch_size):
        obs, _ = env.reset()
        action, log_prob, value = agent.act(obs)
        _, reward, terminated, _, info = env.step(action)
        obs_list.append(obs)
        actions.append(action)
        log_probs.append([log_prob])
        rewards.append([reward])
        values.append([value])
        assert terminated

    observations = torch.tensor(np.array(obs_list), dtype=torch.float32)
    actions_t = torch.tensor(np.array(actions), dtype=torch.float32)
    log_probs_t = torch.tensor(np.array(log_probs), dtype=torch.float32)
    rewards_np = np.array(rewards, dtype=np.float32)
    values_np = np.array(values, dtype=np.float32)
    returns_np, advantages_np = agent._compute_returns_advantages(
        rewards_np,
        values_np,
        config.RL_GAMMA,
    )

    return RolloutBatch(
        observations=observations,
        actions=actions_t,
        log_probs=log_probs_t,
        returns=torch.tensor(returns_np, dtype=torch.float32),
        advantages=torch.tensor(advantages_np, dtype=torch.float32),
    )