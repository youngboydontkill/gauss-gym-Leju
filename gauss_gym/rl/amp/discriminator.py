from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn


class AmpDiscriminator(nn.Module):
  def __init__(
    self,
    input_dim: int,
    hidden_dims: Iterable[int],
    reward_coef: float = 0.3,
    task_reward_lerp: float = 0.0,
  ):
    super().__init__()
    dims = [input_dim, *list(hidden_dims)]
    layers = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
      layers.append(nn.Linear(in_dim, out_dim))
      layers.append(nn.ReLU())
    self.trunk = nn.Sequential(*layers)
    self.head = nn.Linear(dims[-1], 1)
    self.reward_coef = reward_coef
    self.task_reward_lerp = task_reward_lerp

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.head(self.trunk(x))

  def predict_reward(
    self,
    state: torch.Tensor,
    next_state: torch.Tensor,
    task_reward: torch.Tensor,
    normalizer=None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    self.eval()
    with torch.no_grad():
      if normalizer is not None:
        state = normalizer.normalize(state)
        next_state = normalizer.normalize(next_state)
      logits = self(torch.cat([state, next_state], dim=-1))
      disc_reward = torch.clamp(1.0 - 0.25 * (logits - 1.0).pow(2), min=0.0)
      reward = self.reward_coef * disc_reward.squeeze(-1)
      if self.task_reward_lerp > 0:
        reward = (1.0 - self.task_reward_lerp) * reward + self.task_reward_lerp * task_reward
    self.train()
    return reward, logits

  def predict_amp_reward(
    self,
    state: torch.Tensor,
    next_state: torch.Tensor,
    task_reward: torch.Tensor,
    normalizer=None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    return self.predict_reward(state, next_state, task_reward, normalizer=normalizer)

  def compute_grad_penalty(
    self, expert_state: torch.Tensor, expert_next_state: torch.Tensor, lambda_: float = 10.0
  ) -> torch.Tensor:
    expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
    expert_data.requires_grad_(True)
    logits = self(expert_data)
    grad_outputs = torch.ones_like(logits)
    grads = torch.autograd.grad(
      outputs=logits,
      inputs=expert_data,
      grad_outputs=grad_outputs,
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    grad_norm = grads.norm(2, dim=-1)
    return lambda_ * torch.mean((grad_norm - 1.0) ** 2)

  def compute_grad_pen(
    self, expert_state: torch.Tensor, expert_next_state: torch.Tensor, lambda_: float = 10.0
  ) -> torch.Tensor:
    return self.compute_grad_penalty(expert_state, expert_next_state, lambda_=lambda_)
