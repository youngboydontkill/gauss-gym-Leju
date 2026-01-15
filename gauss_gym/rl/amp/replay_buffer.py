from __future__ import annotations

from typing import Tuple

import torch


class AmpReplayBuffer:
  def __init__(self, capacity: int, obs_dim: int, device: str):
    self.capacity = capacity
    self.device = device
    self.obs_dim = obs_dim
    self.ptr = 0
    self.size = 0
    self.states = torch.zeros(capacity, obs_dim, device=device)
    self.next_states = torch.zeros(capacity, obs_dim, device=device)

  def add(self, state: torch.Tensor, next_state: torch.Tensor) -> None:
    if state.ndim == 1:
      state = state.unsqueeze(0)
      next_state = next_state.unsqueeze(0)
    batch = state.shape[0]
    for i in range(batch):
      self.states[self.ptr] = state[i]
      self.next_states[self.ptr] = next_state[i]
      self.ptr = (self.ptr + 1) % self.capacity
      self.size = min(self.size + 1, self.capacity)

  def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if self.size == 0:
      raise ValueError('AMP replay buffer is empty.')
    idx = torch.randint(0, self.size, (batch_size,), device=self.device)
    return self.states[idx], self.next_states[idx]
