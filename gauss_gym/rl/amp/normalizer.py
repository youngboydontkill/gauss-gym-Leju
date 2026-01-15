from __future__ import annotations

import torch


class RunningMeanStd:
  def __init__(self, dim: int, device: str = 'cpu', eps: float = 1e-6):
    self.device = device
    self.eps = eps
    self.count = torch.tensor(0.0, device=device)
    self.mean = torch.zeros(dim, device=device)
    self.var = torch.ones(dim, device=device)

  def update(self, x: torch.Tensor) -> None:
    if x.numel() == 0:
      return
    x = x.detach()
    if x.ndim == 1:
      x = x.unsqueeze(0)
    batch_count = x.shape[0]
    batch_mean = x.mean(dim=0)
    batch_var = x.var(dim=0, unbiased=False)

    delta = batch_mean - self.mean
    total_count = self.count + batch_count

    new_mean = self.mean + delta * batch_count / total_count
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
    new_var = m2 / total_count

    self.mean = new_mean
    self.var = torch.clamp(new_var, min=self.eps)
    self.count = total_count

  def normalize(self, x: torch.Tensor) -> torch.Tensor:
    return (x - self.mean) / torch.sqrt(self.var + self.eps)

  def normalize_torch(self, x: torch.Tensor, device: str) -> torch.Tensor:
    if x.device != self.mean.device:
      x = x.to(self.mean.device)
    return self.normalize(x)

  def state_dict(self) -> dict:
    return {
      'count': self.count,
      'mean': self.mean,
      'var': self.var,
      'eps': self.eps,
    }

  def load_state_dict(self, state: dict) -> None:
    self.count = state['count']
    self.mean = state['mean']
    self.var = state['var']
    self.eps = state.get('eps', self.eps)
