from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class MotionClip:
  frames: torch.Tensor
  frame_duration: float
  weight: float


class AmpMotionLoader:
  JOINT_POS_SIZE = 26
  JOINT_VEL_SIZE = 26
  END_EFFECTOR_POS_SIZE = 12

  JOINT_POSE_START_IDX = 0
  JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

  JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
  JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

  END_POS_START_IDX = JOINT_VEL_END_IDX
  END_POS_END_IDX = END_POS_START_IDX + END_EFFECTOR_POS_SIZE

  def __init__(
    self,
    motion_files: Iterable[str],
    device: str,
    time_between_frames: float,
    obs_slice: Optional[Tuple[int, int]] = None,
    preload_transitions: bool = True,
    num_preload_transitions: int = 200_000,
    seed: int = 0,
  ):
    self.device = device
    self.time_between_frames = time_between_frames
    self.obs_slice = obs_slice
    self.rng = random.Random(seed)

    clips: List[MotionClip] = []
    self.trajectory_lens = []
    self.trajectory_frame_durations = []
    self.trajectory_num_frames = []

    for path in motion_files:
      with open(path, 'r') as f:
        data = json.load(f)
      frames_np = np.array(data['Frames'])
      frames = torch.tensor(frames_np, dtype=torch.float32, device=device)
      if obs_slice is not None:
        frames = frames[:, obs_slice[0] : obs_slice[1]]
      else:
        frames = frames[:, : self.END_POS_END_IDX]
      frame_duration = float(data['FrameDuration'])
      weight = float(data.get('MotionWeight', 1.0))
      traj_len = (frames.shape[0] - 1) * frame_duration
      clips.append(MotionClip(frames=frames, frame_duration=frame_duration, weight=weight))
      self.trajectory_lens.append(traj_len)
      self.trajectory_frame_durations.append(frame_duration)
      self.trajectory_num_frames.append(float(frames.shape[0]))

    if not clips:
      raise ValueError('No motion files loaded for AMP.')

    self.clips = clips
    weights = np.array([c.weight for c in clips], dtype=np.float64)
    self.clip_probs = weights / weights.sum()
    self.trajectory_lens = np.array(self.trajectory_lens)
    self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
    self.trajectory_num_frames = np.array(self.trajectory_num_frames)

    self.preloaded_s = None
    self.preloaded_s_next = None
    if preload_transitions:
      self._preload(num_preload_transitions)

  @property
  def observation_dim(self) -> int:
    return int(self.clips[0].frames.shape[1])

  @property
  def obs_dim(self) -> int:
    return self.observation_dim

  def _sample_clip_indices(self, batch_size: int) -> np.ndarray:
    return np.random.choice(len(self.clips), size=batch_size, p=self.clip_probs)

  def _traj_time_sample_batch(self, traj_idxs: np.ndarray) -> np.ndarray:
    subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
    time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(
      size=len(traj_idxs)
    ) - subst
    return np.maximum(np.zeros_like(time_samples), time_samples)

  def _get_frame_at_time_batch(
    self, traj_idxs: np.ndarray, times: np.ndarray
  ) -> torch.Tensor:
    p = times / self.trajectory_lens[traj_idxs]
    n = self.trajectory_num_frames[traj_idxs]
    idx_low = np.floor(p * n).astype(np.int64)
    idx_high = np.ceil(p * n).astype(np.int64)
    frames_start = torch.zeros(
      len(traj_idxs), self.observation_dim, device=self.device
    )
    frames_end = torch.zeros(
      len(traj_idxs), self.observation_dim, device=self.device
    )
    for traj_idx in set(traj_idxs.tolist()):
      traj_mask = traj_idxs == traj_idx
      trajectory = self.clips[traj_idx].frames
      frames_start[traj_mask] = trajectory[idx_low[traj_mask]]
      frames_end[traj_mask] = trajectory[idx_high[traj_mask]]
    blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
    return (1.0 - blend) * frames_start + blend * frames_end

  def _sample_transitions(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    traj_idxs = self._sample_clip_indices(batch_size)
    times = self._traj_time_sample_batch(traj_idxs)
    s = self._get_frame_at_time_batch(traj_idxs, times)
    s_next = self._get_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
    return s, s_next

  def _preload(self, num_transitions: int) -> None:
    s_list = []
    s_next_list = []
    remaining = num_transitions
    while remaining > 0:
      batch = min(remaining, 4096)
      s, s_next = self._sample_transitions(batch)
      s_list.append(s)
      s_next_list.append(s_next)
      remaining -= s.shape[0]
    self.preloaded_s = torch.cat(s_list, dim=0)
    self.preloaded_s_next = torch.cat(s_next_list, dim=0)

  def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if self.preloaded_s is not None:
      idx = torch.randint(0, self.preloaded_s.shape[0], (batch_size,), device=self.device)
      return self.preloaded_s[idx], self.preloaded_s_next[idx]
    return self._sample_transitions(batch_size)
