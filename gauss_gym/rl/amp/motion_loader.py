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
  frames_full: Optional[torch.Tensor] = None


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
    seed_int = int(seed)
    if seed_int < 0:
      seed_int = seed_int % (2**32)
    self.np_rng = np.random.default_rng(seed_int)

    clips: List[MotionClip] = []
    self.trajectory_lens = []
    self.trajectory_frame_durations = []
    self.trajectory_num_frames = []

    for path in motion_files:
      with open(path, 'r') as f:
        data = json.load(f)
      frames_np = np.array(data['Frames'])
      frames = torch.tensor(frames_np, dtype=torch.float32, device=device)
      frames_full = frames[:, : self.END_POS_END_IDX]
      if obs_slice is not None:
        frames = frames[:, obs_slice[0] : obs_slice[1]]
      else:
        frames = frames_full
      frame_duration = float(data['FrameDuration'])
      weight = float(data.get('MotionWeight', 1.0))
      traj_len = (frames.shape[0] - 1) * frame_duration
      clips.append(
        MotionClip(
          frames=frames,
          frame_duration=frame_duration,
          weight=weight,
          frames_full=frames_full,
        )
      )
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
    return self.np_rng.choice(len(self.clips), size=batch_size, p=self.clip_probs)

  def _traj_time_sample_batch(self, traj_idxs: np.ndarray) -> np.ndarray:
    subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
    time_samples = self.trajectory_lens[traj_idxs] * self.np_rng.uniform(
      size=len(traj_idxs)
    ) - subst
    return np.maximum(np.zeros_like(time_samples), time_samples)

  def slerp(self, frame1: torch.Tensor, frame2: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return (1.0 - blend) * frame1 + blend * frame2

  def get_frame_at_time(self, traj_idx: int, time: float) -> torch.Tensor:
    p = float(time) / self.trajectory_lens[traj_idx]
    n = self.clips[traj_idx].frames.shape[0]
    idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
    frame_start = self.clips[traj_idx].frames[idx_low]
    frame_end = self.clips[traj_idx].frames[idx_high]
    blend = p * n - idx_low
    return self.slerp(frame_start, frame_end, blend)

  def get_frame_at_time_batch(self, traj_idxs: np.ndarray, times: np.ndarray) -> torch.Tensor:
    return self._get_frame_at_time_batch(traj_idxs, times)

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

  def get_full_frame_at_time(self, traj_idx: int, time: float) -> torch.Tensor:
    clip = self.clips[traj_idx]
    if clip.frames_full is None:
      return self.get_frame_at_time(traj_idx, time)
    p = float(time) / self.trajectory_lens[traj_idx]
    n = clip.frames_full.shape[0]
    idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
    frame_start = clip.frames_full[idx_low]
    frame_end = clip.frames_full[idx_high]
    blend = p * n - idx_low
    return self.blend_frame_pose(frame_start, frame_end, blend)

  def get_full_frame_at_time_batch(
    self, traj_idxs: np.ndarray, times: np.ndarray
  ) -> torch.Tensor:
    p = times / self.trajectory_lens[traj_idxs]
    n = self.trajectory_num_frames[traj_idxs]
    idx_low = np.floor(p * n).astype(np.int64)
    idx_high = np.ceil(p * n).astype(np.int64)
    frames_start = torch.zeros(
      len(traj_idxs), self.END_POS_END_IDX - self.JOINT_POSE_START_IDX, device=self.device
    )
    frames_end = torch.zeros(
      len(traj_idxs), self.END_POS_END_IDX - self.JOINT_POSE_START_IDX, device=self.device
    )
    for traj_idx in set(traj_idxs.tolist()):
      traj_mask = traj_idxs == traj_idx
      clip = self.clips[traj_idx]
      trajectory = clip.frames_full if clip.frames_full is not None else clip.frames
      frames_start[traj_mask] = trajectory[idx_low[traj_mask]][
        :, self.JOINT_POSE_START_IDX : self.END_POS_END_IDX
      ]
      frames_end[traj_mask] = trajectory[idx_high[traj_mask]][
        :, self.JOINT_POSE_START_IDX : self.END_POS_END_IDX
      ]
    blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
    return self.slerp(frames_start, frames_end, blend)

  def get_frame(self) -> torch.Tensor:
    traj_idx = int(self._sample_clip_indices(1)[0])
    sampled_time = float(self._traj_time_sample_batch(np.array([traj_idx]))[0])
    return self.get_frame_at_time(traj_idx, sampled_time)

  def get_full_frame(self) -> torch.Tensor:
    traj_idx = int(self._sample_clip_indices(1)[0])
    sampled_time = float(self._traj_time_sample_batch(np.array([traj_idx]))[0])
    return self.get_full_frame_at_time(traj_idx, sampled_time)

  def get_full_frame_batch(self, num_frames: int) -> torch.Tensor:
    traj_idxs = self._sample_clip_indices(num_frames)
    times = self._traj_time_sample_batch(traj_idxs)
    return self.get_full_frame_at_time_batch(traj_idxs, times)

  def blend_frame_pose(self, frame0: torch.Tensor, frame1: torch.Tensor, blend: float) -> torch.Tensor:
    joints0, joints1 = self.get_joint_pose(frame0), self.get_joint_pose(frame1)
    joint_vel_0, joint_vel_1 = self.get_joint_vel(frame0), self.get_joint_vel(frame1)
    blend_joint_q = self.slerp(joints0, joints1, blend)
    blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)
    return torch.cat([blend_joint_q, blend_joints_vel])

  def feed_forward_generator(
    self, num_mini_batch: int, mini_batch_size: int
  ) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    for _ in range(num_mini_batch):
      if self.preloaded_s is not None:
        idx = torch.randint(0, self.preloaded_s.shape[0], (mini_batch_size,), device=self.device)
        s = self.preloaded_s[idx]
        s_next = self.preloaded_s_next[idx]
      else:
        s, s_next = self._sample_transitions(mini_batch_size)
      yield s, s_next

  @property
  def num_motions(self) -> int:
    return len(self.clips)

  @staticmethod
  def get_joint_pose(pose: torch.Tensor) -> torch.Tensor:
    return pose[AmpMotionLoader.JOINT_POSE_START_IDX : AmpMotionLoader.JOINT_POSE_END_IDX]

  @staticmethod
  def get_joint_pose_batch(poses: torch.Tensor) -> torch.Tensor:
    return poses[:, AmpMotionLoader.JOINT_POSE_START_IDX : AmpMotionLoader.JOINT_POSE_END_IDX]

  @staticmethod
  def get_joint_vel(pose: torch.Tensor) -> torch.Tensor:
    return pose[AmpMotionLoader.JOINT_VEL_START_IDX : AmpMotionLoader.JOINT_VEL_END_IDX]

  @staticmethod
  def get_joint_vel_batch(poses: torch.Tensor) -> torch.Tensor:
    return poses[:, AmpMotionLoader.JOINT_VEL_START_IDX : AmpMotionLoader.JOINT_VEL_END_IDX]

  @staticmethod
  def get_end_pos(pose: torch.Tensor) -> torch.Tensor:
    return pose[AmpMotionLoader.END_POS_START_IDX : AmpMotionLoader.END_POS_END_IDX]

  @staticmethod
  def get_end_pos_batch(poses: torch.Tensor) -> torch.Tensor:
    return poses[:, AmpMotionLoader.END_POS_START_IDX : AmpMotionLoader.END_POS_END_IDX]

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
