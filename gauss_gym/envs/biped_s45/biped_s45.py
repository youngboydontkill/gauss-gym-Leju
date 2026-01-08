import isaacgym  # noqa: F401

import torch

from gauss_gym.envs import LeggedRobot
from gauss_gym.utils import math_utils


class BipedS45(LeggedRobot):
  def _init_buffers(self):
    super()._init_buffers()

    # Left/right gait phase (range: [-pi, pi)).
    self.phase_offset = torch.zeros(
      (self.num_envs, 2), dtype=torch.float32, device=self.device, requires_grad=False
    )
    self.phase_offset[:, 0] = math_utils.torch_rand_float(
      -torch.pi, torch.pi, (self.num_envs, 1), device=self.device
    ).squeeze(1)

    # Initialize right leg in anti-phase (offset by pi).
    self.phase_offset[:, 1] = (
      torch.fmod(self.phase_offset[:, 0] + torch.pi + torch.pi, 2 * torch.pi)
      - torch.pi
    )

    self.phase = self.phase_offset
    self.phase_dt = (
      2
      * torch.pi
      * self.dt
      * math_utils.torch_rand_float(
        *self.cfg['commands']['gait_frequency'], (self.num_envs, 1), device=self.device
      )
    )

  def step(self, actions, actions_mean=None):
    phase_tp1 = self.episode_length_buf.unsqueeze(1) * self.phase_dt + self.phase_offset
    self.phase = torch.fmod(phase_tp1 + torch.pi, 2 * torch.pi) - torch.pi
    return super().step(actions, actions_mean)

  def _reset_buffers(self, env_ids):
    super()._reset_buffers(env_ids)

  def _resample_commands(self, env_ids):
    self.phase_dt[env_ids] = (
      2
      * torch.pi
      * self.dt
      * math_utils.torch_rand_float(
        *self.cfg['commands']['gait_frequency'], (len(env_ids), 1), device=self.device
      )
    )
    super()._resample_commands(env_ids)

  def _reward_feet_phase(self, swing_height: float, track_rew: bool = True):
    """Reward for tracking the desired foot height based on gait phase.

    Note: requires the `foot_height_raycaster_grid` sensor to be available.
    """

    foot_sample_heights = self.sensors['foot_height_raycaster_grid'].get_data()
    foot_sample_heights = torch.min(foot_sample_heights, dim=-1)[0]
    foot_z_left = foot_sample_heights[:, 0]
    foot_z_right = foot_sample_heights[:, 1]

    rz_left = math_utils.get_rz(self.phase[:, 0], swing_height)
    rz_right = math_utils.get_rz(self.phase[:, 1], swing_height)

    if track_rew:
      error_left = torch.square(foot_z_left - rz_left)
      error_right = torch.square(foot_z_right - rz_right)
      total_error = error_left + error_right
      return torch.exp(-total_error / 0.01)

    should_contact_left = rz_left < 0.02
    should_contact_right = rz_right < 0.02

    is_contact_left = self.feet_contact[:, 0]
    is_contact_right = self.feet_contact[:, 1]

    exceed_des_height_left = foot_z_left > rz_left
    exceed_des_height_right = foot_z_right > rz_right

    on_track_left = torch.where(should_contact_left, is_contact_left, exceed_des_height_left)
    on_track_right = torch.where(
      should_contact_right, is_contact_right, exceed_des_height_right
    )

    return 0.5 * (on_track_left + on_track_right).float()
