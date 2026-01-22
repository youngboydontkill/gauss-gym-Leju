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

  def _reward_dof_pos_limits(self, soft_dof_pos_limit: float):
    # Penalize dof positions too close to the limit.
    lower = self.dof_pos_limits[:, 0] + 0.5 * (1 - soft_dof_pos_limit) * (
      self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
    )
    upper = self.dof_pos_limits[:, 1] - 0.5 * (1 - soft_dof_pos_limit) * (
      self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
    )
    return torch.sum(((self.dof_pos < lower) | (self.dof_pos > upper)).float(), dim=-1)

  def _reward_knee_bend(self, knee_joint_names, threshold: float):
    idxs = [self.dof_names.index(name) for name in knee_joint_names if name in self.dof_names]
    if len(idxs) == 0:
      return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
    idxs = torch.tensor(idxs, device=self.device)
    dof_pos = self.dof_pos[:, idxs]
    default_pos = self.default_dof_pos[:, idxs]
    delta = torch.abs(dof_pos - default_pos)
    return torch.sum(torch.relu(delta - threshold), dim=-1)

  def _reward_feet_roll(self):
    roll, _, _ = math_utils.get_euler_xyz(self.get_feet_state()[1].reshape(-1, 4))
    roll = (roll.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (
      2 * torch.pi
    ) - torch.pi
    return torch.sum(torch.square(roll), dim=-1)

  def _reward_feet_yaw_diff(self):
    _, _, yaw = math_utils.get_euler_xyz(self.get_feet_state()[1].reshape(-1, 4))
    yaw = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (
      2 * torch.pi
    ) - torch.pi
    return torch.square((yaw[:, 1] - yaw[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)

  def _reward_feet_yaw_mean(self):
    _, _, yaw = math_utils.get_euler_xyz(self.get_feet_state()[1].reshape(-1, 4))
    yaw = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (
      2 * torch.pi
    ) - torch.pi
    feet_yaw_mean = yaw.mean(dim=-1) + torch.pi * (
      torch.abs(yaw[:, 1] - yaw[:, 0]) > torch.pi
    )
    return torch.square(
      (math_utils.get_euler_xyz(self.base_quat)[2] - feet_yaw_mean + torch.pi)
      % (2 * torch.pi)
      - torch.pi
    )

  def _reward_feet_distance(self, close_feet_threshold: float):
    _, _, base_yaw = math_utils.get_euler_xyz(self.base_quat)
    feet_pos = self.get_feet_state()[0]
    feet_distance = torch.abs(
      torch.cos(base_yaw) * (feet_pos[:, 1, 1] - feet_pos[:, 0, 1])
      - torch.sin(base_yaw) * (feet_pos[:, 1, 0] - feet_pos[:, 0, 0])
    )
    return (feet_distance < close_feet_threshold) * 1.0

  def _reward_feet_splay(self, splay_threshold: float):
    """Penalize walking with feet too far apart laterally.

    Computes the feet separation along the robot's left-right axis (in the base yaw
    frame) and returns `relu(separation - splay_threshold)`.
    """

    _, _, base_yaw = math_utils.get_euler_xyz(self.base_quat)
    feet_pos = self.get_feet_state()[0]
    lateral_sep = torch.abs(
      torch.cos(base_yaw) * (feet_pos[:, 1, 1] - feet_pos[:, 0, 1])
      - torch.sin(base_yaw) * (feet_pos[:, 1, 0] - feet_pos[:, 0, 0])
    )
    return torch.relu(lateral_sep - splay_threshold)

  def _reward_feet_distance_clipped(self, feet_distance_ref: float):
    _, _, base_yaw = math_utils.get_euler_xyz(self.base_quat)
    feet_pos = self.get_feet_state()[0]
    feet_distance = torch.abs(
      torch.cos(base_yaw) * (feet_pos[:, 1, 1] - feet_pos[:, 0, 1])
      - torch.sin(base_yaw) * (feet_pos[:, 1, 0] - feet_pos[:, 0, 0])
    )
    return torch.clip(feet_distance - feet_distance_ref, max=0.0)

  def _reward_s45_pose(self):
    # Keep close to the default pose, with stronger penalties on upper-body.
    weights = []
    for name in self.dof_names:
      if name.startswith('zhead_'):
        weights.append(1.0)
      elif name.startswith('zarm_'):
        weights.append(10.0)
      elif name.startswith('leg_'):
        weights.append(0.)
      else:
        raise ValueError(f'Unknown dof name: {name}')
    weights = torch.tensor(weights, device=self.device)[None]
    pose_error = torch.square(self.dof_pos - self.default_dof_pos)
    return torch.sum(pose_error * weights, dim=1)

  def _reward_dof_vel_head(self):
    # Penalize head joint velocity.
    idxs = [
      self.dof_names.index(name)
      for name in ['zhead_1_joint', 'zhead_2_joint']
      if name in self.dof_names
    ]
    if len(idxs) == 0:
      return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
    return torch.sum(torch.square(self.dof_vel[:, idxs]), dim=-1)

  def _reward_dof_acc_head(self, method: str = 'mean'):
    # Penalize head joint acceleration.
    idxs = [
      self.dof_names.index(name)
      for name in ['zhead_1_joint', 'zhead_2_joint']
      if name in self.dof_names
    ]
    if len(idxs) == 0:
      return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
    if method == 'last':
      dof_acc = self.substep_dof_acc[:, -1, :]
    elif method == 'mean':
      dof_acc = torch.mean(self.substep_dof_acc, dim=1)
    else:
      raise ValueError(f'Invalid method: {method}')
    return torch.sum(torch.square(dof_acc[:, idxs]), dim=-1)

  def _reward_feet_swing(self, swing_period: float):
    phase_left = (self.phase[:, 0] + torch.pi) / (2 * torch.pi)
    phase_right = (self.phase[:, 1] + torch.pi) / (2 * torch.pi)
    left_swing = torch.abs(phase_left) < swing_period
    right_swing = torch.abs(phase_right) < swing_period
    return (left_swing & ~self.feet_contact[:, 0]).float() + (
      right_swing & ~self.feet_contact[:, 1]
    ).float()
  def _reward_action_magnitude(self, clip: float = None, **kwargs):
    """Penalty for excessively large/violent actions."""
    if not hasattr(self, "substep_dof_acc") or self.substep_dof_acc is None:
      return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    dof_acc_mean = torch.mean(self.substep_dof_acc, dim=1)  # (num_envs, num_dofs)
    acc_energy = torch.sum(torch.square(dof_acc_mean), dim=-1)  # (num_envs,)
    if clip is not None:
      acc_energy = torch.clamp(acc_energy, max=float(clip))
    return acc_energy

  def _reward_feet_contact_without_cmd(self, use_stance_mask: bool = True):
    # Reward feet contact when no command is issued.
    if use_stance_mask:
      no_command = self.command_manager.ignore_command_mask(self.scene_manager)
    else:
      no_command = torch.linalg.norm(self.commands[:, :3], dim=1) < 0.01
    contact_count = torch.sum(self.feet_contact.float(), dim=1)
    return contact_count * no_command.float()

  def _reward_no_feet_contact(self, use_stance_mask: bool = True):
    # Penalize no feet contact when commands are non-zero.
    if use_stance_mask:
      nonzero_command = ~self.command_manager.ignore_command_mask(self.scene_manager)
    else:
      nonzero_command = torch.linalg.norm(self.commands[:, :3], dim=1) >= 0.01
    no_contact = torch.sum(self.feet_contact.float(), dim=1) == 0
    return no_contact.float() * nonzero_command.float()

  def _reward_track_default_arm_pos(self, joint_names, alpha: float = 5.0):
    # Reward arm joints staying near default positions.
    idxs = [self.dof_names.index(name) for name in joint_names if name in self.dof_names]
    if len(idxs) == 0:
      return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
    idxs = torch.tensor(idxs, device=self.device)
    joint_pos = self.dof_pos[:, idxs]
    default_pos = self.default_dof_pos[:, idxs]
    sq_dist = torch.sum(torch.square(joint_pos - default_pos), dim=1)
    return torch.exp(-alpha * sq_dist)

  def _reward_feet_stumble(self, multiplier: float):
    return self._reward_stumble(multiplier)

  def _reward_fly(self):
    return self._reward_no_fly()

