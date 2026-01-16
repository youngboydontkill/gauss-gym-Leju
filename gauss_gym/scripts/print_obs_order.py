import argparse
import importlib
import pathlib
import types
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

import gauss_gym
from gauss_gym.utils import config


TASK_CLASS_MAP = {
  'biped_s45': 'gauss_gym.envs.biped_s45.biped_s45:BipedS45',
  'biped_s45_vision': 'gauss_gym.envs.biped_s45.biped_s45:BipedS45',
  't1': 'gauss_gym.envs.t1.t1:T1',
  't1_vision': 'gauss_gym.envs.t1.t1:T1',
  'a1': 'gauss_gym.envs.base.legged_robot:LeggedRobot',
  'a1_vision': 'gauss_gym.envs.base.legged_robot:LeggedRobot',
  'go1': 'gauss_gym.envs.base.legged_robot:LeggedRobot',
  'go1_vision': 'gauss_gym.envs.base.legged_robot:LeggedRobot',
  'anymal_c': 'gauss_gym.envs.anymal_c.anymal:Anymal',
  'anymal_c_vision': 'gauss_gym.envs.anymal_c.anymal:Anymal',
}


DOF_OBS_NAMES = {
  'dof_pos',
  'dof_vel',
  'actions',
  'stiffness',
  'damping',
  'motor_strength',
  'motor_error',
}


def _load_class(path: str):
  module_name, class_name = path.split(':', 1)
  module = importlib.import_module(module_name)
  return getattr(module, class_name)


def _resolve_task_class(cfg: config.Config, explicit: str):
  if explicit:
    return _load_class(explicit)
  task_name = cfg['task'] if 'task' in cfg else cfg.task
  if task_name in TASK_CLASS_MAP:
    return _load_class(TASK_CLASS_MAP[task_name])
  raise ValueError(
    f"Unknown task '{task_name}'. Provide --task-class module:Class."
  )


def _flatten_shape(shape: Tuple[int, ...]) -> int:
  return int(np.prod(shape))


def _resolve_urdf_path(cfg: config.Config) -> pathlib.Path:
  asset_key = 'asset.file'
  if asset_key not in cfg.flat:
    return None
  asset_path = cfg['asset']['file']
  asset_path = asset_path.replace('{GAUSS_GYM_ROOT_DIR}', gauss_gym.GAUSS_GYM_ROOT_DIR)
  return pathlib.Path(asset_path)


def _parse_urdf_dof_names(urdf_path: pathlib.Path) -> List[str]:
  if urdf_path is None or not urdf_path.exists():
    return []
  import xml.etree.ElementTree as ET

  tree = ET.parse(urdf_path)
  root = tree.getroot()
  dof_names = []
  for joint in root.findall('joint'):
    joint_type = joint.get('type', 'fixed')
    if joint_type == 'fixed':
      continue
    joint_name = joint.get('name')
    if joint_name:
      dof_names.append(joint_name)
  return dof_names


def _print_dof_order(env, obs_name: str) -> None:
  if not hasattr(env, 'dof_names'):
    return
  if obs_name == 'actions' and hasattr(env, 'action_dof_indices'):
    idxs = env.action_dof_indices.detach().cpu().tolist()
    names = [env.dof_names[i] for i in idxs]
  else:
    names = list(env.dof_names)
  print(f"    dof_order({obs_name}) = {names}")


def _iter_obs_groups(obs_dict: Dict[str, Dict[str, Any]],
                     groups: Iterable[str]):
  for group in groups:
    if group not in obs_dict:
      raise KeyError(f"Observation group '{group}' not found in obs_dict")
    yield group, obs_dict[group]


def main():
  parser = argparse.ArgumentParser(
    description='Print the gym observation order and flat indices.'
  )
  parser.add_argument('--config', type=str, default='', help='Path to config yaml')
  parser.add_argument(
    '--run', type=str, default='', help='Run directory under logs/'
  )
  parser.add_argument(
    '--task-class', type=str, default='', help='Override task class module:Class'
  )
  parser.add_argument(
    '--obs-group', type=str, default='policy', help='Observation group to print'
  )
  parser.add_argument('--all-groups', action='store_true', help='Print all groups')
  parser.add_argument('--num-envs', type=int, default=1)
  parser.add_argument('--headless', action='store_true', default=True)
  parser.add_argument(
    '--keep-noise-latency', action='store_true', help='Do not disable noise/latency'
  )
  parser.add_argument(
    '--keep-remote-scenes',
    action='store_true',
    help='Do not override scene repo_id to local path',
  )
  parser.add_argument(
    '--no-sim',
    action='store_true',
    help='Print order from config only (no IsaacGym sim)',
  )
  args = parser.parse_args()

  if args.run:
    log_root = pathlib.Path(gauss_gym.GAUSS_GYM_ROOT_DIR) / 'logs'
    cfg = config.Config.load(log_root / args.run / 'train_config.yaml')
    cfg = cfg.update({'runner.load_run': args.run})
  elif args.config:
    cfg = config.Config.load(args.config)
  else:
    raise ValueError('Provide --config or --run')

  cfg = cfg.update({'headless': args.headless})
  cfg = cfg.update({'env.num_envs': args.num_envs})
  cfg = cfg.update({'multi_gpu': False})

  if not args.keep_remote_scenes:
    local_scene_root = pathlib.Path(gauss_gym.GAUSS_GYM_ROOT_DIR) / 'scenes'
    scene_repo_key = 'terrain.scenes.iphone_data.repo_id'
    if scene_repo_key in cfg.flat:
      cfg = cfg.update({scene_repo_key: f'local:{local_scene_root}'})

  if not args.keep_noise_latency:
    flat_keys = cfg.flat.keys()
    for group_name in cfg['observations'].keys():
      noise_key = f'observations.{group_name}.add_noise'
      latency_key = f'observations.{group_name}.add_latency'
      if noise_key in flat_keys:
        cfg = cfg.update({noise_key: False})
      if latency_key in flat_keys:
        cfg = cfg.update({latency_key: False})

  if args.no_sim:
    from gauss_gym.utils import observation_groups

    groups_cfg = cfg['observations']
    urdf_path = _resolve_urdf_path(cfg)
    dof_names = _parse_urdf_dof_names(urdf_path)
    groups = list(groups_cfg.keys()) if args.all_groups else [args.obs_group]
    for group_name in groups:
      if group_name not in groups_cfg:
        raise KeyError(f"Observation group '{group_name}' not found in config")
      print(f"\n[{group_name}]")
      for obs_symbol in groups_cfg[group_name]['observations']:
        obs = getattr(observation_groups, obs_symbol)
        print(f"  {obs.name}")
        if obs.name in DOF_OBS_NAMES and dof_names:
          print(f"    dof_order({obs.name}) = {dof_names}")
    return

  import isaacgym  # noqa: F401

  cfg = types.MappingProxyType(dict(cfg))
  task_class = _resolve_task_class(cfg, args.task_class)
  env = task_class(cfg=cfg)

  obs_dict, _ = env.reset()

  groups = list(obs_dict.keys()) if args.all_groups else [args.obs_group]
  for group_name, group_obs in _iter_obs_groups(obs_dict, groups):
    print(f"\n[{group_name}]")
    flat_offset = 0
    for obs_name, obs_tensor in group_obs.items():
      shape = tuple(obs_tensor.shape[1:])
      if len(shape) == 0:
        obs_size = 1
      else:
        obs_size = _flatten_shape(shape)
      end = flat_offset + obs_size
      print(
        f"  {obs_name:<24} shape={shape!s:<16} "
        f"flat=[{flat_offset},{end}) size={obs_size}"
      )
      if obs_name in DOF_OBS_NAMES:
        _print_dof_order(env, obs_name)
      flat_offset = end

    print(f"  total_flat_dim={flat_offset}")


if __name__ == '__main__':
  main()
