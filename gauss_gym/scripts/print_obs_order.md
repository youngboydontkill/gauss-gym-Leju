# Print gym observation order

This helper prints the observation group order and flat index ranges as produced by the environment.

## Usage

From repo root:

```bash
python -m gauss_gym.scripts.print_obs_order --config gauss_gym/envs/biped_s45/config_vision.yaml
```

Print all groups:

```bash
python -m gauss_gym.scripts.print_obs_order --config gauss_gym/envs/biped_s45/config_vision.yaml --all-groups
```

Config-only mode (no IsaacGym sim, no scene downloads):

```bash
python -m gauss_gym.scripts.print_obs_order --config gauss_gym/envs/biped_s45/config_vision.yaml --no-sim
```

Use a past run config:

```bash
python -m gauss_gym.scripts.print_obs_order --run <RUN_NAME>
```

By default it forces local scenes to avoid Hugging Face downloads. To keep remote scenes:

```bash
python -m gauss_gym.scripts.print_obs_order --config gauss_gym/envs/biped_s45/config_vision.yaml \
  --keep-remote-scenes
```

Override task class if needed:

```bash
python -m gauss_gym.scripts.print_obs_order --config <path/to/config.yaml> \
  --task-class gauss_gym.envs.biped_s45.biped_s45:BipedS45
```
