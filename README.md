# Air Raid DQN

Trains a Deep Q-Network on the Atari game Air Raid (`ALE/AirRaid-v5`) with Gymnasium and PyTorch. The training script is [`scripts/train.py`](scripts/train.py). When a run finishes, it writes the model weights and a per-episode performance log to `models/`.

## Prerequisites

- Python 3.14, matching the version in the [Pipfile](Pipfile)
- [pipenv](https://pipenv.pypa.io/)

Install pipenv if you do not already have it:

```bash
pip install pipenv
```

## Install

From the project root, install the packages listed in the Pipfile. That includes PyTorch, pandas, NumPy, ALE, and Gymnasium with the Atari extras. The `accept-rom-license` extra downloads the Atari ROMs during install.

```bash
pipenv install
```

To add another package later:

```bash
pipenv install <package-name>
```

## Run training

`scripts/train.py` saves into `models/`, so create that folder before the first run:

```bash
mkdir -p models
pipenv run python scripts/train.py
```

A finished run writes:

- `models/dqn_base_model.pth` — trained network weights
- `models/training_performance_base.csv` — episode length, step count, and reward

## Changing hyperparameters

The training settings are ordinary Python constants near the bottom of [`scripts/train.py`](scripts/train.py), inside the `if __name__ == "__main__":` block. Change a value, save the file, and run the script again. No other files need to be edited.

| Constant | Default | What it controls |
| --- | --- | --- |
| `LR` | `1e-4` | Adam learning rate |
| `GAMMA` | `0.99` | Discount factor for future rewards |
| `BATCH_SIZE` | `32` | Transitions sampled from the replay buffer each update |
| `EPS_START` | `1.0` | Starting exploration rate (fully random actions) |
| `EPS_END` | `0.05` | Lowest exploration rate after decay |
| `EPS_DECAY` | `50000` | How quickly epsilon falls from `EPS_START` to `EPS_END`, in environment steps |
| `MEMORY_CAPACITY` | `100000` | Maximum transitions stored in the replay buffer |
| `TARGET_UPDATE_FREQ` | `1000` | How often the target network copies the online network, in steps |
| `NUM_EPISODES` | `500` | Number of episodes to train |

For example, a shorter trial run is `NUM_EPISODES = 50`. A more aggressive learner is a larger `LR`, such as `1e-3`.
