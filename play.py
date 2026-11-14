import argparse
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        # input_shape is (4, 84, 84)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # flattened size for 84x84 input: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def make_env():
    env = gym.make("ALE/AirRaid-v5", render_mode="human", frameskip=1)
    env = AtariPreprocessing(
        env=env,
        screen_size=84,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=True,
    )
    env = FrameStackObservation(env, stack_size=4)
    return env


def main():
    parser = argparse.ArgumentParser(description="Play one Air Raid episode with a trained DQN.")
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=Path("dqn_model.pth"),
        help="Path to a saved DQN .pth file (default: dqn_model.pth)",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Model file not found: {args.model}")

    env = make_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = DQN((4, 84, 84), env.action_space.n).to(device)
    net.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    net.eval()

    state, _ = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        state_tensor = torch.tensor(
            np.array(state), dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            action = int(net(state_tensor).argmax(dim=1).item())

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1

        if terminated or truncated:
            break

    env.close()
    print(f"Episode finished. Steps: {steps} | Reward: {total_reward:.1f}")


if __name__ == "__main__":
    main()
