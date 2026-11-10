import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ale_py
import random
from collections import deque
import math
import pandas as pd
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
MODEL_SAVE_DIR = ROOT / "models"

gym.register_envs(ale_py)

def setup_environment(env_name='ALE/AirRaid-v5'):
    # Create the base environment
    env = gym.make(env_name, frameskip=1)

    # Apply Atari preprocessing
    # Convert frames to grasycale, resize, normalize, and skip frames
    env = AtariPreprocessing(
        env=env,
        screen_size=84,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=True
    )

    # Stack the last 4 frames to give the agent a sense of motion
    env = FrameStackObservation(env, stack_size=4)

    return env


class ReplayBuffer:
    def __init__(self, capacity):
        # automatically removes the last element when it reaches 'capacity'
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Saves a transition"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Randomly samples a batch of transitions for training"""
        batch = random.sample(self.buffer, batch_size)

        # unzip the batch into separate tuples
        states, actions, rewards, next_states, dones = zip(*batch)

        # convert to pytorch tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    
    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        # input_shape is (4, 84, 84)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # the flattened size after the 3 conv layers
        # for 84x84 input, the spatial dimensions go: 84 -> 20 -> 9 -> 7
        # flattened size = 64 channels * 7 height * 7 width = 3136

        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, num_actions)
    
    def forward(self, x):
        # apply convolutions with ReLU activation
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # flatten the output for the linear layers
        x = x.view(x.size(0), -1)

        # apply full connected layers
        x = torch.relu(self.fc1(x))

        return self.fc2(x) # return logits as Q-values

class DQNAgent:
    def __init__(
        self, input_shape,
        num_actions, lr,
        gamma, batch_size,
        eps_start, eps_end, 
        eps_decay):
        # store hyperparameters
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        # internal step counter for epsilon decay
        self.steps_done = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize networks and optimizer
        self.online_net = DQN(input_shape, num_actions).to(self.device)
        self.target_net = DQN(input_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

    
    def select_action(self, state):
        """Epsilon-greedy action selection with built-in decay"""

        # calculate the current epsilon based on how many steps we have taken
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        # exploration
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        # exploitation
        else:
            state_tensors = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.online_net(state_tensors)
            return q_values.argmax().item()
    

    def optimize_model(self, memory):
        """Samples from memory and updates the online network"""
        if len(memory) < self.batch_size:
            return None
        
        # get a batch of 32 random samples
        states, actions, rewards, next_states, dones = memory.sample(self.batch_size)

        # Move all data to the hardware (gpu/cpu)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # predict the current Q-value
        current_q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # calculate the target Q-value using the bellman equation
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # calculate the loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_value_(self.online_net.parameters(), 100)

        #adjust weights
        self.optimizer.step()

        return loss.item()

if __name__ == "__main__":
    # initialize hyperparameters
    LR = 1e-4
    GAMMA = 0.99
    BATCH_SIZE =  32
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 50000

    # training loop hyperparameters
    MEMORY_CAPACITY = 100000
    TARGET_UPDATE_FREQ = 1000
    NUM_EPISODES = 500

    # environment initialization
    env = setup_environment("ALE/AirRaid-v5")

    input_shape = (4, 84, 84)
    num_actions = env.action_space.n

    agent = DQNAgent(
        input_shape=input_shape,
        num_actions=num_actions,
        lr=LR,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        eps_start=EPS_START,
        eps_end=EPS_END,
        eps_decay=EPS_DECAY
    )

    memory = ReplayBuffer(MEMORY_CAPACITY)

    performance_data = []

    print("Starting training...")
    
    for episode in range(NUM_EPISODES):
        state, info = env.reset()

        episode_reward = 0
        episode_length = 0

        while True:
            # select action and interact with the environment
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # save experience int the replay buffer
            memory.push(state, action, reward, next_state, done)

            # move to the next state
            state = next_state

            # accumulate stats
            episode_reward += reward
            episode_length += 1

            # sample and optimize the model
            agent.optimize_model(memory)

            # periodically update the target network to match the online network
            if agent.steps_done % TARGET_UPDATE_FREQ == 0:
                agent.target_net.load_state_dict(agent.online_net.state_dict())
            
            if done:
                break
        
        # end of episode summary
        print(f"Episode: {episode + 1}/{NUM_EPISODES} | "
              f"Steps: {episode_length} | "
              f"Total Steps: {agent.steps_done} | "
              f"Reward: {episode_reward:.1f}")
        
        performance_data.append({
            "Episode": episode + 1,
            "Episode Length": episode_length,
            "Total Steps": agent.steps_done,
            "Reward": episode_reward
        })
    
    print("Training complete. Saving results...")

    # save the trained model
    torch.save(agent.online_net.state_dict(), MODEL_SAVE_DIR / "dqn_base_model.pth")

    # save training performance data
    df = pd.DataFrame(performance_data)
    df.to_csv(MODEL_SAVE_DIR / "training_performance_base.csv", index=False)

    env.close()
    print(f"Model saved to '{MODEL_SAVE_DIR / "dqn_base_model.pth"}' and performance saved to '{MODEL_SAVE_DIR / "training_performance_base.csv"}'.")