import numpy as np
import random
import time
import pygame
import logging
import gymnasium as gym
import creepy_catacombs_s1  # assume this triggers environment + renderer registration
from tqdm import tqdm
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.WARNING)

def epsilon_greedy(Q, state, epsilon, n_actions=4):
    """Epsilon-greedy action selection given Q."""
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        return np.argmax(Q[state])

def run_monte_carlo(env, episodes=3000, gamma=0.99, epsilon=0.1):
    # Q[state, action]
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    returns_dict = {}  # (state, action) -> list of returns

    all_rewards = []

    for ep in tqdm(range(episodes)):
        # 1) Generate an episode
        state, info = env.reset(seed=42)
        done = False
        episode = []
        while not done:
            action = epsilon_greedy(Q, state, epsilon, n_actions)
            next_state, reward, done, truncated, _info = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        # 2) Compute returns backward
        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            if (s, a) not in visited:  
                visited.add((s, a))
                if (s, a) not in returns_dict:
                    returns_dict[(s, a)] = []
                returns_dict[(s, a)].append(G)
                # Update Q
                Q[s, a] = np.mean(returns_dict[(s,a)])

        total_ep_return = sum([x[2] for x in episode])
        all_rewards.append(total_ep_return)

    return Q, all_rewards


def monte_carlo_value_estimation(env, episodes=10000, gamma=0.99, epsilon=0.1):
    """
    Perform Monte Carlo Value Estimation to compute the value function.
    Args:
        env: The environment.
        episodes: Number of episodes to run.
        gamma: Discount factor.
        epsilon: Exploration rate for epsilon-greedy policy.
    Returns:
        V: Estimated value function (numpy array).
    """
    n_states = env.observation_space.n
    V = np.zeros(n_states)  # Initialize value function
    returns = {s: [] for s in range(n_states)}  # Store returns for each state
    observes = []
    for ep in tqdm(range(episodes)):
        # Generate an episode
        state, info = env.reset(seed = 42)
        episode = []
        done = False

        while not done:
            # Epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax([V[state]])  # Exploit (use current value estimates)

            next_state, reward, done, truncated, _ = env.step(action)
            observes.append(next_state)
            episode.append((state, reward))
            state = next_state

        # Compute returns and update value estimates
        G = 0
        visited_states = set()
        for t in reversed(range(len(episode))):
            state, reward = episode[t]
            G = gamma * G + reward  # Compute return

            # First-visit Monte Carlo: Update only the first time a state is visited
            if state not in visited_states:
                visited_states.add(state)
                returns[state].append(G)
                V[state] = np.mean(returns[state])  # Update value estimate

    print(np.unique(np.array(observes), return_counts=True))
    return V

def sarsa(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Perform SARSA to compute the optimal Q-values.
    Args:
        env: The environment.
        episodes: Number of episodes to run.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Exploration rate for epsilon-greedy policy.
    Returns:
        Q: Optimal Q-value table (numpy array).
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize Q-values
    Q = np.zeros((n_states, n_actions))

    for ep in range(episodes):
        # Reset the environment
        state, info = env.reset(seed=42)
        # Choose an action using epsilon-greedy policy
        action = epsilon_greedy(Q, state, epsilon, n_actions)

        done = False
        while not done:
            # Take the action and observe the next state and reward
            next_state, reward, done, truncated, _ = env.step(action)

            # Choose the next action using epsilon-greedy policy
            next_action = epsilon_greedy(Q, next_state, epsilon, n_actions)

            # Update Q-value using the SARSA update rule
            Q[state, action] += alpha * (
                reward + gamma * Q[next_state, next_action] - Q[state, action]
            )

            # Move to the next state and action
            state = next_state
            action = next_action

    return Q

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def extract_features(info, env):
    """
    Extract features from the environment's current state using the info dictionary.
    Features:
    - Distance to the nearest plothole.
    - Distance to the goal.
    - Agent's normalized position (row, col).
    """
    agent_pos = np.array(info["agent_pos"])
    goal_pos = np.array(info["goal_pos"])
    plotholes = np.array(info["plotholes"])
    height = info["grid_shape"][0]
    width = info["grid_shape"][1]
    # Distance to the goal
    dist_to_goal = np.linalg.norm(agent_pos - goal_pos)

    # Distance to the nearest plothole
    if len(plotholes) > 0:
        dist_to_plothole = np.min(np.linalg.norm(plotholes - agent_pos, axis=1))
    else:
        dist_to_plothole = float('inf')

    # Normalize agent position
    normalized_pos = agent_pos / np.array([height, width])

    return np.array([dist_to_goal, *normalized_pos])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)
    
def train_dqn(env, episodes=1000, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, batch_size=64, lr=0.001):
    input_dim = 3  # Number of features (distance to goal, distance to plothole, normalized position)
    output_dim = env.action_space.n  # Number of actions

    # Initialize DQN and target network
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=10000)

    rewards_per_episode = []

    for ep in range(episodes):
        env.reset(seed=42)
        state_features = extract_features(info, env)
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state_features).unsqueeze(0)
                    q_values = policy_net(state_tensor)
                    action = torch.argmax(q_values).item()  # Exploit

            # Take action in the environment
            next_state, reward, done, truncated, next_info = env.step(action)
            next_state_features = extract_features(next_info, env)

            # Store transition in replay buffer
            replay_buffer.push(state_features, action, reward, next_state_features, done)

            state_features = next_state_features
            total_reward += reward

            # Train the network if the replay buffer has enough samples
            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                # Convert to tensors
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                # Compute Q-values and targets
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    targets = rewards + gamma * next_q_values * (1 - dones)

                # Compute loss
                loss = nn.MSELoss()(q_values, targets)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Update target network periodically
        if ep % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        rewards_per_episode.append(total_reward)
        print(f"Episode {ep + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    return policy_net, rewards_per_episode

if __name__ == "__main__":
    env = gym.make(
        "CreepyCatacombs-v0",
        render_mode="human",
        verbosity=logging.WARNING,
        n_zombies=2,
        zombie_movement="random",
    )
    env.reset(seed=42)
    
    action_sequence = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    for action in action_sequence:
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        time.sleep(1)
        if done:
            break
    env.close()
    #Q_mc, rewards_mc = run_monte_carlo(env, episodes=10000, gamma=0.99, epsilon=0.1)
    #print("MC training done!")

    # Now we want an arrow overlay of Q-values as an image.
    # We'll call the separate renderer:
    #  - env.renderer is the Pygame class
    #  - pass return_surface=True to get a surface back

    # Convert Q (NumPy array) to a dictionary
    #env.reset()
    #Q_dict = {(state, action): Q_mc[state, action] for state in range(Q_mc.shape[0]) for action in range(Q_mc.shape[1])}

    #surface = env.unwrapped.render_q_values(
    #    Q_dict,
    #)
    # V = monte_carlo_value_estimation(env, episodes=10000, gamma=0.99, epsilon=0.25)
    # V_dict = {(state): V[state] for state in range(V.shape[0])}

    # surface = env.unwrapped.render_values(
    #     V_dict,
    # )
    # # Print the results
    # print("Optimal Value Function:")
    # print(V.reshape(env.unwrapped.height, env.unwrapped.width))

    # # Visualize the policy
    # env.render()
    # env.close()
    # Q = sarsa(env, episodes=10000, alpha=0.2, gamma=0.99, epsilon=0.15)
    # env.reset(seed=42)
    # Q_dict = {(state, action): Q[state, action] for state in range(Q.shape[0]) for action in range(Q.shape[1])}
    
    # rgb_array = env.unwrapped.render_q_values(
    #     Q_dict,
    # )
    # print("size of rgb_array", rgb_array.shape) 
    # policy = np.argmax(Q, axis=1)
    
    # env.reset(seed=42)
    # # render optimal path
    # rgb_array_2 = env.unwrapped.render_optimal_path(
    #     policy,
    # )
    
    # # rgb array is a numpy array of shape (height, width, 3)
    # # lets increase the size of it so it matches original pygame window
    # # and show it
    # # remove whites from the image
    
    # plt.figure(figsize=(20, 20))  # Adjust the figsize to make the image larger

    # # Display the rgb_array
    # plt.imshow(rgb_array_2, interpolation='nearest')  # Use 'nearest' to avoid smoothing
    # plt.title("Q-values")
    # plt.axis('off')  # Turn off the axis

    # # Adjust the layout to remove white borders
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # plt.show()
    policy_net, rewards = train_dqn(env, episodes=3000)

    # Save the trained model
    torch.save(policy_net.state_dict(), "dqn_model.pth")

    # Plot rewards
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Rewards")
    plt.show()