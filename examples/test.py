import numpy as np
import random
import time
import pygame
import gymnasium as gym
import creepy_catacombs_s1  # assume this triggers environment + renderer registration
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        state, info = env.reset()
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


if __name__ == "__main__":
    env = gym.make(
        "CreepyCatacombs-v0",
        render_mode="human",  # or "human" if you'd like to see step-by-step
    )
    env.reset(seed=42)
    env.render()
    time.sleep(100)
    env.close()

    # Q_mc, rewards_mc = run_monte_carlo(env, episodes=10000, gamma=0.99, epsilon=0.1)
    # print("MC training done!")

    # # Watch a final run
    # state, info = env.reset(seed=42)
    # done = False
    # cumulative_reward = 0
    # while not done:
    #     action = np.argmax(Q_mc[state])  # greedy
    #     state, reward, done, trunc, _info = env.step(action)
    #     cumulative_reward += reward
    # print("Final run (MC) total reward =", cumulative_reward)

    # # Now we want an arrow overlay of Q-values as an image.
    # # We'll call the separate renderer:
    # #  - env.renderer is the Pygame class
    # #  - pass return_surface=True to get a surface back

    # surface = env.unwrapped.renderer.render_q_values_arrows(
    #     env.unwrapped,  # pass the real environment, not the wrapper
    #     Q_mc,
    # )


    # # Finally, display with matplotlib
    # plt.imshow(surface)
    # plt.axis('off')  # optional, to hide axis ticks
    # plt.title('Q-Value Arrows')
    # plt.show()

    # env.close()
