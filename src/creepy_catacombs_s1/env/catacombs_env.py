# src/creepy_catacombs/env/catacombs_env.py

import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces
from typing import Optional, Any

from creepy_catacombs_s1.param.env_params import EnvParams
from creepy_catacombs_s1.map.generator import generate_tunnel, place_zombies, move_zombies

# If you'd like to reference your separate renderer:
from creepy_catacombs_s1.render.pygame_render import CreepyCatacombsPygameRenderer

class CreepyCatacombsEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10
    }

    def __init__(
        self, 
        render_mode: Optional[str] = None,
        fixed_env_params: EnvParams = EnvParams(),
        **kwargs: Any
    ):
        """
        Creepy Catacombs environment for the local AI Olympiad in Hungary.
        - render_mode: 'human' or 'rgb_array'.
        - fixed_env_params: EnvParams, environment configuration.
        """
        super().__init__()
        self.params = fixed_env_params
        self.render_mode = render_mode

        # Extract fields from params
        self.width = self.params.width
        self.height = self.params.height
        self.tile_size = self.params.tile_size
        self.corridor_width = self.params.corridor_width
        self.n_zombies = self.params.n_zombies

        self.observation_space = spaces.Discrete(self.width * self.height)
        self.action_space = spaces.Discrete(4)

        self.grid = None
        self.agent_pos = (0, 0)
        self.goal_pos = (0, 0)
        self.zombie_positions = []

        # We'll store a separate renderer instance if needed
        if render_mode in ["human", "rgb_array"]:
            self.renderer = CreepyCatacombsPygameRenderer()
        else:
            self.renderer = None

        self.reset()

    def _get_obs(self):
        r, c = self.agent_pos
        return r * self.width + c

    def _get_info(self):
        return {
            "agent_pos": self.agent_pos,
            "zombies": self.zombie_positions
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed if seed is not None else 42)

        # Generate the map & place zombies
        self.grid = generate_tunnel(self.height, self.width, self.corridor_width)
        self.zombie_positions = place_zombies(self.grid, self.n_zombies)

        # Find start & goal
        start_cells = np.argwhere(self.grid == 1)
        goal_cells = np.argwhere(self.grid == 2)
        if len(start_cells) == 0:
            start_cells = np.array([[self.height - 1, 0]])
        if len(goal_cells) == 0:
            goal_cells = np.array([[0, 0]])
        (start_r, start_c) = start_cells[0]
        (goal_r, goal_c) = goal_cells[0]
        self.agent_pos = (start_r, start_c)
        self.goal_pos = (goal_r, goal_c)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        reward = -1
        terminated = False

        r, c = self.agent_pos
        nr, nc = r, c
        if action == 0:
            nr, nc = r - 1, c   # up
        elif action == 1:
            nr, nc = r, c + 1   # right
        elif action == 2:
            nr, nc = r + 1, c   # down
        elif action == 3:
            nr, nc = r, c - 1   # left

        # Check bounds or walls/zombies
        if not (0 <= nr < self.height and 0 <= nc < self.width):
            reward = -10
            terminated = True
        else:
            cell_val = self.grid[nr, nc]
            if cell_val == -1:
                reward = -10
                terminated = True
            elif (nr, nc) in self.zombie_positions:
                reward = -10
                terminated = True
            elif cell_val == 2:
                self.agent_pos = (nr, nc)
                reward = 100
                terminated = True
            else:
                self.agent_pos = (nr, nc)

        # Move zombies if not done
        if not terminated:
            self.zombie_positions = move_zombies(self.grid, self.zombie_positions)
            if self.agent_pos in self.zombie_positions:
                reward = -10
                terminated = True

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, False, info

    def render(self):
        """Just calls the separate renderer if available."""
        if self.renderer is None:
            return
        return self.renderer.render(
            env=self,
            render_mode=self.render_mode
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    def render_q_values_arrows(self, Q, scale=0.2):
        """Pass Q-values to the external renderer if needed."""
        if self.renderer:
            return self.renderer.render_q_values_arrows(self, Q, scale=scale)


