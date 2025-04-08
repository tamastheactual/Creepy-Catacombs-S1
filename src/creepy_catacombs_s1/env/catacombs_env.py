import gymnasium as gym
import numpy as np
import random
import logging
from gymnasium import spaces
from typing import Optional, Any, List, Tuple

from creepy_catacombs_s1.param.env_params import EnvParams
from creepy_catacombs_s1.map.generator import generate_tunnel, place_zombies
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
        verbosity: int = logging.WARNING,
        **kwargs: Any
    ):
        """
        Creepy Catacombs environment for the local AI Olympiad in Hungary.
        - render_mode: 'human' or 'rgb_array'.
        - fixed_env_params: EnvParams, environment configuration.
        - verbosity: Logging verbosity level (e.g., logging.DEBUG, logging.WARNING).
        """
        super().__init__()
        self.params = fixed_env_params
        self.render_mode = render_mode
        self.verbosity = verbosity
        self.kwargs = kwargs
        
        logging.basicConfig(
        level=self.verbosity,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("pygame").setLevel(logging.WARNING)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.verbosity)

        self.width = self.params.width
        self.height = self.params.height
        self.tile_size = self.params.tile_size
        self.logger.info(f"Tile size: {self.tile_size}")
        self.logger.info(f"Width: {self.width}, Height: {self.height}")
        self.corridor_width = self.params.corridor_width
        self.n_zombies = self.kwargs.get("n_zombies", self.params.n_zombies)
        self.zombie_movement = self.kwargs.get("zombie_movement", self.params.zombie_movement)

        self.observation_space = spaces.Discrete(self.width * self.height)
        self.action_space = spaces.Discrete(4)

        self.grid = None
        self.agent_pos = (0, 0)
        self.goal_pos = (0, 0)
        self.zombie_positions = []
        self.plotholes = []

        if render_mode in self.metadata["render_modes"]:
            self.logger.info(f"Initializing renderer with mode: {render_mode}")
            self.renderer = CreepyCatacombsPygameRenderer(verbosity=self.verbosity)
        else:
            self.renderer = None

        self.reset()

    def _get_obs(self):
        r, c = self.agent_pos
        return r * self.width + c

    def _get_info(self):
        return {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "grid": self.grid,
            "grid_shape": self.grid.shape,
            "corridor_width": self.corridor_width,
            "n_zombies": self.n_zombies,
            "plotholes": self.plotholes,
            "zombies": self.zombie_positions
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed if seed is not None else 42)

        self.logger.info("Resetting environment...")
        self.grid = generate_tunnel(
            self.height, 
            self.width, 
            self.corridor_width, 
            verbosity=self.verbosity
        )
        self.original_grid = self.grid.copy()
        self.zombie_positions = place_zombies(self.grid, self.n_zombies, verbosity=self.verbosity)
        self.plotholes = np.argwhere(self.grid == -2)
        self.logger.debug("Generated grid:\n%s", self.grid)
        self.logger.debug("Zombie positions: %s", self.zombie_positions)

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

        self.logger.info(f"Agent start position: {self.agent_pos}")
        self.logger.info(f"Goal position: {self.goal_pos}")
        self.logger.info(f"Plotholes: {self.plotholes}")
        self.logger.info(f"Zombies: {self.zombie_positions}")

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

        if self._is_illegal_move(nr, nc):
            reward = -2
            self.logger.debug("Illegal move attempted to (%d, %d).", nr, nc)
        else:
            cell_val = self.grid[nr, nc]
            if cell_val == -2: # Plothole
                reward = -5
                terminated = True
                self.logger.debug("Agent fell into a plothole at (%d, %d).", nr, nc)
            elif cell_val == 2: # Goal
                reward = 10
                terminated = True
                self.logger.info("Agent reached the goal at (%d, %d).", nr, nc)
            else:
                # Valid move
                self.agent_pos = (nr, nc)

        if not terminated:
            self._move_zombies()
            if self.agent_pos in self.zombie_positions:
                reward = -5  # Caught by a zombie
                terminated = True
                self.logger.debug("Agent was caught by a zombie at %s.", self.agent_pos)

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, False, info
    
    def _is_illegal_move(self, nr: int, nc: int) -> bool:
        if not (0 <= nr < self.height and 0 <= nc < self.width):
            return True
        if self.grid[nr, nc] == -1:  # Wall
            return True
        return False
    
    def _move_zombies(self):
        """
        Moves zombies based on the selected movement strategy ('random' or 'towards_player').
        Updates zombie positions in the environment.
        """
        self.logger.debug("Moving zombies with strategy: %s", self.zombie_movement)
        height, width = self.grid.shape
        new_positions: List[Tuple[int, int]] = []
        directions: List[Tuple[int, int]] = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]  # Stay, Right, Left, Down, Up

        random.shuffle(self.zombie_positions)

        for (zr, zc) in self.zombie_positions:
            if self.grid[zr, zc] == 3:
                self.grid[zr, zc] = self.original_grid[zr, zc]

            nr, nc = zr, zc

            if self.zombie_movement == "random":
                random.shuffle(directions)
                for dr, dc in directions:
                    temp_nr, temp_nc = zr + dr, zc + dc
                    if (0 <= temp_nr < height and 0 <= temp_nc < width and
                        self.grid[temp_nr, temp_nc] not in [-1, -2] and
                        (temp_nr, temp_nc) not in new_positions):
                        nr, nc = temp_nr, temp_nc
                        break
            elif self.zombie_movement == "towards_player":
                ar, ac = self.agent_pos
                valid_moves = []
                fallback_moves = []
                for dr, dc in directions:
                    temp_nr, temp_nc = zr + dr, zc + dc
                    if (0 <= temp_nr < height and 0 <= temp_nc < width and
                        self.grid[temp_nr, temp_nc] not in [-1, -2] and
                        (temp_nr, temp_nc) not in new_positions):
                        distance = abs(temp_nr - ar) + abs(temp_nc - ac)
                        fallback_moves.append((temp_nr, temp_nc))
                        if distance < abs(zr - ar) + abs(zc - ac):
                            valid_moves.append((distance, temp_nr, temp_nc))
                if valid_moves:
                    _, nr, nc = min(valid_moves, key=lambda x: x[0])
                elif fallback_moves:
                    nr, nc = random.choice(fallback_moves)

            new_positions.append((nr, nc))

        for (zr, zc) in new_positions:
            self.grid[zr, zc] = 3

        self.zombie_positions = new_positions
        self.logger.debug("Zombies moved to new positions: %s", self.zombie_positions)

        return new_positions

    def render(self):
        """Just calls the separate renderer if available."""
        if self.renderer is None:
            self.logger.warning("Render called, but no renderer is initialized.")
            return
        return self.renderer.render(
            env=self,
            render_mode=self.render_mode
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.logger.info("Renderer closed.")

    def render_q_values(self, Q):
        """Pass Q-values to the external renderer if needed."""
        print("Rendering Q-values...")
        surface = self.renderer.render_q_values_arrows(self, Q, verbosity=self.verbosity)
        rgb_image = self.renderer.display_surface_with_matplotlib(surface)            
        return rgb_image
    
    def render_values(self, V):
        """Pass V-values to the external renderer if needed."""
        print("Rendering V-values...")
        surface = self.renderer.render_v_values(self, V, verbosity=self.verbosity)
        rgb_image = self.renderer.display_surface_with_matplotlib(surface)
        return rgb_image
        
    def render_optimal_path(self, policy):
        """Pass policy to the external renderer if needed."""
        print("Rendering optimal path...")
        surface = self.renderer.render_optimal_path(self, policy, verbosity=self.verbosity)
        rgb_image = self.renderer.display_surface_with_matplotlib(surface)
        return rgb_image
    
    