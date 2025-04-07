# src/creepy_catacombs/env/catacombs_pygame_renderer.py

import pygame
import numpy as np

class CreepyCatacombsPygameRenderer:
    def __init__(self):
        self.window = None
        self.clock = None
        self.surface = None
        self.initialized = False

    def render(self, env, render_mode="human"):
        """
        Renders the environment's current state using Pygame.
        env: the CreepyCatacombsEnv instance
        render_mode: 'human' or 'rgb_array'
        """
        if not self.initialized:
            self._init_pygame(env)
            self.initialized = True

        # Create a surface each time or reuse self.surface
        self.surface.fill((0, 0, 0))

        # Draw the environment
        for r in range(env.height):
            for c in range(env.width):
                val = env.grid[r, c]
                rect = pygame.Rect(
                    c * env.tile_size,
                    r * env.tile_size,
                    env.tile_size,
                    env.tile_size
                )
                color = (100, 100, 100)
                if val == -1:
                    color = (50, 50, 50)
                elif val == 1:
                    color = (0, 200, 0)
                elif val == 2:
                    color = (200, 0, 0)
                elif val == 0:
                    color = (150, 150, 150)
                elif val == 3:
                    color = (0, 255, 0)
                pygame.draw.rect(self.surface, color, rect)

        # Draw agent
        ar, ac = env.agent_pos
        agent_rect = pygame.Rect(
            ac * env.tile_size, 
            ar * env.tile_size,
            env.tile_size,
            env.tile_size
        )
        pygame.draw.rect(self.surface, (0, 0, 255), agent_rect)

        # If 'human', update the main window
        if render_mode == "human" and self.window is not None:
            self.window.blit(self.surface, (0, 0))
            pygame.display.update()
            self.clock.tick(env.metadata["render_fps"])
            return None

        elif render_mode == "rgb_array":
            arr = pygame.surfarray.array3d(self.surface)
            return arr.transpose((1, 0, 2))

    def render_q_values_arrows(self, env, Q, scale=0.2):
        """
        Similar to your old method: Draw environment + arrows for best actions.
        """
        if not self.initialized:
            self._init_pygame(env)
            self.initialized = True

        self.surface.fill((0, 0, 0))

        # Draw environment same as above
        for r in range(env.height):
            for c in range(env.width):
                val = env.grid[r, c]
                rect = pygame.Rect(
                    c * env.tile_size,
                    r * env.tile_size,
                    env.tile_size,
                    env.tile_size
                )
                color = (100, 100, 100)
                if val == -1:
                    color = (50, 50, 50)
                elif val == 1:
                    color = (0, 200, 0)
                elif val == 2:
                    color = (200, 0, 0)
                elif val == 0:
                    color = (150, 150, 150)
                elif val == 3:
                    color = (0, 255, 0)
                pygame.draw.rect(self.surface, color, rect)

        # Now overlay the best-action arrows
        for r in range(env.height):
            for c in range(env.width):
                val = env.grid[r, c]
                if val >= 0 and val not in [1, 2]:
                    state_id = r * env.width + c
                    best_action = np.argmax(Q[state_id])
                    cx = c * env.tile_size + env.tile_size // 2
                    cy = r * env.tile_size + env.tile_size // 2
                    if best_action == 0:
                        dx, dy = 0, -1
                    elif best_action == 1:
                        dx, dy = 1, 0
                    elif best_action == 2:
                        dx, dy = 0, 1
                    else:
                        dx, dy = -1, 0
                    tip_x = cx + int(dx * env.tile_size * scale)
                    tip_y = cy + int(dy * env.tile_size * scale)

                    pygame.draw.line(self.surface, (0,0,0), (cx, cy), (tip_x, tip_y), 3)
                    pygame.draw.circle(self.surface, (0,0,0), (tip_x, tip_y), 4)

        if env.render_mode == "human" and self.window:
            self.window.blit(self.surface, (0, 0))
            pygame.display.update()
            self.clock.tick(env.metadata["render_fps"])
            return None
        else:
            # Return an rgb array
            arr = pygame.surfarray.array3d(self.surface)
            return arr.transpose((1, 0, 2))

    def _init_pygame(self, env):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.window_size = (env.width * env.tile_size, env.height * env.tile_size)
        self.window = pygame.display.set_mode(self.window_size)
        self.surface = pygame.Surface(self.window.get_size(), pygame.SRCALPHA)
        pygame.display.set_caption("Creepy Catacombs")

    def close(self):
        if self.window:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.surface = None
            self.clock = None
            self.initialized = False
