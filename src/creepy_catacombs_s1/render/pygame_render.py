import pygame
import logging
import os
import matplotlib.pyplot as plt

class CreepyCatacombsPygameRenderer:
    def __init__(self, verbosity: int = logging.WARNING):
        """
        Initializes the Pygame renderer.
        verbosity: Logging verbosity level (e.g., logging.DEBUG, logging.WARNING).
        """
        self.window = None
        self.clock = None
        self.surface = None
        self.initialized = False
        self.assets = {}  # Dictionary to store loaded assets
        self.verbosity = verbosity
        
        logging.basicConfig(
        level=self.verbosity,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        self.logger = logging.getLogger(__name__)
        

    def render(self, env, render_mode="human"):
        """
        Renders the environment's current state using Pygame.
        env: the CreepyCatacombsEnv instance
        render_mode: 'human' or 'rgb_array'
        """
        if not self.initialized:
            self._init_pygame(env)
            self.initialized = True

        self.logger.debug("Rendering environment with render_mode=%s", render_mode)
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
                if val == -1:
                    self._draw_asset("wall", rect)
                elif val == 1:
                    self._draw_asset("start", rect)
                elif val == 2:
                    self._draw_asset("goal", rect)
                elif val == 0:
                    self._draw_asset("floor", rect)
                elif val == 3:
                    self._draw_asset("zombie", rect)
                elif val == -2:
                    self._draw_asset("plothole", rect)

        # Draw agent
        ar, ac = env.agent_pos
        agent_rect = pygame.Rect(
            ac * env.tile_size,
            ar * env.tile_size,
            env.tile_size,
            env.tile_size
        )
        self._draw_asset("agent", agent_rect)

        if render_mode == "human" and self.window is not None:
            self.window.blit(self.surface, (0, 0))
            pygame.display.update()
            self.clock.tick(env.metadata["render_fps"])
            self.logger.debug("Rendered frame to human display.")
            return None

        elif render_mode == "rgb_array":
            arr = pygame.surfarray.array3d(self.surface)
            self.logger.debug("Rendered frame as RGB array.")
            return arr.transpose((1, 0, 2))

    def _draw_asset(self, asset_name, rect):
        """
        Draws an asset (image or fallback color) at the given rect.
        """
        if asset_name in self.assets:
            self.surface.blit(self.assets[asset_name], rect)
        else:
            # Fallback: Draw a simple rectangle with a default color
            fallback_colors = {
                "wall": (50, 50, 50),
                "start": (0, 200, 0),
                "goal": (200, 0, 0),
                "floor": (150, 150, 150),
                "zombie": (0, 255, 0),
                "plothole": (100, 0, 100),
                "agent": (0, 0, 255),
            }
            color = fallback_colors.get(asset_name, (255, 255, 255))
            pygame.draw.rect(self.surface, color, rect)

    def _init_pygame(self, env):
        """
        Initializes Pygame and loads assets.
        If `create_window` is False, only initializes the surface and assets.
        """
        self.logger.info("Initializing Pygame renderer.")
        pygame.init()
        self.clock = pygame.time.Clock()
        self.window_size = (env.width * env.tile_size, env.height * env.tile_size)
        self.window = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Creepy-Catacombs-S1")
        self.surface = pygame.Surface(self.window_size, pygame.SRCALPHA)

        # Load assets
        assets_path = os.path.join(os.path.dirname(__file__), "assets")
        self.assets = {
            "wall": pygame.image.load(os.path.join(assets_path, "wall.png")).convert_alpha(),
            "start": pygame.image.load(os.path.join(assets_path, "start.png")).convert_alpha(),
            "goal": pygame.image.load(os.path.join(assets_path, "goal.png")).convert_alpha(),
            "floor": pygame.image.load(os.path.join(assets_path, "floor.png")).convert_alpha(),
            "zombie": pygame.image.load(os.path.join(assets_path, "zombie.png")).convert_alpha(),
            "plothole": pygame.image.load(os.path.join(assets_path, "plothole.png")).convert_alpha(),
            "agent": pygame.image.load(os.path.join(assets_path, "agent.png")).convert_alpha(),
        }
        for key in self.assets:
            self.assets[key] = pygame.transform.scale(self.assets[key], (env.tile_size, env.tile_size))

        self.logger.info("Assets loaded and scaled to tile size.")
        
    def render_q_values_arrows(self, env, Q, scale=0.01, verbosity: int = logging.WARNING):
        """
        Render Q-values as arrows on the grid.
        - Q: A dictionary mapping state-action pairs to Q-values.
        - scale: Scaling factor for arrow size.
        - verbosity: Logging verbosity level (e.g., logging.DEBUG, logging.WARNING).
        """
        self.logger.setLevel(verbosity)
        self.logger.info("Rendering Q-values with scale=%.2f and verbosity=%d.", scale, verbosity)

        try:
            rgb_array = self.render(env, render_mode="rgb_array")
            surface = pygame.surfarray.make_surface(rgb_array.transpose((1, 0, 2)))
            font = pygame.font.SysFont("Arial", 18)

            for r in range(env.height):
                for c in range(env.width):
                    state = r * env.width + c
                    rect = pygame.Rect(
                        c * env.tile_size,
                        r * env.tile_size,
                        env.tile_size,
                        env.tile_size
                    )
                    
                    if env.grid[r, c] == -1:
                        continue
                    if env.grid[r, c] == -2:
                        continue
                    if env.grid[r, c] == 2:
                        continue

                    q_values = [Q.get((state, action), 0) for action in range(4)]
                    max_q = max(q_values)

                    for action, q_value in enumerate(q_values):
                        color = (255, 0, 0) if q_value == max_q else (255, 255, 255)
                        self._draw_arrow_on_surface(surface, rect, action, color)
                        text = font.render(f"{q_value:.2f}", True, color)
                        if action == 0:  # Up
                            surface.blit(text, (rect.centerx - text.get_width() // 2, rect.top + text.get_height() // 4))
                        elif action == 1:  # Right
                            surface.blit(text, (rect.right - text.get_width() - 5, rect.centery - text.get_height()))
                        elif action == 2:  # Down
                            surface.blit(text, (rect.centerx - text.get_width() // 2, rect.bottom - text.get_height() - 5 ))
                        elif action == 3:  # Left
                            surface.blit(text, (rect.left + 5, rect.centery - text.get_height()))
            
            self.close()
            self.logger.debug("Q-values rendered successfully.")
            return surface
        except Exception as e:
            self.logger.error("Error while rendering Q-values: %s", str(e))
            return None
            
    def display_surface_with_matplotlib(self, surface):
        """
        Displays a Pygame surface using Matplotlib.
        - surface: The Pygame surface to display.
        """
        if surface is None:
            self.logger.error("Cannot display surface: surface is None.")
            return

        array = pygame.surfarray.array3d(surface)
        array = array.transpose((1, 0, 2))
        return array
    
    def render_v_values(self, env, V, max_value=None, verbosity: int = logging.WARNING):
        """
        Render V-values as gradient squares on the grid.
        - V: A dictionary mapping states to V-values.
        - max_value: The maximum value for normalization (optional).
        - verbosity: Logging verbosity level (e.g., logging.DEBUG, logging.WARNING).
        """
        self.logger.setLevel(verbosity)
        self.logger.info("Rendering V-values with verbosity=%d.", verbosity)

        try:
            rgb_array = self.render(env, render_mode="rgb_array")
            surface = pygame.surfarray.make_surface(rgb_array.transpose((1, 0, 2)))
            font = pygame.font.SysFont("Arial", 18)

            # Determine max and min values for normalization
            max_value = max_value or max(V.values())
            min_value = min(V.values())

            for r in range(env.height):
                for c in range(env.width):
                    state = r * env.width + c
                    rect = pygame.Rect(
                        c * env.tile_size,
                        r * env.tile_size,
                        env.tile_size,
                        env.tile_size
                    )

                    # Skip walls, plotholes, and goal cells
                    if env.grid[r, c] in [-1, -2, 2]:
                        continue

                    # Get the V-value for the current state
                    v_value = V.get(state, 0)

                    # Normalize the value to a range between 0 and 1
                    normalized_value = (v_value - min_value) / (max_value - min_value) if max_value != min_value else 0

                    # Calculate the red intensity (higher values = deeper red)
                    red_intensity = int(255 * normalized_value)
                    overlay_color = (red_intensity, 255 - red_intensity, 255 - red_intensity, 128)  # Red gradient

                    # Create a semi-transparent overlay
                    overlay = pygame.Surface((env.tile_size, env.tile_size), pygame.SRCALPHA)
                    overlay.fill(overlay_color)
                    surface.blit(overlay, rect.topleft)

                    # Render the V-value as text
                    text = font.render(f"{v_value:.2f}", True, (255, 255, 255))
                    surface.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))

            self.close()
            self.logger.debug("V-values rendered successfully.")
            return surface
        except Exception as e:
            self.logger.error("Error while rendering V-values: %s", str(e))
            return None
        
    def render_optimal_path(self, env, policy, verbosity: int = logging.WARNING):
        """
        Render the optimal path from the agent's position to the goal using red arrows.
        - env: The environment instance.
        - policy: A dictionary or array mapping states to actions (optimal policy).
        - verbosity: Logging verbosity level (e.g., logging.DEBUG, logging.WARNING).
        """
        self.logger.setLevel(verbosity)
        self.logger.info("Rendering optimal path with verbosity=%d.", verbosity)

        try:
            rgb_array = self.render(env, render_mode="rgb_array")
            surface = pygame.surfarray.make_surface(rgb_array.transpose((1, 0, 2)))

            # Start from the agent's current position
            current_pos = env.agent_pos
            visited_states = set()

            while True:
                r, c = current_pos
                state = r * env.width + c

                # Avoid infinite loops by tracking visited states
                if state in visited_states:
                    self.logger.warning("Detected a loop in the policy. Stopping path rendering.")
                    break
                visited_states.add(state)

                # Get the action from the policy
                action = policy[state]

                # Draw an arrow for the action
                rect = pygame.Rect(
                    c * env.tile_size,
                    r * env.tile_size,
                    env.tile_size,
                    env.tile_size
                )
                self._draw_arrow_on_surface(surface, rect, action, (255, 0, 0))  # Red arrow

                # Compute the next position based on the action
                if action == 0:  # Up
                    next_pos = (r - 1, c)
                elif action == 1:  # Right
                    next_pos = (r, c + 1)
                elif action == 2:  # Down
                    next_pos = (r + 1, c)
                elif action == 3:  # Left
                    next_pos = (r, c - 1)
                else:
                    self.logger.error(f"Invalid action {action} at state {state}.")
                    break

                # Check if the next position is the goal
                if env.grid[next_pos[0], next_pos[1]] == 2:  # Goal cell
                    self.logger.info("Reached the goal. Stopping path rendering.")
                    break

                # Update the current position
                current_pos = next_pos

            self.close()
            self.logger.debug("Optimal path rendered successfully.")
            return surface
        except Exception as e:
            self.logger.error("Error while rendering optimal path: %s", str(e))
            return None

    def _draw_arrow_on_surface(self, surface, rect, action, color):
        """
        Draw an arrow for a specific action on a given surface.
        - surface: The Pygame surface to draw on.
        - rect: The grid cell rectangle.
        - action: The action (0=up, 1=right, 2=down, 3=left).
        - length: The relative length of the arrow (scaled to rect dimensions).
        - color: The color of the arrow.
        """
        center = rect.center
        scaled_length = 0.3 * rect.height
        if action == 0:  # Up
            start = (center[0], center[1])
            end = (center[0], center[1] - scaled_length)
        elif action == 1:  # Right
            start = (center[0], center[1])
            end = (center[0] + scaled_length, center[1])
        elif action == 2:  # Down
            start = (center[0], center[1])
            end = (center[0], center[1] + scaled_length)
        elif action == 3:  # Left
            start = (center[0], center[1])
            end = (center[0] - scaled_length, center[1])

        pygame.draw.line(surface, color, start, end, 2)
        pygame.draw.circle(surface, color, end, 3)
        
    
        
    def close(self):
        """
        Closes the Pygame window and cleans up resources.
        """
        if self.window:
            self.logger.info("Closing Pygame renderer.")
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.surface = None
            self.clock = None
            self.initialized = False