import numpy as np
import random
from typing import List, Tuple
from collections import deque

import numpy as np
import random
from typing import List, Tuple
from collections import deque
import matplotlib.pyplot as plt

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def configure_logger(verbosity: int):
    """
    Configures the logger dynamically based on the verbosity level.
    """
    logger.setLevel(verbosity)

def is_reachable(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], verbosity: int = logging.WARNING) -> bool:
    """
    Check if the tunnel is reachable from start to goal using BFS.
    """
    configure_logger(verbosity)
    logger.info("Starting BFS to check reachability from %s to %s", start, goal)
    height, width = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    queue = deque([start])

    while queue:
        r, c = queue.popleft()
        logger.debug("Visiting cell: (%d, %d)", r, c)
        if (r, c) == goal:
            logger.info("Goal reached at %s", goal)
            return True
        if visited[r, c]:
            continue
        visited[r, c] = True

        # Explore neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc] and grid[nr, nc] in [0, 2]:
                logger.debug("Adding neighbor cell: (%d, %d) to queue", nr, nc)
                queue.append((nr, nc))

    logger.debug("No path found from %s to %s", start, goal)
    return False

def generate_tunnel(height: int, width: int, corridor_width: int, n_plotholes: int = 3, verbosity: int = logging.WARNING) -> np.ndarray:
    """
    Create the 2D array with a 'tunnel' from bottom row to top row, with optional plotholes:
    -1 = wall, 0 = empty, 1 = start, 2 = goal, -2 = plothole
    Returns the generated grid as np.array of shape (height, width).
    """
    configure_logger(verbosity)
    logger.info("Generating tunnel with height=%d, width=%d, corridor_width=%d, n_plotholes=%d", height, width, corridor_width, n_plotholes)

    assert corridor_width > 0, "Corridor width must be greater than 0."
    assert corridor_width < width, "Corridor width must be smaller than the total width."
    assert height > 2 and width > 2, "Height and width must be greater than 2."

    grid: np.ndarray = -1 * np.ones((height, width), dtype=int)
    center: int = width // 2

    # Generate the main tunnel
    for row in range(height):
        half_w: int = (corridor_width - 1) // 2
        left: int = max(0, center - half_w)
        right: int = min(width, center + half_w + 1)
        grid[row, left:right] = 0
        logger.debug("Row %d: Tunnel created from column %d to %d", row, left, right)
        center += random.choice([-1, 0, 1])
        center = np.clip(center, 1, width - 2)

    # Mark start (bottom)
    bottom_valid: np.ndarray = np.where(grid[height - 1] == 0)[0]
    if len(bottom_valid) > 0:
        start_col: int = random.choice(bottom_valid)
        grid[height - 1, start_col] = 1
        start = (height - 1, start_col)
        logger.info("Start position set at %s", start)
    else:
        logger.error("No valid start position found.")
        raise ValueError("No valid start position found.")

    # Mark goal (top)
    top_valid: np.ndarray = np.where(grid[0] == 0)[0]
    if len(top_valid) > 0:
        goal_col: int = random.choice(top_valid)
        grid[0, goal_col] = 2
        goal = (0, goal_col)
        logger.info("Goal position set at %s", goal)
    else:
        logger.error("No valid goal position found.")
        raise ValueError("No valid goal position found.")

    # Add plotholes
    for i in range(n_plotholes):
        plothole_row = random.randint(1, height - 2)
        plothole_col = random.randint(1, width - 2)
        logger.debug("Placing plothole %d at (%d, %d)", i + 1, plothole_row, plothole_col)
        if grid[plothole_row, plothole_col] == 0:
            grid[plothole_row, plothole_col] = -2

    if not is_reachable(grid, start, goal, verbosity=verbosity):
        logger.debug("Tunnel is disconnected. Retrying with fewer plotholes.")
        return generate_tunnel(height, width, corridor_width, max(0, n_plotholes - 1), verbosity=verbosity)

    logger.info("Tunnel generation complete.")
    return grid

def place_zombies(grid: np.ndarray, n_zombies: int, verbosity: int = logging.WARNING) -> List[Tuple[int, int]]:
    """
    Place n_zombies in random empty cells (3) at top 70% rows.
    Returns a list of zombie positions and modifies grid in-place.
    """
    configure_logger(verbosity)
    logger.info("Placing %d zombies in the grid.", n_zombies)

    height, _ = grid.shape
    spawn_cutoff: int = int(height * 0.7)

    empty_cells: List[Tuple[int, int]] = list(zip(*np.where(grid == 0)))
    valid_zombie_cells: List[Tuple[int, int]] = [
        (r, c) for (r, c) in empty_cells if r < spawn_cutoff
    ]

    random.shuffle(valid_zombie_cells)
    zombie_positions: List[Tuple[int, int]] = []
    for i in range(min(n_zombies, len(valid_zombie_cells))):
        zr, zc = valid_zombie_cells[i]
        zombie_positions.append((zr, zc))
        grid[zr, zc] = 3
        logger.debug("Zombie placed at (%d, %d)", zr, zc)
    return zombie_positions

