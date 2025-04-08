import numpy as np
import random
from typing import List, Tuple
from collections import deque

import numpy as np
import random
from typing import List, Tuple
from collections import deque
import matplotlib.pyplot as plt

def is_reachable(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    """
    Check if the tunnel is reachable from start to goal using BFS.
    """
    height, width = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    queue = deque([start])

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        if visited[r, c]:
            continue
        visited[r, c] = True

        # Explore neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            # Ensure the neighbor is within bounds, not visited, and is an empty cell (0)
            if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc] and grid[nr, nc] in [0, 2]:
                queue.append((nr, nc))

    # If we exhaust the queue without finding the goal, return False
    print("No path found from start to goal.")
    return False

def generate_tunnel(height: int, width: int, corridor_width: int, n_islands: int = 3) -> np.ndarray:
    """
    Create the 2D array with a 'tunnel' from bottom row to top row, with optional islands:
    -1 = wall, 0 = empty, 1 = start, 2 = goal
    Returns the generated grid as np.array of shape (height, width).
    """
    # Assertions to validate input parameters
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
        center += random.choice([-1, 0, 1])
        center = np.clip(center, 1, width - 2)

    # Mark start (bottom)
    bottom_valid: np.ndarray = np.where(grid[height - 1] == 0)[0]
    if len(bottom_valid) > 0:
        start_col: int = bottom_valid[0]
        grid[height - 1, start_col] = 1
        start = (height - 1, start_col)
    else:
        raise ValueError("No valid start position found.")

    # Mark goal (top)
    top_valid: np.ndarray = np.where(grid[0] == 0)[0]
    if len(top_valid) > 0:
        goal_col: int = top_valid[-1]
        grid[0, goal_col] = 2
        goal = (0, goal_col)
    else:
        raise ValueError("No valid goal position found.")

    # Add smaller islands
    for _ in range(n_islands):
        island_row = random.randint(1, height - 2)
        island_col = random.randint(1, width - 2)

        # Ensure the island is a single cell or very small (1x1 or 2x2)
        island_size = random.choice([1, 2, 3, 4])  # Smaller islands
        for dr in range(-island_size + 1, island_size):
            for dc in range(-island_size + 1, island_size):
                nr, nc = island_row + dr, island_col + dc
                if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] == 0:
                    grid[nr, nc] = -1
    
    # plot the grid
    plt.imshow(grid, cmap='gray')
    plt.show()
    # Ensure the start and goal are not blocked by islands
    # Ensure the tunnel is still reachable
    if not is_reachable(grid, start, goal):
        # If disconnected, retry with fewer islands to avoid infinite recursion
        return generate_tunnel(height, width, corridor_width, max(0, n_islands - 1))

    return grid

def place_zombies(grid: np.ndarray, n_zombies: int) -> List[Tuple[int, int]]:
    """
    Place n_zombies in random empty cells (3) at top 70% rows.
    Returns a list of zombie positions and modifies grid in-place.
    """
    height, width = grid.shape
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
    return zombie_positions

def move_zombies(grid: np.ndarray, zombie_positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Each zombie can move randomly up/down/left/right/stay if valid.
    Returns updated list of zombie positions, modifies grid in-place.
    """
    height, width = grid.shape
    new_positions: List[Tuple[int, int]] = []
    directions: List[Tuple[int, int]] = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]

    for (zr, zc) in zombie_positions:
        if grid[zr, zc] == 3:
            grid[zr, zc] = 0

        dr, dc = random.choice(directions)
        nr, nc = zr + dr, zc + dc
        if (0 <= nr < height and 0 <= nc < width and
            grid[nr, nc] not in [-1, 1, 2]):  # not a wall or start/goal
            new_positions.append((nr, nc))
        else:
            new_positions.append((zr, zc))

    for (zr, zc) in new_positions:
        grid[zr, zc] = 3

    return new_positions