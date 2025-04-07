import numpy as np
import random
from typing import List, Tuple

def generate_tunnel(height: int, width: int, corridor_width: int) -> np.ndarray:
    """
    Create the 2D array with a 'tunnel' from bottom row to top row:
    -1 = wall, 0 = empty, 1 = start, 2 = goal
    Returns the generated grid as np.array of shape (height, width).
    """
    grid: np.ndarray = -1 * np.ones((height, width), dtype=int)
    center: int = width // 2

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

    # Mark goal (top)
    top_valid: np.ndarray = np.where(grid[0] == 0)[0]
    if len(top_valid) > 0:
        goal_col: int = top_valid[-1]
        grid[0, goal_col] = 2

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