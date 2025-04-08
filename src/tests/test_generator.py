import pytest
import gymnasium as gym
import numpy as np
import creepy_catacombs_s1  # assume this triggers environment + renderer registration


@pytest.fixture
def env():
    """Fixture to create a fresh environment instance for each test."""
    return gym.make(
        "CreepyCatacombs-v0",
        render_mode="rgb_array",
        width=5,
        height=5,
        corridor_width=3,
        n_zombies=1,
        zombie_movement="random"
    )


def test_zombie_placement(env):
    """Test that zombies are placed correctly in the environment."""
    env.reset()
    zombie_positions = env.unwrapped.zombie_positions
    grid = env.unwrapped.grid

    for zr, zc in zombie_positions:
        assert grid[zr, zc] == 3, "Zombies should be placed correctly on the grid"


def test_goal_placement(env):
    """Test that the goal is placed correctly in the environment."""
    env.reset()
    grid = env.unwrapped.grid
    goal_positions = np.argwhere(grid == 2)

    assert len(goal_positions) == 1, "There should be exactly one goal in the environment"


def test_map_generation(env):
    """Test that the map is generated correctly with walls and corridors."""
    env.reset()
    grid = env.unwrapped.grid

    # Check that walls are present
    assert np.any(grid == -1), "There should be walls in the environment"

    # Check that corridors are present
    assert np.any(grid == 0), "There should be walkable corridors in the environment"