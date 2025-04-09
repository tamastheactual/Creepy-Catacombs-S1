import pytest
import numpy as np
import creepy_catacombs_s1  # assume this triggers environment + renderer registration


@pytest.fixture
def env():
    """Fixture to create a fresh environment instance for each test."""
    from creepy_catacombs_s1.env.catacombs_env import CreepyCatacombsEnv
    return CreepyCatacombsEnv(
        render_mode="rgb_array",
        width=5,
        height=5,
        corridor_width=3,
        n_zombies=1,
        zombie_movement="random"
    )