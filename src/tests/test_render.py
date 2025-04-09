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

def test_render_q_values_arrows(env):
    """Test that rendering Q-values as arrows works without errors."""
    Q = np.zeros((env.observation_space.n, env.action_space.n))  # Example Q-table
    Q_dict = {(state, action): Q[state, action] for state in range(Q.shape[0]) for action in range(Q.shape[1])}
    renderer = env.unwrapped.renderer

    surface = renderer.render_q_values_arrows(env.unwrapped, Q_dict)
    assert surface is not None, "render_q_values_arrows should return a pygame.Surface"