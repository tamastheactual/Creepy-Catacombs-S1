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


def test_env_creation(env):
    """Test that the environment is created successfully and reset returns valid initial state."""
    assert env is not None, "Environment should be created successfully"

    obs, info = env.reset()
    assert isinstance(obs, np.int64), "Observation should be an integer (discrete space)"
    assert isinstance(info, dict), "Info should be a dictionary"


def test_observation_space(env):
    """Check that the observation space matches the expected discrete dimension."""
    unwrapped = env.unwrapped
    assert env.observation_space.shape == (), "Observation space should be scalar (Discrete)"
    assert env.observation_space.n == unwrapped.width * unwrapped.height, (
        "Discrete space size should match width * height"
    )


def test_action_space(env):
    """Check that the action space is Discrete(4)."""
    assert env.action_space.n == 4, "Action space should have 4 discrete actions: up, right, down, left"


def test_reset_seed_consistency(env):
    """Test that resetting with the same seed produces consistent initial states."""
    obs1, info1 = env.reset(seed=42)
    obs2, info2 = env.reset(seed=42)
    assert obs1 == obs2, "Same seed should produce the same initial observation"

    obs3, info3 = env.reset(seed=999)
    assert obs3 != obs1, "Different seeds should produce different initial observations (likely)"


def test_random_steps(env):
    """Take random actions and ensure step transitions are valid."""
    env.reset()
    done = False
    steps = 0
    max_steps = 20

    while not done and steps < max_steps:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        # Basic checks
        assert isinstance(obs, np.int64), "Observation should remain an integer"
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert isinstance(done, bool), "Done should be a boolean"
        steps += 1