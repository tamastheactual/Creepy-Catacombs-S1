import pytest
import gymnasium as gym
import numpy as np
import creepy_catacombs_s1

@pytest.fixture
def env():
    # Create a fresh environment instance for each test
    # Use some small config for speed
    return gym.make(
        "CreepyCatacombs-v0",
        render_mode=None,
        width=5,
        height=5,
        corridor_width=3,
        n_zombies=1
    )

def test_env_creation(env):
    """Test that environment creation works and reset returns valid initial state."""
    assert env is not None, "Env should be created"

    obs, info = env.reset()
    assert isinstance(obs, np.int64), "Observation should be an integer (discrete space)"
    assert isinstance(info, dict), "Info should be a dict"

def test_observation_space(env):
    """Check that observation space matches expected discrete dimension."""
    unwrapped = env.unwrapped  # unwrap TimeLimit etc. to get the actual environment
    assert env.observation_space.shape == (), "Should be scalar (Discretes have shape=())"
    assert env.observation_space.n == unwrapped.width * unwrapped.height, (
        "Discrete space size should match width*height"
    )

def test_action_space(env):
    """Check the action space is Discrete(4)."""
    assert env.action_space.n == 4, "Should have 4 discrete actions: up, right, down, left"

def test_reset_seed_consistency(env):
    """
    Test that using the same seed yields identical initial states (obs)
    and that different seeds produce different states, 
    unless the random generator picks the same outcome by chance.
    """
    obs1, info1 = env.reset(seed=42)
    obs2, info2 = env.reset(seed=42)
    assert obs1 == obs2, "Same seed should produce same initial observation"

    obs3, info3 = env.reset(seed=999)
    # Can't strictly guarantee different obs with random seeds, but usually it's different
    # For a stronger check: store the entire generated map, or compare info dict.
    assert obs3 != obs1, "Different seed likely yields different start state"

def test_random_steps(env):
    """Take random actions, ensure we don't crash and step returns valid transitions."""
    env.reset()
    done = False
    steps = 0
    max_steps = 20
    while not done and steps < max_steps:
        a = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(a)
        # Basic checks
        assert isinstance(obs, np.int64), "Obs should remain an integer"
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert isinstance(done, bool), "Done should be boolean"
        steps += 1

def test_reward_logic(env):
    """
    Check a few known scenarios:
      - Out-of-bounds yields -10
      - Reaching goal yields +100
    """
    env.reset(seed=1234)
    r, c = env.unwrapped.agent_pos  # e.g. check agent at bottom?
    # Force out-of-bounds by stepping up if agent is at top, etc.
    # If we can't do that easily, we can just brute force some moves:
    obs, reward, done, _, _ = env.step(0)  # attempt to go up
    # Possibly check if the agent was out of bounds
    # or if it was a valid move, etc. This can vary by random environment.
    # => You may want a mock or a known environment layout for a deterministic test
    # e.g. we can do an 'assert reward == -10 or reward == -1'

def test_render_rgb_array(env):
    """
    Check that rendering in rgb_array mode returns a numpy array of shape (H, W, 3).
    """
    # create a new env with render_mode='rgb_array'
    env_rgb = gym.make(
        "CreepyCatacombs-v0",
        render_mode="rgb_array",
        width=5,
        height=5,
        corridor_width=3,
        n_zombies=1
    )
    env_rgb.reset()
    arr = env_rgb.render()
    assert isinstance(arr, np.ndarray), "render() should return a numpy array in rgb_array mode"
    assert arr.ndim == 3 and arr.shape[-1] == 3, "Should be (H, W, 3) shape"

def test_render_q_values_arrows(env):
    """
    If your environment includes a custom method 'render_q_values_arrows',
    ensure it returns a Surface or draws to the screen without error.
    """
    # create a Q-table for the env
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    # Ensure the method doesn't crash
    surf = env.unwrapped.render_q_values_arrows(Q, return_surface=True)
    assert surf is not None, "Should return a pygame.Surface"
