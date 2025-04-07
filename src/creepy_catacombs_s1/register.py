from gymnasium.envs.registration import register

register(
    id="CreepyCatacombs-v0",
    entry_point="creepy_catacombs_s1.env.catacombs_env:CreepyCatacombsEnv",
    max_episode_steps=500,
)