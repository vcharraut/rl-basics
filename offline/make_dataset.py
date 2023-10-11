import gymnasium as gym
from minari import DataCollectorV0, create_dataset_from_collector_env
from stable_baselines3 import SAC


if __name__ == "__main__":
    env_name = "HalfCheetah-v4"
    run_dir = f"logs/{env_name}"
    seed = 42

    env = DataCollectorV0(gym.make(env_name))

    model = SAC("MlpPolicy", env, buffer_size=100_000, learning_starts=25_000, tensorboard_log=run_dir, seed=seed)
    model.learn(total_timesteps=1_000_000, progress_bar=True)

    dataset = create_dataset_from_collector_env(
        dataset_id="HalfCheetah-expert-v4",
        collector_env=env,
    )
