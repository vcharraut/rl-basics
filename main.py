from tqdm import tqdm
import gymnasium as gym
import numpy as np
import torch
from rl_gym.policy import Policy


def make_env(env_id, seed):

    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


if __name__ == "__main__":
    ENV_NAME = "LunarLanderContinuous-v2"
    ALGO = "ppo"
    N_EPISODE = 1000
    LEARNING_RATE = 3e-4
    LIST_LAYER = [128, 128]
    IS_SHARED_NETWORK = False

    N_ENV = 16
    N_STEPS = 2048

    device = torch.device("cuda")

    envs = gym.vector.AsyncVectorEnv(
        [make_env(ENV_NAME, 0 + i) for i in range(N_ENV)])

    obversation_space = envs.single_observation_space
    action_space = envs.single_action_space

    policy = Policy(ALGO, obversation_space, action_space, LEARNING_RATE,
                    LIST_LAYER, IS_SHARED_NETWORK)

    obversation_shape, action_shape = policy.get_obs_and_action_shape()

    current_state, _ = envs.reset()

    for episode in tqdm(range(N_EPISODE)):
        states = torch.zeros((N_STEPS, N_ENV) + obversation_shape).to(device)
        actions = torch.zeros((N_STEPS, N_ENV) + action_shape).to(device)
        logprobs = torch.zeros((N_STEPS, N_ENV)).to(device)
        rewards = torch.zeros((N_STEPS, N_ENV)).to(device)
        flags = torch.zeros((N_STEPS, N_ENV)).to(device)
        next_states = torch.zeros((N_STEPS, N_ENV) +
                                  obversation_space.shape).to(device)

        list_rewards = []

        for i in range(N_STEPS):
            action, log_prob = policy.get_action(current_state)

            next_state, reward, terminated, _, info = envs.step(action)

            if info:
                idx = info["_final_info"].nonzero()[0]
                values = info["final_info"][idx]
                for x in values:
                    list_rewards.append(x["episode"]["r"])

            states[i] = torch.from_numpy(current_state).float()
            actions[i] = torch.from_numpy(action).float()
            next_states[i] = torch.from_numpy(next_state).float()
            rewards[i] = torch.from_numpy(reward).float()
            flags[i] = torch.from_numpy(terminated * 1).float()
            logprobs[i] = log_prob

            current_state = next_state

        policy.update_policy({
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "rewards": rewards,
            "flags": flags,
            "log_probs": logprobs
        })

        mean_rewards = np.array(list_rewards).mean()
        policy.add_reward_to_tensorboard(mean_rewards)

    policy.add_hparams_to_tensorboard(LEARNING_RATE, LIST_LAYER[0],
                                      mean_rewards)
    envs.close()
    policy.close()
