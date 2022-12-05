import argparse
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",
                        type=str,
                        default="LunarLander-v2",
                        help="name of the environment")
    parser.add_argument("--algo",
                        type=str,
                        default="ppo",
                        help="name of the RL algorithm")
    parser.add_argument("--episode",
                        type=int,
                        default=1000,
                        help="number of episodes to run")
    parser.add_argument("--num-envs",
                        type=int,
                        default=8,
                        help="the number of parallel game environments")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2048,
        help="the number of steps to run in each environment per policy rollout"
    )
    parser.add_argument("--learning-rate",
                        type=float,
                        default=3e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument('--layers',
                        nargs="+",
                        type=int,
                        default=[64, 64],
                        help="the list of the layers for the neural network")
    parser.add_argument(
        "--shared-network",
        action="store_true",
        help="if toggled, actor and critic will share the same network",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # LunarLander LunarLanderContinuous
    args = parse_args()

    device = torch.device("cuda")

    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env, 0 + i) for i in range(args.num_envs)])

    obversation_space = envs.single_observation_space
    action_space = envs.single_action_space

    policy = Policy(args.algo, obversation_space, action_space,
                    args.learning_rate, args.layers, args.shared_network)

    obversation_shape, action_shape = policy.get_obs_and_action_shape()

    current_state, _ = envs.reset()

    for _ in tqdm(range(args.episode)):
        states = torch.zeros((args.num_steps, args.num_envs) +
                             obversation_shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) +
                              action_shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        flags = torch.zeros((args.num_steps, args.num_envs)).to(device)
        next_states = torch.zeros((args.num_steps, args.num_envs) +
                                  obversation_space.shape).to(device)

        list_rewards = []

        for i in range(args.num_steps):
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

    envs.close()
    policy.algorithm.writer.close()
