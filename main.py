import argparse
import time
from tqdm import tqdm
import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from rl_gym.agent import Agent


def make_env(env_id):

    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.FlattenObservation(env)
        # env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=0.99)
        env = gym.wrappers.TransformReward(
            env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",
                        type=str,
                        default="CartPole-v1",
                        help="name of the environment")
    parser.add_argument("--algo",
                        type=str,
                        default="a2c",
                        help="name of the RL algorithm")
    parser.add_argument("--episode",
                        type=int,
                        default=300,
                        help="number of episodes to run")
    parser.add_argument("--num-envs",
                        type=int,
                        default=8,
                        help="number of parallel game environments")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2048,
        help="number of steps to run in each environment per rollout")
    parser.add_argument("--learning-rate",
                        type=float,
                        default=3e-4,
                        help="learning rate of the optimizer")
    parser.add_argument("--gamma",
                        type=float,
                        default=0.99,
                        help="the discount factor gamma")
    parser.add_argument('--layers',
                        nargs="+",
                        type=int,
                        default=[64, 64],
                        help="list of the layers for the neural network")
    parser.add_argument(
        "--shared-network",
        action="store_true",
        help="if toggled, actor and critic will share the same network",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda")

    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env) for _ in range(args.num_envs)])

    run_name = f"{args.env}_{args.algo}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" %
        ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    obversation_space = envs.single_observation_space
    action_space = envs.single_action_space

    agent = Agent(args, obversation_space, action_space, writer)

    torch.backends.cudnn.benchmark = True

    obversation_shape, action_shape = agent.get_obs_and_action_shape()

    states = torch.zeros((args.num_steps, args.num_envs) +
                         obversation_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) +
                          action_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    flags = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_states = torch.zeros((args.num_steps, args.num_envs) +
                              obversation_space.shape).to(device)

    state, _ = envs.reset()

    for _ in tqdm(range(args.episode)):
        log_rewards, log_eps_length = [], []

        for i in range(args.num_steps):
            action, log_prob = agent.get_action(state)

            next_state, reward, terminated, _, info = envs.step(action)

            if info:
                idx = info["_final_info"].nonzero()[0]
                values = info["final_info"][idx]
                for x in values:
                    log_rewards.append(x["episode"]["r"])
                    log_eps_length.append(x["episode"]["l"])

            states[i] = torch.from_numpy(state).float()
            actions[i] = torch.from_numpy(action).float()
            next_states[i] = torch.from_numpy(next_state).float()
            rewards[i] = torch.from_numpy(reward).float()
            flags[i] = torch.from_numpy(terminated * 1).float()
            logprobs[i] = log_prob

            state = next_state

        agent.update_policy({
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "rewards": rewards,
            "flags": flags,
            "log_probs": logprobs
        })

        writer.add_scalar("rollout/reward",
                          np.array(log_rewards).mean(),
                          agent.get_global_step())
        writer.add_scalar("rollout/episode_length",
                          np.array(log_eps_length).mean(),
                          agent.get_global_step())

    envs.close()
    writer.close()
