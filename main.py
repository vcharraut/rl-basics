import argparse
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from rlgym.policy import Policy

plt.style.use("bmh")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",
                        type=str,
                        default="a2c",
                        help="Algortihms to be used for the learning")
    parser.add_argument("--env",
                        type=str,
                        default="CartPole-v1",
                        help="Gym environment")
    parser.add_argument("--lr", type=int, default=1e-3, help="Learning rite")
    parser.add_argument("--size_hidden",
                        type=int,
                        default=128,
                        help="Size of hidden layers")
    parser.add_argument("--number_hidden",
                        type=int,
                        default=2,
                        help="Number of hidden layers")
    parser.add_argument("--is_shared_network",
                        action="store_true",
                        help="Display or not the agent in the environment")

    args = parser.parse_args()

    env = gym.make(args.env, new_step_api=True)
    n_episode = 1500
    log_every_n = n_episode / 100

    obversation_space = env.observation_space.shape[0]
    action_space = env.action_space
    action_space_type = type(env.action_space).__name__.lower()

    if action_space_type == "discrete":
        is_continuous = False
    elif action_space_type == "box":
        is_continuous = True
    else:
        raise Exception("Unvalid type of action_space")

    policy = Policy(args.algo, obversation_space, action_space, is_continuous,
                    args.lr, args.size_hidden, args.number_hidden,
                    args.is_shared_network)

    log_rewards = []

    for t in tqdm(range(n_episode)):

        current_state = env.reset()
        next_state = None
        states, actions, next_states, rewards, flags, logprobs = [], [], [], [], [], []
        terminated, truncated = False, False

        while not terminated and not truncated:
            action, log_prob = policy.get_action(current_state)

            if is_continuous:
                action = np.array(action, ndmin=1)

            next_state, reward, terminated, truncated, _ = env.step(action)

            states.append(current_state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            flags.append(int(terminated))
            logprobs.append(log_prob)

            current_state = next_state

        policy.update_policy({
            "states":
            torch.from_numpy(np.array(states)).to(torch.device("cuda")),
            "actions":
            torch.from_numpy(np.array(actions)).to(torch.device("cuda")),
            "next_states":
            torch.from_numpy(np.array(next_states)).to(torch.device("cuda")),
            "rewards":
            torch.from_numpy(np.array(rewards)).to(torch.device("cuda")),
            "flags":
            torch.from_numpy(np.array(flags)).to(torch.device("cuda")),
            "logprobs":
            torch.stack(logprobs).squeeze()
        })

        if not t % log_every_n:
            log_rewards.append(np.sum(rewards))

    return log_rewards


if __name__ == "__main__":
    main()
