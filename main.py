import argparse
import time
from datetime import datetime
from warnings import simplefilter

import torch
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from rl_gym.agent import Agent

simplefilter(action="ignore", category=DeprecationWarning)

SEED = 4


def make_env(env_id, idx, run_name, capture_video):

    def thunk():

        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        if env.spec.entry_point == "shimmy.atari_env:AtariEnv":
            env = gym.wrappers.AtariPreprocessing(env, scale_obs=True)
            env = gym.wrappers.FrameStack(env, 4)
        else:
            env = gym.wrappers.FlattenObservation(env)
            if type(env.action_space).__name__.lower() == "box":
                env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env, lambda obs: np.clip(obs, -10, 10))
            env = gym.wrappers.NormalizeReward(env, gamma=0.99)
            env = gym.wrappers.TransformReward(
                env, lambda reward: np.clip(reward, -10, 10))

        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env=env,
                video_folder=f"runs/{run_name}/videos/",
                disable_logger=True)
        return env

    return thunk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",
                        type=str,
                        default="BreakoutNoFrameskip-v4",
                        help="name of the environment")
    parser.add_argument("--algo",
                        type=str,
                        default="ppo",
                        help="name of the RL algorithm")
    parser.add_argument("--total-timesteps",
                        type=int,
                        default=int(1e7),
                        help="number of episodes to run")
    parser.add_argument("--num-envs",
                        type=int,
                        default=4,
                        help="number of parallel game environments")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2048,
        help="number of steps to run in each environment per rollout")
    parser.add_argument("--num-minibatches",
                        type=int,
                        default=32,
                        help="the number of mini-batches")
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
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="if toggled, the model will run on CPU",
    )
    parser.add_argument(
        "--capture-video",
        action="store_true",
        help="if toggled, videos will be recorded",
    )

    _args = parser.parse_args()

    _args.device = torch.device(
        "cpu" if _args.cpu or not torch.cuda.is_available() else "cuda")
    _args.batch_size = int(_args.num_envs * _args.num_steps)
    _args.minibatch_size = int(_args.batch_size // _args.num_minibatches)

    return _args


def main():
    args = parse_args()

    date = str(datetime.now().strftime("%d-%m_%H:%M:%S"))
    run_name = f"{args.env}__{args.algo}__{date}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" %
        ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    envs = gym.vector.SyncVectorEnv([
        make_env(args.env, i, run_name, args.capture_video)
        for i in range(args.num_envs)
    ])

    args.cnn = envs.get_attr(
        "spec")[0].entry_point == "shimmy.atari_env:AtariEnv"

    obversation_space = envs.single_observation_space
    action_space = envs.single_action_space

    agent = Agent(args, obversation_space, action_space, writer)

    obversation_shape, action_shape = agent.get_obs_and_action_shape()

    states = np.zeros((args.num_steps, args.num_envs) + obversation_shape)
    actions = np.zeros((args.num_steps, args.num_envs) + action_shape)
    rewards = np.zeros((args.num_steps, args.num_envs))
    flags = np.zeros((args.num_steps, args.num_envs))
    state_values = torch.zeros((args.num_steps, args.num_envs)).to(args.device)

    num_updates = int(args.total_timesteps // args.num_steps)
    global_step = 0

    state, _ = envs.reset(seed=SEED)
    next_done = np.zeros(args.num_envs)

    for _ in tqdm(range(num_updates)):
        log_probs = []
        start = time.perf_counter()

        for i in range(args.num_steps):
            global_step += 1
            flags[i] = next_done

            action, log_prob, state_value = agent.get_action(state)

            next_state, reward, terminated, truncated, infos = envs.step(
                action)

            states[i] = state
            actions[i] = action
            rewards[i] = reward
            state_values[i] = state_value
            log_probs.append(log_prob)

            state = next_state
            next_done = np.logical_or(terminated, truncated)

            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                if info is None:
                    continue

                writer.add_scalar("rollout/episodic_return",
                                  info["episode"]["r"], global_step)
                writer.add_scalar("rollout/episodic_length",
                                  info["episode"]["l"], global_step)

        agent.update_policy(
            {
                "states": torch.from_numpy(states).float().to(args.device),
                "actions": torch.from_numpy(actions).float().to(args.device),
                "last_state": torch.from_numpy(next_state).float().to(
                    args.device),
                "last_flag": torch.from_numpy(next_done).float().to(
                    args.device),
                "rewards": torch.from_numpy(rewards).float().to(args.device),
                "flags": torch.from_numpy(flags).float().to(args.device),
                "state_values": state_values,
                "log_probs": torch.stack(log_probs).squeeze()
            }, global_step)

        end = time.perf_counter()
        writer.add_scalar("rollout/time", end - start, global_step)

    envs.close()
    writer.close()


if __name__ == '__main__':
    main()
