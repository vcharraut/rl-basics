import argparse
import math
import random
from collections import deque, namedtuple
from datetime import datetime
from pathlib import Path
from warnings import simplefilter

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.nn.functional import mse_loss
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

simplefilter(action="ignore", category=DeprecationWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--total-timesteps", type=int, default=int(10e6))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=500000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=5e-3)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-start", type=int, default=1)
    parser.add_argument("--learning-start", type=int, default=100000)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    _args = parser.parse_args()

    _args.device = torch.device("cuda")
    _args.eps_decay = int(_args.total_timesteps * 0.10)

    return _args


def make_env(env_id, run_dir, capture_video):
    def thunk():

        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env=env, video_folder=f"{run_dir}/videos/", disable_logger=True
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.AtariPreprocessing(env, scale_obs=True)
        env = gym.wrappers.FrameStack(env, 4)

        return env

    return thunk


class ReplayMemory:
    def __init__(self, buffer_size, batch_size, obversation_shape, device):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.obversation_shape = obversation_shape
        self.device = device

        self.transition = namedtuple(
            "Transition", field_names=["state", "action", "reward", "flag"]
        )

    @property
    def size(self):
        return len(self.buffer)

    def push(self, state, action, reward, flag):
        state = state.to(dtype=torch.uint8).squeeze()
        self.buffer.append(self.transition(state[-1], action, reward, flag))

    def get_states(self, index_ns):
        index_s = index_ns - 1

        i_range = range(0, 4)
        state = torch.stack(
            [self.buffer[index_s - i].state.to(dtype=torch.float32) for i in i_range]
        )
        next_state = torch.stack(
            [self.buffer[index_ns - i].state.to(dtype=torch.float32) for i in i_range]
        )

        return state, next_state

    def sample(self):
        sample_indices = np.random.choice(range(5, self.size), size=self.batch_size, replace=False)

        samples = [
            (
                *self.get_states(index),
                self.buffer[index].action,
                self.buffer[index].reward,
                self.buffer[index].flag,
            )
            for index in sample_indices
        ]

        states, next_states, actions, rewards, flags = zip(*samples)

        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.tensor(actions).to(self.device).unsqueeze(-1)
        rewards = torch.tensor(rewards).to(self.device)
        flags = torch.tensor(flags).to(self.device)

        return states, actions, rewards, next_states, flags


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class QNetwork(nn.Module):
    def __init__(self, args, action_shape):

        super().__init__()

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, action_shape), std=0.01),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate)

        if args.device.type == "cuda":
            self.cuda()

    def forward(self, state):
        return self.network(state)


def get_exploration_prob(args, step):
    return args.eps_end + (args.eps_start - args.eps_end) * math.exp(-1.0 * step / args.eps_decay)


def main():
    args = parse_args()

    date = str(datetime.now().strftime("%d-%m_%H:%M:%S"))
    run_dir = Path(Path(__file__).parent.resolve().parent, "runs", f"{args.env}__dqn__{date}")
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Create vectorized environment(s)
    env = gym.vector.AsyncVectorEnv([make_env(args.env, run_dir, args.capture_video)])

    obversation_shape = env.single_observation_space.shape
    action_shape = env.single_action_space.n

    policy_net = QNetwork(args, action_shape)
    target_net = QNetwork(args, action_shape)
    target_net.load_state_dict(policy_net.state_dict())

    replay_memory = ReplayMemory(args.buffer_size, args.batch_size, obversation_shape, args.device)

    state, _ = env.reset(seed=args.seed) if args.seed > 0 else env.reset()

    for global_step in tqdm(range(args.total_timesteps)):

        state = torch.from_numpy(state).to(args.device).float()

        # Generate transitions
        with torch.no_grad():
            exploration_prob = get_exploration_prob(args, global_step)

            # Log metrics on Tensorboard
            writer.add_scalar("rollout/eps_threshold", exploration_prob, global_step)

            # Choice between exploration or intensification
            if np.random.rand() < exploration_prob:
                action = torch.tensor([env.single_action_space.sample()]).to(args.device)
            else:
                action = torch.argmax(policy_net(state), dim=1)

        next_state, reward, terminated, truncated, infos = env.step(action.cpu().numpy())

        reward = torch.from_numpy(reward).to(args.device).float()
        flag = torch.from_numpy(np.logical_or(terminated, truncated)).to(args.device).float()

        replay_memory.push(state, action, reward, flag)

        state = next_state

        if "final_info" in infos:
            info = infos["final_info"][0]
            writer.add_scalar("rollout/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("rollout/episodic_length", info["episode"]["l"], global_step)

        # Update policy
        if global_step > args.learning_start:
            if global_step % args.train_frequency == 0:
                states, actions, rewards, next_states, flags = replay_memory.sample()

                td_predict = policy_net(states).gather(1, actions).squeeze()

                with torch.no_grad():
                    action_by_qvalue = policy_net(next_states).argmax(1).unsqueeze(-1)
                    max_q_target = target_net(next_states).gather(1, action_by_qvalue).squeeze()

                td_target = rewards + (1.0 - flags) * args.gamma * max_q_target

                loss = mse_loss(td_predict, td_target)

                policy_net.optimizer.zero_grad()
                loss.backward()
                policy_net.optimizer.step()

                # Update target network
                target_net_dict = dict(target_net.state_dict())
                policy_net_dict = dict(policy_net.state_dict())

                for key, value in policy_net_dict.items():
                    target_net_dict[key] = value * args.tau + target_net_dict[key] * (
                        1.0 - args.tau
                    )
                target_net.load_state_dict(target_net_dict)

                # Log metrics on Tensorboard
                writer.add_scalar("update/loss", loss, global_step)

    env.close()
    writer.close()


if __name__ == "__main__":
    main()
