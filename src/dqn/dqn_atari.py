import argparse
import math
import random
import time
from collections import deque, namedtuple
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.nn.functional import mse_loss
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--total_timesteps", type=int, default=5_000_000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--buffer_size", type=int, default=500_000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_start", type=int, default=1)
    parser.add_argument("--eps_decay", type=int, default=200_000)
    parser.add_argument("--learning_start", type=int, default=50_000)
    parser.add_argument("--train_frequency", type=int, default=4)
    parser.add_argument("--target_update_frequency", type=int, default=10_000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    args.device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    return args


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
        env = gym.wrappers.AtariPreprocessing(env)
        env = gym.wrappers.FrameStack(env, 4)

        return env

    return thunk


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

        self.transition = namedtuple(
            "Transition", field_names=["state", "action", "reward", "flag"]
        )

    @property
    def size(self):
        return len(self.buffer)

    def push(self, state, action, reward, flag):
        state = state.to(dtype=torch.uint8).squeeze()
        self.buffer.append(self.transition(state, action, reward, flag))

    def sample(self):
        sample_indices = np.random.choice(self.size - 1, size=self.batch_size)

        samples = [
            (
                self.buffer[index].state,
                self.buffer[index].action,
                self.buffer[index].reward,
                self.buffer[index + 1].state,
                self.buffer[index].flag,
            )
            for index in sample_indices
        ]

        states, actions, rewards, next_states, flags = zip(*samples)

        states = torch.stack(states).to(self.device).float()
        next_states = torch.stack(next_states).to(self.device).float()
        actions = torch.cat(actions).to(self.device).type(torch.int64).unsqueeze(-1)
        rewards = torch.cat(rewards).to(self.device)
        flags = torch.cat(flags).to(self.device)

        return states, actions, rewards, next_states, flags


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class QNetwork(nn.Module):
    def __init__(self, args, action_shape):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_shape),
        )

        if args.device.type == "cuda":
            self.cuda()

    def forward(self, state):
        return self.network(state)


def get_exploration_prob(args, step):
    return args.eps_end + (args.eps_start - args.eps_end) * math.exp(-1.0 * step / args.eps_decay)


def main():
    args = parse_args()

    date = str(datetime.now().strftime("%d-%m_%H:%M"))
    algo_name = Path(__file__).stem.split("_")[0].upper()
    run_dir = Path(
        Path(__file__).parent.resolve().parents[1], "runs", f"{args.env}__{algo_name}__{date}"
    )

    if args.wandb:
        wandb.init(
            project=args.env,
            name=algo_name,
            sync_tensorboard=True,
            config=vars(args),
            dir=run_dir,
            save_code=True,
        )

    # Create writer for Tensorboard
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

    # Create vectorized environment
    env = gym.vector.SyncVectorEnv([make_env(args.env, run_dir, args.capture_video)])

    # Metadata about the environment
    action_shape = env.single_action_space.n

    # Create the policy networks
    policy_net = QNetwork(args, action_shape)
    target_net = QNetwork(args, action_shape)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)

    # Create the replay buffer
    replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size, args.device)

    # Generate the initial state of the environment
    state, _ = env.reset(seed=args.seed) if args.seed > 0 else env.reset()

    start_time = time.process_time()

    for global_step in tqdm(range(args.total_timesteps)):

        state = torch.from_numpy(state).to(args.device).float()

        # Generate transitions
        with torch.no_grad():
            exploration_prob = get_exploration_prob(args, global_step)

            # Log metrics on Tensorboard
            writer.add_scalar("rollout/eps_threshold", exploration_prob, global_step)

            # Choice between exploration or intensification
            if np.random.rand() < exploration_prob:
                action = torch.randint(action_shape, (1,)).to(args.device)
            else:
                action = torch.argmax(policy_net(state), dim=1)

        next_state, reward, terminated, truncated, infos = env.step(action.cpu().numpy())

        reward = torch.from_numpy(reward).to(args.device).float()
        flag = torch.from_numpy(np.logical_or(terminated, truncated)).to(args.device).float()

        replay_buffer.push(state, action, reward, flag)

        state = next_state

        if "final_info" in infos:
            info = infos["final_info"][0]
            writer.add_scalar("rollout/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("rollout/episodic_length", info["episode"]["l"], global_step)

        # Update policy
        if global_step > args.learning_start:
            if not global_step % args.train_frequency:
                states, actions, rewards, next_states, flags = replay_buffer.sample()

                td_predict = policy_net(states).gather(1, actions).squeeze()

                with torch.no_grad():
                    # Double Q-Learning
                    action_by_qvalue = policy_net(next_states).argmax(1).unsqueeze(-1)
                    max_q_target = target_net(next_states).gather(1, action_by_qvalue).squeeze()

                td_target = rewards + (1.0 - flags) * args.gamma * max_q_target

                loss = mse_loss(td_predict, td_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update target network
                for param, target_param in zip(policy_net.parameters(), target_net.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

                # Log metrics on Tensorboard
                writer.add_scalar("train/loss", loss, global_step)

            # if not global_step % args.target_update_frequency:
            #     target_net.load_state_dict(policy_net.state_dict())

        writer.add_scalar(
            "rollout/SPS", int(global_step / (time.process_time() - start_time)), global_step
        )

    env.close()
    writer.close()


if __name__ == "__main__":
    main()
