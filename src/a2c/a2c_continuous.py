import argparse
import random
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import wandb
from torch import nn, optim
from torch.distributions import Normal
from torch.nn.functional import mse_loss
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--total_timesteps", type=int, default=int(1e6))
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=7e-4)
    parser.add_argument("--list_layer", nargs="+", type=int, default=[64, 64])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--value_factor", type=float, default=0.5)
    parser.add_argument("--entropy_factor", type=float, default=0.01)
    parser.add_argument("--clip_grad_norm", type=float, default=0.5)
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_updates = int(args.total_timesteps // args.batch_size)

    return args


def make_env(env_id, idx, run_dir, capture_video):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=0.99)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env=env, video_folder=f"{run_dir}/videos/", disable_logger=True
            )
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticNet(nn.Module):
    def __init__(self, args, obversation_shape, action_shape):
        super().__init__()

        fc_layer_value = np.prod(obversation_shape)
        action_shape = np.prod(action_shape)

        self.actor_net = nn.Sequential()
        self.critic_net = nn.Sequential()

        for layer_value in args.list_layer:
            self.actor_net.append(layer_init(nn.Linear(fc_layer_value, layer_value)))
            self.actor_net.append(nn.Tanh())

            self.critic_net.append(layer_init(nn.Linear(fc_layer_value, layer_value)))
            self.critic_net.append(nn.Tanh())

            fc_layer_value = layer_value

        self.actor_net.append(layer_init(nn.Linear(args.list_layer[-1], action_shape), std=0.01))

        self.critic_net.append(layer_init(nn.Linear(args.list_layer[-1], 1), std=1.0))

        self.actor_logstd = nn.Parameter(torch.zeros(1, action_shape))

    def forward(self, state):
        action_mean = self.actor_net(state)
        action_std = self.actor_logstd.expand_as(action_mean).exp()
        distribution = Normal(action_mean, action_std)

        action = distribution.sample()

        return action.cpu().numpy()

    def evaluate(self, state, action):
        action_mean = self.actor_net(state)
        action_std = self.actor_logstd.expand_as(action_mean).exp()
        distribution = Normal(action_mean, action_std)

        log_probs = distribution.log_prob(action).sum(-1)
        dist_entropy = distribution.entropy().sum(-1)

        critic_values = self.critic_net(state).squeeze()

        return log_probs, critic_values, dist_entropy


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

    # Create vectorized environment(s)
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env, i, run_dir, args.capture_video) for i in range(args.num_envs)]
    )

    # Metadata about the environment
    obversation_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape

    # Create the policy network
    policy_net = ActorCriticNet(args, obversation_shape, action_shape)

    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)

    # Initialize batch variables
    states = torch.zeros((args.num_steps, args.num_envs) + obversation_shape)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape)
    rewards = torch.zeros((args.num_steps, args.num_envs))
    flags = torch.zeros((args.num_steps, args.num_envs))

    # Generate the initial state of the environment
    state, _ = envs.reset(seed=args.seed) if args.seed > 0 else envs.reset()

    global_step = 0
    start_time = time.process_time()

    for _ in tqdm(range(args.num_updates)):

        # Generate transitions
        for i in range(args.num_steps):
            global_step += 1 * args.num_envs

            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float()
                action = policy_net(state_tensor)

            next_state, reward, terminated, truncated, infos = envs.step(action)

            states[i] = state_tensor
            actions[i] = torch.from_numpy(action)
            rewards[i] = torch.from_numpy(reward)
            flags[i] = torch.from_numpy(np.logical_or(terminated, truncated))

            state = next_state

            if "final_info" not in infos:
                continue

            # Log episode metrics on Tensorboard
            for info in infos["final_info"]:
                if info is None:
                    continue

                writer.add_scalar("rollout/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("rollout/episodic_length", info["episode"]["l"], global_step)

        # Compute values
        td_target = torch.zeros(rewards.size())
        gain = torch.zeros(rewards.size(1))

        for i in reversed(range(td_target.size(0))):
            terminal = 1.0 - flags[i]
            gain = rewards[i] + gain * args.gamma * terminal
            td_target[i] = gain

        td_target = (td_target - td_target.mean()) / (td_target.std() + 1e-7)
        td_target = td_target.squeeze()

        # Flatten batch
        states_batch = states.flatten(0, 1)
        actions_batch = actions.flatten(0, 1)
        td_target_batch = td_target.reshape(-1)

        # Update policy
        log_probs, td_predict, dist_entropy = policy_net.evaluate(states_batch, actions_batch)
        advantages = td_target_batch - td_predict

        actor_loss = (-log_probs * advantages.detach()).mean()
        critic_loss = mse_loss(td_target_batch, td_predict) * args.value_factor
        entropy_bonus = dist_entropy.mean() * args.entropy_factor

        loss = actor_loss + critic_loss - entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(policy_net.parameters(), args.clip_grad_norm)
        optimizer.step()

        # Log metrics on Tensorboard
        writer.add_scalar("train/actor_loss", actor_loss, global_step)
        writer.add_scalar("train/critic_loss", critic_loss, global_step)
        writer.add_scalar(
            "rollout/SPS", int(global_step / (time.process_time() - start_time)), global_step
        )

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
