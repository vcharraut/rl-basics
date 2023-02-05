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
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=7e-4)
    parser.add_argument("--list_layer", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    args.device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
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

        if args.device.type == "cuda":
            self.cuda()

    def forward(self):
        pass

    def get_action(self, state):
        action_mean = self.actor_net(state)
        action_std = self.actor_logstd.expand_as(action_mean).exp()
        distribution = Normal(action_mean, action_std)

        action = distribution.sample()

        return action.cpu().numpy()

    def get_logprob_value(self, state, action):
        action_mean = self.actor_net(state)
        action_std = self.actor_logstd.expand_as(action_mean).exp()
        distribution = Normal(action_mean, action_std)

        log_prob = distribution.log_prob(action).sum(-1)

        critic_value = self.critic_net(state).squeeze()

        return log_prob, critic_value


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
    states = torch.zeros((args.num_steps, args.num_envs) + obversation_shape).to(args.device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape).to(args.device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    flags = torch.zeros((args.num_steps, args.num_envs)).to(args.device)

    # Generate the initial state of the environment
    state, _ = envs.reset(seed=args.seed) if args.seed > 0 else envs.reset()

    global_step = 0
    start_time = time.process_time()

    for _ in tqdm(range(args.num_updates)):

        # Generate transitions
        for i in range(args.num_steps):
            global_step += 1 * args.num_envs

            with torch.no_grad():
                state_tensor = torch.from_numpy(state).to(args.device).float()
                action = policy_net.get_action(state_tensor)

            next_state, reward, terminated, truncated, infos = envs.step(action)

            states[i] = state_tensor
            actions[i] = torch.from_numpy(action).to(args.device)
            rewards[i] = torch.from_numpy(reward).to(args.device)
            flags[i] = torch.from_numpy(np.logical_or(terminated, truncated)).to(args.device)

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
        td_target = torch.zeros(rewards.size()).to(args.device)
        gain = torch.zeros(rewards.size(1)).to(args.device)

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

        # Shuffle batch
        batch_indexes = torch.randperm(args.batch_size)

        states_batch = states_batch[batch_indexes]
        actions_batch = actions_batch[batch_indexes]
        td_target_batch = td_target_batch[batch_indexes]

        # Update policy
        log_probs, td_predict = policy_net.get_logprob_value(states_batch, actions_batch)

        advantages = td_target_batch - td_predict

        actor_loss = (-log_probs * advantages).mean()
        critic_loss = mse_loss(td_target_batch, td_predict)

        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(policy_net.parameters(), 0.5)
        optimizer.step()

        # Log metrics on Tensorboard
        writer.add_scalar("update/actor_loss", actor_loss, global_step)
        writer.add_scalar("update/critic_loss", critic_loss, global_step)
        writer.add_scalar(
            "rollout/SPS", int(global_step / (time.process_time() - start_time)), global_step
        )

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
