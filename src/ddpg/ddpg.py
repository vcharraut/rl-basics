import argparse
import random
import time
from collections import deque, namedtuple
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Uniform
from torch.nn.functional import mse_loss
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v4")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--list_layer", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration_noise", type=float, default=0.1)
    parser.add_argument("--learning_start", type=int, default=25_000)
    parser.add_argument("--policy_frequency", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    args.device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    return args


def make_env(env_id, capture_video=False):
    def thunk():

        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env=env,
                video_folder=f"{run_dir}/videos/",
                episode_trigger=lambda x: x,
                disable_logger=True,
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        return env

    return thunk


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

        self.transition = namedtuple(
            "Transition", field_names=["state", "action", "reward", "next_state", "flag"]
        )

    def push(self, state, action, reward, next_state, flag):
        self.buffer.append(self.transition(state, action, reward, next_state, flag))

    def sample(self):
        batch = self.transition(*zip(*random.sample(self.buffer, self.batch_size)))

        states = torch.cat(batch.state).to(self.device)
        actions = torch.cat(batch.action).to(self.device)
        rewards = torch.cat(batch.reward).to(self.device)
        next_states = torch.cat(batch.next_state).to(self.device)
        flags = torch.cat(batch.flag).to(self.device)

        return states, actions, rewards, next_states, flags


class ActorNet(nn.Module):
    def __init__(self, args, obversation_shape, action_shape, action_low, action_high):
        super().__init__()

        fc_layer_value = np.prod(obversation_shape)
        action_shape = np.prod(action_shape)

        self.network = nn.Sequential()

        for layer_value in args.list_layer:
            self.network.append(nn.Linear(fc_layer_value, layer_value))
            self.network.append(nn.ReLU())
            fc_layer_value = layer_value

        self.network.append(nn.Linear(fc_layer_value, action_shape))

        # Scale and bias the output of the network to match the action space
        self.register_buffer("action_scale", ((action_high - action_low) / 2.0))
        self.register_buffer("action_bias", ((action_high + action_low) / 2.0))

        if args.device.type == "cuda":
            self.cuda()

    def forward(self, state):
        output = torch.tanh(self.network(state))
        return output * self.action_scale + self.action_bias


class CriticNet(nn.Module):
    def __init__(self, args, obversation_shape, action_shape):
        super().__init__()

        fc_layer_value = np.prod(obversation_shape) + np.prod(action_shape)

        self.network = nn.Sequential()

        for layer_value in args.list_layer:
            self.network.append(nn.Linear(fc_layer_value, layer_value))
            self.network.append(nn.ReLU())
            fc_layer_value = layer_value

        self.network.append(nn.Linear(fc_layer_value, 1))

        if args.device.type == "cuda":
            self.cuda()

    def forward(self, state, action):
        return self.network(torch.cat([state, action], 1))


if __name__ == "__main__":
    args = parse_args()

    date = str(datetime.now().strftime("%d-%m_%H:%M"))
    # These variables are specific to the repo "rl-gym-zoo"
    # You should change them if you are just copy/paste the code
    algo_name = Path(__file__).stem.split("_")[0].upper()
    run_dir = Path(
        Path(__file__).parent.resolve().parents[1], "runs", f"{args.env_id}__{algo_name}__{date}"
    )

    # Initialize wandb if needed (https://wandb.ai/)
    if args.wandb:
        import wandb

        wandb.init(project=args.env_id, name=algo_name, sync_tensorboard=True, config=vars(args))

    # Create tensorboard writer and save hyperparameters
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Set seed for reproducibility
    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Create vectorized environment
    env = gym.vector.SyncVectorEnv([make_env(args.env_id)])

    # Metadata about the environment
    obversation_shape = env.single_observation_space.shape
    action_shape = env.single_action_space.shape
    action_low = torch.from_numpy(env.single_action_space.low).to(args.device)
    action_high = torch.from_numpy(env.single_action_space.high).to(args.device)

    # Create the networks and the optimizer
    actor = ActorNet(args, obversation_shape, action_shape, action_low, action_high)
    critic = CriticNet(args, obversation_shape, action_shape)

    target_actor = ActorNet(args, obversation_shape, action_shape, action_low, action_high)
    target_critic = CriticNet(args, obversation_shape, action_shape)

    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    optimizer_actor = optim.Adam(actor.parameters(), lr=args.learning_rate)
    optimizer_critic = optim.Adam(critic.parameters(), lr=args.learning_rate)

    # Create the replay buffer
    replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size, args.device)

    # Initialize environment
    state, _ = env.reset(seed=args.seed) if args.seed > 0 else env.reset()
    state = torch.from_numpy(state).to(args.device).float()

    log_episodic_returns = []

    start_time = time.process_time()

    # Main loop
    for global_step in tqdm(range(args.total_timesteps)):

        if global_step < args.learning_start:
            action = Uniform(action_low, action_high).sample().unsqueeze(0)
        else:
            with torch.no_grad():
                action = actor(state)
                action += torch.normal(0, actor.action_scale * args.exploration_noise)

        # Perform action
        next_state, reward, terminated, truncated, infos = env.step(action.cpu().numpy())

        # Convert transition to torch tensors
        next_state = torch.from_numpy(next_state).to(args.device).float()
        reward = torch.from_numpy(reward).to(args.device).float()
        flag = torch.from_numpy(np.logical_or(terminated, truncated)).to(args.device).float()

        # Store transition in the replay buffer
        replay_buffer.push(state, action, reward, next_state, flag)

        state = next_state

        # Log episodic return and length
        if "final_info" in infos:
            info = infos["final_info"][0]

            log_episodic_returns.append(info["episode"]["r"])
            writer.add_scalar("rollout/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("rollout/episodic_length", info["episode"]["l"], global_step)

        # Perform training step
        if global_step > args.learning_start:
            # Sample a batch from the replay buffer
            states, actions, rewards, next_states, flags = replay_buffer.sample()

            # Update critic
            with torch.no_grad():
                next_state_actions = target_actor(next_states)
                critic_next_target = target_critic(next_states, next_state_actions).squeeze()

            td_target = rewards + (1.0 - flags) * args.gamma * critic_next_target
            td_predict = critic(states, actions).squeeze()
            critic_loss = mse_loss(td_predict, td_target)

            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            # Update actor
            if not global_step % args.policy_frequency:
                actor_loss = -critic(states, actor(states)).mean()
                optimizer_actor.zero_grad()
                actor_loss.backward()
                optimizer_actor.step()

                # Update the target network (soft update)
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

                # Log training metrics
                writer.add_scalar("train/actor_loss", actor_loss, global_step)
                writer.add_scalar("train/critic_loss", critic_loss, global_step)

        writer.add_scalar(
            "rollout/SPS", int(global_step / (time.process_time() - start_time)), global_step
        )

    # Average of episodic returns (for the last 5% of the training)
    indexes = int(len(log_episodic_returns) * 0.05)
    avg_final_rewards = np.mean(log_episodic_returns[-indexes:])
    print(f"Average of the last {indexes} episodic returns: {round(avg_final_rewards, 2)}")
    writer.add_scalar("rollout/avg_final_rewards", avg_final_rewards, global_step)

    # Close the environment
    env.close()
    writer.close()
    if args.wandb:
        wandb.finish()

    # Capture video of the policy
    if args.capture_video:
        print(f"Capturing videos and saving them to {run_dir}/videos ...")
        env_test = gym.vector.SyncVectorEnv([make_env(args.env_id, capture_video=True)])
        state, _ = env_test.reset()
        count_episodes = 0

        while count_episodes < 10:
            with torch.no_grad():
                state = torch.from_numpy(state).to(args.device).float()
                action = actor(state).detach()

            action += torch.normal(0, actor.action_scale * args.exploration_noise)

            state, _, terminated, truncated, _ = env_test.step(action.cpu().numpy())

            if terminated or truncated:
                count_episodes += 1

        env_test.close()
        print("Done!")
