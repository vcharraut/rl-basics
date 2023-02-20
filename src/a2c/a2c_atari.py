import argparse
import random
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical
from torch.nn.functional import mse_loss
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--total_timesteps", type=int, default=5_000_000)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--clip_grad_norm", type=float, default=0.5)
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_updates = int(args.total_timesteps // args.batch_size)

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
        env = gym.wrappers.AtariPreprocessing(env)
        env = gym.wrappers.FrameStack(env, 4)

        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticNet(nn.Module):
    def __init__(self, action_shape):
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
        )

        self.actor_net = layer_init(nn.Linear(512, action_shape), std=0.01)
        self.critic_net = layer_init(nn.Linear(512, 1), std=1)

    def forward(self, state):
        output = self.network(state)
        actor_value = self.actor_net(output)
        distribution = Categorical(logits=actor_value)

        action = distribution.sample()

        return action.cpu().numpy()

    def evaluate(self, state, action):
        output = self.network(state)
        actor_value = self.actor_net(output)

        distribution = Categorical(logits=actor_value)

        log_probs = distribution.log_prob(action)
        dist_entropy = distribution.entropy()

        critic_values = self.critic_net(output).squeeze()

        return log_probs, critic_values, dist_entropy


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

    # Create vectorized environment(s)
    envs = gym.vector.AsyncVectorEnv([make_env(args.env_id) for _ in range(args.num_envs)])

    # Metadata about the environment
    obversation_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.n

    # Create policy network and optimizer
    policy_net = ActorCriticNet(action_shape)
    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)

    # Create buffers
    states = torch.zeros((args.num_steps, args.num_envs) + obversation_shape)
    actions = torch.zeros((args.num_steps, args.num_envs))
    rewards = torch.zeros((args.num_steps, args.num_envs))
    flags = torch.zeros((args.num_steps, args.num_envs))

    log_episodic_returns = []

    # Initialize environment
    state, _ = envs.reset(seed=args.seed) if args.seed > 0 else envs.reset()

    global_step = 0
    start_time = time.process_time()

    # Main loop
    for _ in tqdm(range(args.num_updates)):

        for i in range(args.num_steps):
            # Update global step
            global_step += 1 * args.num_envs

            with torch.no_grad():
                # Get action
                state_tensor = torch.from_numpy(state).float()
                action = policy_net(state_tensor)

            # Perform action
            next_state, reward, terminated, truncated, infos = envs.step(action)

            # Store transition
            states[i] = state_tensor
            actions[i] = torch.from_numpy(action)
            rewards[i] = torch.from_numpy(reward)
            flags[i] = torch.from_numpy(np.logical_or(terminated, truncated))

            state = next_state

            if "final_info" not in infos:
                continue

            # Log episodic return and length
            for info in infos["final_info"]:
                if info is None:
                    continue

                log_episodic_returns.append(info["episode"]["r"])
                writer.add_scalar("rollout/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("rollout/episodic_length", info["episode"]["l"], global_step)

                break

        td_target = torch.zeros(rewards.size())
        gain = torch.zeros(rewards.size(1))

        # Compute TD target
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

        # Compute losses
        log_probs, td_predict, dist_entropy = policy_net.evaluate(states_batch, actions_batch)
        advantages = td_target_batch - td_predict

        actor_loss = (-log_probs * advantages.detach()).mean()
        critic_loss = mse_loss(td_target_batch, td_predict)
        entropy_bonus = dist_entropy.mean()

        loss = actor_loss + critic_loss * args.value_coef - entropy_bonus * args.entropy_coef

        # Update policy network
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(policy_net.parameters(), args.clip_grad_norm)
        optimizer.step()

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
    envs.close()
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
                action = policy_net(torch.from_numpy(state).float())
                state, _, terminated, truncated, _ = env_test.step(action)

            state_tensor = torch.from_numpy(state).float()
            action = policy_net(state_tensor)

            if terminated or truncated:
                count_episodes += 1

        env_test.close()
        print("Done!")
