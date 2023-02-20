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
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--num_optims", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae", type=float, default=0.95)
    parser.add_argument("--eps_clip", type=float, default=0.1)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--clip_grad_norm", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    args.device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
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
        )

        self.actor_net = layer_init(nn.Linear(512, action_shape), std=0.01)
        self.critic_net = layer_init(nn.Linear(512, 1), std=1)

        if args.device.type == "cuda":
            self.cuda()

    def forward(self, state):
        output = self.network(state)
        actor_value = self.actor_net(output)
        distribution = Categorical(logits=actor_value)

        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        critic_value = self.critic_net(output).squeeze()

        return action.cpu().numpy(), log_prob, critic_value

    def evaluate(self, states, actions):
        output = self.network(states)
        actor_value = self.actor_net(output)
        distribution = Categorical(logits=actor_value)

        log_probs = distribution.log_prob(actions)
        dist_entropy = distribution.entropy()

        critic_values = self.critic_net(output).squeeze()

        return log_probs, critic_values, dist_entropy

    def critic(self, state):
        return self.critic_net(self.network(state)).squeeze(-1)


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
    policy_net = ActorCriticNet(args, action_shape)
    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 1.0 - (epoch - 1.0) / args.num_updates
    )

    # Create buffers
    states = torch.zeros((args.num_steps, args.num_envs) + obversation_shape).to(args.device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    flags = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    log_probs = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    state_values = torch.zeros((args.num_steps, args.num_envs)).to(args.device)

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
                state_tensor = torch.from_numpy(state).to(args.device).float()
                action, log_prob, state_value = policy_net(state_tensor)

            # Perform action
            next_state, reward, terminated, truncated, infos = envs.step(action)

            # Store transition
            states[i] = state_tensor
            actions[i] = torch.from_numpy(action).to(args.device)
            rewards[i] = torch.from_numpy(reward).to(args.device)
            log_probs[i] = log_prob
            state_values[i] = state_value
            flags[i] = torch.from_numpy(np.logical_or(terminated, truncated)).to(args.device)

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

        # Compute TD target and advantages with GAE
        with torch.no_grad():
            next_state_tensor = torch.from_numpy(next_state).to(args.device).float()
            next_state_value = policy_net.critic(next_state_tensor)

        advantages = torch.zeros(rewards.size()).to(args.device)
        adv = torch.zeros(rewards.size(1)).to(args.device)

        for i in reversed(range(rewards.size(0))):
            terminal = 1.0 - flags[i]

            returns = rewards[i] + args.gamma * next_state_value * terminal
            delta = returns - state_values[i]

            adv = args.gamma * args.gae * adv * terminal + delta
            advantages[i] = adv

            next_state_value = state_values[i]

        td_target = (advantages + state_values).squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        advantages = advantages.squeeze()

        # Flatten batch
        states_batch = states.flatten(0, 1)
        actions_batch = actions.flatten(0, 1)
        logprobs_batch = log_probs.reshape(-1)
        td_target_batch = td_target.reshape(-1)
        advantages_batch = advantages.reshape(-1)

        batch_indexes = np.arange(args.batch_size)

        clipfracs = []

        # Perform PPO update
        for _ in range(args.num_optims):

            # Shuffle batch
            np.random.shuffle(batch_indexes)

            # Perform minibatch updates
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                index = batch_indexes[start:end]

                # Calculate new values from minibatch
                new_log_probs, td_predict, dist_entropy = policy_net.evaluate(
                    states_batch[index], actions_batch[index]
                )

                # Calculate ratios
                logratio = new_log_probs - logprobs_batch[index]
                ratios = logratio.exp()

                # Calculate approx_kl (http://joschu.net/blog/kl-approx.html)
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratios - 1) - logratio).mean()
                    clipfracs += [((ratios - 1.0).abs() > 0.2).float().mean().item()]

                # Calculate surrogates
                surr1 = advantages_batch[index] * ratios
                surr2 = advantages_batch[index] * torch.clamp(
                    ratios, 1.0 - args.eps_clip, 1.0 + args.eps_clip
                )

                # Calculate losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = mse_loss(td_predict, td_target_batch[index])
                entropy_bonus = dist_entropy.mean()

                loss = (
                    actor_loss + critic_loss * args.value_coef - entropy_bonus * args.entropy_coef
                )

                # Update policy network
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(policy_net.parameters(), args.clip_grad_norm)
                optimizer.step()

        # Annealing learning rate
        scheduler.step()

        # Log training metrics
        writer.add_scalar("train/actor_loss", actor_loss, global_step)
        writer.add_scalar("train/critic_loss", critic_loss, global_step)
        writer.add_scalar("train/old_approx_kl", old_approx_kl, global_step)
        writer.add_scalar("train/approx_kl", approx_kl, global_step)
        writer.add_scalar("train/clipfrac", np.mean(clipfracs), global_step)
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
                state_tensor = torch.from_numpy(state).to(args.device).float()
                action, _, _ = policy_net(state_tensor)

            state, _, terminated, truncated, _ = env_test.step(action)

            if terminated or truncated:
                count_episodes += 1

        env_test.close()
        print("Done!")
