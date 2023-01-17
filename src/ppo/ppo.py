import argparse
import random
import time
from datetime import datetime
from pathlib import Path
from warnings import simplefilter

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical
from torch.nn.functional import mse_loss
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

simplefilter(action="ignore", category=DeprecationWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLander-v2")
    parser.add_argument("--total-timesteps", type=int, default=int(5e5))
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--num-minibatches", type=int, default=32)
    parser.add_argument("--num-optims", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--list-layer", nargs="+", type=int, default=[64, 64])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae", type=float, default=0.95)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--value-factor", type=float, default=0.5)
    parser.add_argument("--entropy-factor", type=float, default=0.01)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    args.device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = int(args.total_timesteps // args.num_steps)

    return args


def make_env(env_id, idx, run_dir, capture_video):
    def thunk():

        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
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

        current_layer_value = np.prod(obversation_shape)

        self.actor_net = nn.Sequential()
        self.critic_net = nn.Sequential()

        for layer_value in args.list_layer:
            self.actor_net.append(layer_init(nn.Linear(current_layer_value, layer_value)))
            self.actor_net.append(nn.Tanh())

            self.critic_net.append(layer_init(nn.Linear(current_layer_value, layer_value)))
            self.critic_net.append(nn.Tanh())

            current_layer_value = layer_value

        self.actor_net.append(layer_init(nn.Linear(args.list_layer[-1], action_shape), std=0.01))

        self.critic_net.append(layer_init(nn.Linear(args.list_layer[-1], 1), std=1.0))

        self.optimizer = optim.AdamW(self.parameters(), lr=args.learning_rate)

        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: 1.0 - (epoch - 1.0) / args.num_updates,
        )

        if args.device.type == "cuda":
            self.cuda()

    def forward(self):
        pass

    def get_action_value(self, state, action=None):

        actor_value = self.actor_net(state)
        distribution = Categorical(logits=actor_value)

        if action is None:
            action = distribution.sample()

        log_prob = distribution.log_prob(action)
        dist_entropy = distribution.entropy()

        critic_value = self.critic_net(state).squeeze()

        return action.cpu().numpy(), log_prob, critic_value, dist_entropy

    def get_value(self, state):

        return self.critic_net(state)


def main():
    args = parse_args()

    date = str(datetime.now().strftime("%d-%m_%H:%M:%S"))
    run_dir = Path(Path(__file__).parent.resolve().parents[1], "runs", f"{args.env}__ppo__{date}")

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
    action_shape = envs.single_action_space.n

    # Create the policy network
    policy_net = ActorCriticNet(args, obversation_shape, action_shape)

    # Initialize batch variables
    states = torch.zeros((args.num_steps, args.num_envs) + obversation_shape).to(args.device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    flags = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    log_probs = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    state_values = torch.zeros((args.num_steps, args.num_envs)).to(args.device)

    # Generate the initial state of the environment
    state, _ = envs.reset(seed=args.seed) if args.seed > 0 else envs.reset()

    global_step = 0
    start_time = time.process_time()

    for _ in tqdm(range(args.num_updates)):

        # Generate transitions
        for i in range(args.num_steps):
            global_step += 1

            with torch.no_grad():
                state_tensor = torch.from_numpy(state).to(args.device).float()
                action, log_prob, state_value, _ = policy_net.get_action_value(state_tensor)

            next_state, reward, terminated, truncated, infos = envs.step(action)

            states[i] = state_tensor
            actions[i] = torch.from_numpy(action).to(args.device)
            rewards[i] = torch.from_numpy(reward).to(args.device)
            log_probs[i] = log_prob
            state_values[i] = state_value
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

        # Compute values with GAE
        with torch.no_grad():
            next_state_tensor = torch.from_numpy(next_state).to(args.device).float()
            next_state_value = policy_net.get_value(next_state_tensor).squeeze(-1)

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

        # Update policy
        for _ in range(args.num_optims):

            # Shuffle batch
            np.random.shuffle(batch_indexes)

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                index = batch_indexes[start:end]

                _, new_log_probs, td_predict, dist_entropy = policy_net.get_action_value(
                    states_batch[index], actions_batch[index]
                )

                logratio = new_log_probs - logprobs_batch[index]
                ratios = logratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratios - 1) - logratio).mean()
                    clipfracs += [((ratios - 1.0).abs() > 0.2).float().mean().item()]

                surr1 = advantages_batch[index] * ratios

                surr2 = advantages_batch[index] * torch.clamp(
                    ratios, 1.0 - args.eps_clip, 1.0 + args.eps_clip
                )

                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = args.value_factor * mse_loss(td_predict, td_target_batch[index])

                entropy_bonus = args.entropy_factor * dist_entropy.mean()

                loss = actor_loss + critic_loss - entropy_bonus

                policy_net.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(policy_net.parameters(), 0.5)
                policy_net.optimizer.step()

        # Annealing learning rate
        policy_net.scheduler.step()

        # Log metrics on Tensorboard
        writer.add_scalar("update/actor_loss", actor_loss, global_step)
        writer.add_scalar("update/critic_loss", critic_loss, global_step)
        writer.add_scalar("debug/old_approx_kl", old_approx_kl, global_step)
        writer.add_scalar("debug/approx_kl", approx_kl, global_step)
        writer.add_scalar("debug/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar(
            "rollout/SPS", int(global_step / (time.process_time() - start_time)), global_step
        )

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
