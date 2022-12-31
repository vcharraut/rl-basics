import argparse
import time
from datetime import datetime
from warnings import simplefilter

import torch
from torch import optim, nn
from torch.nn.functional import mse_loss
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions import Categorical
from torch.utils.tensorboard.writer import SummaryWriter
import gymnasium as gym
import numpy as np
from tqdm import tqdm

simplefilter(action="ignore", category=DeprecationWarning)

SEED = 24


def make_env(env_id, idx, run_name, capture_video):

    def thunk():

        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.AtariPreprocessing(env, scale_obs=True)
        env = gym.wrappers.FrameStack(env, 4)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env=env,
                video_folder=f"../runs/{run_name}/videos/",
                disable_logger=True)
        return env

    return thunk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLander-v2")
    parser.add_argument("--total-timesteps", type=int, default=int(1e6))
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--num-minibatches", type=int, default=32)
    parser.add_argument("--num-updates", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument('--list-layer', nargs="+", type=int, default=[64, 64])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae", type=float, default=0.95)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--value-factor", type=float, default=0.5)
    parser.add_argument("--entropy-factor", type=float, default=0.01)
    parser.add_argument("--shared-network", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--capture-video", action="store_true")

    _args = parser.parse_args()

    _args.device = torch.device(
        "cpu" if _args.cpu or not torch.cuda.is_available() else "cuda")
    _args.batch_size = int(_args.num_envs * _args.num_steps)
    _args.minibatch_size = int(_args.batch_size // _args.num_minibatches)

    return _args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):

    def __init__(self, args, action_shape: tuple):

        super().__init__()

        num_actions = np.prod(action_shape)

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
        self.actor_neural_net = layer_init(nn.Linear(512, num_actions),
                                           std=0.01)
        self.critic_neural_net = layer_init(nn.Linear(512, 1), std=1)

        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate)

        if args.device.type == "cuda":
            self.cuda()

    def forward(self, state: torch.Tensor, action=None) -> torch.Tensor:

        output = self.network(state)
        actor_value = self.actor_neural_net(output)
        distribution = Categorical(logits=actor_value)

        if action is None:
            action = distribution.sample()

        log_prob = distribution.log_prob(action)
        dist_entropy = distribution.entropy()

        critic_value = self.critic_neural_net(output)

        return action.cpu().numpy(), log_prob, critic_value.squeeze(
        ), dist_entropy

    def critic(self, state):
        output = self.network(state)
        return self.critic_neural_net(output)


def main():
    args = parse_args()

    date = str(datetime.now().strftime("%d-%m_%H:%M:%S"))
    run_name = f"{args.env}__ppo__{date}"
    writer = SummaryWriter(f"../runs/{run_name}")
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

    action_space = envs.single_action_space

    agent = Agent(args, action_space)

    obversation_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape

    states = torch.zeros((args.num_steps, args.num_envs) +
                         obversation_shape).to(args.device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape).to(
        args.device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    flags = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    log_probs = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    state_values = torch.zeros((args.num_steps, args.num_envs)).to(args.device)

    num_updates = int(args.total_timesteps // args.num_steps)
    global_step = 0

    state, _ = envs.reset(seed=SEED)

    for _ in tqdm(range(num_updates)):
        start = time.perf_counter()

        # Generate transitions
        for i in range(args.num_steps):
            global_step += 1

            with torch.no_grad():
                state_torch = torch.from_numpy(state).to(args.device).float()
                action, log_prob, state_value, _ = agent(state_torch)

            next_state, reward, terminated, truncated, infos = envs.step(
                action)

            states[i] = state_torch
            actions[i] = torch.from_numpy(action).to(args.device)
            rewards[i] = torch.from_numpy(reward).to(args.device)
            log_probs[i] = log_prob
            state_values[i] = state_value

            done = np.logical_or(terminated, truncated)
            flags[i] = torch.from_numpy(done).to(args.device)

            state = next_state

            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                if info is None:
                    continue

                writer.add_scalar("rollout/episodic_return",
                                  info["episode"]["r"], global_step)
                writer.add_scalar("rollout/episodic_length",
                                  info["episode"]["l"], global_step)

        end = time.perf_counter()
        writer.add_scalar("rollout/time", end - start, global_step)

        # Compute values
        with torch.no_grad():
            state_torch = torch.from_numpy(state).to(args.device).float()
            next_state_value = agent.critic(state_torch).squeeze(-1)

        advantages = torch.zeros(rewards.size()).to(args.device)
        adv = torch.zeros(rewards.size(1)).to(args.device)

        for i in reversed(range(rewards.size(0))):
            terminal = 1. - flags[i]

            returns = rewards[i] + args.gamma * next_state_value * terminal
            delta = returns - state_values[i]

            adv = args.gamma * args.gae * adv * terminal + delta
            advantages[i] = adv

            next_state_value = state_values[i]

        td_target = (advantages + state_values).squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() +
                                                         1e-8)
        advantages = advantages.squeeze()

        # Shuffle batch
        b_states = states.flatten(0, 1)
        b_actions = actions.flatten(0, 1)
        b_log_probs = log_probs.reshape(-1)
        b_td_target = td_target.reshape(-1)
        b_advantages = advantages.reshape(-1)

        batch_index = torch.randperm(args.batch_size)

        b_states = b_states[batch_index]
        b_ations = b_actions[batch_index]
        b_log_probs = b_log_probs[batch_index]
        b_td_target = b_td_target[batch_index]
        b_advantages = b_advantages[batch_index]

        # Update policy
        clipfracs = []
        for _ in range(args.num_updates):
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                index = slice(start, end)

                _, new_log_probs, td_predict, dist_entropy = agent(
                    b_states[index], b_ations[index])

                logratio = new_log_probs - b_log_probs[index]
                ratios = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratios - 1) - logratio).mean()
                    clipfracs += [
                        ((ratios - 1.0).abs() > 0.2).float().mean().item()
                    ]

                surr1 = b_advantages[index] * ratios

                surr2 = b_advantages[index] * torch.clamp(
                    ratios, 1. - args.eps_clip, 1. + args.eps_clip)

                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = args.value_factor * mse_loss(
                    td_predict, b_td_target[index])

                entropy_bonus = args.entropy_factor * dist_entropy.mean()

                loss = policy_loss + value_loss - entropy_bonus

                agent.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(agent.parameters(), 0.5)
                agent.optimizer.step()

        writer.add_scalar("update/policy_loss", policy_loss, global_step)
        writer.add_scalar("update/value_loss", value_loss, global_step)
        writer.add_scalar("debug/old_approx_kl", old_approx_kl, global_step)
        writer.add_scalar("debug/approx_kl", approx_kl, global_step)
        writer.add_scalar("debug/clipfrac", np.mean(clipfracs), global_step)

    envs.close()
    writer.close()


if __name__ == '__main__':
    main()
