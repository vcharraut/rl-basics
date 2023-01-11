import argparse
import random
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
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--list-layer", nargs="+", type=int, default=[64, 64])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    _args = parser.parse_args()

    _args.device = torch.device("cpu" if _args.cpu or not torch.cuda.is_available() else "cuda")
    _args.batch_size = int(_args.num_envs * _args.num_steps)
    _args.num_updates = int(_args.total_timesteps // _args.num_steps)

    return _args


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

        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate)

        if args.device.type == "cuda":
            self.cuda()

    def forward(self):
        pass

    def get_action(self, state):

        actor_value = self.actor_net(state)
        distribution = Categorical(logits=actor_value)
        action = distribution.sample()

        return action.cpu().numpy()

    def get_logprob_value(self, state, action):

        actor_value = self.actor_net(state)
        distribution = Categorical(logits=actor_value)
        log_prob = distribution.log_prob(action)

        critic_value = self.critic_net(state).squeeze()

        return log_prob, critic_value


def main():
    args = parse_args()

    date = str(datetime.now().strftime("%d-%m_%H:%M:%S"))
    run_dir = Path(Path(__file__).parent.resolve().parent, "../runs", f"{args.env}__a2c__{date}")
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

    obversation_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.n

    policy_net = ActorCriticNet(args, obversation_shape, action_shape)

    # Initialize batch variables
    states = torch.zeros((args.num_steps, args.num_envs) + obversation_shape).to(args.device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    flags = torch.zeros((args.num_steps, args.num_envs)).to(args.device)

    state, _ = envs.reset(seed=args.seed) if args.seed > 0 else envs.reset()

    global_step = 0

    for _ in tqdm(range(args.num_updates)):

        # Generate transitions
        for i in range(args.num_steps):
            global_step += 1

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

        policy_loss = (-log_probs * advantages).mean()
        value_loss = mse_loss(td_target_batch, td_predict)

        loss = policy_loss + value_loss

        policy_net.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(policy_net.parameters(), 0.5)
        policy_net.optimizer.step()

        # Log metrics on Tensorboard
        writer.add_scalar("update/policy_loss", policy_loss, global_step)
        writer.add_scalar("update/value_loss", value_loss, global_step)

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
