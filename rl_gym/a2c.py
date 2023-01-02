import argparse
import time
from datetime import datetime
from warnings import simplefilter

import gymnasium as gym
import numpy as np
from tqdm import tqdm

import torch
from torch import optim, nn
from torch.nn.functional import mse_loss
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions import Categorical
from torch.utils.tensorboard.writer import SummaryWriter

simplefilter(action="ignore", category=DeprecationWarning)

SEED = 24


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLander-v2")
    parser.add_argument("--total-timesteps", type=int, default=int(1e6))
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument('--list-layer', nargs="+", type=int, default=[64, 64])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--shared-network", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--capture-video", action="store_true")

    _args = parser.parse_args()

    _args.device = torch.device(
        "cpu" if _args.cpu or not torch.cuda.is_available() else "cuda")
    _args.batch_size = int(_args.num_envs * _args.num_steps)
    _args.minibatch_size = int(_args.batch_size // _args.num_minibatches)

    return _args


def make_env(env_id, idx, run_name, capture_video):

    def thunk():

        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=0.99)
        env = gym.wrappers.TransformReward(
            env, lambda reward: np.clip(reward, -10, 10))
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env=env,
                video_folder=f"../runs/{run_name}/videos/",
                disable_logger=True)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):

    def __init__(self, args, obversation_space, action_space):

        super().__init__()

        current_layer_value = np.array(obversation_space.shape).prod()
        num_actions = action_space.n

        if args.shared_network:
            base_neural_net = nn.Sequential()

            for layer_value in args.list_layer:
                base_neural_net.append(
                    layer_init(nn.Linear(current_layer_value, layer_value)))
                base_neural_net.append(nn.Tanh())

                current_layer_value = layer_value

            self.actor_neural_net = nn.Sequential(
                base_neural_net,
                layer_init(nn.Linear(args.list_layer[-1], num_actions),
                           std=0.01))

            self.critic_neural_net = nn.Sequential(
                base_neural_net,
                layer_init(nn.Linear(args.list_layer[-1], 1), std=1.0))

        else:
            self.actor_neural_net = nn.Sequential()
            self.critic_neural_net = nn.Sequential()

            for layer_value in args.list_layer:
                self.actor_neural_net.append(
                    layer_init(nn.Linear(current_layer_value, layer_value)))
                self.actor_neural_net.append(nn.Tanh())

                self.critic_neural_net.append(
                    layer_init(nn.Linear(current_layer_value, layer_value)))
                self.critic_neural_net.append(nn.Tanh())

                current_layer_value = layer_value

            self.actor_neural_net.append(
                layer_init(nn.Linear(args.list_layer[-1], num_actions),
                           std=0.01))

            self.critic_neural_net.append(
                layer_init(nn.Linear(args.list_layer[-1], 1), std=1.0))

        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate)

        if args.device.type == "cuda":
            self.cuda()

    def forward(self):
        pass

    def get_action(self, state):

        actor_value = self.actor_neural_net(state)
        distribution = Categorical(logits=actor_value)
        action = distribution.sample()

        return action.cpu().numpy()

    def get_logprob_value(self, state, action):

        actor_value = self.actor_neural_net(state)
        distribution = Categorical(logits=actor_value)
        log_prob = distribution.log_prob(action)

        critic_value = self.critic_neural_net(state).squeeze()

        return log_prob, critic_value


def main():
    args = parse_args()

    date = str(datetime.now().strftime("%d-%m_%H:%M:%S"))
    run_name = f"{args.env}__a2c__{date}"
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

    obversation_space = envs.single_observation_space
    action_space = envs.single_action_space

    agent = Agent(args, obversation_space, action_space)

    obversation_shape = obversation_space.shape
    action_shape = action_space.shape

    states = torch.zeros((args.num_steps, args.num_envs) +
                         obversation_shape).to(args.device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape).to(
        args.device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    flags = torch.zeros((args.num_steps, args.num_envs)).to(args.device)

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
                action = agent.get_action(state_torch)

            next_state, reward, terminated, truncated, infos = envs.step(
                action)

            states[i] = state_torch
            actions[i] = torch.from_numpy(action).to(args.device)
            rewards[i] = torch.from_numpy(reward).to(args.device)

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
        td_target = torch.zeros(rewards.size()).to(args.device)
        gain = torch.zeros(rewards.size(1)).to(args.device)

        for i in reversed(range(td_target.size(0))):
            terminal = 1. - flags[i]
            gain = rewards[i] + gain * args.gamma * terminal
            td_target[i] = gain

        td_target = (td_target - td_target.mean()) / (td_target.std() + 1e-7)
        td_target = td_target.squeeze()

        # Shuffle batch
        b_states = states.flatten(0, 1)
        b_actions = actions.flatten(0, 1)
        b_td_target = td_target.reshape(-1)

        batch_index = torch.randperm(args.batch_size)

        b_states = b_states[batch_index]
        b_ations = b_actions[batch_index]
        b_td_target = b_td_target[batch_index]

        # Update policy
        log_probs, td_predict = agent.get_logprob_value(b_states, b_ations)

        advantages = b_td_target - td_predict

        policy_loss = (-log_probs * advantages).mean()
        value_loss = mse_loss(td_target, td_predict)

        loss = policy_loss + value_loss

        agent.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(agent.parameters(), 0.5)
        agent.optimizer.step()

        writer.add_scalar("update/policy_loss", policy_loss, global_step)
        writer.add_scalar("update/value_loss", value_loss, global_step)

    envs.close()
    writer.close()


if __name__ == '__main__':
    main()
