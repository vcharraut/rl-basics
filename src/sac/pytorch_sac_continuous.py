import argparse
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal, Uniform
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
    parser.add_argument("--actor_layers", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--critic_layers", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--learning_start", type=int, default=25_000)
    parser.add_argument("--policy_frequency", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    args.device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    return args


def make_env(env_id, capture_video=False, run_dir=""):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env=env,
                video_folder=f"{run_dir}/videos",
                episode_trigger=lambda x: x,
                disable_logger=True,
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.FlattenObservation(env)

        return env

    return thunk


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, observation_shape, action_shape, numpy_rng, device):
        self.state_buffer = np.zeros((buffer_size, *observation_shape), dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size, *action_shape), dtype=np.float32)
        self.reward_buffer = np.zeros((buffer_size,), dtype=np.float32)
        self.flag_buffer = np.zeros((buffer_size,), dtype=np.float32)

        self.batch_size = batch_size
        self.max_size = buffer_size
        self.idx = 0
        self.size = 0

        self.numpy_rng = numpy_rng
        self.device = device

    def push(self, state, action, reward, flag):
        self.state_buffer[self.idx] = state
        self.action_buffer[self.idx] = action
        self.reward_buffer[self.idx] = reward
        self.flag_buffer[self.idx] = flag

        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        idxs = self.numpy_rng.integers(0, self.size - 1, size=self.batch_size)

        return (
            torch.from_numpy(self.state_buffer[idxs]).to(self.device),
            torch.from_numpy(self.action_buffer[idxs]).to(self.device),
            torch.from_numpy(self.reward_buffer[idxs]).to(self.device),
            torch.from_numpy(self.state_buffer[idxs + 1]).to(self.device),
            torch.from_numpy(self.flag_buffer[idxs]).to(self.device),
        )


class ActorCriticNet(nn.Module):
    def __init__(self, observation_shape, action_shape, actor_layers, critic_layers, action_low, action_high, device):
        super().__init__()

        action_shape = np.prod(action_shape)

        self.actor_net = self._build_net(observation_shape, actor_layers)
        self.actor_mean = self._build_linear(actor_layers[-1], action_shape)
        self.actor_logstd = self._build_linear(actor_layers[-1], action_shape)

        self.critic_net1 = self._build_net(np.prod(observation_shape) + np.prod(action_shape), critic_layers)
        self.critic_net1.append(self._build_linear(critic_layers[-1], 1))

        self.critic_net2 = self._build_net(np.prod(observation_shape) + np.prod(action_shape), critic_layers)
        self.critic_net2.append(self._build_linear(critic_layers[-1], 1))

        # Scale and bias the output of the network to match the action space
        self.register_buffer("action_scale", ((action_high - action_low) / 2.0))
        self.register_buffer("action_bias", ((action_high + action_low) / 2.0))

        if device.type == "cuda":
            self.cuda()

    def _build_linear(self, in_size, out_size, apply_init=False, std=np.sqrt(2), bias_const=0.0):
        layer = nn.Linear(in_size, out_size)

        if apply_init:
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)

        return layer

    def _build_net(self, observation_shape, hidden_layers):
        layers = nn.Sequential()
        in_size = np.prod(observation_shape)

        for out_size in hidden_layers:
            layers.append(self._build_linear(in_size, out_size))
            layers.append(nn.ReLU())
            in_size = out_size

        return layers

    def actor(self, state):
        log_std_max = 2
        log_std_min = -5

        output = self.actor_net(state)
        mean = self.actor_mean(output)
        log_std = torch.tanh(self.actor_logstd(output))

        # Rescale log_std to ensure it is within range [log_std_min, log_std_max].
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        # Sample action using reparameterization trick.
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        # Rescale and shift the action.
        action = y_t * self.action_scale + self.action_bias

        # Calculate the log probability of the sampled action.
        log_prob = normal.log_prob(x_t)

        # Enforce action bounds.
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        # Rescale mean and shift it to match the action range.
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob.squeeze()

    def critic(self, state, action):
        critic1 = self.critic_net1(torch.cat([state, action], 1)).squeeze()
        critic2 = self.critic_net2(torch.cat([state, action], 1)).squeeze()
        return critic1, critic2


def train(args, run_name, run_dir):
    # Initialize wandb if needed (https://wandb.ai/)
    if args.wandb:
        import wandb

        wandb.init(project=args.env_id, name=run_name, sync_tensorboard=True, config=vars(args))

    # Create tensorboard writer and save hyperparameters
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Create vectorized environment
    env = gym.vector.SyncVectorEnv([make_env(args.env_id)])

    # Metadata about the environment
    observation_shape = env.single_observation_space.shape
    action_shape = env.single_action_space.shape
    action_low = torch.from_numpy(env.single_action_space.low).to(args.device)
    action_high = torch.from_numpy(env.single_action_space.high).to(args.device)

    # Set seed for reproducibility
    if args.seed:
        numpy_rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        state, _ = env.reset(seed=args.seed)
    else:
        numpy_rng = np.random.default_rng()
        state, _ = env.reset()

    # Create the networks and the optimizer
    policy = ActorCriticNet(
        observation_shape,
        action_shape,
        args.actor_layers,
        args.critic_layers,
        action_low,
        action_high,
        args.device,
    )
    target = ActorCriticNet(
        observation_shape,
        action_shape,
        args.actor_layers,
        args.critic_layers,
        action_low,
        action_high,
        args.device,
    )
    target.load_state_dict(policy.state_dict())

    optimizer_actor = optim.Adam(policy.actor_net.parameters(), lr=args.learning_rate)
    optimizer_critic = optim.Adam(
        list(policy.critic_net1.parameters()) + list(policy.critic_net2.parameters()),
        lr=args.learning_rate,
    )

    alpha = args.alpha

    # Create the replay buffer
    replay_buffer = ReplayBuffer(
        args.buffer_size,
        args.batch_size,
        observation_shape,
        action_shape,
        numpy_rng,
        args.device,
    )

    log_episodic_returns = []

    start_time = time.process_time()

    # Main loop
    for global_step in tqdm(range(args.total_timesteps)):
        if global_step < args.learning_start:
            action = Uniform(action_low, action_high).sample().unsqueeze(0)
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).to(args.device).float()
                action, _ = policy.actor(state_tensor)

        # Perform action
        action = action.cpu().numpy()
        next_state, reward, terminated, truncated, infos = env.step(action)

        # Store transition in the replay buffer
        flag = 1.0 - np.logical_or(terminated, truncated)
        replay_buffer.push(state, action, reward, flag)

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
                next_state_actions, next_state_log_pi = policy.actor(next_states)
                critic1_next_target, critic2_next_target = target.critic(next_states, next_state_actions)
                min_qf_next_target = torch.min(critic1_next_target, critic2_next_target) - alpha * next_state_log_pi
                next_q_value = rewards + args.gamma * flags * min_qf_next_target

            qf1_a_values, qf2_a_values = policy.critic(states, actions)
            qf1_loss = mse_loss(qf1_a_values, next_q_value)
            qf2_loss = mse_loss(qf2_a_values, next_q_value)
            critic_loss = qf1_loss + qf2_loss

            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            # Update actor
            if not global_step % args.policy_frequency:
                for _ in range(args.policy_frequency):
                    pi, log_pi = policy.actor(states)
                    qf1_pi, qf2_pi = policy.critic(states, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = (alpha * log_pi - min_qf_pi).mean()

                    optimizer_actor.zero_grad()
                    actor_loss.backward()
                    optimizer_actor.step()

                # Update the target network (soft update)
                for param, target_param in zip(policy.critic_net1.parameters(), target.critic_net1.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(policy.critic_net2.parameters(), target.critic_net2.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                # Log training metrics
                writer.add_scalar("train/actor_loss", actor_loss, global_step)
                writer.add_scalar("train/critic_loss", critic_loss, global_step)
                writer.add_scalar("train/qf1_a_values", qf1_a_values.mean(), global_step)
                writer.add_scalar("train/qf2_a_values", qf2_a_values.mean(), global_step)
                writer.add_scalar("train/critic1_next_target", critic1_next_target.mean(), global_step)
                writer.add_scalar("train/critic2_next_target", critic2_next_target.mean(), global_step)
                writer.add_scalar("train/qf1_loss", qf1_loss, global_step)
                writer.add_scalar("train/qf2_loss", qf2_loss, global_step)
                writer.add_scalar("train/min_qf_next_target", min_qf_next_target.mean(), global_step)
                writer.add_scalar("train/next_q_value", next_q_value.mean(), global_step)

        writer.add_scalar("rollout/SPS", int(global_step / (time.process_time() - start_time)), global_step)

    # Save final policy
    torch.save(policy.state_dict(), f"{run_dir}/policy.pt")
    print(f"Saved policy to {run_dir}/policy.pt")

    # Close the environment
    env.close()
    writer.close()

    # Average of episodic returns (for the last 5% of the training)
    indexes = int(len(log_episodic_returns) * 0.05)
    mean_train_return = np.mean(log_episodic_returns[-indexes:])
    writer.add_scalar("rollout/mean_train_return", mean_train_return, global_step)

    return mean_train_return


def eval_and_render(args, run_dir):
    # Create environment
    env = gym.vector.SyncVectorEnv([make_env(args.env_id, capture_video=True, run_dir=run_dir)])

    # Metadata about the environment
    observation_shape = env.single_observation_space.shape
    action_shape = env.single_action_space.shape
    action_low = torch.from_numpy(env.single_action_space.low).to(args.device)
    action_high = torch.from_numpy(env.single_action_space.high).to(args.device)

    # Load policy
    policy = ActorCriticNet(
        observation_shape,
        action_shape,
        args.actor_layers,
        args.critic_layers,
        action_low,
        action_high,
        args.device,
    )
    policy.load_state_dict(torch.load(f"{run_dir}/actor.pt"))
    policy.eval()

    count_episodes = 0
    list_rewards = []

    state, _ = env.reset(seed=args.seed) if args.seed else env.reset()

    # Run episodes
    while count_episodes < 30:
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).to(args.device).float()
            action, _ = policy(state_tensor)

        action = action.cpu().numpy()
        state, _, _, _, infos = env.step(action)

        if "final_info" in infos:
            info = infos["final_info"][0]
            returns = info["episode"]["r"][0]
            count_episodes += 1
            list_rewards.append(returns)
            print(f"-> Episode {count_episodes}: {returns} returns")

    env.close()

    return np.mean(list_rewards)


if __name__ == "__main__":
    args_ = parse_args()

    # Create run directory
    run_time = str(datetime.now().strftime("%d-%m_%H:%M:%S"))
    run_name = "SAC_PyTorch"
    run_dir = f"runs/{args_.env_id}__{run_name}__{run_time}"

    print(f"Commencing training of {run_name} on {args_.env_id} for {args_.total_timesteps} timesteps.")
    print(f"Results will be saved to: {run_dir}")
    mean_train_return = train(args=args_, run_name=run_name, run_dir=run_dir)
    print(f"Training - Mean returns achieved: {mean_train_return}.")

    if args_.capture_video:
        print(f"Evaluating and capturing videos of {run_name} on {args_.env_id}.")
        mean_eval_return = eval_and_render(args=args_, run_dir=run_dir)
        print(f"Evaluation - Mean returns achieved: {mean_eval_return}.")
