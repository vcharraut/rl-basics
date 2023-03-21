import argparse
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.nn.functional import mse_loss
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="LunarLander-v2")
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=25_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--list_layer", nargs="+", type=int, default=[64, 64])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_start", type=int, default=1)
    parser.add_argument("--eps_decay", type=int, default=50_000)
    parser.add_argument("--learning_start", type=int, default=10_000)
    parser.add_argument("--train_frequency", type=int, default=10)
    parser.add_argument("--target_update_frequency", type=int, default=1_000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    args.device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    return args


def make_env(env_id, capture_video=False, run_dir="."):
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
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda state: np.clip(state, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        return env

    return thunk


def get_exploration_prob(eps_start, eps_end, eps_decay, step):
    return eps_end + (eps_start - eps_end) * np.exp(-1.0 * step / eps_decay)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, observation_shape, numpy_rng, device):
        self.states = np.zeros((buffer_size, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.flags = np.zeros((buffer_size,), dtype=np.float32)

        self.batch_size = batch_size
        self.max_size = buffer_size
        self.idx = 0
        self.size = 0

        self.numpy_rng = numpy_rng
        self.device = device

    def push(self, state, action, reward, flag):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.flags[self.idx] = flag

        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        idxs = self.numpy_rng.integers(0, self.size - 1, size=self.batch_size)

        return (
            torch.from_numpy(self.states[idxs]).to(self.device),
            torch.from_numpy(self.actions[idxs]).unsqueeze(-1).to(self.device),
            torch.from_numpy(self.rewards[idxs]).to(self.device),
            torch.from_numpy(self.states[idxs + 1]).to(self.device),
            torch.from_numpy(self.flags[idxs]).to(self.device),
        )


class QNetwork(nn.Module):
    def __init__(self, observation_shape, action_dim, list_layer, device):
        super().__init__()

        self.network = self._build_net(observation_shape, list_layer)
        self.network.append(self._build_linear(list_layer[-1], action_dim))

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

    def forward(self, state):
        return self.network(state)


def train(args, run_name, run_dir):
    # Initialize wandb if needed (https://wandb.ai/)
    if args.wandb:
        import wandb

        wandb.init(
            project=args.env_id,
            name=run_name,
            sync_tensorboard=True,
            config=vars(args),
            monitor_gym=True,
            save_code=True,
        )

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
    action_dim = env.single_action_space.n

    # Set seed for reproducibility
    if args.seed:
        numpy_rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        state, _ = env.reset(seed=args.seed)
    else:
        numpy_rng = np.random.default_rng()
        state, _ = env.reset()

    # Create the networks and the optimizer
    policy = QNetwork(observation_shape, action_dim, args.list_layer, args.device)
    target_policy = QNetwork(observation_shape, action_dim, args.list_layer, args.device)
    target_policy.load_state_dict(policy.state_dict())

    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)

    # Create the replay buffer
    replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size, observation_shape, numpy_rng, args.device)

    # Remove unnecessary variables
    del observation_shape

    log_episodic_returns, log_episodic_lengths = [], []
    start_time = time.process_time()

    # Main loop
    for global_step in tqdm(range(args.total_timesteps)):
        with torch.no_grad():
            # Exploration or intensification
            exploration_prob = get_exploration_prob(args.eps_start, args.eps_end, args.eps_decay, global_step)

            # Log exploration probability
            writer.add_scalar("rollout/eps_threshold", exploration_prob, global_step)

            if numpy_rng.random() < exploration_prob:
                # Exploration
                action = torch.randint(action_dim, (1,)).to(args.device)
            else:
                # Intensification
                action = policy(torch.from_numpy(state).to(args.device).float()).argmax(axis=1)

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
            log_episodic_lengths.append(info["episode"]["l"])
            writer.add_scalar("rollout/episodic_return", np.mean(info["episode"]["r"][-10:]), global_step)
            writer.add_scalar("rollout/episodic_length", np.mean(info["episode"]["l"][-10:]), global_step)

        # Perform training step
        if global_step > args.learning_start:
            if not global_step % args.train_frequency:
                # Sample a batch from the replay buffer
                states, actions, rewards, next_states, flags = replay_buffer.sample()

                # Compute TD error
                td_predict = policy(states).gather(1, actions).squeeze()

                # Compute TD target
                with torch.no_grad():
                    # Double Q-Learning
                    action_by_qvalue = policy(next_states).argmax(1).unsqueeze(-1)
                    max_q_target = target_policy(next_states).gather(1, action_by_qvalue).squeeze()

                td_target = rewards + args.gamma * flags * max_q_target

                # Compute loss
                loss = mse_loss(td_predict, td_target)

                # Update policy network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Log training metrics
                writer.add_scalar("train/loss", loss, global_step)

            # Update target network
            if not global_step % args.target_update_frequency:
                target_policy.load_state_dict(policy.state_dict())

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
    action_dim = env.single_action_space.n

    # Load policy
    policy = QNetwork(observation_shape, action_dim, args.list_layer, args.device)
    policy.load_state_dict(torch.load(f"{run_dir}/policy.pt"))
    policy.eval()

    count_episodes = 0
    list_rewards = []

    numpy_rng = np.random.default_rng()
    state, _ = env.reset(seed=args.seed) if args.seed else env.reset()

    # Run episodes
    while count_episodes < 30:
        with torch.no_grad():
            if numpy_rng.random() < 0.05:
                # Exploration
                action = torch.randint(action_dim, (1,)).to(args.device)
            else:
                # Intensification
                action = policy(torch.from_numpy(state).to(args.device).float()).argmax(axis=1)

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
    run_name = "DQN_PyTorch"
    run_dir = f"runs/{args_.env_id}__{run_name}__{run_time}"

    print(f"Commencing training of {run_name} on {args_.env_id} for {args_.total_timesteps} timesteps.")
    print(f"Results will be saved to: {run_dir}")
    mean_train_return = train(args=args_, run_name=run_name, run_dir=run_dir)
    print(f"Training - Mean returns achieved: {mean_train_return}.")

    if args_.capture_video:
        print(f"Evaluating and capturing videos of {run_name} on {args_.env_id}.")
        mean_eval_return = eval_and_render(args=args_, run_dir=run_dir)
        print(f"Evaluation - Mean returns achieved: {mean_eval_return}.")
