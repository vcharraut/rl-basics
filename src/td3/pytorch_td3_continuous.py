import argparse
import time
from datetime import datetime

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
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--learning_start", type=int, default=25_000)
    parser.add_argument("--policy_frequency", type=int, default=4)
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
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        return env

    return thunk


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, state_dim, act_dim, device):
        self.state_buffer = np.zeros((buffer_size,) + state_dim, dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size,) + act_dim, dtype=np.float32)
        self.reward_buffer = np.zeros((buffer_size), dtype=np.float32)
        self.flag_buffer = np.zeros((buffer_size), dtype=np.float32)

        self.batch_size = batch_size
        self.max_size = buffer_size
        self.idx = 0
        self.size = 0
        self.device = device

    def push(self, state, action, reward, flag):
        self.state_buffer[self.idx] = state
        self.action_buffer[self.idx] = action
        self.reward_buffer[self.idx] = reward
        self.flag_buffer[self.idx] = flag

        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        idxs = np.random.randint(0, self.size - 1, size=self.batch_size)

        return (
            torch.from_numpy(self.state_buffer[idxs]).to(self.device),
            torch.from_numpy(self.action_buffer[idxs]).to(self.device),
            torch.from_numpy(self.reward_buffer[idxs]).to(self.device),
            torch.from_numpy(self.state_buffer[idxs + 1]).to(self.device),
            torch.from_numpy(self.flag_buffer[idxs]).to(self.device),
        )


class ActorNet(nn.Module):
    def __init__(self, obversation_shape, action_shape, list_layer, action_low, action_high, device):
        super().__init__()

        fc_layer_value = np.prod(obversation_shape)
        action_shape = np.prod(action_shape)

        self.network = nn.Sequential()

        for layer_value in list_layer:
            self.network.append(nn.Linear(fc_layer_value, layer_value))
            self.network.append(nn.ReLU())
            fc_layer_value = layer_value

        self.network.append(nn.Linear(fc_layer_value, action_shape))

        # Scale and bias the output of the network to match the action space
        self.register_buffer("action_scale", ((action_high - action_low) / 2.0))
        self.register_buffer("action_bias", ((action_high + action_low) / 2.0))

        if device.type == "cuda":
            self.cuda()

    def forward(self, state):
        output = torch.tanh(self.network(state))
        return output * self.action_scale + self.action_bias


class CriticNet(nn.Module):
    def __init__(self, obversation_shape, action_shape, list_layer, device):
        super().__init__()

        fc_layer_value = np.prod(obversation_shape) + np.prod(action_shape)

        self.network = nn.Sequential()

        for layer_value in list_layer:
            self.network.append(nn.Linear(fc_layer_value, layer_value))
            self.network.append(nn.ReLU())
            fc_layer_value = layer_value

        self.network.append(nn.Linear(fc_layer_value, 1))

        if device.type == "cuda":
            self.cuda()

    def forward(self, state, action):
        return self.network(torch.cat([state, action], 1))


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

    # Set seed for reproducibility
    if args.seed:
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
    actor = ActorNet(obversation_shape, action_shape, args.list_layer, action_low, action_high, args.device)
    critic1 = CriticNet(obversation_shape, action_shape, args.list_layer, args.device)
    critic2 = CriticNet(obversation_shape, action_shape, args.list_layer, args.device)

    target_actor = ActorNet(obversation_shape, action_shape, args.list_layer, action_low, action_high, args.device)
    target_critic1 = CriticNet(obversation_shape, action_shape, args.list_layer, args.device)
    target_critic2 = CriticNet(obversation_shape, action_shape, args.list_layer, args.device)

    target_actor.load_state_dict(actor.state_dict())
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())

    optimizer_actor = optim.Adam(actor.parameters(), lr=args.learning_rate)
    optimizer_critic = optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=args.learning_rate)

    # Create the replay buffer
    replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size, obversation_shape, action_shape, args.device)

    # Initialize environment
    state, _ = env.reset(seed=args.seed) if args.seed else env.reset()

    log_episodic_returns = []

    start_time = time.process_time()

    # Main loop
    for global_step in tqdm(range(args.total_timesteps)):
        if global_step < args.learning_start:
            action = Uniform(action_low, action_high).sample().unsqueeze(0)
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).to(args.device).float()
                action = actor(state_tensor)
                action += torch.normal(0, actor.action_scale * args.exploration_noise)

        # Perform action
        action = action.cpu().numpy()
        next_state, reward, terminated, truncated, infos = env.step(action)

        # Store transition in the replay buffer
        replay_buffer.push(state, action, reward, np.logical_or(terminated, truncated))

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
                clipped_noise = (
                    torch.clamp(
                        (torch.randn_like(actions) * args.policy_noise),
                        -args.noise_clip,
                        args.noise_clip,
                    )
                    * target_actor.action_scale
                )

                next_state_actions = torch.clamp((target_actor(next_states) + clipped_noise), action_low, action_high)
                critic1_next_target = target_critic1(next_states, next_state_actions).squeeze()
                critic2_next_target = target_critic2(next_states, next_state_actions).squeeze()
                min_qf_next_target = torch.min(critic1_next_target, critic2_next_target)
                next_q_value = rewards + (1.0 - flags) * args.gamma * min_qf_next_target

            qf1_a_values = critic1(states, actions).squeeze()
            qf2_a_values = critic2(states, actions).squeeze()
            qf1_loss = mse_loss(qf1_a_values, next_q_value)
            qf2_loss = mse_loss(qf2_a_values, next_q_value)
            critic_loss = qf1_loss + qf2_loss

            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            # Update actor
            if not global_step % args.policy_frequency:
                actor_loss = -critic1(states, actor(states)).mean()
                optimizer_actor.zero_grad()
                actor_loss.backward()
                optimizer_actor.step()

                # Update the target network (soft update)
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(critic1.parameters(), target_critic1.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(critic2.parameters(), target_critic2.parameters()):
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
    torch.save(actor.state_dict(), f"{run_dir}/actor.pt")
    torch.save(critic1.state_dict(), f"{run_dir}/critic1.pt")
    torch.save(critic2.state_dict(), f"{run_dir}/critic2.pt")
    print(f"Saved policy to {run_dir}/actor.pt, {run_dir}/critic1.pt, {run_dir}/critic2.pt")

    # Close the environment
    env.close()
    writer.close()
    if args.wandb:
        wandb.finish()

    # Average of episodic returns (for the last 5% of the training)
    indexes = int(len(log_episodic_returns) * 0.05)
    mean_train_return = np.mean(log_episodic_returns[-indexes:])
    writer.add_scalar("rollout/mean_train_return", mean_train_return, global_step)

    return mean_train_return


def eval_and_render(args, run_dir):
    # Create environment
    env = gym.vector.SyncVectorEnv([make_env(args.env_id, capture_video=True, run_dir=run_dir)])

    # Metadata about the environment
    obversation_shape = env.single_observation_space.shape
    action_shape = env.single_action_space.shape
    action_low = torch.from_numpy(env.single_action_space.low).to(args.device)
    action_high = torch.from_numpy(env.single_action_space.high).to(args.device)

    # Load policy
    policy = ActorNet(obversation_shape, action_shape, args.list_layer, action_low, action_high, args.device)
    policy.load_state_dict(torch.load(f"{run_dir}/actor.pt"))
    policy.eval()

    count_episodes = 0
    list_rewards = []

    state, _ = env.reset(seed=args.seed) if args.seed else env.reset()

    # Run episodes
    while count_episodes < 30:
        with torch.no_grad():
            state = torch.from_numpy(state).to(args.device).float()
            action = policy(state)
            action += torch.normal(0, policy.action_scale * args.exploration_noise)

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
    args = parse_args()

    # Create run directory
    run_time = str(datetime.now().strftime("%d-%m_%H:%M:%S"))
    run_name = "TD3_PyTorch"
    run_dir = f"runs/{args.env_id}__{run_name}__{run_time}"

    print(f"Commencing training of {run_name} on {args.env_id} for {args.total_timesteps} timesteps.")
    print(f"Results will be saved to: {run_dir}")
    mean_train_return = train(args=args, run_name=run_name, run_dir=run_dir)
    print(f"Training - Mean returns achieved: {mean_train_return}.")

    if args.capture_video:
        print(f"Evaluating and capturing videos of {run_name} on {args.env_id}.")
        mean_eval_return = eval_and_render(args=args, run_dir=run_dir)
        print(f"Evaluation - Mean returns achieved: {mean_eval_return}.")
