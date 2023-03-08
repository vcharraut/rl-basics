import argparse
import time
from datetime import datetime

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
    parser.add_argument("--env_id", type=str, default="LunarLander-v2")
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--list_layer", nargs="+", type=int, default=[64, 64])
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


def normalize(value):
    return (value - value.mean()) / (value.std() + 1e-7)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticNet(nn.Module):
    def __init__(self, obversation_shape, action_shape, list_layer):
        super().__init__()

        fc_layer_value = np.prod(obversation_shape)

        self.actor_net = nn.Sequential()
        self.critic_net = nn.Sequential()

        for layer_value in list_layer:
            self.actor_net.append(layer_init(nn.Linear(fc_layer_value, layer_value)))
            self.actor_net.append(nn.Tanh())

            self.critic_net.append(layer_init(nn.Linear(fc_layer_value, layer_value)))
            self.critic_net.append(nn.Tanh())

            fc_layer_value = layer_value

        self.actor_net.append(layer_init(nn.Linear(list_layer[-1], action_shape), std=0.01))
        self.critic_net.append(layer_init(nn.Linear(list_layer[-1], 1), std=1.0))

    def forward(self, state):
        actor_value = self.actor_net(state)
        distribution = Categorical(logits=actor_value)

        action = distribution.sample()
        return action

    def evaluate(self, states, actions):
        actor_value = self.actor_net(states)
        distribution = Categorical(logits=actor_value)

        log_probs = distribution.log_prob(actions)
        dist_entropy = distribution.entropy()

        critic_values = self.critic_net(states).squeeze(-1)

        return log_probs, critic_values, dist_entropy


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

    # Create vectorized environment(s)
    envs = gym.vector.AsyncVectorEnv([make_env(args.env_id) for _ in range(args.num_envs)])

    # Metadata about the environment
    obversation_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.n

    # Create policy network and optimizer
    policy_net = ActorCriticNet(obversation_shape, action_shape, args.list_layer)
    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)

    # Create buffers
    states = np.zeros((args.num_steps, args.num_envs) + obversation_shape, dtype=np.float32)
    actions = np.zeros((args.num_steps, args.num_envs), dtype=np.int32)
    rewards = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    flags = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)

    log_episodic_returns = []

    # Initialize environment
    state, _ = envs.reset(seed=args.seed) if args.seed else envs.reset()

    global_step = 0
    start_time = time.process_time()

    # Main loop
    for _ in tqdm(range(args.num_updates)):
        for i in range(args.num_steps):
            # Update global step
            global_step += 1 * args.num_envs

            with torch.no_grad():
                # Get action
                state = normalize(state)
                state_tensor = torch.from_numpy(state).float()
                action = policy_net(state_tensor)

            # Perform action
            action = action.cpu().numpy()
            next_state, reward, terminated, truncated, infos = envs.step(action)

            # Store transition
            states[i] = state
            actions[i] = action
            rewards[i] = reward
            flags[i] = np.logical_or(terminated, truncated)

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

        td_target = np.zeros_like(rewards, dtype=np.float32)
        gain = np.zeros(rewards.shape[1], dtype=np.float32)

        # Compute TD target
        for i in reversed(range(td_target.shape[0])):
            terminal = 1.0 - flags[i]
            gain = rewards[i] + gain * args.gamma * terminal
            td_target[i] = gain

        td_target = normalize(td_target)

        # Flatten batch
        batch_states = states.reshape(-1, *obversation_shape)
        batch_actions = actions.reshape(-1)
        batch_td_targets = td_target.reshape(-1)

        # Convert to tensor
        batch_states = torch.from_numpy(batch_states)
        batch_actions = torch.from_numpy(batch_actions)
        batch_td_targets = torch.from_numpy(batch_td_targets)

        # Compute losses
        log_probs, td_predict, dist_entropy = policy_net.evaluate(batch_states, batch_actions)
        advantages = batch_td_targets - td_predict

        actor_loss = (-log_probs * advantages.detach()).mean()
        critic_loss = mse_loss(batch_td_targets, td_predict)
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
        writer.add_scalar("rollout/SPS", int(global_step / (time.process_time() - start_time)), global_step)

    # Average of episodic returns (for the last 5% of the training)
    indexes = int(len(log_episodic_returns) * 0.05)
    avg_final_rewards = np.mean(log_episodic_returns[-indexes:])
    print(f"Average of the last {indexes} episodic returns: {round(avg_final_rewards, 2)}")
    writer.add_scalar("rollout/avg_final_rewards", avg_final_rewards, global_step)

    # Save final policy
    torch.save(policy_net.state_dict(), f"{run_dir}/policy.pt")
    print(f"Saved policy to {run_dir}/policy.pt")

    # Close the environment
    envs.close()
    writer.close()
    if args.wandb:
        wandb.finish()


def test_and_render(args, run_dir):
    # Create environment
    env = gym.vector.SyncVectorEnv([make_env(args.env_id, capture_video=True, run_dir=run_dir)])

    # Metadata about the environment
    obversation_shape = env.single_observation_space.shape
    action_shape = env.single_action_space.n

    # Load policy
    policy = ActorCriticNet(obversation_shape, action_shape, args.list_layer)
    policy.load_state_dict(torch.load(f"{run_dir}/policy.pt"))
    policy.eval()

    count_episodes = 0
    list_rewards = []

    state, _ = env.reset(seed=args.seed) if args.seed else env.reset()

    # Run episodes
    while count_episodes < 30:
        with torch.no_grad():
            state = normalize(state)
            state_tensor = torch.from_numpy(state).float()
            action = policy(state_tensor)

        action = action.cpu().numpy()
        state, _, _, _, infos = env.step(action)

        if "final_info" in infos:
            info = infos["final_info"][0]
            returns = info["episode"]["r"][0]
            count_episodes += 1
            list_rewards.append(returns)
            print(f"Episode {count_episodes}: {returns} returns")

    print(f"Average returns: {np.mean(list_rewards)}")

    env.close()


if __name__ == "__main__":
    args = parse_args()

    # Create run directory
    run_time = str(datetime.now().strftime("%d-%m_%H:%M:%S"))
    run_name = "A2C_PyTorch"
    run_dir = f"runs/{args.env_id}__{run_name}__{run_time}"

    print(f"Training {run_name} on {args.env_id} for {args.total_timesteps} timesteps")
    print(f"Saving results to {run_dir}")
    train(args=args, run_name=run_name, run_dir=run_dir)

    if args.capture_video:
        print(f"Testing and capturing videos for {run_name} on {args.env_id}")
        test_and_render(args=args, run_dir=run_dir)
