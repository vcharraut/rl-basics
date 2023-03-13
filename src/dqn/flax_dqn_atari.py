import argparse
import time
from datetime import datetime

import flax
import gymnasium as gym
import jax
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import numpy as jnp
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--buffer_size", type=int, default=600_000)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_start", type=int, default=1)
    parser.add_argument("--eps_decay", type=int, default=300_000)
    parser.add_argument("--learning_start", type=int, default=50_000)
    parser.add_argument("--train_frequency", type=int, default=4)
    parser.add_argument("--target_update_frequency", type=int, default=10_000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    return args


def make_env(env_id, capture_video=False):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env=env,
                video_folder="/videos/",
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


class QNetwork(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        dtype = jnp.float32
        x = x.astype(dtype) / 255.0
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), name="conv1", dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), name="conv2", dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), name="conv3", dtype=dtype)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=512, name="hidden", dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        return x


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, state_dim):
        self.state_buffer = np.zeros((buffer_size,) + state_dim, dtype=np.int8)
        self.action_buffer = np.zeros((buffer_size), dtype=np.int32)
        self.reward_buffer = np.zeros((buffer_size), dtype=np.float32)
        self.flag_buffer = np.zeros((buffer_size), dtype=np.float32)

        self.batch_size = batch_size
        self.max_size = buffer_size
        self.idx = 0
        self.size = 0

    def push(self, state, action, reward, flag):
        self.state_buffer[self.idx] = state
        self.action_buffer[self.idx] = action
        self.reward_buffer[self.idx] = reward
        self.flag_buffer[self.idx] = flag

        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        idxs = np.random.randint(0, self.size - 1, size=self.batch_size)

        return {
            "states": self.state_buffer[idxs],
            "actions": self.action_buffer[idxs],
            "rewards": self.reward_buffer[idxs],
            "next_states": self.state_buffer[idxs + 1],
            "flags": self.flag_buffer[idxs],
        }


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def loss_fn(params, target_params, apply_fn, batch, gamma):
    # Compute TD error
    q_predict = apply_fn(params, batch["states"])
    td_predict = jax.vmap(lambda qp, a: qp[a])(q_predict, batch["actions"])

    # Compute TD target with Double Q-Learning
    action_by_qvalue = apply_fn(params, batch["next_states"]).argmax(axis=1)
    q_target = apply_fn(target_params, batch["next_states"])
    max_q_target = jax.vmap(lambda qt, a: qt[a])(q_target, action_by_qvalue)

    td_target = batch["rewards"] + (1.0 - batch["flags"]) * gamma * max_q_target

    return jnp.mean((td_predict - td_target) ** 2)


@jax.jit
def train_step(train_state, batch, gamma):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(train_state.params, train_state.target_params, train_state.apply_fn, batch, gamma)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss


def get_exploration_prob(eps_start, eps_end, eps_decay, step):
    return eps_end + (eps_start - eps_end) * np.exp(-1.0 * step / eps_decay)


def main():
    args = parse_args()

    # Create run directory
    run_time = str(datetime.now().strftime("%d-%m_%H:%M:%S"))
    run_name = "DQN_Flax"
    run_dir = f"runs/{args.env_id}__{run_name}__{run_time}"

    print(f"Training {run_name} on {args.env_id} for {args.total_timesteps} timesteps")
    print(f"Saving results to {run_dir}")

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
    if args.seed > 0:
        np.random.seed(args.seed)

    # Create vectorized environment
    env = gym.vector.SyncVectorEnv([make_env(args.env_id)])

    # Metadata about the environment
    observation_shape = env.single_observation_space.shape
    action_shape = env.single_action_space.n

    # Initialize environment
    state, _ = env.reset(seed=args.seed) if args.seed > 0 else env.reset()

    # Create the networks and the optimizer
    policy_net = QNetwork(num_actions=action_shape)
    initial_params = policy_net.init(jax.random.PRNGKey(args.seed), state)

    optimizer = optax.adam(learning_rate=args.learning_rate)

    # Jit the train step
    policy_net.apply = jax.jit(policy_net.apply)

    # Create the train state
    train_state = TrainState.create(
        apply_fn=policy_net.apply,
        params=initial_params,
        target_params=initial_params,
        tx=optimizer,
    )

    del initial_params

    # Create the replay buffer
    replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size, observation_shape)

    log_episodic_returns = []

    start_time = time.process_time()

    # Main loop
    for global_step in tqdm(range(args.total_timesteps)):
        # Exploration or intensification
        exploration_prob = get_exploration_prob(args.eps_start, args.eps_end, args.eps_decay, global_step)

        # Log exploration probability
        writer.add_scalar("rollout/eps_threshold", exploration_prob, global_step)

        if np.random.rand() < exploration_prob:
            # Exploration
            action = np.random.randint(0, action_shape, size=env.num_envs)
        else:
            # Intensification
            q_values = policy_net.apply(train_state.params, state)
            action = np.asarray(q_values.argmax(axis=1))

        # Perform action
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
            if not global_step % args.train_frequency:
                # Sample a batch from the replay buffer
                batch = replay_buffer.sample()

                # Train
                train_state, loss = train_step(train_state, batch, args.gamma)

                # Log training metrics
                writer.add_scalar("train/loss", np.asarray(loss), global_step)

            # Update target network
            if not global_step % args.target_update_frequency:
                train_state = train_state.replace(target_params=train_state.params)

        writer.add_scalar("rollout/SPS", int(global_step / (time.process_time() - start_time)), global_step)

    # Average of episodic returns (for the last 5% of the training)
    indexes = int(len(log_episodic_returns) * 0.05)
    avg_final_rewards = np.mean(log_episodic_returns[-indexes:])
    print(f"Average of the last {indexes} episodic returns: {round(avg_final_rewards, 2)}")
    writer.add_scalar("rollout/avg_final_rewards", avg_final_rewards, global_step)

    # Close the environment
    env.close()
    writer.close()
    if args.wandb:
        wandb.finish()

    # Capture video of the policy
    # if args.capture_video:
    #     print(f"Capturing videos and saving them to {run_dir}/videos ...")
    #     env_test = gym.vector.SyncVectorEnv([make_env(args.env_id, capture_video=True)])
    #     state, _ = env_test.reset()
    #     count_episodes = 0

    #     while count_episodes < 10:
    #         with torch.no_grad():
    #             state = torch.from_numpy(state).to(args.device).float()
    #             action = torch.argmax(policy_net(state), dim=1).cpu().numpy()

    #         state, _, terminated, truncated, _ = env_test.step(action)

    #         if terminated or truncated:
    #             count_episodes += 1

    #     env_test.close()
    #     print("Done!")


if __name__ == "__main__":
    main()
