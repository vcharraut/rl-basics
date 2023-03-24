import argparse
import functools
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
    parser.add_argument("--env_id", type=str, default="ALE/Pong-v5")
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


def make_env(env_id, capture_video=False, run_dir="."):
    def thunk():
        if capture_video:
            env = gym.make(
                env_id,
                frameskip=1,
                full_action_space=False,
                repeat_action_probability=0.0,
                render_mode="rgb_array",
            )
            env = gym.wrappers.RecordVideo(
                env=env,
                video_folder=f"{run_dir}/videos",
                episode_trigger=lambda x: x,
                disable_logger=True,
            )
        else:
            env = gym.make(env_id, frameskip=1, full_action_space=False, repeat_action_probability=0.0)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.AtariPreprocessing(env)
        env = gym.wrappers.FrameStack(env, 4)

        return env

    return thunk


def get_exploration_prob(eps_start, eps_end, eps_decay, step):
    return eps_end + (eps_start - eps_end) * np.exp(-1.0 * step / eps_decay)


@functools.partial(jax.jit, static_argnums=0)
def policy_output(apply_fn, params, state):
    return apply_fn(params, state)


@functools.partial(jax.jit, static_argnums=2)
def train_step(train_state, batch, gamma):
    def loss_fn(params):
        states, actions, rewards, next_states, flags = batch

        # Compute TD error
        q_predict = policy_output(train_state.apply_fn, params, states)
        td_predict = jax.vmap(lambda qp, a: qp[a])(q_predict, actions)

        # Compute TD target with Double Q-Learning
        action_by_qvalue = policy_output(train_state.apply_fn, params, next_states).argmax(axis=1)
        q_target = policy_output(train_state.apply_fn, train_state.target_params, next_states)
        max_q_target = jax.vmap(lambda qt, a: qt[a])(q_target, action_by_qvalue)

        td_target = rewards + (1.0 - flags) * gamma * max_q_target

        return jnp.mean((td_predict - td_target) ** 2)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)

    return train_state, loss


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, observation_shape, numpy_rng):
        self.states = np.zeros((buffer_size, *observation_shape), dtype=np.int8)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.flags = np.zeros((buffer_size,), dtype=np.float32)

        self.batch_size = batch_size
        self.max_size = buffer_size
        self.idx = 0
        self.size = 0

        self.numpy_rng = numpy_rng

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
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.states[idxs + 1],
            self.flags[idxs],
        )


class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, state):
        output = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(state)
        output = nn.relu(output)
        output = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(output)
        output = nn.relu(output)
        output = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(output)
        output = nn.relu(output)
        output = output.reshape((output.shape[0], -1))
        output = nn.Dense(features=512)(output)
        output = nn.relu(output)
        output = nn.Dense(features=self.action_dim)(output)
        return output


def train(args, run_name, run_dir):
    # Initialize wandb if needed (https://wandb.ai/)
    if args.wandb:
        import wandb

        wandb.init(
            project=args_.env_id.split("/")[1],
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
        state, _ = env.reset(seed=args.seed)
    else:
        numpy_rng = np.random.default_rng()
        state, _ = env.reset()

    key = jax.random.PRNGKey(args.seed)

    # Create the networks and the optimizer
    policy_net = QNetwork(action_dim=action_dim)
    init_params = policy_net.init(key, state)

    optimizer = optax.adam(learning_rate=args.learning_rate)

    train_state = TrainState.create(
        apply_fn=policy_net.apply,
        params=init_params,
        target_params=init_params,
        tx=optimizer,
    )

    # Create the replay buffer
    replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size, observation_shape, numpy_rng)

    # Remove unnecessary variables
    del policy_net, init_params, optimizer, observation_shape, key

    log_episodic_returns, log_episodic_lengths = [], []
    start_time = time.process_time()

    # Main loop
    for global_step in tqdm(range(args.total_timesteps)):
        # Exploration or intensification
        exploration_prob = get_exploration_prob(args.eps_start, args.eps_end, args.eps_decay, global_step)

        # Log exploration probability
        writer.add_scalar("rollout/eps_threshold", exploration_prob, global_step)

        if numpy_rng.random() < exploration_prob:
            # Exploration
            action = numpy_rng.integers(0, action_dim, size=env.num_envs)
        else:
            # Intensification
            q_values = policy_output(train_state.apply_fn, train_state.params, state)
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
            log_episodic_lengths.append(info["episode"]["l"])
            writer.add_scalar("rollout/episodic_return", np.mean(info["episode"]["r"][-5:]), global_step)
            writer.add_scalar("rollout/episodic_length", np.mean(info["episode"]["l"][-5:]), global_step)

        # Perform training step
        if global_step > args.learning_start:
            if not global_step % args.train_frequency:
                # Sample a batch from the replay buffer
                batch = replay_buffer.sample()

                # Train
                train_state, loss = train_step(train_state, batch, args.gamma)

            # Update target network
            if not global_step % args.target_update_frequency:
                train_state = train_state.replace(target_params=train_state.params)

            # Log training metrics
            writer.add_scalar("rollout/SPS", int(global_step / (time.process_time() - start_time)), global_step)
            writer.add_scalar("train/loss", np.array(loss), global_step)

    # Close the environment
    env.close()
    writer.close()

    # Average of episodic returns (for the last 5% of the training)
    indexes = int(len(log_episodic_returns) * 0.05)
    mean_train_return = np.mean(log_episodic_returns[-indexes:])
    writer.add_scalar("rollout/mean_train_return", mean_train_return, global_step)

    return mean_train_return


if __name__ == "__main__":
    args_ = parse_args()

    # Create run directory
    run_time = str(datetime.now().strftime("%d-%m_%H:%M:%S"))
    run_name = "DQN_Flax"
    env_name = args_.env_id.split("/")[1]
    run_dir = f"runs/{env_name}__{run_name}__{run_time}"

    print(f"Commencing training of {run_name} on {args_.env_id} for {args_.total_timesteps} timesteps.")
    print(f"Results will be saved to: {run_dir}")
    mean_train_return = train(args=args_, run_name=run_name, run_dir=run_dir)
    print(f"Training - Mean returns achieved: {mean_train_return}.")
