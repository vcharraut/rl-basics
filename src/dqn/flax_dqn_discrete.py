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
    parser.add_argument("--env_id", type=str, default="LunarLander-v2")
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=25_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
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
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        return env

    return thunk


class QNetwork(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_actions)(x)

        return x


@functools.partial(jax.jit, static_argnums=(0,))
def policy_output(apply_fn, params, state):
    return apply_fn(params, state)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, observation_shape, numpy_rng):
        self.state_buffer = np.zeros((buffer_size, *observation_shape), dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size,), dtype=np.int64)
        self.reward_buffer = np.zeros((buffer_size,), dtype=np.float32)
        self.flag_buffer = np.zeros((buffer_size,), dtype=np.float32)

        self.batch_size = batch_size
        self.max_size = buffer_size
        self.idx = 0
        self.size = 0

        self.numpy_rng = numpy_rng

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
            self.state_buffer[idxs],
            self.action_buffer[idxs],
            self.reward_buffer[idxs],
            self.state_buffer[idxs + 1],
            self.flag_buffer[idxs],
        )


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def loss_fn(params, target_params, apply_fn, batch, gamma):
    states, actions, rewards, next_states, flags = batch

    # Compute TD error
    q_predict = policy_output(apply_fn, params, states)
    td_predict = jax.vmap(lambda qp, a: qp[a])(q_predict, actions)

    # Compute TD target with Double Q-Learning
    action_by_qvalue = policy_output(apply_fn, params, next_states).argmax(axis=1)
    q_target = policy_output(apply_fn, target_params, next_states)
    max_q_target = jax.vmap(lambda qt, a: qt[a])(q_target, action_by_qvalue)

    td_target = rewards + (1.0 - flags) * gamma * max_q_target

    return jnp.mean((td_predict - td_target) ** 2)


@jax.jit
def train_step(train_state, batch, gamma):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(train_state.params, train_state.target_params, train_state.apply_fn, batch, gamma)
    train_state = train_state.apply_gradients(grads=grads)

    return train_state, loss


def get_exploration_prob(eps_start, eps_end, eps_decay, step):
    return eps_end + (eps_start - eps_end) * np.exp(-1.0 * step / eps_decay)


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
    action_shape = env.single_action_space.n

    # Set seed for reproducibility
    if args.seed:
        numpy_rng = np.random.default_rng(args.seed)
        state, _ = env.reset(seed=args.seed)
    else:
        numpy_rng = np.random.default_rng()
        state, _ = env.reset()

    key = jax.random.PRNGKey(args.seed)

    # Create the networks and the optimizer
    policy = QNetwork(num_actions=action_shape)
    initial_params = policy.init(key, state)

    optimizer = optax.adam(learning_rate=args.learning_rate)

    train_state = TrainState.create(
        apply_fn=policy.apply,
        params=initial_params,
        target_params=initial_params,
        tx=optimizer,
    )

    del initial_params

    # Create the replay buffer
    replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size, observation_shape, numpy_rng)

    log_episodic_returns = []

    start_time = time.process_time()

    # Main loop
    for global_step in tqdm(range(args.total_timesteps)):
        # Exploration or intensification
        exploration_prob = get_exploration_prob(args.eps_start, args.eps_end, args.eps_decay, global_step)

        # Log exploration probability
        writer.add_scalar("rollout/eps_threshold", exploration_prob, global_step)

        if numpy_rng.random() < exploration_prob:
            # Exploration
            action = numpy_rng.integers(0, action_shape, size=env.num_envs)
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

    # Save the final policy
    # flax.training.checkpoints.save_checkpoint(ckpt_dir=run_dir, target=train_state, step=0)

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
    # state, _ = env.reset(seed=args.seed) if args.seed else env.reset()

    # Metadata about the environment
    # action_shape = env.single_action_space.n

    # Load policy
    # policy = QNetwork(num_actions=action_shape)
    # params = policy.init(jax.random.PRNGKey(args.seed), state)

    # train_state = TrainState.create(apply_fn=policy.apply, params=params)
    # train_state = flax.training.checkpoints.restore_checkpoint(ckpt_dir=run_dir, target=train_state)

    # count_episodes = 0
    list_rewards = []

    # Run episodes
    # while count_episodes < 30:
    #     if np.random.rand() < 0.05:
    #         action = np.random.randint(0, action_shape, size=1)
    #     else:
    #         q_values = policy.apply(train_state.params, state)
    #         action = np.asarray(q_values.argmax(axis=1))

    #     state, _, _, _, infos = env.step(action)

    #     if "final_info" in infos:
    #         info = infos["final_info"][0]
    #         returns = info["episode"]["r"][0]
    #         count_episodes += 1
    #         list_rewards.append(returns)
    #         print(f"-> Episode {count_episodes}: {returns} returns")

    env.close()

    return np.mean(list_rewards)


if __name__ == "__main__":
    args = parse_args()

    # Create run directory
    run_time = str(datetime.now().strftime("%d-%m_%H:%M:%S"))
    run_name = "DQN_Flax"
    run_dir = f"runs/{args.env_id}__{run_name}__{run_time}"

    print(f"Commencing training of {run_name} on {args.env_id} for {args.total_timesteps} timesteps.")
    print(f"Results will be saved to: {run_dir}")
    mean_train_return = train(args=args, run_name=run_name, run_dir=run_dir)
    print(f"Training - Mean returns achieved: {mean_train_return}.")

    if args.capture_video:
        print(f"Evaluating and capturing videos of {run_name} on {args.env_id}.")
        mean_eval_return = eval_and_render(args=args, run_dir=run_dir)
        print(f"Evaluation - Mean returns achieved: {mean_eval_return}.")
