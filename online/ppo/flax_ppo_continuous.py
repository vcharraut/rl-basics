import argparse
import functools
from datetime import datetime
from time import perf_counter

import gymnasium as gym
import jax
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import numpy as jnp
from tensorflow_probability.substrates.jax.distributions import Normal
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v4")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=2048)
    parser.add_argument("--num_optims", type=int, default=10)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--actor_layers", nargs="+", type=int, default=[64, 64])
    parser.add_argument("--critic_layers", nargs="+", type=int, default=[64, 64])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae", type=float, default=0.95)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--clip_grad_norm", type=float, default=0.5)
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_minibatches = int(args.batch_size // args.minibatch_size)
    args.num_updates = int(args.total_timesteps // args.batch_size)

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
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.FlattenObservation(env)

        return env

    return thunk


@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7))
def compute_advantages(rewards, values, flags, last_value, gamma, gae, num_steps, num_envs):
    advantages = jnp.zeros((num_steps, num_envs))
    adv = jnp.zeros(num_envs)

    for i in reversed(range(num_steps)):
        returns = rewards[i] + gamma * flags[i] * last_value
        delta = returns - values[i]

        adv = delta + gamma * gae * flags[i] * adv
        advantages = advantages.at[i].set(adv)

        last_value = values[i]

    return advantages


@functools.partial(jax.jit, static_argnums=0)
def policy_predict(apply_fn, params, state, key):
    dist, value = apply_fn(params, state)
    key, action_key = jax.random.split(key)
    action = dist.sample(seed=action_key)
    log_prob = dist.log_prob(action).sum(-1)

    return action, log_prob, value, key


@functools.partial(jax.jit, static_argnums=0)
def policy_critic(apply_fn, params, state):
    _, value = apply_fn(params, state)

    return value


@functools.partial(jax.jit, static_argnums=0)
def policy_evaluate(apply_fn, params, states, actions):
    dist, value = apply_fn(params, states)
    log_probs = dist.log_prob(actions).sum(-1)
    entropy = dist.entropy().sum(-1)

    return log_probs, entropy, value


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def train_step(train_state, trajectories, num_minibatches, minibatch_size, value_coef, entropy_coef, eps_clip):
    def loss_fn(params, batch):
        states, actions, old_log_probs, advantages, td_target = batch

        log_probs, entropy, td_predict = policy_evaluate(train_state.apply_fn, params, states, actions)

        ratios = jnp.exp(log_probs - old_log_probs)

        surr1 = advantages * ratios
        surr2 = advantages * jax.lax.clamp(1.0 - eps_clip, ratios, 1.0 + eps_clip)

        actor_loss = -jnp.minimum(surr1, surr2).mean()
        critic_loss = jnp.square(td_target - td_predict).mean()
        entropy_loss = entropy.mean()

        loss = actor_loss + critic_loss * value_coef - entropy_loss * entropy_coef

        return loss

    trajectories = jax.tree_util.tree_map(
        lambda x: x.reshape((num_minibatches, minibatch_size) + x.shape[1:]),
        trajectories,
    )

    for batch in zip(*trajectories):
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(train_state.params, batch)
        train_state = train_state.apply_gradients(grads=grads)

    return train_state, loss


class RolloutBuffer:
    def __init__(self, num_steps, num_envs, observation_shape, action_shape):
        self.states = np.zeros((num_steps, num_envs, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.flags = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)

        self.step = 0
        self.num_steps = num_steps

    def push(self, state, action, reward, flag, log_prob, value):
        self.states[self.step] = state
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.flags[self.step] = flag
        self.log_probs[self.step] = log_prob
        self.values[self.step] = value

        self.step = (self.step + 1) % self.num_steps

    def get(self):
        return (self.states, self.actions, self.rewards, self.flags, self.log_probs, self.values)


class ActorCriticNet(nn.Module):
    action_dim: int
    actor_layers: list
    critic_layers: list

    @nn.compact
    def __call__(self, state):
        actor_output = state
        for layer in self.actor_layers:
            actor_output = nn.Dense(features=layer)(actor_output)
            actor_output = nn.tanh(actor_output)

        critic_output = state
        for layer in self.critic_layers:
            critic_output = nn.Dense(features=layer)(critic_output)
            critic_output = nn.tanh(critic_output)

        mean = nn.Dense(features=self.action_dim)(actor_output)
        std = nn.sigmoid(nn.Dense(features=self.action_dim)(actor_output)) + 1e-7

        distribution = Normal(loc=mean, scale=std)

        value = nn.Dense(features=1)(critic_output)

        return distribution, value.squeeze()


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

    # Create vectorized environment(s)
    envs = gym.vector.AsyncVectorEnv([make_env(args.env_id) for _ in range(args.num_envs)])

    # Metadata about the environment
    observation_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    action_dim = np.prod(action_shape)

    # Initialize state
    state, _ = envs.reset(seed=args.seed) if args.seed else envs.reset()

    key, model_key = jax.random.split(jax.random.PRNGKey(args.seed))

    # Create policy network and optimizer
    policy_net = ActorCriticNet(action_dim=action_dim, actor_layers=args.actor_layers, critic_layers=args.critic_layers)
    init_params = policy_net.init(model_key, state)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=args.clip_grad_norm),
        optax.adam(learning_rate=args.learning_rate),
    )

    train_state = TrainState.create(params=init_params, apply_fn=policy_net.apply, tx=optimizer)

    # Create buffers
    rollout_buffer = RolloutBuffer(args.num_steps, args.num_envs, observation_shape, action_shape)

    # Remove unnecessary variables
    del action_dim, model_key, policy_net, init_params, optimizer

    global_step = 0
    log_episodic_returns, log_episodic_lengths = [], []
    start_time = perf_counter()

    # Main loop
    for _ in tqdm(range(args.num_updates)):
        for _ in range(args.num_steps):
            # Update global step
            global_step += 1 * args.num_envs

            # Get action
            action, log_prob, value, key = policy_predict(train_state.apply_fn, train_state.params, state, key)

            # Perform action
            next_state, reward, terminated, truncated, infos = envs.step(jax.device_get(action))

            # Store transition
            flag = 1.0 - np.logical_or(terminated, truncated)
            rollout_buffer.push(state, action, reward, flag, log_prob, value)

            state = next_state

            if "final_info" not in infos:
                continue

            # Log episodic return and length
            for info in infos["final_info"]:
                if info is None:
                    continue

                log_episodic_returns.append(info["episode"]["r"])
                log_episodic_lengths.append(info["episode"]["l"])
                writer.add_scalar("rollout/episodic_return", np.mean(log_episodic_returns[-5:]), global_step)
                writer.add_scalar("rollout/episodic_length", np.mean(log_episodic_lengths[-5:]), global_step)

        # Get transition batch
        states, actions, rewards, flags, log_probs, values = rollout_buffer.get()

        last_value = policy_critic(train_state.apply_fn, train_state.params, next_state)

        # Calculate advantages and TD target
        advantages = compute_advantages(
            rewards,
            values,
            flags,
            last_value,
            args.gamma,
            args.gae,
            args.num_steps,
            args.num_envs,
        )
        td_target = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten batch
        batch = (
            states.reshape(-1, *observation_shape),
            actions.reshape(-1, *action_shape),
            log_probs.reshape(-1),
            advantages.reshape(-1),
            td_target.reshape(-1),
        )

        # Perform PPO update
        for _ in range(args.num_optims):
            key, subkey = jax.random.split(key)
            permutation = jax.random.permutation(subkey, args.batch_size)
            batch = tuple(x[permutation] for x in batch)

            train_state, loss = train_step(
                train_state,
                batch,
                args.num_minibatches,
                args.minibatch_size,
                args.value_coef,
                args.entropy_coef,
                args.eps_clip,
            )

        # Log training metrics
        writer.add_scalar("rollout/SPS", int(global_step / (perf_counter() - start_time)), global_step)
        writer.add_scalar("train/loss", jax.device_get(loss), global_step)

    # Close the environment
    envs.close()
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
    run_name = "PPO_Flax"
    run_dir = f"runs/{args_.env_id}__{run_name}__{run_time}"

    print(f"Commencing training of {run_name} on {args_.env_id} for {args_.total_timesteps} timesteps.")
    print(f"Results will be saved to: {run_dir}")
    mean_train_return = train(args=args_, run_name=run_name, run_dir=run_dir)
    print(f"Training - Mean returns achieved: {mean_train_return}.")
