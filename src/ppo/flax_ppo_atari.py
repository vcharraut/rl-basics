import argparse
import functools
import time
from datetime import datetime

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
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--num_optims", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae", type=float, default=0.95)
    parser.add_argument("--eps_clip", type=float, default=0.1)
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
        env = gym.wrappers.AtariPreprocessing(env)
        env = gym.wrappers.FrameStack(env, 4)

        return env

    return thunk


@jax.jit
def collect_log_probs(log_probs, actions):
    return jax.vmap(lambda log_prob, action: log_prob[action])(log_probs, actions)


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


@functools.partial(jax.jit, static_argnums=(0,))
def policy_output(apply_fn, params, state):
    return apply_fn(params, state)


def loss_fn(params, apply_fn, batch, value_coef, entropy_coef, eps_clip):
    states, actions, old_log_probs, advantages, td_target = batch

    log_probs, td_predict = policy_output(apply_fn, params, states)
    log_probs_act_taken = collect_log_probs(log_probs, actions)

    ratios = jnp.exp(log_probs_act_taken - old_log_probs)

    surr1 = advantages * ratios
    surr2 = advantages * jax.lax.clamp(1.0 - eps_clip, ratios, 1.0 + eps_clip)

    actor_loss = -jnp.minimum(surr1, surr2).mean()
    critic_loss = jnp.square(td_target - td_predict).mean()
    entropy_loss = jnp.sum(-log_probs * jnp.exp(log_probs), axis=1).mean()

    return actor_loss + critic_loss * value_coef - entropy_loss * entropy_coef


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def train_step(train_state, trajectories, num_minibatches, minibatch_size, value_coef, entropy_coef, eps_clip):
    trajectories = jax.tree_util.tree_map(
        lambda x: x.reshape((num_minibatches, minibatch_size) + x.shape[1:]),
        trajectories,
    )

    for batch in zip(*trajectories):
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(train_state.params, train_state.apply_fn, batch, value_coef, entropy_coef, eps_clip)
        train_state = train_state.apply_gradients(grads=grads)

    return train_state, loss


class RolloutBuffer:
    def __init__(self, num_steps, num_envs, observation_shape):
        self.states = np.zeros((num_steps, num_envs, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs), dtype=np.int64)
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
    num_actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), name="conv1")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), name="conv2")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), name="conv3")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=512, name="hidden")(x)
        x = nn.relu(x)

        logits = nn.Dense(features=self.num_actions, name="actor")(x)
        log_prob = nn.log_softmax(logits)

        value = nn.Dense(features=1, name="critic")(x)

        return log_prob, value.squeeze()


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

    # Create vectorized environment(s)
    envs = gym.vector.AsyncVectorEnv([make_env(args.env_id) for _ in range(args.num_envs)])

    # Metadata about the environment
    observation_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.n

    # Set seed for reproducibility
    if args.seed:
        numpy_rng = np.random.default_rng(args.seed)
        state, _ = envs.reset(seed=args.seed)
    else:
        numpy_rng = np.random.default_rng()
        state, _ = envs.reset()

    key = jax.random.PRNGKey(args.seed)

    # Create policy network and optimizer
    policy = ActorCriticNet(num_actions=action_shape)
    initial_params = policy.init(key, state)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=args.clip_grad_norm), optax.adam(learning_rate=args.learning_rate)
    )

    train_state = TrainState.create(params=initial_params, apply_fn=policy.apply, tx=optimizer)

    del initial_params

    # Create buffers
    rollout_buffer = RolloutBuffer(args.num_steps, args.num_envs, observation_shape)

    log_episodic_returns = []

    global_step = 0
    start_time = time.process_time()

    # Main loop
    for _ in tqdm(range(args.num_updates)):
        for _ in range(args.num_steps):
            # Update global step
            global_step += 1 * args.num_envs

            # Get action
            log_prob, value = policy_output(train_state.apply_fn, train_state.params, state)
            probs = np.exp(log_prob)
            action = np.array([numpy_rng.choice(action_shape, p=probs[i]) for i in range(args.num_envs)])

            # Perform action
            next_state, reward, terminated, truncated, infos = envs.step(action)

            # Store transition
            flag = 1.0 - np.logical_or(terminated, truncated)
            rollout_buffer.push(state, action, reward, flag, collect_log_probs(log_prob, action), value)

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

        # Get transition batch
        states, actions, rewards, flags, log_probs, values = rollout_buffer.get()

        _, last_value = policy_output(train_state.apply_fn, train_state.params, next_state)

        # Calculate advantages and TD target
        advantages = compute_advantages(
            rewards, values, flags, last_value, args.gamma, args.gae, args.num_steps, args.num_envs
        )
        td_target = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten batch
        batch = (
            states.reshape(-1, *observation_shape),
            actions.reshape(-1),
            log_probs.reshape(-1),
            advantages.reshape(-1),
            td_target.reshape(-1),
        )

        # Perform PPO update
        for _ in range(args.num_optims):
            permutation = numpy_rng.permutation(args.batch_size)
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
        writer.add_scalar("train/loss", np.asarray(loss), global_step)
        writer.add_scalar("rollout/SPS", int(global_step / (time.process_time() - start_time)), global_step)

    # Save the final policy
    # checkpoints.save_checkpoint(ckpt_dir=run_dir, target=train_state, step=0)

    # Close the environment
    envs.close()
    writer.close()

    # Average of episodic returns (for the last 5% of the training)
    indexes = int(len(log_episodic_returns) * 0.05)
    mean_train_return = np.mean(log_episodic_returns[-indexes:])
    writer.add_scalar("rollout/mean_train_return", mean_train_return, global_step)

    return mean_train_return


def eval_and_render(args, run_dir):
    # Create environment
    env = gym.vector.SyncVectorEnv([make_env(args.env_id, capture_video=True, run_dir=run_dir)])
    state, _ = env.reset(seed=args.seed) if args.seed else env.reset()

    # Metadata about the environment
    action_shape = env.single_action_space.n

    # Load policy
    policy = ActorCriticNet(num_actions=action_shape, list_layer=args.list_layer)
    params = policy.init(jax.random.PRNGKey(args.seed), state)

    train_state = TrainState.create(apply_fn=policy.apply, params=params)
    # train_state = checkpoints.restore_checkpoint(ckpt_dir=run_dir, target=train_state)

    count_episodes = 0
    list_rewards = []
    numpy_rng = np.random.default_rng()

    # Run episodes
    while count_episodes < 30:
        log_probs, state_values = policy_output(train_state.apply_fn, train_state.params, state)
        probs = np.exp(log_probs)
        action = np.array([numpy_rng.choice(action_shape, p=probs[i]) for i in range(args.num_envs)])

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
    run_name = "PPO_Flax"
    run_dir = f"runs/{args.env_id}__{run_name}__{run_time}"

    print(f"Commencing training of {run_name} on {args.env_id} for {args.total_timesteps} timesteps.")
    print(f"Results will be saved to: {run_dir}")
    mean_train_return = train(args=args, run_name=run_name, run_dir=run_dir)
    print(f"Training - Mean returns achieved: {mean_train_return}.")

    if args.capture_video:
        print(f"Evaluating and capturing videos of {run_name} on {args.env_id}.")
        mean_eval_return = eval_and_render(args=args, run_dir=run_dir)
        print(f"Evaluation - Mean returns achieved: {mean_eval_return}.")
