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
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v4")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
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


def make_env(env_id, capture_video=False, run_dir="."):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env=env, video_folder=f"{run_dir}/videos", episode_trigger=lambda x: x, disable_logger=True
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda state: np.clip(state, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        return env

    return thunk


@jax.jit
def sample_action(mean, std, key):
    key, subkey = jax.random.split(key)
    return jax.random.normal(subkey, shape=mean.shape) * std + mean, key


@jax.jit
def get_log_prob(mean, std, value):
    var = std**2
    log_scale = jnp.log(std)
    return -((value - mean) ** 2) / (2 * var) - log_scale - jnp.log(jnp.sqrt(2 * jnp.pi))


@jax.jit
@functools.partial(jax.vmap, in_axes=(1, 1, None), out_axes=1)
def compute_td_target(rewards, flags, gamma):
    td_target = []
    gain = 0.0
    for i in reversed(range(len(rewards))):
        gain = rewards[i] + gamma * flags[i] * gain
        td_target.append(gain)

    td_target = td_target[::-1]
    return jnp.array(td_target)


@functools.partial(jax.jit, static_argnums=0)
def policy_output(apply_fn, params, state):
    return apply_fn(params, state)


@functools.partial(jax.jit, static_argnums=(2, 3))
def train_step(train_state, batch, value_coef, entropy_coef):
    def loss_fn(params):
        states, actions, td_target = batch
        log_probs, td_predict = policy_output(train_state.apply_fn, params, states)

        log_probs_by_actions = jax.vmap(lambda lp, a: lp[a])(log_probs, actions)

        advantages = td_target - td_predict

        actor_loss = (-log_probs_by_actions * advantages).mean()
        critic_loss = jnp.square(advantages).mean()
        entropy_loss = -(log_probs * jnp.exp(log_probs)).sum(axis=-1).mean()

        return actor_loss + critic_loss * value_coef - entropy_loss * entropy_coef

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)

    return train_state, loss


class RolloutBuffer:
    def __init__(self, num_steps, num_envs, observation_shape, action_shape):
        self.states = np.zeros((num_steps, num_envs, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.flags = np.zeros((num_steps, num_envs), dtype=np.float32)

        self.step = 0
        self.num_steps = num_steps

    def push(self, state, action, reward, flag):
        self.states[self.step] = state
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.flags[self.step] = flag

        self.step = (self.step + 1) % self.num_steps

    def get(self):
        return self.states, self.actions, self.rewards, self.flags


class ActorCriticNet(nn.Module):
    action_dim: int
    list_layer: list

    @nn.compact
    def __call__(self, x):
        for layer in self.list_layer:
            x = nn.Dense(features=layer)(x)
            x = nn.tanh(x)

        action_mean = nn.Dense(features=self.action_dim)(x)
        action_std = nn.Dense(features=self.action_dim)(x)
        action_std = nn.sigmoid(action_std) + 1e-7

        value = nn.Dense(features=1)(x)

        return action_mean, action_std, value.squeeze()


def train(args, run_name, run_dir):
    # Initialize wandb if needed (https://wandb.ai/)
    if args.wandb:
        import wandb

        wandb.init(project=args.env_id, name=run_name, sync_tensorboard=True, config=vars(args))

    # Create tensorboard writer and save hyperparameters
    writer = SummaryWriter(run_dir)
    hyperparameters = "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
    table = f"|param|value|\n|-|-|\n{hyperparameters}"
    writer.add_text("hyperparameters", table)

    # Create vectorized environment(s)
    envs = gym.vector.AsyncVectorEnv([make_env(args.env_id) for _ in range(args.num_envs)])

    # Metadata about the environment
    observation_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    action_dim = np.prod(action_shape)

    # Initialize environment(s)
    state, _ = envs.reset(seed=args.seed) if args.seed else envs.reset()

    key, subkey = jax.random.split(jax.random.PRNGKey(args.seed))

    # Create policy network and optimizer
    policy_net = ActorCriticNet(action_dim=action_dim, list_layer=args.list_layer)
    initial_params = policy_net.init(subkey, state)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=args.clip_grad_norm), optax.adam(learning_rate=args.learning_rate)
    )

    train_state = TrainState.create(params=initial_params, apply_fn=policy_net.apply, tx=optimizer)

    del initial_params

    # Create buffers
    rollout_buffer = RolloutBuffer(args.num_steps, args.num_envs, observation_shape, action_shape)

    log_episodic_returns = []

    global_step = 0
    start_time = time.process_time()

    # Main loop
    for _ in tqdm(range(args.num_updates)):
        for _ in range(args.num_steps):
            # Update global step
            global_step += 1 * args.num_envs

            # Get action
            action_mean, action_std, _ = policy_output(train_state.apply_fn, train_state.params, state)
            action, key = sample_action(action_mean, action_std, key)

            action = np.array(action)

            # Perform action
            next_state, reward, terminated, truncated, infos = envs.step(action)

            # Store transition
            flag = 1.0 - np.logical_or(terminated, truncated)
            rollout_buffer.push(state, action, reward, flag)

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
        states, actions, rewards, flags = rollout_buffer.get()

        td_target = compute_td_target(rewards, flags, args.gamma)

        # Normalize td_target
        td_target = (td_target - td_target.mean()) / (td_target.std() + 1e-7)

        # Create batch
        batch = (states.reshape(-1, *observation_shape), actions.reshape(-1, *action_shape), td_target.reshape(-1))

        # Train
        train_state, loss = train_step(train_state, batch, args.value_coef, args.entropy_coef)

        # Log training metrics
        writer.add_scalar("train/loss", np.asarray(loss), global_step)
        writer.add_scalar("rollout/SPS", int(global_step / (time.process_time() - start_time)), global_step)

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

    # Metadata about the environment
    # observation_shape = env.single_observation_space.shape
    # action_shape = env.single_action_space.n

    # Load policy
    # policy = ActorCriticNet(observation_shape, action_shape, args.list_layer)
    # policy.load_state_dict(torch.load(f"{run_dir}/policy.pt"))
    # policy.eval()

    # count_episodes = 0
    list_rewards = []

    # state, _ = env.reset(seed=args.seed) if args.seed else env.reset()

    # # Run episodes
    # while count_episodes < 30:
    #     log_probs, _ = policy_output(train_state.apply_fn, train_state.params, state)
    #     probs = np.exp(log_probs)
    #     action = np.array([np.random.choice(action_shape, p=probs[i]) for i in range(args.num_envs)])

    #     action = action.cpu().numpy()
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
    args_ = parse_args()

    # Create run directory
    run_time = str(datetime.now().strftime("%d-%m_%H:%M:%S"))
    run_name = "A2C_Flax"
    run_dir = f"runs/{args_.env_id}__{run_name}__{run_time}"

    print(f"Commencing training of {run_name} on {args_.env_id} for {args_.total_timesteps} timesteps.")
    print(f"Results will be saved to: {run_dir}")
    mean_train_return = train(args=args_, run_name=run_name, run_dir=run_dir)
    print(f"Training - Mean returns achieved: {mean_train_return}.")

    if args_.capture_video:
        print(f"Evaluating and capturing videos of {run_name} on {args_.env_id}.")
        mean_eval_return = eval_and_render(args=args_, run_dir=run_dir)
        print(f"Evaluation - Mean returns achieved: {mean_eval_return}.")
