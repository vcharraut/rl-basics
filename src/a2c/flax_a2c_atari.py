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
    parser.add_argument("--env_id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--total_timesteps", type=int, default=5_000_000)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
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
            env = gym.make(env_id, frameskip=1, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env=env,
                video_folder=f"{run_dir}/videos",
                episode_trigger=lambda x: x,
                disable_logger=True,
            )
        else:
            env = gym.make(env_id, frameskip=1)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.AtariPreprocessing(env)
        env = gym.wrappers.FrameStack(env, 4)

        return env

    return thunk


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
    def __init__(self, num_steps, num_envs, observation_shape):
        self.states = np.zeros((num_steps, num_envs, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs), dtype=np.int64)
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

        logits = nn.Dense(features=self.action_dim)(output)
        log_prob = nn.log_softmax(logits)

        value = nn.Dense(features=1)(output)

        return log_prob, value.squeeze()


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
    action_dim = envs.single_action_space.n

    # Set seed for reproducibility
    if args.seed:
        numpy_rng = np.random.default_rng(args.seed)
        state, _ = envs.reset(seed=args.seed)
    else:
        numpy_rng = np.random.default_rng()
        state, _ = envs.reset()

    key = jax.random.PRNGKey(args.seed)

    # Create policy network and optimizer
    policy_net = ActorCriticNet(action_dim=action_dim)
    init_params = policy_net.init(key, state)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=args.clip_grad_norm),
        optax.adam(learning_rate=args.learning_rate),
    )

    train_state = TrainState.create(params=init_params, apply_fn=policy_net.apply, tx=optimizer)

    # Create buffers
    rollout_buffer = RolloutBuffer(args.num_steps, args.num_envs, observation_shape)

    # Remove unnecessary variables
    del key, policy_net, init_params, optimizer

    global_step = 0
    log_episodic_returns, log_episodic_lengths = [], []
    start_time = time.process_time()

    # Main loop
    for _ in tqdm(range(args.num_updates)):
        for _ in range(args.num_steps):
            # Update global step
            global_step += 1 * args.num_envs

            # Get action
            log_prob, _ = policy_output(train_state.apply_fn, train_state.params, state)
            probs = np.exp(log_prob)
            action = np.array([numpy_rng.choice(action_dim, p=probs[i]) for i in range(args.num_envs)])

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
                log_episodic_lengths.append(info["episode"]["l"])
                writer.add_scalar("rollout/episodic_return", np.mean(info["episode"]["r"][-10:]), global_step)
                writer.add_scalar("rollout/episodic_length", np.mean(info["episode"]["l"][-10:]), global_step)

                break

        # Get transition batch
        states, actions, rewards, flags = rollout_buffer.get()

        td_target = compute_td_target(rewards, flags, args.gamma)

        # Normalize td_target
        td_target = (td_target - td_target.mean()) / (td_target.std() + 1e-7)

        # Create batch
        batch = (states.reshape(-1, *observation_shape), actions.reshape(-1), td_target.reshape(-1))

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


if __name__ == "__main__":
    args_ = parse_args()

    # Create run directory
    run_time = str(datetime.now().strftime("%d-%m_%H:%M:%S"))
    run_name = "A2C_Flax"
    env_name = args_.env_id.split("/")[1]
    run_dir = f"runs/{env_name}__{run_name}__{run_time}"

    print(f"Commencing training of {run_name} on {args_.env_id} for {args_.total_timesteps} timesteps.")
    print(f"Results will be saved to: {run_dir}")
    mean_train_return = train(args=args_, run_name=run_name, run_dir=run_dir)
    print(f"Training - Mean returns achieved: {mean_train_return}.")
