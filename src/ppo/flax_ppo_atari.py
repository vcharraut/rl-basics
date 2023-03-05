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
    parser.add_argument("--env_id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--num_optims", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae", type=float, default=0.95)
    parser.add_argument("--eps_clip", type=float, default=0.1)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = int(args.total_timesteps // args.batch_size)

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


class ActorCriticNet(nn.Module):
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
        logits = nn.Dense(features=self.num_actions, name="logits", dtype=dtype)(x)
        policy_log_probabilities = nn.log_softmax(logits)
        value = nn.Dense(features=1, name="value", dtype=dtype)(x)
        return policy_log_probabilities, value.squeeze()


@functools.partial(jax.jit, static_argnums=(0,))
def get_policy(apply_fn, params, state):
    return apply_fn(params, state)


@jax.jit
def collect_log_probs(log_probs, actions):
    return jax.vmap(lambda log_prob, action: log_prob[action])(log_probs, actions)


@functools.partial(jax.jit, static_argnums=(4, 5))
def compute_gae(rewards, state_values, flags, next_state_value, gamma, gae):
    advantages = jnp.zeros_like(rewards)
    adv = jnp.zeros(rewards.shape[1])

    for i in reversed(range(rewards.shape[0])):
        terminal = 1.0 - flags[i]

        returns = rewards[i] + gamma * next_state_value * terminal
        delta = returns - state_values[i]

        adv = gamma * gae * adv * terminal + delta
        advantages = advantages.at[i].set(adv)

        next_state_value = state_values[i]

    return advantages


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def train_step(train_state, batch, value_coef, entropy_coef, eps_clip):
    def loss_fn(params, apply_fn, batch, value_coef, entropy_coef, eps_clip):
        states, actions, old_log_probs, advantages, td_target = batch

        log_probs, td_predict = get_policy(apply_fn, params, states)
        log_probs_act_taken = collect_log_probs(log_probs, actions)

        ratios = jnp.exp(log_probs_act_taken - old_log_probs)

        surr1 = advantages * ratios
        surr2 = advantages * jax.lax.clamp(1.0 - eps_clip, ratios, 1.0 + eps_clip)

        actor_loss = -jnp.minimum(surr1, surr2).mean()
        critic_loss = jnp.square(td_target - td_predict).mean()
        entropy_loss = jnp.sum(-log_probs * jnp.exp(log_probs), axis=1).mean()

        return actor_loss + critic_loss * value_coef - entropy_loss * entropy_coef

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(
        train_state.params, train_state.apply_fn, batch, value_coef, entropy_coef, eps_clip
    )
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss


def main():
    args = parse_args()

    # Create run directory
    run_time = str(datetime.now().strftime("%d-%m_%H:%M:%S"))
    run_name = "PPO_Flax"
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
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Set seed for reproducibility
    if args.seed > 0:
        np.random.seed(args.seed)

    # Create vectorized environment(s)
    envs = gym.vector.AsyncVectorEnv([make_env(args.env_id) for _ in range(args.num_envs)])

    # Metadata about the environment
    obversation_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.n

    # Initialize environment
    state, _ = envs.reset(seed=args.seed) if args.seed > 0 else envs.reset()

    # Create policy network and optimizer
    policy_net = ActorCriticNet(num_actions=action_shape)

    optimizer = optax.adam(learning_rate=args.learning_rate)

    initial_params = policy_net.init(jax.random.PRNGKey(args.seed), state)

    train_state = TrainState.create(
        params=initial_params,
        apply_fn=policy_net.apply,
        tx=optimizer,
    )

    del initial_params

    # Create buffers
    states = np.zeros((args.num_steps, args.num_envs) + obversation_shape, dtype=np.float32)
    actions = np.zeros((args.num_steps, args.num_envs), dtype=np.int32)
    rewards = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    flags = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    list_log_probs = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    list_state_values = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)

    log_episodic_returns = []

    global_step = 0
    start_time = time.process_time()

    # Main loop
    for _ in tqdm(range(args.num_updates)):
        for i in range(args.num_steps):
            # Update global step
            global_step += 1 * args.num_envs

            # Get action
            log_probs, state_values = get_policy(train_state.apply_fn, train_state.params, state)
            probs = np.exp(log_probs)
            action = np.array(
                [np.random.choice(action_shape, p=probs[i]) for i in range(args.num_envs)]
            )

            # Perform action
            next_state, reward, terminated, truncated, infos = envs.step(action)

            # Store transition
            states[i] = state
            actions[i] = action
            rewards[i] = reward
            flags[i] = np.logical_or(terminated, truncated)
            list_log_probs[i] = collect_log_probs(log_probs, action)
            list_state_values[i] = state_values

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

        _, next_state_value = get_policy(train_state.apply_fn, train_state.params, state)

        advantages = compute_gae(
            rewards, list_state_values, flags, next_state_value, args.gamma, args.gae
        )

        td_target = advantages + list_state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # Create batch
        _states = states.reshape(-1, *obversation_shape)
        _actions = actions.reshape(-1)
        _log_probs = list_log_probs.reshape(-1)
        _advantages = advantages.reshape(-1)
        _td_target = td_target.reshape(-1)

        batch_indexes = np.arange(args.batch_size)

        # Update policy network
        for _ in range(args.num_optims):
            np.random.shuffle(batch_indexes)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                idxes = batch_indexes[start:end]

                batch = (
                    _states[idxes],
                    _actions[idxes],
                    _log_probs[idxes],
                    _advantages[idxes],
                    _td_target[idxes],
                )

                train_state, loss = train_step(
                    train_state,
                    batch,
                    args.value_coef,
                    args.entropy_coef,
                    args.eps_clip,
                )

        # Log training metrics
        writer.add_scalar("train/loss", np.asarray(loss), global_step)
        writer.add_scalar(
            "rollout/SPS", int(global_step / (time.process_time() - start_time)), global_step
        )

    # Average of episodic returns (for the last 5% of the training)
    indexes = int(len(log_episodic_returns) * 0.05)
    avg_final_rewards = np.mean(log_episodic_returns[-indexes:])
    print(f"Average of the last {indexes} episodic returns: {round(avg_final_rewards, 2)}")
    writer.add_scalar("rollout/avg_final_rewards", avg_final_rewards, global_step)

    # Close the environment
    envs.close()
    writer.close()
    if args.wandb:
        wandb.finish()

    # Capture video of the policy
    if args.capture_video:
        print(f"Capturing videos and saving them to {run_dir}/videos ...")
        env_test = gym.vector.SyncVectorEnv([make_env(args.env_id, capture_video=True)])
        state, _ = env_test.reset()
        count_episodes = 0

        while count_episodes < 10:
            action = policy_net(state)
            state, _, terminated, truncated, _ = env_test.step(action)

            action = policy_net(state)

            if terminated or truncated:
                count_episodes += 1

        env_test.close()
        print("Done!")


if __name__ == "__main__":
    main()
