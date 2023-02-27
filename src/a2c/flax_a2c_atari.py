import argparse
import random
import time
from datetime import datetime
from pathlib import Path

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


def make_env(env_id, capture_video=False):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env=env,
                video_folder=f"{run_dir}/videos/",
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
        return policy_log_probabilities, value


@jax.jit
def compute_td_target(rewards, flags, gamma):
    td_target = jnp.zeros_like(rewards)
    gain = jnp.zeros(rewards.shape[1])

    for i in reversed(range(td_target.shape[0])):
        terminal = 1.0 - flags[i]
        gain = rewards[i] + gain * gamma * terminal
        td_target = td_target.at[i].set(gain)

    td_target = (td_target - td_target.mean()) / (td_target.std() + 1e-7)
    td_target = td_target.squeeze()

    return td_target


def loss_fn(params, apply_fn, batch, value_coef, entropy_coef):
    log_probs, td_predict = apply_fn(params, batch["states"])
    log_probs_act_taken = jax.vmap(lambda lp, a: lp[a])(log_probs, batch["actions"])

    advantages = batch["td_target"] - td_predict

    actor_loss = (-log_probs_act_taken * advantages).mean()
    critic_loss = jnp.square(advantages).mean()

    return actor_loss + critic_loss * value_coef


@jax.jit
def train_step(train_state, batch, value_coef, entropy_coef):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(
        train_state.params,
        train_state.apply_fn,
        batch,
        value_coef,
        entropy_coef,
    )
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss


if __name__ == "__main__":
    args = parse_args()

    date = str(datetime.now().strftime("%d-%m_%H:%M"))
    # These variables are specific to the repo "rl-gym-zoo"
    # You should change them if you are just copy/paste the code
    algo_name = Path(__file__).stem.split("_")[0].upper()
    run_dir = Path(
        Path(__file__).parent.resolve().parents[1], "runs", f"{args.env_id}__{algo_name}__{date}"
    )

    # Initialize wandb if needed (https://wandb.ai/)
    if args.wandb:
        import wandb

        wandb.init(project=args.env_id, name=algo_name, sync_tensorboard=True, config=vars(args))

    # Create tensorboard writer and save hyperparameters
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Set seed for reproducibility
    if args.seed > 0:
        random.seed(args.seed)
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

    policy_net.apply = jax.jit(policy_net.apply)

    del initial_params

    # Create buffers
    states = np.zeros((args.num_steps, args.num_envs) + obversation_shape)
    actions = np.zeros((args.num_steps, args.num_envs), dtype=np.int32)
    rewards = np.zeros((args.num_steps, args.num_envs))
    flags = np.zeros((args.num_steps, args.num_envs))

    log_episodic_returns = []

    global_step = 0
    start_time = time.process_time()

    # Main loop
    for _ in tqdm(range(args.num_updates)):
        for i in range(args.num_steps):
            # Update global step
            global_step += 1 * args.num_envs

            # Get action
            log_probs, _ = policy_net.apply(train_state.params, state)
            probs = np.exp(np.array(log_probs))
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

        td_target = compute_td_target(rewards, flags, args.gamma)

        # Create batch
        batch = {
            "states": states.reshape(-1, *obversation_shape),
            "actions": actions.reshape(-1),
            "td_target": td_target.reshape(-1),
        }

        # Train
        train_state, loss = train_step(
            train_state,
            batch,
            args.value_coef,
            args.entropy_coef,
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
