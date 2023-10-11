import gymnasium as gym
import minari
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def collate_fn(batch):
    return {
        "id": torch.Tensor([x.id for x in batch]),
        "total_timesteps": torch.Tensor([x.total_timesteps for x in batch]),
        "observations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.observations) for x in batch],
            batch_first=True,
        ),
        "actions": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.actions) for x in batch],
            batch_first=True,
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.rewards) for x in batch],
            batch_first=True,
        ),
        "terminations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.terminations) for x in batch],
            batch_first=True,
        ),
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.truncations) for x in batch],
            batch_first=True,
        ),
    }


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    minari_dataset = minari.load_dataset("HalfCheetah-expert-v4")
    dataloader = DataLoader(minari_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)

    torch.manual_seed(42)

    env = minari_dataset.recover_environment()
    observation_space = env.observation_space
    action_space = env.action_space

    policy_net = PolicyNetwork(np.prod(observation_space.shape), np.prod(action_space.shape))
    optimizer = torch.optim.Adam(policy_net.parameters())
    loss_fn = nn.MSELoss()

    num_epochs = 32

    for epoch in range(num_epochs):
        for batch in dataloader:
            a_pred = policy_net(batch["observations"][:, :-1].float())
            a_hat = batch["actions"]
            loss = loss_fn(a_pred, a_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch}/{num_epochs}, Loss: {loss.item()}")

    env = gym.make("HalfCheetah-v4")

    obs, _ = env.reset(seed=42)
    rewards_bc = []

    for _ in range(100):
        obs, _ = env.reset()
        accumulated_rew = 0
        done = False

        while not done:
            action = policy_net(torch.Tensor(obs).float()).detach().numpy()
            obs, rew, ter, tru, _ = env.step(action)
            done = ter or tru
            accumulated_rew += rew

        rewards_bc.append(accumulated_rew)

    env.close()

    print(f"Mean rewards: {np.mean(rewards_bc)}")
