import torch
import gymnasium as gym
import numpy as np
from rl_gym.algorithm.a2c import A2C
from rl_gym.algorithm.ppo import PPO


class Agent:

    def __init__(self, args, obversation_space: gym.spaces.box.Box,
                 action_space: gym.spaces.box.Box, writer):

        type_algorithm = args.algo.lower()

        self.is_continuous = type(action_space).__name__.lower() == "box"
        self.device = args.device

        action_shape = action_space.shape if self.is_continuous else (
            action_space.n, )

        args = [
            args, obversation_space.shape, action_shape, writer, self.is_continuous
        ]

        if type_algorithm == "a2c":
            self.algorithm = A2C(*args)

        elif type_algorithm == "a3c":
            raise NotImplementedError()

        elif type_algorithm == "ppo":
            self.algorithm = PPO(*args)

        elif type_algorithm == "ddpg":
            raise NotImplementedError()

        elif type_algorithm == "td3":
            raise NotImplementedError()

        elif type_algorithm == "sac":
            raise NotImplementedError()

    def get_action(self, state: np.ndarray) -> tuple:

        state_torch = torch.from_numpy(state).to(self.device).float()
        return self.algorithm.act(state_torch)

    def get_obs_and_action_shape(self):

        return self.algorithm.get_obs_and_action_shape()

    def update_policy(self, batch: dict, step: int):

        self.algorithm.update_policy(batch, step)

    def save(self, path: str):

        self.algorithm.save_model(path)

    def load(self, path: str):

        self.algorithm.load_model(path)
