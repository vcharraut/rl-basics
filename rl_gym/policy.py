import torch
import gymnasium as gym
import numpy as np
from rl_gym.algorithm.ppo import PPOContinuous, PPODiscrete
from rl_gym.algorithm.a2c import A2CDiscrete, A2CContinuous


class Policy:
    """
    Interface class between any Gym environment and any type of
    algortihms implemented.
    """

    def __init__(self, type_algorithm: str,
                 obversation_space: gym.spaces.box.Box,
                 action_space: gym.spaces.box.Box, learning_rate: float,
                 list_layer: list, is_shared_network: bool):
        """
        Constructs the Policy class.

        Args:
            type_algorithm: type of algorithm for the learning method.
            obversation_space: dimension of the observation.
            action_space: box object of the action space.
            learning_rate: learning rate to use during learning.
            list_layer: list of layers' size.
            is_shared_network: boolean to chose between shared or
                                      seperated network.
        """

        type_algorithm = type_algorithm.lower()

        self.is_continuous = type(action_space).__name__.lower() == "box"

        action_shape = action_space.shape if self.is_continuous else (
            action_space.n, )

        args = [
            obversation_space.shape, action_shape, learning_rate, list_layer,
            is_shared_network
        ]

        if type_algorithm == "a2c":
            self.algorithm = A2CContinuous(
                *args) if self.is_continuous else A2CDiscrete(*args)

        elif type_algorithm == "a3c":
            raise NotImplementedError()

        elif type_algorithm == "ppo":
            self.algorithm = PPOContinuous(
                *args) if self.is_continuous else PPODiscrete(*args)

        elif type_algorithm == "ddpg":
            raise NotImplementedError()

        elif type_algorithm == "td3":
            raise NotImplementedError()

        elif type_algorithm == "sac":
            raise NotImplementedError()

    def get_action(self, state: np.ndarray) -> tuple:
        """
        Get the optimal action based on the current state.

        Args:
            state: current state of the environment.

        Returns:
            action chosed by the model.
        """

        state_torch = torch.from_numpy(state).float().to(torch.device("cuda"))
        return self.algorithm.act(state_torch)

    def get_obs_and_action_shape(self):
        obversation_shape, action_shape = self.algorithm.get_obs_and_action_shape(
        )
        if not self.is_continuous:
            action_shape = ()
        return obversation_shape, action_shape

    def add_reward_to_tensorboard(self, reward):
        self.algorithm.writer.add_scalar("Rollout/reward", reward,
                                         self.algorithm.global_step)

    def add_hparams_to_tensorboard(self, learning_rate, layer, reward):
        self.algorithm.writer.add_hparams(
            {
                "learning_rate": learning_rate,
                "layer": layer
            }, {"Reward": reward})

    def close_writer(self):
        self.algorithm.writer.close()

    def update_policy(self, minibatch: dict):
        """
        Update the model based on the training data.

        Args:
            minibatch: batch of data from training.
        """

        self.algorithm.update_policy(minibatch)

    def save(self, path: str):
        """
        Save the model to disk.

        Args:
            path: path file to save the model.
        """

        self.algorithm.save_model(path)

    def load(self, path: str):
        """
        Load a model from disk.

        Args:
            path: path file to load the model.
        """

        self.algorithm.load_model(path)
