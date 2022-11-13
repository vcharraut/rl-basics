import torch
import gym
import numpy
from rlgym.algorithm.reinforce import REINFORCEContinuous, REINFORCEDiscrete
from rlgym.algorithm.ppo import PPOContinuous, PPODiscrete
from rlgym.algorithm.a2c import A2CDiscrete, A2CContinuous


class Policy:
    """
    Interface class between any Gym environment and any type of
    algortihms implemented.
    """

    def __init__(self, type_algorithm: str, obversation_space: int,
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
        args = [
            obversation_space, action_space, learning_rate, list_layer,
            is_shared_network
        ]

        action_space_type = type(action_space).__name__.lower()
        self.is_continuous = True if action_space_type == "box" else False

        if type_algorithm == "reinforce":
            self.algorithm = REINFORCEContinuous(
                *args[:-1]) if self.is_continuous else REINFORCEDiscrete(
                    *args[:-1])

        elif type_algorithm == "a2c":
            self.algorithm = A2CContinuous(
                *args) if self.is_continuous else A2CDiscrete(*args)

        elif type_algorithm == "a3c":
            raise NotImplementedError()

        elif type_algorithm == "ppo":
            self.algorithm = PPOContinuous(
                *args) if self.is_continuous else PPODiscrete(*args)

        elif type_algorithm == "dqn":
            raise NotImplementedError()

        elif type_algorithm == "ddpg":
            raise NotImplementedError()

        elif type_algorithm == "td3":
            raise NotImplementedError()

        elif type_algorithm == "sac":
            raise NotImplementedError()

    def get_action(self, state: numpy.ndarray) -> tuple:
        """
        Get the optimal action based on the current state.

        Args:
            state: current state of the environment.

        Returns:
            action chosed by the model.
        """

        state_torch = torch.from_numpy(state).float().to(torch.device("cuda"))
        return self.algorithm.act(state_torch)

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
