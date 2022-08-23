import torch
from rlgym.algorithm.reinforce import REINFORCE_Continuous, REINFORCE_Discrete
from rlgym.algorithm.ppo import PPO_Continuous, PPO_Discrete
from rlgym.algorithm.a2c import A2C_Discrete, A2C_Continuous


class Policy:

    def __init__(self, algorithm, obversation_space, action_space,
                 learning_rate, list_layer, is_shared_network):
        """Interface class between any Gym environment and any type of algortihms implemented.

        Args:
            algorithm (str): algorithm for the learning method.
            obversation_space (int): dimension of the observation.
            action_space (gym.spaces.box.Box): box object of the action space.
            learning_rate (float): learning rate to use during learning.
            list_layer (list):
            is_shared_network (bool): boolean to chose between shared or seperated network.
        """

        algorithm = algorithm.lower()
        args = [
            obversation_space, action_space, learning_rate, list_layer,
            is_shared_network
        ]

        action_space_type = type(action_space).__name__.lower()
        self.is_continuous = True if action_space_type == "box" else False

        if algorithm == "reinforce":
            self.policy = REINFORCE_Continuous(
                *args) if self.is_continuous else REINFORCE_Discrete(*args)

        elif algorithm == "a2c":
            self.policy = A2C_Continuous(
                *args) if self.is_continuous else A2C_Discrete(*args)

        elif algorithm == "a3c":
            print("Not implemented")

        elif algorithm == "ppo":
            self.policy = PPO_Continuous(
                *args) if self.is_continuous else PPO_Discrete(*args)

        elif algorithm == "dqn":
            print("Not implemented")

        elif algorithm == "ddpg":
            print("Not implemented")

        elif algorithm == "td3":
            print("Not implemented")

        elif algorithm == "sac":
            print("Not implemented")

    def get_action(self, state):
        """Get the optimal action based on the current state.

        Args:
            state (numpy.ndarray): current state of the environment.

        Returns:
            tuple: action chosed by the model.
        """
        state_torch = torch.from_numpy(state).float().to(torch.device("cuda"))
        return self.policy.act(state_torch)

    def update_policy(self, minibatch):
        """Update the model based on the training data.

        Args:
            minibatch (dict): Minibatch of data from training.
        """
        self.policy.update_policy(minibatch)

    def save(self, path):
        """Save the model to disk.

        Args:
            path (str): path file to save the model.
        """
        self.policy.save_model(path)

    def load(self, path):
        """Load a model from disk.

        Args:
            path (str): path file to load the model.
        """
        self.policy.load_model(path)
