from abc import ABC, abstractmethod
import torch
from torch.utils.tensorboard import SummaryWriter


class Base(ABC):
    """
    Base class for any reinforcement learning algorithms.
    """

    @abstractmethod
    def __init__(self, obversation_space: tuple, action_space: tuple):
        """
        Constructs Base class.
        """

        self._model = None
        self._obversation_space = obversation_space
        self._action_space = action_space
        self.__gamma = 0.99
        self.__lmbda = 0.95

        self.writer = SummaryWriter()
        self._global_step = 0

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def update_policy(self, minibatch):
        pass

    def _normalize_tensor(self,
                          tensor: torch.Tensor,
                          eps=1e-9) -> torch.Tensor:
        """
        _summary_

        Args:
            tensor: tensor to normalize.
            eps: epsilon value to prevent zero division. Defaults to 1e-9.

        Returns:
            normalized tensor.
        """

        return (tensor - tensor.mean()) / (tensor.std() + eps)

    def _discounted_rewards(self, rewards: torch.Tensor,
                            flags: torch.Tensor) -> torch.Tensor:
        """
        Computes the rewards with a discount factor.

        Args:
            rewards: rewards collected during an episode

        Returns:
            rewards with a discount factor
        """

        # rewards = self._normalize_tensor(rewards)

        discounted_rewards = torch.zeros(rewards.size()).to(
            torch.device("cuda"))
        gain = torch.zeros(rewards.size(1)).to(torch.device("cuda"))

        for i in reversed(range(rewards.size(0))):
            gain = rewards[i] * self.__gamma + (gain * (1. - flags[i]))
            discounted_rewards[i] = gain

        return discounted_rewards

    def _gae(self, states: torch.Tensor, next_states: torch.Tensor,
             rewards: torch.Tensor,
             flags: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the rewards with the GAE method.

        Args:
            state: states collected during an episode
            next_state: next states collected during an episode
            reward: collected during an episode
            flags: flags collected during an episode

        Returns:
            td_target: values from the value function
            advantages: values from the advantage function
        """

        # rewards = self._normalize_tensor(rewards)

        with torch.no_grad():
            value_next_states = self._model.critic(next_states).squeeze()
            value_states = self._model.critic(states).squeeze()

        td_target = rewards + self.__gamma * value_next_states * (1. - flags)
        delta = td_target - value_states

        advantages = torch.zeros(rewards.size()).to(torch.device("cuda"))
        adv = torch.zeros(rewards.size(1)).to(torch.device("cuda"))

        for i in reversed(range(delta.size(0))):
            adv = self.__gamma * self.__lmbda * adv + delta[i]
            advantages[i] = adv

        return td_target, advantages

    def increment_step(self):
        self._global_step += 1

    @property
    def global_step(self):
        return self._global_step

    def get_obs_and_action_shape(self):
        return self._obversation_space, self._action_space

    def save_model(self, path: str):
        """
        Save the model parameter to a pt file.

        Args:
            path: path for the pt file.
        """

        torch.save(self._model.state_dict(), path)

    def load_model(self, path: str):
        """
        Load a model parameter from a pt file.

        Args:
            path: path for the pt file.
        """

        self._model.load_state_dict(torch.load(path))
