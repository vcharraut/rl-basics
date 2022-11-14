import torch
from abc import ABC, abstractmethod


class Base(ABC):
    """
    Base class for any reinforcement learning algorithms.
    """

    def __init__(self):
        """
        Constructs Base class.
        """

        self._model = None
        self.__gamma = 0.99
        self.__lmbda = 0.95

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
            tensor: _description_
            eps: _description_. Defaults to 1e-9.

        Returns:
            _description_
        """

        return (tensor - tensor.mean()) / (tensor.std() + eps)

    def _discounted_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Computes the rewards with a discount factor.

        Args:
            rewards: rewards collected during an episode

        Returns:
            rewards with a discount factor
        """

        discounted_rewards = torch.zeros(rewards.size()).to(
            torch.device("cuda"))

        gain = 0
        for i in range(rewards.size(0) - 1, -1, -1):
            gain = rewards[i] * self.__gamma + gain
            discounted_rewards[i] = gain

        discounted_rewards = self._normalize_tensor(discounted_rewards)

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

        with torch.no_grad():
            value_next_states = self._model.critic(next_states).squeeze()
            value_states = self._model.critic(states).squeeze()

        td_target = rewards + self.__gamma * value_next_states * (1.0 - flags)
        delta = td_target - value_states

        advantages = torch.zeros(rewards.size()).to(torch.device("cuda"))
        adv = 0

        for i in range(delta.size(0) - 1, -1, -1):
            adv = self.__gamma * self.__lmbda * adv + delta[i]
            advantages[i] = adv

        td_target = self._normalize_tensor(td_target)
        advantages = self._normalize_tensor(advantages)

        return td_target, advantages

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
