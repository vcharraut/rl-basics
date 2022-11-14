import torch
import gymnasium as gym
import numpy as np
from torch.nn.functional import softmax
from torch.distributions import Categorical, Normal
from rl_gym.algorithm.base import Base
from rl_gym.neuralnet import ActorCriticNet
from abc import abstractmethod


class PPO(Base):
    """
    _summary_
    """

    def __init__(self):
        """
        _summary_
        """

        super(PPO, self).__init__()

        self.__n_optim = 3
        self.__eps_clip = 0.2
        self.__value_constant = 0.5
        self.__entropy_constant = 0.01

    @abstractmethod
    def _evaluate(self, states, actions):
        pass

    def update_policy(self, minibatch: dict):
        """
        Computes new gradients based from an episode and
        update the neural network.

        Args:
            minibatch (dict): dict containing information from the episode,
            keys are (states, actions, next_states, rewards, flags, log_probs)
        """

        states = minibatch["states"]
        actions = minibatch["actions"]
        next_states = minibatch["next_states"]
        rewards = minibatch["rewards"]
        flags = minibatch["flags"]
        old_log_probs = minibatch["log_probs"]

        returns, advantages = self._gae(states, next_states, rewards, flags)

        for _ in range(self.__n_optim):
            log_probs, dist_entropy, state_values = self._evaluate(
                states, actions)

            ratios = torch.exp(log_probs - old_log_probs)

            surr1 = ratios * advantages

            surr2 = torch.clamp(ratios, 1. - self.__eps_clip,
                                1. + self.__eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2)

            value_loss = self.__value_constant * ((state_values - returns)**2)

            entropy_bonus = self.__entropy_constant * dist_entropy

            loss = (policy_loss + value_loss - entropy_bonus).mean()

            self._model.optimizer.zero_grad()
            loss.backward()
            self._model.optimizer.step()


class PPODiscrete(PPO):
    """
    _summary_

    Args:
        PPO: _description_
    """

    def __init__(self, obs_space: int,
                 action_space: gym.spaces.discrete.Discrete,
                 learning_rate: float, list_layer: list,
                 is_shared_network: bool):
        """
        _summary_

        Args:
            obs_space: _description_
            action_space: _description_
            learning_rate: _description_
            list_layer: _description_
            is_shared_network: _description_
        """

        super(PPODiscrete, self).__init__()

        num_actionss = action_space.n

        self._model = ActorCriticNet(obs_space,
                                     num_actionss,
                                     learning_rate,
                                     list_layer,
                                     is_shared_network,
                                     is_continuous=False)

        self._model.cuda()

    def _evaluate(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        _summary_

        Args:
            states: _description_
            actions: _description_

        Returns:
            _type_: _description_
        """

        action_values = self._model.actor(states)
        state_values = self._model.critic(states).squeeze()

        action_prob = softmax(action_values, dim=1)
        action_dist = Categorical(action_prob)

        log_prob = action_dist.log_prob(actions)
        dist_entropy = action_dist.entropy()

        return log_prob, dist_entropy, state_values

    def act(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:
        """
        _summary_

        Args:
            state: _description_

        Returns:
            _type_: _description_
        """

        with torch.no_grad():
            actor_value = self._model.actor(state)

        probs = softmax(actor_value, dim=0)
        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob


class PPOContinuous(PPO):
    """
    _summary_

    Args:
        PPO: _description_
    """

    def __init__(self, obs_space: int, action_space: gym.spaces.box.Box,
                 learning_rate: float, list_layer: list,
                 is_shared_network: bool):
        """
        _summary_

        Args:
            obs_space: _description_
            action_space: _description_
            learning_rate: _description_
            list_layer: _description_
            is_shared_network: _description_
        """

        super(PPOContinuous, self).__init__()

        self.bound_interval = torch.Tensor(action_space.high).cuda()

        self._model = ActorCriticNet(obs_space,
                                     action_space,
                                     learning_rate,
                                     list_layer,
                                     is_shared_network,
                                     is_continuous=True)
        self._model.cuda()

    def _evaluate(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        _summary_

        Args:
            states: _description_
            actions: _description_

        Returns:
            _type_: _description_
        """

        action_values = self._model.actor(states)
        state_values = self._model.critic(states).squeeze()

        mean = torch.tanh(action_values[0])
        variance = torch.sigmoid(action_values[1])
        dist = Normal(mean, variance)

        log_prob = dist.log_prob(actions).sum(dim=1)
        dist_entropy = dist.entropy().sum(dim=1)

        return log_prob, dist_entropy, state_values

    def act(self, state: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
        """
        _summary_

        Args:
            state: _description_

        Returns:
            _type_: _description_
        """

        with torch.no_grad():
            actor_value = self._model.actor(state)

        mean = torch.tanh(actor_value[0]) * self.bound_interval
        variance = torch.sigmoid(actor_value[1])
        dist = Normal(mean, variance)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        return action.cpu().numpy(), log_prob
