import torch
import gym
import numpy
from torch.nn.functional import softmax, mse_loss
from torch.distributions import Categorical, Normal
from rlgym.algorithm.base import Base
from rlgym.neuralnet import ActorCriticNet


class A2C(Base):
    """
    Advantage Actor Critic implementation.
    """

    def update_policy(self, minibatch: dict):
        """
        Computes new gradients based from an episode and
        update the neural network.

        Args:
            minibatch (dict): dict containing information from the episode,
            keys are (states, actions, next_states, rewards, flags, log_probs)
        """

        states = minibatch["states"]
        rewards = minibatch["rewards"]
        log_probs = minibatch["log_probs"]

        discounted_rewards = self._discounted_rewards(rewards)

        values = self._model.critic(states).squeeze()

        with torch.no_grad():
            advantages = (discounted_rewards - values)

        policy_loss = (-log_probs * advantages).mean()
        value_loss = mse_loss(values, discounted_rewards)

        loss = (policy_loss + value_loss)

        self._model.optimizer.zero_grad()
        loss.backward()
        self._model.optimizer.step()


class A2CDiscrete(A2C):
    """
    A2C implementation for discrete environments.
    """

    def __init__(self, obs_space: int,
                 action_space: gym.spaces.discrete.Discrete,
                 learning_rate: float, list_layer: list,
                 is_shared_network: bool):
        """
        Constructs a A2C algorithm for discrete environments.

        Args:
            obs_space: size of the observation
            action_space: number of actions
            learning_rate: _description_
            list_layer: _description_
            is_shared_network: _description_
        """

        super(A2CDiscrete, self).__init__()

        num_actionss = action_space.n

        self._model = ActorCriticNet(obs_space,
                                     num_actionss,
                                     learning_rate,
                                     list_layer,
                                     is_shared_network,
                                     is_continuous=False)
        self._model.cuda()

    def act(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:
        """
        _summary_

        Args:
            state: _description_

        Returns:
            _description_
        """

        actor_value = self._model.actor(state)

        probs = softmax(actor_value, dim=0)
        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob


class A2CContinuous(A2C):
    """
    A2C implementation for continuous environments.
    """

    def __init__(self, obs_space: int, action_space: gym.spaces.box.Box,
                 learning_rate: float, list_layer: list,
                 is_shared_network: bool):
        """
        Constructs a A2C algorithm for continuous environments.

        Args:
            obs_space: size of the observation
            action_space: number of actions
            learning_rate: _description_
            list_layer: _description_
            is_shared_network: _description_
        """

        super(A2CContinuous, self).__init__()

        self.bound_interval = torch.Tensor(action_space.high).cuda()

        self._model = ActorCriticNet(obs_space,
                                     action_space,
                                     learning_rate,
                                     list_layer,
                                     is_shared_network,
                                     is_continuous=True)
        self._model.cuda()

    def act(self, state: torch.Tensor) -> tuple[numpy.ndarray, torch.Tensor]:
        """
        _summary_

        Args:
            state: _description_

        Returns:
            _description_
        """

        actor_value = self._model.actor(state)

        mean = torch.tanh(actor_value[0]) * self.bound_interval
        variance = torch.sigmoid(actor_value[1])
        dist = Normal(mean, variance)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        return action.cpu().numpy(), log_prob
