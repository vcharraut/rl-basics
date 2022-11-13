import torch
import gym
import numpy
from torch.distributions import Categorical, Normal
from torch.nn.functional import softmax
from rlgym.algorithm.base import Base
from rlgym.neuralnet import LinearNet


class REINFORCE(Base):
    """
    Base class for the implementation of the REINFORCE algorithm.

    Args:
        Base (Class): parent class
    """

    def update_policy(self, minibatch: dict):
        """
        Computes new gradients based from an episode and
        update the neural network.

        Args:
            minibatch (dict): dict containing information from the episode,
            keys are (states, actions, next_states, rewards, flags, log_probs)
        """

        rewards = minibatch["rewards"]
        log_probs = minibatch["log_probs"]

        discounted_rewards = self._discounted_rewards(rewards)

        loss = (-log_probs * discounted_rewards).mean()

        self._model.optimizer.zero_grad()
        loss.backward()
        self._model.optimizer.step()


class REINFORCEDiscrete(REINFORCE):
    """
    REINFORCE implementation to use in a discrete environment.

    Args:
        REINFORCE -- parent class
    """

    def __init__(self, obs_space: int,
                 action_space: gym.spaces.discrete.Discrete,
                 learning_rate: float, list_layer: list):
        """
        Constructs REINFORCEDiscrete class

        Args:
            obs_space: size of the observation
            action_space: number of actions
            learning_rate: learning rate's value
            list_layer: _description_
        """

        super(REINFORCEDiscrete, self).__init__()

        num_actions = action_space.n

        self._model = LinearNet(obs_space,
                                num_actions,
                                learning_rate,
                                list_layer,
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

        actor_value = self._model(state)

        probs = softmax(actor_value, dim=0)
        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob


class REINFORCEContinuous(REINFORCE):
    """
    _summary_

    Args:
        REINFORCE: _description_
    """

    def __init__(self, obs_space: int, action_space: gym.spaces.box.Box,
                 learning_rate: float, list_layer: list):
        """
        _summary_

        Args:
            obs_space: size of the observation
            action_space: _description_
            learning_rate: _description_
            list_layer: _description_
        """

        super(REINFORCEContinuous, self).__init__()

        self.bound_interval = torch.Tensor(action_space.high).cuda()

        self._model = LinearNet(obs_space,
                                action_space,
                                learning_rate,
                                list_layer,
                                is_continuous=True)
        self._model.cuda()

    def act(self, state: torch.Tensor) -> tuple[numpy.ndarray, torch.Tensor]:
        """
        _summary_

        Args:
            state: _description_

        Returns:
            _type_: _description_
        """

        actor_value = self._model(state)

        mean = torch.tanh(actor_value[0]) * self.bound_interval
        variance = torch.sigmoid(actor_value[1])
        dist = Normal(mean, variance)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        return action.cpu().numpy(), log_prob
