import torch
import numpy as np
from torch.nn.functional import softmax, mse_loss
from torch.distributions import Categorical, Normal
from rl_gym.algorithm.base import Base
from rl_gym.neuralnet import ActorCriticNet


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
        flags = minibatch["flags"]

        discounted_rewards = self._discounted_rewards(rewards, flags)
        values = self._model.critic(states).squeeze()

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

    def __init__(self, obversation_space: tuple, action_space: tuple,
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

        super(A2CDiscrete, self).__init__(obversation_space, action_space)

        self._model = ActorCriticNet(obversation_space,
                                     action_space,
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

        actor_value = self._model.actor_discrete(state)

        probs = softmax(actor_value, dim=-1)
        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.cpu().numpy(), log_prob


class A2CContinuous(A2C):
    """
    A2C implementation for continuous environments.
    """

    def __init__(self, obversation_space: tuple, action_space: tuple,
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

        super(A2CContinuous, self).__init__(obversation_space, action_space)

        # self.bound_interval = torch.Tensor(action_space.high).cuda()

        self._model = ActorCriticNet(obversation_space,
                                     action_space,
                                     learning_rate,
                                     list_layer,
                                     is_shared_network,
                                     is_continuous=True)
        self._model.cuda()

    def act(self, state: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
        """
        _summary_

        Args:
            state: _description_

        Returns:
            _description_
        """

        with torch.no_grad():
            actor_value = self._model.actor_continuous(state)

        mean = actor_value[0]
        variance = actor_value[1]
        dist = Normal(mean, variance)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)

        return action.cpu().numpy(), log_prob
