from abc import abstractmethod
import numpy as np
import torch
from torch.nn.functional import softmax, mse_loss
from torch.distributions import Categorical, Normal
from rl_gym.algorithm.base import Base
from rl_gym.neuralnet import ActorCriticNet


class PPO(Base):
    """
    _summary_
    """

    def __init__(self, obversation_space: tuple, action_space: tuple):
        """
        _summary_
        """

        super(PPO, self).__init__(obversation_space, action_space)

        self.__n_optim = 1
        self.__eps_clip = 0.2
        self.__value_constant = 0.5
        self.__entropy_constant = 0.01

    @abstractmethod
    def _evaluate(self, states, actions):
        pass

    def shuffle_batch(self, minibatch, returns, advantages):
        states = minibatch["states"].reshape((-1, ) + self._obversation_space)
        actions = minibatch["actions"].reshape((-1, ) + self._action_space)
        old_log_probs = minibatch["log_probs"].reshape(-1)
        returns = returns.reshape(-1)
        advantages = advantages.reshape(-1)

        batch_size = len(minibatch["rewards"])
        batch_index = np.arange(batch_size)
        np.random.shuffle(batch_index)
        batch_index = batch_index[:2048]

        return states[batch_index], actions[batch_index], old_log_probs[
            batch_index], returns[batch_index], advantages[batch_index]

    def update_policy(self, minibatch: dict):
        """
        Computes new gradients based from an episode and
        update the neural network.

        Args:
            minibatch (dict): dict containing information from the episode,
            keys are (states, actions, next_states, rewards, flags, log_probs)
        """

        returns, advantages = self._gae(minibatch["states"],
                                        minibatch["next_states"],
                                        minibatch["rewards"],
                                        minibatch["flags"])

        states, actions, old_log_probs, returns, advantages = self.shuffle_batch(
            minibatch, returns, advantages)

        for _ in range(self.__n_optim):
            log_probs, dist_entropy, state_values = self._evaluate(
                states, actions)

            ratios = torch.exp(log_probs - old_log_probs)

            surr1 = ratios * advantages

            surr2 = torch.clamp(ratios, 1. - self.__eps_clip,
                                1. + self.__eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = self.__value_constant * mse_loss(
                state_values, returns)

            entropy_bonus = self.__entropy_constant * dist_entropy.mean()

            loss = policy_loss + value_loss - entropy_bonus

            self._model.optimizer.zero_grad()
            loss.backward()
            self._model.optimizer.step()

        self.writer.add_scalar("Update/policy_loss", policy_loss,
                               self.global_step)
        self.writer.add_scalar("Update/value_loss", value_loss,
                               self.global_step)
        self.writer.add_scalar("Update/loss", loss, self.global_step)


class PPODiscrete(PPO):
    """
    _summary_

    Args:
        PPO: _description_
    """

    def __init__(self, obversation_space: tuple, action_space: tuple,
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

        super(PPODiscrete, self).__init__(obversation_space, ())

        self._model = ActorCriticNet(obversation_space,
                                     action_space,
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

        action_values = self._model.actor_discrete(states)
        state_values = self._model.critic(states).squeeze()

        action_prob = softmax(action_values, dim=-1)
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
            actor_value = self._model.actor_discrete(state)

        probs = softmax(actor_value, dim=-1)
        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.increment_step()

        return action.cpu().numpy(), log_prob


class PPOContinuous(PPO):
    """
    _summary_

    Args:
        PPO: _description_
    """

    def __init__(self, obversation_space: tuple, action_space: tuple,
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

        super(PPOContinuous, self).__init__(obversation_space, action_space)

        # self.bound_interval = torch.Tensor(action_space.high).cuda()

        self._model = ActorCriticNet(obversation_space,
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

        action_values = self._model.actor_continuous(states)
        state_values = self._model.critic(states).squeeze()

        mean = action_values[0]
        variance = action_values[1]
        dist = Normal(mean, variance)

        log_prob = dist.log_prob(actions).sum(-1)
        dist_entropy = dist.entropy().sum(-1)

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
            actor_value = self._model.actor_continuous(state)

        mean = actor_value[0]
        variance = actor_value[1]
        dist = Normal(mean, variance)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)

        self.writer.add_scalar("Rollout/Mean", mean.mean(), self.global_step)
        self.writer.add_scalar("Rollout/Variance", variance.mean(),
                               self.global_step)

        self.increment_step()

        return action.cpu().numpy(), log_prob
