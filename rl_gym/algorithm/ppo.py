from abc import abstractmethod
import numpy as np
import torch
from torch.nn.functional import softmax, mse_loss
from torch.distributions import Categorical, Normal
from torch.nn.utils import clip_grad_norm_
from rl_gym.algorithm.base import Base
from rl_gym.neuralnet import ActorCriticNet


class PPO(Base):

    def __init__(self, args, obversation_space: tuple, action_space: tuple,
                 writer):

        super(PPO, self).__init__(args, obversation_space, action_space,
                                  writer)

        self.__n_optim = 10
        self.__eps_clip = 0.2
        self.__value_constant = 0.5
        self.__entropy_constant = 0.01
        self._lmbda = 0.95

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

    def _gae(self, states: torch.Tensor, next_states: torch.Tensor,
             rewards: torch.Tensor,
             flags: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            value_next_states = self._model.critic(next_states).squeeze(-1)
            value_states = self._model.critic(states).squeeze(-1)

        returns = rewards + self._gamma * value_next_states * (1. - flags)
        delta = returns - value_states

        advantages = torch.zeros(rewards.size()).to(torch.device("cuda"))
        adv = torch.zeros(rewards.size(1)).to(torch.device("cuda"))

        for i in reversed(range(rewards.size(0))):
            adv = self._gamma * self._lmbda * adv * (1. - flags[i]) + delta[i]
            advantages[i] = adv

        advantages = self._normalize_tensor(advantages)

        td_target = advantages + value_states

        return td_target, advantages

    def update_policy(self, minibatch: dict):

        returns, advantages = self._gae(minibatch["states"],
                                        minibatch["next_states"],
                                        minibatch["rewards"],
                                        minibatch["flags"])

        states, actions, old_log_probs, returns, advantages = self.shuffle_batch(
            minibatch, returns, advantages)

        b_inds = np.arange(2048)

        for _ in range(self.__n_optim):
            for start in range(0, 128, 2048):
                end = start + 128
                mb_inds = b_inds[start:end]

                log_probs, dist_entropy, state_values = self._evaluate(
                    states[mb_inds], actions[mb_inds])

                ratios = torch.exp(log_probs[mb_inds] - old_log_probs[mb_inds])

                surr1 = -advantages[mb_inds] * ratios

                surr2 = -advantages[mb_inds] * torch.clamp(
                    ratios, 1. - self.__eps_clip, 1. + self.__eps_clip)

                policy_loss = torch.max(surr1, surr2).mean()

                value_loss = self.__value_constant * mse_loss(
                    state_values, returns[mb_inds])

                entropy_bonus = self.__entropy_constant * dist_entropy.mean()

                loss = policy_loss + value_loss - entropy_bonus

                self._model.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self._model.parameters(), 0.5)
                self._model.optimizer.step()

        self.writer.add_scalar("update/policy_loss", policy_loss,
                               self.global_step)
        self.writer.add_scalar("update/value_loss", value_loss,
                               self.global_step)
        self.writer.add_scalar("update/loss", loss, self.global_step)


class PPODiscrete(PPO):

    def __init__(self, args, obversation_space: tuple, action_space: tuple,
                 writer):

        super(PPODiscrete, self).__init__(args, obversation_space, (), writer)

        self._model = ActorCriticNet(obversation_space,
                                     action_space,
                                     args.learning_rate,
                                     args.layers,
                                     args.shared_network,
                                     is_continuous=False)

        self._model.cuda()

    def _evaluate(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        action_values = self._model.actor_discrete(states)
        state_values = self._model.critic(states).squeeze()

        action_prob = softmax(action_values, dim=-1)
        action_dist = Categorical(action_prob)

        log_prob = action_dist.log_prob(actions)
        dist_entropy = action_dist.entropy()

        return log_prob, dist_entropy, state_values

    def act(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:

        with torch.no_grad():
            actor_value = self._model.actor_discrete(state)

        probs = softmax(actor_value, dim=-1)
        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.increment_step()

        return action.cpu().numpy(), log_prob


class PPOContinuous(PPO):

    def __init__(self, args, obversation_space: tuple, action_space: tuple,
                 writer):

        super(PPOContinuous, self).__init__(args, obversation_space,
                                            action_space, writer)

        # self.bound_interval = torch.Tensor(action_space.high).cuda()

        self._model = ActorCriticNet(obversation_space,
                                     action_space,
                                     args.learning_rate,
                                     args.layers,
                                     args.shared_network,
                                     is_continuous=True)

        self._model.cuda()

    def _evaluate(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        action_values = self._model.actor_continuous(states)
        state_values = self._model.critic(states).squeeze()

        mean = action_values[0]
        variance = action_values[1]
        dist = Normal(mean, variance)

        log_prob = dist.log_prob(actions).sum(-1)
        dist_entropy = dist.entropy().sum(-1)

        return log_prob, dist_entropy, state_values

    def act(self, state: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:

        with torch.no_grad():
            actor_value = self._model.actor_continuous(state)

        mean = actor_value[0]
        variance = actor_value[1]
        dist = Normal(mean, variance)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)

        self.writer.add_scalar("rollout/mean", mean.mean(), self.global_step)
        self.writer.add_scalar("rollout/variance", variance.mean(),
                               self.global_step)

        self.increment_step()

        return action.cpu().numpy(), log_prob
