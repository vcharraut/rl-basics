import torch
import numpy as np
from torch.nn.functional import softmax, mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical, Normal
from rl_gym.algorithm.base import Base
from rl_gym.neuralnet import ActorCriticNet


class A2C(Base):

    def _shuffle_batch(self, minibatch, returns):

        states = minibatch["states"].reshape((-1, ) + self._obversation_space)
        old_log_probs = minibatch["log_probs"].reshape(-1)
        returns = returns.reshape(-1)

        batch_size = len(minibatch["rewards"])
        batch_index = np.arange(batch_size)
        np.random.shuffle(batch_index)
        batch_index = batch_index[:2048]

        return states[batch_index], old_log_probs[batch_index], returns[
            batch_index]

    def _get_returns(self, rewards: torch.Tensor,
                     flags: torch.Tensor) -> torch.Tensor:

        returns = torch.zeros(rewards.size()).to(torch.device("cuda"))
        gain = torch.zeros(rewards.size(1)).to(torch.device("cuda"))

        for i in reversed(range(returns.size(0))):
            gain = rewards[i] + gain * self._gamma * (1. - flags[i])
            returns[i] = gain

        return self._normalize_tensor(returns)

    def update_policy(self, minibatch: dict):

        torch.autograd.set_detect_anomaly(True)

        states, log_probs, returns = self._shuffle_batch(
            minibatch,
            self._get_returns(minibatch["rewards"], minibatch["flags"]))

        values = self._model.critic(states).squeeze(-1)

        advantages = (returns - values)

        policy_loss = (-log_probs * advantages).mean()
        value_loss = mse_loss(values, returns)

        loss = (policy_loss + value_loss)

        self._model.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        clip_grad_norm_(self._model.parameters(), 0.5)
        self._model.optimizer.step()

        self.writer.add_scalar("update/policy_loss", policy_loss,
                               self.global_step)
        self.writer.add_scalar("update/value_loss", value_loss,
                               self.global_step)
        self.writer.add_scalar("update/loss", loss, self.global_step)


class A2CDiscrete(A2C):

    def __init__(self, args, obversation_space: tuple, action_space: tuple,
                 writer):
        super(A2CDiscrete, self).__init__(args, obversation_space, (), writer)

        self._model = ActorCriticNet(obversation_space,
                                     action_space,
                                     args.learning_rate,
                                     args.layers,
                                     args.shared_network,
                                     is_continuous=False)
        self._model.cuda()

    def act(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:

        actor_value = self._model.actor_discrete(state)

        probs = softmax(actor_value, dim=-1)
        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.increment_step()

        return action.cpu().numpy(), log_prob


class A2CContinuous(A2C):

    def __init__(self, args, obversation_space: tuple, action_space: tuple,
                 writer):
        super(A2CContinuous, self).__init__(args, obversation_space,
                                            action_space, writer)

        # self.bound_interval = torch.Tensor(action_space.high).cuda()

        self._model = ActorCriticNet(obversation_space,
                                     action_space,
                                     args.learning_rate,
                                     args.layers,
                                     args.shared_network,
                                     is_continuous=True)
        self._model.cuda()

    def act(self, state: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:

        actor_value = self._model.actor_continuous(state)

        mean = actor_value[0]
        variance = actor_value[1]
        dist = Normal(mean, variance)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)

        self.increment_step()
        self.writer.add_scalar("rollout/mean", mean.mean(), self.global_step)
        self.writer.add_scalar("rollout/variance", variance.mean(),
                               self.global_step)

        return action.cpu().numpy(), log_prob
