import torch
from torch.nn.functional import softmax, mse_loss
from torch.distributions import Categorical, Normal
from rlgym.utils.normalization import normalize
from rlgym.algorithm.base import Base
from rlgym.neuralnet import ActorCriticNet_Continuous, ActorCriticNet_Discrete


class A2C(Base):

    def __init__(self):
        super(A2C, self).__init__()

        self.__gamma = 0.99

    def _discounted_rewards(self, rewards):
        discounted_rewards = torch.zeros(rewards.size()).to(
            torch.device("cuda"))
        Gt = 0

        for i in range(rewards.size(0) - 1, -1, -1):
            Gt = rewards[i] * self.__gamma + Gt
            discounted_rewards[i] = Gt

        discounted_rewards = normalize(discounted_rewards)

        return discounted_rewards

    def update_policy(self, minibatch):
        states = minibatch["states"]
        rewards = minibatch["rewards"]
        log_probs = minibatch["logprobs"]

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


class A2C_Discrete(A2C):

    def __init__(self, num_inputs, action_space, learning_rate, list_layer, is_shared_network):
        super(A2C_Discrete, self).__init__()

        num_actionss = action_space.n

        self._model = ActorCriticNet_Discrete(num_inputs, num_actionss,
                                              learning_rate, list_layer,
                                              is_shared_network)
        self._model.cuda()

    def act(self, state):
        actor_value = self._model.actor(state)

        probs = softmax(actor_value, dim=0)
        dist = Categorical(probs)

        action = dist.sample()
        logprob = dist.log_prob(action)

        return action.item(), logprob


class A2C_Continuous(A2C):

    def __init__(self, num_inputs, action_space, learning_rate, list_layer, is_shared_network):
        super(A2C_Continuous, self).__init__()

        self.bound_interval = torch.Tensor(action_space.high).cuda()

        self._model = ActorCriticNet_Continuous(num_inputs, action_space,
                                                learning_rate, list_layer,
                                                is_shared_network)
        self._model.cuda()

    def act(self, state):
        actor_value = self._model.actor(state)

        mu = torch.tanh(actor_value[0]) * self.bound_interval
        sigma = torch.sigmoid(actor_value[1])
        dist = Normal(mu, sigma)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        return action.cpu().numpy(), log_prob
