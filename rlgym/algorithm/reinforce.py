import torch
from torch.distributions import Categorical, Normal
from torch.nn.functional import softmax
from rlgym.utils.normalization import normalize
from rlgym.algorithm.base import Base
from rlgym.neuralnet import LinearNet_Discrete, LinearNet_Continuous


class REINFORCE(Base):

    def __init__(self):
        super(REINFORCE, self).__init__()

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
        rewards = minibatch["rewards"]
        log_probs = minibatch["logprobs"]

        discounted_rewards = self._discounted_rewards(rewards)

        loss = (-log_probs * discounted_rewards).mean()

        self._model.optimizer.zero_grad()
        loss.backward()
        self._model.optimizer.step()


class REINFORCE_Discrete(REINFORCE):

    def __init__(self, num_inputs, action_space, learning_rate, list_layer, is_shared_network):
        super(REINFORCE_Discrete, self).__init__()

        num_actionss = action_space.n

        self._model = LinearNet_Discrete(num_inputs, num_actionss,
                                         learning_rate, list_layer)
        self._model.cuda()

    def act(self, state):
        actor_value = self._model(state)

        probs = softmax(actor_value, dim=0)
        dist = Categorical(probs)

        action = dist.sample()
        logprob = dist.log_prob(action)

        return action.item(), logprob


class REINFORCE_Continuous(REINFORCE):

    def __init__(self, num_inputs, action_space, learning_rate, list_layer, is_shared_network):
        super(REINFORCE_Continuous, self).__init__()

        self.bound_interval = torch.Tensor(action_space.high).cuda()

        self._model = LinearNet_Continuous(num_inputs, action_space,
                                           learning_rate, list_layer)
        self._model.cuda()

    def act(self, state):
        actor_value = self._model(state)

        mu = torch.tanh(actor_value[0]) * self.bound_interval
        sigma = torch.sigmoid(actor_value[1])
        dist = Normal(mu, sigma)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        return action.cpu().numpy(), log_prob
