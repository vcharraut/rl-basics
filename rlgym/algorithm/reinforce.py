import torch
from torch.distributions import Categorical, Normal
from torch.nn.functional import softmax
from rlgym.algorithm.base import Base
from rlgym.neuralnet import LinearNet_Discrete, LinearNet_Continuous


class REINFORCE:

    def __init__(self):
        super(REINFORCE, self).__init__()

        self.model = None
        self.gamma = 0.99

    def _discounted_rewards(self, rewards):
        discounted_rewards = torch.zeros(rewards.size()).to(
            torch.device("cuda"))
        Gt = 0

        for i in range(rewards.size(0) - 1, -1, -1):
            Gt = rewards[i] * self.gamma + Gt
            discounted_rewards[i] = Gt

        discounted_rewards = (discounted_rewards - discounted_rewards.mean()
                              ) / (discounted_rewards.std() + 1e-9)

        return discounted_rewards

    def update_policy(self, minibatch):
        rewards = minibatch["rewards"]
        log_probs = minibatch["logprobs"]

        discounted_rewards = self._discounted_rewards(rewards)

        loss = (-log_probs * discounted_rewards).mean()

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()


class REINFORCE_Discrete(REINFORCE):

    def __init__(self, num_inputs, action_space, learning_rate, hidden_size,
                 number_of_layers, shared_layers):
        super(REINFORCE_Discrete, self).__init__()

        num_actions = action_space.n

        self.model = LinearNet_Discrete(num_inputs, num_actions, learning_rate,
                                        hidden_size, number_of_layers)
        self.model.cuda()

    def act(self, state):
        state_torch = torch.from_numpy(state).float().to(torch.device("cuda"))

        actor_value = self.model(state_torch)

        probs = softmax(actor_value, dim=0)
        dist = Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)

        return action.item(), logprob


class REINFORCE_Continuous(REINFORCE):

    def __init__(self, num_inputs, action_space, learning_rate, hidden_size,
                 number_of_layers, shared_layers):
        super(REINFORCE_Continuous, self).__init__()

        self.bound_interval = torch.Tensor(action_space.high).cuda()

        self.model = LinearNet_Continuous(num_inputs, action_space,
                                          learning_rate, hidden_size,
                                          number_of_layers)
        self.model.cuda()

    def act(self, state):
        state_torch = torch.from_numpy(state).float().to(torch.device("cuda"))

        actor_value = self.model(state_torch)

        mu = torch.tanh(actor_value[0]) * self.bound_interval
        sigma = torch.sigmoid(actor_value[1])
        dist = Normal(mu, sigma)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        return action.cpu().numpy(), log_prob
