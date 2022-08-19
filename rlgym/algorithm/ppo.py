import torch
from torch.nn.functional import softmax
from torch.distributions import Categorical, Normal
from rlgym.algorithm.base import Base
from rlgym.neuralnet import ActorCriticNet_Continuous, ActorCriticNet_Discrete


class PPO:

    def __init__(self):
        super(PPO, self).__init__()

        self.model = None
        self.n_optim = 3
        self.gamma = 0.99
        self.lmbda = 0.95
        self.eps_clip = 0.2
        self.value_constant = 0.5
        self.entropy_constant = 0.01

    def _evaluate(self, states, actions):
        pass

    def _gae(self, state, next_state, reward, flags):
        reward = (reward - reward.mean()) / (reward.std() + 1e-9)

        with torch.no_grad():
            value_next_state = self.model.critic(next_state).squeeze()
            value_state = self.model.critic(state).squeeze()

        td_target = reward + self.gamma * value_next_state * (1. - flags)
        delta = td_target - value_state

        advantages = torch.zeros(reward.size()).to(torch.device("cuda"))
        adv = 0

        for i in range(delta.size(0) - 1, -1, -1):
            adv = self.gamma * self.lmbda * adv + delta[i]
            advantages[i] = adv

        return td_target, advantages

    def update_policy(self, minibatch):
        states = minibatch["states"]
        actions = minibatch["actions"]
        next_states = minibatch["next_states"]
        rewards = minibatch["rewards"]
        flags = minibatch["flags"]
        old_logprobs = minibatch["logprobs"]

        returns, advantages = self._gae(states, next_states, rewards, flags)

        for _ in range(self.n_optim):
            logprobs, dist_entropy, state_values = self._evaluate(
                states, actions)

            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages

            surr2 = torch.clamp(ratios, 1. - self.eps_clip,
                                1. + self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2)

            value_loss = self.value_constant * ((state_values - returns)**2)

            entropy_bonus = self.entropy_constant * dist_entropy

            loss = (policy_loss + value_loss - entropy_bonus).mean()

            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()


class PPO_Discrete(PPO):

    def __init__(self, num_inputs, action_space, learning_rate, hidden_size,
                 number_of_layers, shared_layers):
        super(PPO_Discrete, self).__init__()

        num_actions = action_space.n

        self.model = ActorCriticNet_Discrete(num_inputs, num_actions,
                                             learning_rate, hidden_size,
                                             number_of_layers, shared_layers)
        self.model.cuda()

    def _evaluate(self, states, actions):
        action_values = self.model.actor(states)
        state_values = self.model.critic(states).squeeze()

        action_prob = softmax(action_values, dim=1)

        action_dist = Categorical(action_prob)

        log_prob = action_dist.log_prob(actions)

        dist_entropy = action_dist.entropy()

        return log_prob, dist_entropy, state_values

    def act(self, state):
        state_torch = torch.from_numpy(state).float().to(torch.device("cuda"))

        with torch.no_grad():
            actor_value = self.model.actor(state_torch)

        probs = softmax(actor_value, dim=0)
        dist = Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)

        return action.item(), logprob


class PPO_Continuous(PPO):

    def __init__(self, num_inputs, action_space, learning_rate, hidden_size,
                 number_of_layers, shared_layers):
        super(PPO_Continuous, self).__init__()

        self.bound_interval = torch.Tensor(action_space.high).cuda()

        self.model = ActorCriticNet_Continuous(num_inputs, action_space,
                                               learning_rate, hidden_size,
                                               number_of_layers, shared_layers)
        self.model.cuda()

    def _evaluate(self, states, actions):
        action_values = self.model.actor(states)
        state_values = self.model.critic(states).squeeze()

        mu = torch.tanh(action_values[0])
        sigma = torch.sigmoid(action_values[1])
        dist = Normal(mu, sigma)
        log_prob = dist.log_prob(actions).sum(dim=1)
        dist_entropy = dist.entropy().sum(dim=1)

        return log_prob, dist_entropy, state_values

    def act(self, state):
        state_torch = torch.from_numpy(state).float().to(torch.device("cuda"))

        with torch.no_grad():
            actor_value = self.model.actor(state_torch)

        mu = torch.tanh(actor_value[0]) * self.bound_interval
        sigma = torch.sigmoid(actor_value[1])
        dist = Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        return action.cpu().numpy(), log_prob
