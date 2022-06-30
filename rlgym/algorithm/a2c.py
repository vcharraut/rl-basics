import torch
from torch.distributions import Categorical, Normal
from rlgym.neuralnet import ActorCriticNet_Discrete, ActorCriticNet_Continuous


class A2C_Base:
    def __init__(self):
        self.model = None
        self.gamma = 0.99
        self.lmbda = 0.95

    def gae(self, state, next_state, reward, flags):
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

    def discounted_rewards(self, rewards):
        discounted_rewards = torch.zeros(
            rewards.size()).to(torch.device("cuda"))
        Gt = 0

        for i in range(rewards.size(0) - 1, -1, -1):
            Gt = rewards[i] * self.gamma + Gt
            discounted_rewards[i] = Gt

        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-9
        )

        return discounted_rewards

    def update_policy(self, minibatch):
        states = minibatch["states"]
        rewards = minibatch["rewards"]
        log_probs = minibatch["logprobs"]

        discounted_rewards = self.discounted_rewards(rewards)

        values = self.model.critic(states)

        advantages = values - discounted_rewards

        policy_loss = (-log_probs * advantages)
        value_loss = (advantages**2)

        loss = (policy_loss + value_loss).mean()

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


class A2C_Discrete(A2C_Base):
    def __init__(self, num_inputs, action_space, hidden_size, learning_rate):
        super(A2C_Discrete, self).__init__()

        num_actions = action_space.n

        self.model = ActorCriticNet_Discrete(
            num_inputs, num_actions, hidden_size, learning_rate)
        self.model.cuda()

    def act(self, state):
        # with torch.no_grad():
        state = torch.from_numpy(state).float().unsqueeze(
            0).to(torch.device("cuda"))
        probs = self.model.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action.item(), logprob


class A2C_Continuous(A2C_Base):
    def __init__(self, num_inputs, action_space, hidden_size, learning_rate):
        super(A2C_Base, self).__init__()

        self.model = ActorCriticNet_Continuous(num_inputs, action_space, hidden_size, learning_rate)
        self.model.cuda()

    def act(self, state):
        state_torch = torch.from_numpy(
            state).float().unsqueeze(0).to(torch.device("cuda"))

        log_prob = 0
        list_action = []

        list_probs = self.model.forward(state_torch)

        for probs in list_probs:
            mu = torch.tanh(probs[0])
            sigma = torch.sigmoid(probs[1])
            dist = Normal(mu, sigma)
            action = dist.sample()
            log_prob += dist.log_prob(action)
            list_action.append(action.item())

        return list_action, log_prob
