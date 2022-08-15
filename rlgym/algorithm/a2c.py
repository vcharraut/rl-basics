import torch
from torch.nn.functional import softmax, mse_loss
from torch.distributions import Categorical, Normal
from rlgym.neuralnet import ActorCriticNet_Continuous, ActorCriticNet_Discrete


class A2C_Base:

    def __init__(self):
        self.model = None
        self.gamma = 0.9

    def discounted_rewards(self, rewards):
        discounted_rewards = torch.zeros(rewards.size()).to(
            torch.device("cuda"))
        Gt = 0

        for i in range(rewards.size(0) - 1, -1, -1):
            Gt = rewards[i] * self.gamma + Gt
            discounted_rewards[i] = Gt

        discounted_rewards = (discounted_rewards - discounted_rewards.mean()
                              ) / (discounted_rewards.std() + 1e-7)

        return discounted_rewards

    def update_policy(self, minibatch):
        states = minibatch["states"]
        rewards = minibatch["rewards"]
        log_probs = minibatch["logprobs"]

        discounted_rewards = self.discounted_rewards(rewards)

        values = self.model.critic(states).squeeze()

        advantages = (discounted_rewards - values).detach()

        policy_loss = (-log_probs * advantages).mean()
        value_loss = mse_loss(values, discounted_rewards)

        loss = (policy_loss + value_loss)

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


class A2C_Discrete(A2C_Base):

    def __init__(self, num_inputs, action_space, learning_rate, hidden_size,
                 number_of_layers, is_dual):
        super(A2C_Discrete, self).__init__()

        num_actions = action_space.n

        self.model = ActorCriticNet_Discrete(num_inputs, num_actions,
                                             learning_rate, hidden_size,
                                             number_of_layers, is_dual)
        self.model.cuda()

    def act(self, state):
        state_torch = torch.from_numpy(state).float().to(torch.device("cuda"))
        actor_value = self.model.actor(state_torch)
        probs = softmax(actor_value, dim=0)
        dist = Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action.item(), logprob


class A2C_Continuous(A2C_Base):

    def __init__(self, num_inputs, action_space, learning_rate, hidden_size,
                 number_of_layers):
        super(A2C_Continuous, self).__init__()

        self.model = ActorCriticNet_Continuous(num_inputs, action_space,
                                               learning_rate, hidden_size,
                                               number_of_layers)
        self.model.cuda()

    def act(self, state):
        state_torch = torch.from_numpy(state).float().to(torch.device("cuda"))

        actor_value = self.model.actor(state_torch)
        mu = torch.tanh(actor_value[0]) * 2.0
        sigma = torch.sigmoid(actor_value[1])
        dist = Normal(mu, sigma)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob
