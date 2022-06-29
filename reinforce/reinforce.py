import torch
from torch.distributions import Categorical, Normal
from neuralnet import LinearNet_Discrete, LinearNet_Continuous


class REINFORCE_Base:
    def __init__(self):
        self.model = None
        self.gamma = 0.99
        self.log_probs = []

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

    def learn(self, minibatch):
        rewards = minibatch["rewards"]
        log_probs = minibatch["logprobs"]

        discounted_rewards = self.discounted_rewards(rewards)

        loss = (-log_probs * discounted_rewards).mean()

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


class REINFORCE_Discrete(REINFORCE_Base):
    def __init__(self, num_inputs, action_space, hidden_size, learning_rate):
        super(REINFORCE_Discrete, self).__init__()

        num_actions = action_space.n

        self.model = LinearNet_Discrete(
            num_inputs, num_actions, hidden_size, learning_rate)
        self.model.cuda()

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(
            0).to(torch.device("cuda"))
        probs = self.model.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action.item(), logprob


class REINFORCE_Continuous(REINFORCE_Base):
    def __init__(self, num_inputs, action_space, hidden_size, learning_rate):
        super(REINFORCE_Continuous, self).__init__()

        print(action_space)
        self.lows = action_space.low

        self.model = LinearNet_Continuous(
            num_inputs, action_space, hidden_size, learning_rate)
        self.model.cuda()

    def get_action(self, state):
        state_c = state.copy()
        state_torch = torch.from_numpy(
            state_c).float().unsqueeze(0).to(torch.device("cuda"))

        log_prob = 0
        list_action = []

        list_probs = self.model.forward(state_torch)

        for i, probs in enumerate(list_probs):
            if self.lows[i] == -1:
                mu = torch.tanh(probs[0])
            elif self.lows[i] == 0:
                mu = torch.sigmoid(probs[0])
            else:
                print("action_space.low is not accepted: ", i)
            sigma = torch.sigmoid(probs[1])
            dist = Normal(mu, sigma)
            action = dist.sample()
            log_prob += dist.log_prob(action)
            list_action.append(action.item())

        self.log_probs.append(log_prob)

        return list_action
