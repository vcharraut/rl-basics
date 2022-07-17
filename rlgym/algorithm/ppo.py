import torch
from torch.distributions import Categorical, Normal
from rlgym.neuralnet import ActorCriticNet_Discrete, ActorCriticNet_Continuous


class PPO_Base:

    def __init__(self):
        self.model = None
        self.n_optim = 1
        self.gamma = 0.99
        self.lmbda = 0.95
        self.eps_clip = 0.2
        self.value_constant = 0.5
        self.entropy_constant = 0.01

    def evaluate(self, states, actions):
        pass

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

    def update_policy(self, minibatch):
        states = minibatch["states"]
        actions = minibatch["actions"]
        next_states = minibatch["next_states"]
        rewards = minibatch["rewards"]
        flags = minibatch["flags"]
        old_logprobs = minibatch["logprobs"]

        returns, advantages = self.gae(states, next_states, rewards, flags)

        for _ in range(self.n_optim):
            logprobs, dist_entropy, state_values = self.evaluate(
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

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


class PPO_Discrete(PPO_Base):

    def __init__(self, num_inputs, action_space, learning_rate, hidden_size,
                 number_of_layers):
        super(PPO_Discrete, self).__init__()

        num_actions = action_space.n

        self.model = ActorCriticNet_Discrete(num_inputs, num_actions,
                                             learning_rate, hidden_size,
                                             number_of_layers)
        self.model.cuda()

    def evaluate(self, states, actions):
        action_prob, state_values = self.model.actor_critic(states)

        action_dist = Categorical(action_prob)

        log_prob = action_dist.log_prob(actions)

        dist_entropy = action_dist.entropy()

        return log_prob, dist_entropy, state_values

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(
            torch.device("cuda"))
        probs = self.model.actor(state).detach()
        dist = Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action.item(), logprob


class PPO_Continuous(PPO_Base):

    def __init__(self, num_inputs, action_space, learning_rate, hidden_size,
                 number_of_layers):
        super(PPO_Continuous, self).__init__()

        self.model = ActorCriticNet_Continuous(num_inputs, action_space,
                                               learning_rate, hidden_size,
                                               number_of_layers)
        self.model.cuda()

    def evaluate(self, states, actions):
        actor_value, state_values = self.model.actor_critic(states)

        mu = torch.tanh(actor_value[:, 0])
        sigma = torch.sigmoid(actor_value[:, 1])
        dist = Normal(mu, sigma)
        log_prob = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        return log_prob, dist_entropy, state_values

    def act(self, state):
        state_torch = torch.from_numpy(state).float().unsqueeze(0).to(
            torch.device("cuda"))

        actor_value = self.model.actor(state_torch)

        mu = torch.tanh(actor_value[:, 0])
        sigma = torch.sigmoid(actor_value[:, 1])
        dist = Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob
