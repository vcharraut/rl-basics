import torch
from rlgym.utils.normalization import normalize


class Base:

    def __init__(self):
        self._model = None
        self.__gamma = 0.99

    def _discounted_rewards(self, rewards):
        discounted_rewards = torch.zeros(rewards.size()).to(
            torch.device("cuda"))

        gain = 0
        for i in range(rewards.size(0) - 1, -1, -1):
            gain = rewards[i] * self.__gamma + gain
            discounted_rewards[i] = gain

        discounted_rewards = normalize(discounted_rewards)

        return discounted_rewards

    def _gae(self, state, next_state, reward, flags):
        with torch.no_grad():
            value_next_state = self._model.critic(next_state).squeeze()
            value_state = self._model.critic(state).squeeze()

        td_target = reward + self.__gamma * value_next_state * (1. - flags)
        delta = td_target - value_state

        advantages = torch.zeros(reward.size()).to(torch.device("cuda"))
        adv = 0

        for i in range(delta.size(0) - 1, -1, -1):
            adv = self.__gamma * self.__lmbda * adv + delta[i]
            advantages[i] = adv

        td_target = normalize(td_target)
        advantages = normalize(advantages)

        return td_target, advantages

    def act(self, state):
        pass

    def update_policy(self, minibatch):
        pass

    def save_model(self, path):
        torch.save(self._model.state_dict(), path)

    def load_model(self, path):
        self._model.load_state_dict(torch.load(path))
