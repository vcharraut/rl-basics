import torch
from rl_gym.utils.normalization import normalize


class Base:

    def __init__(self):
        """_summary_
        """

        self._model = None
        self.__gamma = 0.99
        self.__lmbda = 0.95

    def _discounted_rewards(self, rewards):
        """_summary_

        Args:
            rewards (_type_): _description_

        Returns:
            _type_: _description_
        """

        discounted_rewards = torch.zeros(rewards.size()).to(
            torch.device("cuda"))

        gain = 0
        for i in range(rewards.size(0) - 1, -1, -1):
            gain = rewards[i] * self.__gamma + gain
            discounted_rewards[i] = gain

        discounted_rewards = normalize(discounted_rewards)

        return discounted_rewards

    def _gae(self, state, next_state, reward, flags):
        """_summary_

        Args:
            state (_type_): _description_
            next_state (_type_): _description_
            reward (_type_): _description_
            flags (_type_): _description_

        Returns:
            _type_: _description_
        """

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
        """Save the model parameter to a pt file.

        Args:
            path (str): path for the pt file.
        """

        torch.save(self._model.state_dict(), path)

    def load_model(self, path):
        """Load a model parameter from a pt file.

        Args:
            path (str): path for the pt file.
        """

        self._model.load_state_dict(torch.load(path))
