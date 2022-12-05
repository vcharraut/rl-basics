import numpy as np
import torch
from torch import optim, nn


def layer_init(in_shape, out_shape, std=np.sqrt(2), bias_const=0.0):
    layer = nn.Linear(int(in_shape), int(out_shape))
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticNet(nn.Module):
    """
    Actor Critic neural network.
    """

    def __init__(self, obversation_space: tuple, action_space: tuple,
                 learning_rate: float, list_layer: list,
                 is_shared_network: bool, is_continuous: bool):
        """
        _summary_

        Args:
            obs_space: size of the observation
            action_space: _description_
            learning_rate: _description_
            list_layer: _description_
            is_shared_network: _description_
            is_continuous: _description_
        """

        super(ActorCriticNet, self).__init__()

        self.actor_neural_net = None
        self.critic_neural_net = None

        current_layer_value = np.array(obversation_space).prod()
        num_actions = np.prod(action_space)

        if is_shared_network:
            base_neural_net = nn.Sequential()

            for layer_value in list_layer:
                base_neural_net.append(
                    layer_init(current_layer_value, layer_value))
                base_neural_net.append(nn.Tanh())

                current_layer_value = layer_value

            self.actor_neural_net = nn.Sequential(
                base_neural_net,
                layer_init(list_layer[-1], num_actions, std=0.01))

            self.critic_neural_net = nn.Sequential(
                base_neural_net, layer_init(list_layer[-1], 1, std=1.0))

        else:
            self.actor_neural_net = nn.Sequential()
            self.critic_neural_net = nn.Sequential()

            for layer_value in list_layer:
                self.actor_neural_net.append(
                    layer_init(current_layer_value, layer_value))
                self.actor_neural_net.append(nn.Tanh())

                self.critic_neural_net.append(
                    layer_init(current_layer_value, layer_value))
                self.critic_neural_net.append(nn.Tanh())

                current_layer_value = layer_value

            self.actor_neural_net.append(
                layer_init(list_layer[-1], num_actions, std=0.01))

            self.critic_neural_net.append(
                layer_init(list_layer[-1], 1, std=1.0))

        if is_continuous:
            self.actor_logstd = nn.Parameter(torch.zeros(1, num_actions))

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self):
        pass

    def actor_discrete(self, state: torch.Tensor) -> torch.Tensor:
        """
        _summary_

        Args:
            state: _description_

        Returns:
            _description_
        """

        return self.actor_neural_net(state)

    def actor_continuous(self, state: torch.Tensor) -> torch.Tensor:
        """
        _summary_

        Args:
            state: _description_

        Returns:
            _description_
        """

        action_mean = self.actor_neural_net(state)
        action_std = self.actor_logstd.expand_as(action_mean).exp()
        return action_mean, action_std

    def critic(self, state: torch.Tensor) -> torch.Tensor:
        """
        _summary_

        Args:
            state: _description_

        Returns:
            _description_
        """

        return self.critic_neural_net(state)
