import numpy as np
import torch
from torch import optim, nn
from torch.distributions import Categorical, Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticNet(nn.Module):

    def __init__(self, obversation_space: tuple, action_space: tuple,
                 learning_rate: float, list_layer: list,
                 is_shared_network: bool, is_continuous: bool):

        super().__init__()

        current_layer_value = np.array(obversation_space).prod()
        num_actions = np.prod(action_space)

        if is_shared_network:
            base_neural_net = nn.Sequential()

            for layer_value in list_layer:
                base_neural_net.append(
                    layer_init(nn.Linear(current_layer_value, layer_value)))
                base_neural_net.append(nn.Tanh())

                current_layer_value = layer_value

            self.actor_neural_net = nn.Sequential(
                base_neural_net,
                layer_init(nn.Linear(list_layer[-1], num_actions), std=0.01))

            self.critic_neural_net = nn.Sequential(
                base_neural_net,
                layer_init(nn.Linear(list_layer[-1], 1), std=1.0))

        else:
            self.actor_neural_net = nn.Sequential()
            self.critic_neural_net = nn.Sequential()

            for layer_value in list_layer:
                self.actor_neural_net.append(
                    layer_init(nn.Linear(current_layer_value, layer_value)))
                self.actor_neural_net.append(nn.Tanh())

                self.critic_neural_net.append(
                    layer_init(nn.Linear(current_layer_value, layer_value)))
                self.critic_neural_net.append(nn.Tanh())

                current_layer_value = layer_value

            self.actor_neural_net.append(
                layer_init(nn.Linear(list_layer[-1], num_actions), std=0.01))

            self.critic_neural_net.append(
                layer_init(nn.Linear(list_layer[-1], 1), std=1.0))

        if is_continuous:
            self.actor_logstd = nn.Parameter(torch.zeros(1, num_actions))

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self):
        pass

    def forward_discrete(self, state: torch.Tensor) -> torch.Tensor:

        actor_value = self.actor_neural_net(state)
        critic_value = self.critic_neural_net(state)
        distribution = Categorical(logits=actor_value)

        return distribution, critic_value

    def forward_continuous(
            self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        action_mean = self.actor_neural_net(state)
        action_std = self.actor_logstd.expand_as(action_mean).exp()
        distribution = Normal(action_mean, action_std)

        return distribution, self.critic_neural_net(state)

    def critic(self, state: torch.Tensor) -> torch.Tensor:

        return self.critic_neural_net(state)


class CNNActorCritic(nn.Module):

    def __init__(self, action_space: tuple, learning_rate: float):

        super().__init__()

        num_actions = np.prod(action_space)

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor_neural_net = layer_init(nn.Linear(512, num_actions),
                                           std=0.01)
        self.critic_neural_net = layer_init(nn.Linear(512, 1), std=1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self):
        pass

    def forward_discrete(self, state):
        output = self.network(state)
        actor_value = self.actor_neural_net(output)
        critic_value = self.critic_neural_net(output)

        distribution = Categorical(logits=actor_value)

        return distribution, critic_value

    def critic(self, state):
        output = self.network(state)
        return self.critic_neural_net(output)
