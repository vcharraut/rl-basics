
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax, relu


class LinearNet_Discrete(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate):
        super(LinearNet_Discrete, self).__init__()

        self.num_actions = num_actions
        self.nn = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Linear(hidden_size, num_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = softmax(self.nn(state), dim=1)
        return x


class LinearNet_Continuous(nn.Module):
    def __init__(self, num_inputs, action_space, hidden_size, learning_rate):
        super(LinearNet_Continuous, self).__init__()

        num_action_layers = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.action_layers = nn.ModuleList(
            [nn.Linear(hidden_size, 2) for _ in range(num_action_layers)])

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = self.linear1(state)

        list_mu_sigma = []

        for operation in self.action_layers:
            output = operation(x)
            list_mu_sigma.append(output.squeeze())

        return list_mu_sigma


class ActorCriticNet_Discrete(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate):
        super(ActorCriticNet_Discrete, self).__init__()

        self.num_actions = num_actions
        self.nn = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )

        self.actor_layer = nn.Linear(hidden_size, num_actions)
        self.critic_layer = nn.Linear(hidden_size, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def actor_critic(self, state):
        output = self.nn(state)

        actor_value = softmax(self.actor_layer(output), dim=1)
        critic_value = self.critic_layer(output)
        return actor_value, critic_value

    def actor(self, state):
        output = self.nn(state)

        return softmax(self.actor_layer(output), dim=1)

    def critic(self, state):
        output = self.nn(state)

        return self.critic_layer(output)


class ActorCriticNet_Continuous(nn.Module):
    def __init__(self, num_inputs, action_space, hidden_size, learning_rate):
        super(ActorCriticNet_Continuous, self).__init__()

        num_action_layers = action_space.shape[0]

        self.nn = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )

        self.actor_layers = nn.ModuleList(
            [nn.Linear(hidden_size, 2) for _ in range(num_action_layers)])

        self.critic_layer = nn.Linear(hidden_size, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def actor_critic(self, state):
        output = self.nn(state)

        list_mu_sigma = []

        for operation in self.action_layers:
            x = operation(output)
            list_mu_sigma.append(x.squeeze())

        critic_value = self.critic_layer(output)
        return list_mu_sigma, critic_value

    def actor(self, state):
        output = self.nn(state)

        list_mu_sigma = []

        for operation in self.action_layers:
            x = operation(output)
            list_mu_sigma.append(x.squeeze())

        return list_mu_sigma

    def critic(self, state):
        output = self.nn(state)

        return self.critic_layer(output)
