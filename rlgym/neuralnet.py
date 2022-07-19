import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax


class LinearNet_Discrete(nn.Module):

    def __init__(self, num_inputs, num_actions, learning_rate, hidden_size,
                 number_of_layers):
        super(LinearNet_Discrete, self).__init__()

        self.num_actions = num_actions

        self.nn = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), *[
                nn.Linear(hidden_size, hidden_size)
                for _ in range(number_of_layers - 1)
            ])

        self.actor_layer = nn.Linear(hidden_size, num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        output = self.nn(state)
        return self.actor_layer(output)


class LinearNet_Continuous(nn.Module):

    def __init__(self, num_inputs, action_space, learning_rate, hidden_size,
                 number_of_layers):
        super(LinearNet_Continuous, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), *[
                nn.Linear(hidden_size, hidden_size)
                for _ in range(number_of_layers - 1)
            ])

        self.actor_layer = nn.Linear(hidden_size, 2)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        output = self.nn(state)
        return self.actor_layer(output)


class ActorCriticNet_Discrete(nn.Module):

    def __init__(self, num_inputs, num_actions, learning_rate, hidden_size,
                 number_of_layers):
        super(ActorCriticNet_Discrete, self).__init__()

        self.num_actions = num_actions

        self.nn = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), *[
                nn.Linear(hidden_size, hidden_size)
                for _ in range(number_of_layers - 1)
            ])

        self.actor_layer = nn.Linear(hidden_size, num_actions)
        self.critic_layer = nn.Linear(hidden_size, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def actor_critic(self, state):
        output = self.nn(state)
        return self.actor_layer(output), self.critic_layer(output)

    def actor(self, state):
        output = self.nn(state)
        return self.actor_layer(output)

    def critic(self, state):
        output = self.nn(state)
        return self.critic_layer(output)


class ActorCriticNet_Continuous(nn.Module):

    def __init__(self, num_inputs, action_space, learning_rate, hidden_size,
                 number_of_layers):
        super(ActorCriticNet_Continuous, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), *[
                nn.Linear(hidden_size, hidden_size)
                for _ in range(number_of_layers - 1)
            ])

        self.actor_layer = nn.Linear(hidden_size, 2)

        self.critic_layer = nn.Linear(hidden_size, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def actor_critic(self, state):
        output = self.nn(state)
        return self.actor_layer(output), self.critic_layer(output)

    def actor(self, state):
        output = self.nn(state)
        return self.actor_layer(output)

    def critic(self, state):
        output = self.nn(state)
        return self.critic_layer(output)
