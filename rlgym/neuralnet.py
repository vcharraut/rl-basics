import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax, relu


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

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = softmax(self.nn(state), dim=1)
        return x


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

        actor_value = self.actor_layer(output)

        return actor_value


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

        actor_value = self.actor_layer(output)

        critic_value = self.critic_layer(output)

        return actor_value, critic_value

    def actor(self, state):
        output = self.nn(state)

        actor_value = self.actor_layer(output)


        # if mu_value != mu_value:
        #     print("- state: ", state)
        #     print("- output: ", output)

        return actor_value

    def critic(self, state):
        output = self.nn(state)

        return self.critic_layer(output)
