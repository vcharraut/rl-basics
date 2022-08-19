import torch.nn as nn
import torch.optim as optim


class Parallel(nn.Module):

    def __init__(self, mean_layer, sigma_layer):
        super().__init__()
        self.list_module = nn.ModuleList([mean_layer, sigma_layer])

    def forward(self, inputs):
        return [module(inputs) for module in self.list_module]


class LinearNet_Discrete(nn.Module):

    def __init__(self, num_inputs, num_actions, learning_rate, hidden_size,
                 number_of_layers):
        super(LinearNet_Discrete, self).__init__()

        self.num_actions = num_actions

        self.nn = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), *[
                nn.Linear(hidden_size, hidden_size)
                for _ in range(number_of_layers - 1)
            ], nn.Linear(hidden_size, num_actions))

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        return self.nn(state)


class LinearNet_Continuous(nn.Module):

    def __init__(self, num_inputs, action_space, learning_rate, hidden_size,
                 number_of_layers):
        super(LinearNet_Continuous, self).__init__()

        num_action = action_space.shape[0]

        mean_sigma_layer = Parallel(nn.Linear(hidden_size, num_action),
                                    nn.Linear(hidden_size, num_action))

        self.nn = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), *[
                nn.Linear(hidden_size, hidden_size)
                for _ in range(number_of_layers - 1)
            ], mean_sigma_layer)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        return self.nn(state)


class ActorCriticNet_Discrete(nn.Module):

    def __init__(self, num_inputs, num_actions, learning_rate, hidden_size,
                 number_of_layers, shared_layers):
        super(ActorCriticNet_Discrete, self).__init__()

        self.actor_nn = None
        self.critic_nn = None
        self.optimizer = None

        if shared_layers:
            base_nn = nn.Sequential(
                nn.Linear(num_inputs, hidden_size), *[
                    nn.Linear(hidden_size, hidden_size)
                    for _ in range(number_of_layers - 1)
                ])

            self.actor_nn = nn.Sequential(base_nn,
                                          nn.Linear(hidden_size, num_actions))

            self.critic_nn = nn.Sequential(base_nn, nn.Linear(hidden_size, 1))

            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        else:
            self.actor_nn = nn.Sequential(
                nn.Linear(num_inputs, hidden_size), *[
                    nn.Linear(hidden_size, hidden_size)
                    for _ in range(number_of_layers - 1)
                ], nn.Linear(hidden_size, num_actions))

            self.critic_nn = nn.Sequential(
                nn.Linear(num_inputs, hidden_size), *[
                    nn.Linear(hidden_size, hidden_size)
                    for _ in range(number_of_layers - 1)
                ], nn.Linear(hidden_size, 1))

            self.optimizer = optim.Adam([{
                'params': self.actor_nn.parameters(),
                'lr': learning_rate
            }, {
                'params': self.critic_nn.parameters(),
                'lr': 0.001
            }])

    def actor(self, state):
        return self.actor_nn(state)

    def critic(self, state):
        return self.critic_nn(state)


class ActorCriticNet_Continuous(nn.Module):

    def __init__(self, num_inputs, action_space, learning_rate, hidden_size,
                 number_of_layers, shared_layers):
        super(ActorCriticNet_Continuous, self).__init__()

        self.actor_nn = None
        self.critic_nn = None
        self.optimizer = None

        num_action = action_space.shape[0]

        mean_sigma_layer = Parallel(nn.Linear(hidden_size, num_action),
                                    nn.Linear(hidden_size, num_action))

        if shared_layers:
            base_nn = nn.Sequential(
                nn.Linear(num_inputs, hidden_size), *[
                    nn.Linear(hidden_size, hidden_size)
                    for _ in range(number_of_layers - 1)
                ])

            self.actor_nn = nn.Sequential(base_nn, mean_sigma_layer)

            self.critic_nn = nn.Sequential(base_nn, nn.Linear(hidden_size, 1))

            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        else:
            self.actor_nn = nn.Sequential(
                nn.Linear(num_inputs, hidden_size), *[
                    nn.Linear(hidden_size, hidden_size)
                    for _ in range(number_of_layers - 1)
                ], mean_sigma_layer)

            self.critic_nn = nn.Sequential(
                nn.Linear(num_inputs, hidden_size), *[
                    nn.Linear(hidden_size, hidden_size)
                    for _ in range(number_of_layers - 1)
                ], nn.Linear(hidden_size, 1))

            self.optimizer = optim.Adam([{
                'params': self.actor_nn.parameters(),
                'lr': learning_rate
            }, {
                'params': self.critic_nn.parameters(),
                'lr': 0.001
            }])

    def actor(self, state):
        return self.actor_nn(state)

    def critic(self, state):
        state = state.float()
        return self.critic_nn(state)
