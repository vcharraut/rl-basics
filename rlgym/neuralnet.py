import torch.nn as nn
import torch.optim as optim


class Parallel(nn.Module):

    def __init__(self, mean_layer, sigma_layer):
        super().__init__()
        self.list_module = nn.ModuleList([mean_layer, sigma_layer])

    def forward(self, inputs):
        return [module(inputs) for module in self.list_module]


class LinearNet_Discrete(nn.Module):

    def __init__(self, num_inputs, num_actions, learning_rate, list_layer):
        super(LinearNet_Discrete, self).__init__()

        self.num_actionss = num_actions

        self.nn = nn.Sequential()

        current_layer_value = num_inputs

        for layer_value in list_layer:
            self.nn.append(nn.Linear(current_layer_value, layer_value))
            current_layer_value = layer_value

        self.nn.append(nn.Linear(list_layer[-1], num_actions))

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        return self.nn(state)


class LinearNet_Continuous(nn.Module):

    def __init__(self, num_inputs, action_space, learning_rate, list_layer):
        super(LinearNet_Continuous, self).__init__()

        num_actions = action_space.shape[0]

        mean_sigma_layer = Parallel(nn.Linear(list_layer[-1], num_actions),
                                    nn.Linear(list_layer[-1], num_actions))

        self.nn = nn.Sequential()

        current_layer_value = num_inputs

        for layer_value in list_layer:
            self.nn.append(nn.Linear(current_layer_value, layer_value))
            current_layer_value = layer_value

        self.nn.append(mean_sigma_layer)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        return self.nn(state)


class ActorCriticNet_Discrete(nn.Module):

    def __init__(self, num_inputs, num_actions, learning_rate, list_layer,
                 is_shared_network):
        super(ActorCriticNet_Discrete, self).__init__()

        self.actor_nn = None
        self.critic_nn = None
        self.optimizer = None

        current_layer_value = num_inputs

        if is_shared_network:
            base_nn = nn.Sequential()

            for layer_value in list_layer:
                base_nn.append(nn.Linear(current_layer_value, layer_value))
                current_layer_value = layer_value

            self.actor_nn = nn.Sequential(
                base_nn, nn.Linear(list_layer[-1], num_actions))

            self.critic_nn = nn.Sequential(base_nn,
                                           nn.Linear(list_layer[-1], 1))

            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        else:
            self.actor_nn = nn.Sequential()
            self.critic_nn = nn.Sequential()

            for layer_value in list_layer:
                self.actor_nn.append(
                    nn.Linear(current_layer_value, layer_value))
                self.critic_nn.append(
                    nn.Linear(current_layer_value, layer_value))
                current_layer_value = layer_value

            self.actor_nn.append(nn.Linear(list_layer[-1], num_actions))
            self.critic_nn.append(nn.Linear(list_layer[-1], 1))

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

    def __init__(self, num_inputs, action_space, learning_rate, list_layer,
                 is_shared_network):
        super(ActorCriticNet_Continuous, self).__init__()

        self.actor_nn = None
        self.critic_nn = None
        self.optimizer = None

        num_actionss = action_space.shape[0]
        current_layer_value = num_inputs

        mean_sigma_layer = Parallel(nn.Linear(list_layer[-1], num_actionss),
                                    nn.Linear(list_layer[-1], num_actionss))

        if is_shared_network:
            base_nn = nn.Sequential()

            for layer_value in list_layer:
                base_nn.append(nn.Linear(current_layer_value, layer_value))
                current_layer_value = layer_value

            self.actor_nn = nn.Sequential(base_nn, mean_sigma_layer)

            self.critic_nn = nn.Sequential(base_nn,
                                           nn.Linear(list_layer[-1], 1))

            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        else:
            self.actor_nn = nn.Sequential()
            self.critic_nn = nn.Sequential()

            for layer_value in list_layer:
                self.actor_nn.append(
                    nn.Linear(current_layer_value, layer_value))
                self.critic_nn.append(
                    nn.Linear(current_layer_value, layer_value))
                current_layer_value = layer_value

            self.actor_nn.append(mean_sigma_layer)
            self.critic_nn.append(nn.Linear(list_layer[-1], 1))
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
