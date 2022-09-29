import torch.nn as nn
import torch.optim as optim


class Parallel(nn.Module):

    def __init__(self, mean_layer, sigma_layer):
        """_summary_

        Args:
            mean_layer (_type_): _description_
            sigma_layer (_type_): _description_
        """

        super().__init__()
        self.list_module = nn.ModuleList([mean_layer, sigma_layer])

    def forward(self, inputs):
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """

        return [module(inputs) for module in self.list_module]


class LinearNet(nn.Module):

    def __init__(self, num_inputs, action_space, learning_rate, list_layer,
                 is_continuous):
        """_summary_

        Args:
            num_inputs (_type_): _description_
            action_space (_type_): _description_
            learning_rate (_type_): _description_
            list_layer (_type_): _description_
            is_continuous (bool): _description_
        """

        super(LinearNet, self).__init__()

        num_actions = action_space.shape[0]

        if is_continuous:
            last_layer = Parallel(nn.Linear(list_layer[-1], num_actions),
                                  nn.Linear(list_layer[-1], num_actions))
        else:
            last_layer = nn.Linear(list_layer[-1], num_actions)

        self.neural_net = nn.Sequential()

        current_layer_value = num_inputs

        for layer_value in list_layer:
            self.neural_net.append(nn.Linear(current_layer_value, layer_value))
            current_layer_value = layer_value

        self.neural_net.append(last_layer)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        """_summary_

        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """

        return self.neural_net(state)


class ActorCriticNet(nn.Module):

    def __init__(self, num_inputs, action_space, learning_rate, list_layer,
                 is_shared_network, is_continuous):
        """_summary_

        Args:
            num_inputs (_type_): _description_
            action_space (_type_): _description_
            learning_rate (_type_): _description_
            list_layer (_type_): _description_
            is_shared_network (bool): _description_
            is_continuous (bool): _description_
        """

        super(ActorCriticNet, self).__init__()

        self.actor_neural_net = None
        self.critic_neural_net = None
        self.optimizer = None

        current_layer_value = num_inputs

        if is_continuous:
            num_actions = action_space.shape[0]
            last_layer = Parallel(nn.Linear(list_layer[-1], num_actions),
                                  nn.Linear(list_layer[-1], num_actions))
        else:
            num_actions = action_space
            last_layer = nn.Linear(list_layer[-1], num_actions)

        if is_shared_network:
            base_neural_net = nn.Sequential()

            for layer_value in list_layer:
                base_neural_net.append(
                    nn.Linear(current_layer_value, layer_value))
                current_layer_value = layer_value

            self.actor_neural_net = nn.Sequential(base_neural_net, last_layer)

            self.critic_neural_net = nn.Sequential(
                base_neural_net, nn.Linear(list_layer[-1], 1))

            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        else:
            self.actor_neural_net = nn.Sequential()
            self.critic_neural_net = nn.Sequential()

            for layer_value in list_layer:
                self.actor_neural_net.append(
                    nn.Linear(current_layer_value, layer_value))
                self.critic_neural_net.append(
                    nn.Linear(current_layer_value, layer_value))
                current_layer_value = layer_value

            self.actor_neural_net.append(last_layer)
            self.critic_neural_net.append(nn.Linear(list_layer[-1], 1))

            # self.optimizer = optim.Adam([{
            #     'params': self.actor_neural_net.parameters(),
            #     'lr': learning_rate
            # }, {
            #     'params': self.critic_neural_net.parameters(),
            #     'lr': 0.001
            # }])

            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def foward(self):
        raise NotImplementedError(self.__class__.__name__)

    def actor(self, state):
        """_summary_

        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """

        return self.actor_neural_net(state)

    def critic(self, state):
        """_summary_

        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """

        return self.critic_neural_net(state)
