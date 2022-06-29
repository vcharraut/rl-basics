
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
        x = relu(self.linear1(state))

        list_mu_sigma = []

        for operation in self.action_layers:
            output = operation(x)
            list_mu_sigma.append(output.squeeze())

        return list_mu_sigma