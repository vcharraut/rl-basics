
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax, relu

class ActorCriticNet(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate):
        super(ActorCriticNet, self).__init__()

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
