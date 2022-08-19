from rlgym.algorithm.reinforce import REINFORCE_Continuous, REINFORCE_Discrete
from rlgym.algorithm.ppo import PPO_Continuous, PPO_Discrete
from rlgym.algorithm.a2c import A2C_Discrete, A2C_Continuous


class Policy:

    def __init__(self, algorithm, num_inputs, action_space, is_continuous,
                 learning_rate, hidden_size, number_of_layers, shared_layers):

        algorithm = algorithm.lower()
        args = [
            num_inputs, action_space, learning_rate, hidden_size,
            number_of_layers, shared_layers
        ]

        if algorithm == "reinforce":
            self.policy = REINFORCE_Continuous(
                *args) if is_continuous else REINFORCE_Discrete(*args)

        elif algorithm == "a2c":
            self.policy = A2C_Continuous(
                *args) if is_continuous else A2C_Discrete(*args)

        elif algorithm == "a3c":
            print("Not implemented")

        elif algorithm == "ppo":
            self.policy = PPO_Continuous(
                *args) if is_continuous else PPO_Discrete(*args)

        elif algorithm == "dqn":
            print("Not implemented")

        elif algorithm == "ddpg":
            print("Not implemented")

        elif algorithm == "td3":
            print("Not implemented")

        elif algorithm == "sac":
            print("Not implemented")

    def get_action(self, state):
        return self.policy.act(state)

    def update_policy(self, minibatch):
        self.policy.update_policy(minibatch)

    def save(self, path):
        self.policy.save_model(path)

    def load(self, path):
        self.policy.load_model(path)
