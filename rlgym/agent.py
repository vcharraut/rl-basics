from rlgym.algorithm.reinforce import REINFORCE_Continuous, REINFORCE_Discrete
from rlgym.algorithm.ppo import PPO_Continuous, PPO_Discrete
from rlgym.algorithm.a2c import A2C_Discrete, A2C_Continuous



class Agent:
    def __init__(self, algorithm, num_inputs, action_space, action_space_type, hidden_size, learning_rate):
        self.action_space = action_space

        algorithm = algorithm.lower()
        action_space_type = action_space_type.lower()

        self.continuous = False

        if algorithm == "reinforce":
            if action_space_type == "discrete":
                self.policy = REINFORCE_Discrete(
                    num_inputs, action_space, hidden_size, learning_rate)
            elif action_space_type == "box":
                self.continuous = True
                self.policy = REINFORCE_Continuous(
                    num_inputs, action_space, hidden_size, learning_rate)

        elif algorithm == "a2c":
            if action_space_type == "discrete":
                self.policy = A2C_Discrete(
                    num_inputs, action_space, hidden_size, learning_rate)
            elif action_space_type == "box":
                self.continuous = True
                self.policy = A2C_Continuous(
                    num_inputs, action_space, hidden_size, learning_rate)


        elif algorithm == "a3c":
            print("Not implemented")

        elif algorithm == "ppo":
            if action_space_type == "discrete":
                self.policy = PPO_Discrete(
                    num_inputs, action_space, hidden_size, learning_rate)
            elif action_space_type == "box":
                self.continuous = True
                self.policy = PPO_Continuous(
                    num_inputs, action_space, hidden_size, learning_rate)

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
