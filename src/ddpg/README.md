# DDPG - Deep Deterministic Policy Gradient

DDPG (Deep Deterministic Policy Gradients) is a reinforcement learning algorithm used to solve problems in fields such as game playing, robotics, and autonomous control. It is a model-free, off-policy algorithm that combines the strengths of both value-based and policy-based methods.

In DDPG, the agent uses two deep neural networks: an actor network and a critic network. The actor network maps states to actions, while the critic network maps states and actions to the expected reward. The agent collects experiences through interactions with its environment and uses these experiences to update both networks.

The actor network is updated using policy gradients, which are derived from the gradient of the expected reward with respect to the actions. The critic network is updated using the TD error, which measures the difference between the estimated and true expected rewards.

DDPG is known for its stability and sample efficiency, making it a popular choice for solving complex and challenging problems in reinforcement learning. Additionally, it is relatively simple to implement and is able to handle large and high-dimensional action spaces.
