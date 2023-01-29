# TD3 - Twin Delayed DDPG

TD3 (Twin Delayed Deep Deterministic Policy Gradients) is a reinforcement learning algorithm used to solve problems in fields such as game playing, robotics, and autonomous control. It is a model-free, off-policy algorithm that builds on the Deep Deterministic Policy Gradients (DDPG) algorithm.

TD3 uses two separate critic networks to estimate the expected reward and to reduce over-estimation bias in the value function. The two critic networks are updated at different frequencies, with one network being updated less frequently than the other.

In TD3, the agent uses two deep neural networks: an actor network and two critic networks. The actor network maps states to actions, while the critic networks map states and actions to the expected reward. The agent collects experiences through interactions with its environment and uses these experiences to update both the actor and critic networks.

The actor network is updated using policy gradients, which are derived from the gradient of the expected reward with respect to the actions. The critic networks are updated using the TD error, which measures the difference between the estimated and true expected rewards.

TD3 is known for its stability, sample efficiency, and low over-estimation bias, making it a popular choice for solving complex and challenging problems in reinforcement learning. Additionally, it is relatively simple to implement and is able to handle large and high-dimensional action spaces.
