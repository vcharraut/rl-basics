# DQN - Deep Q-Network

DQN (Deep Q-Network) is a reinforcement learning algorithm used for solving problems in game playing, robotics, and autonomous control. It is a value-based method that uses deep neural networks to represent the Q-function, which maps states and actions to the expected future reward.

In DQN, the agent collects experiences through interactions with its environment and uses these experiences to update its Q-function. The Q-function is updated using a variant of Q-learning, which is a temporal difference (TD) learning algorithm. The TD error, which measures the difference between the estimated and true Q-values, is used to update the Q-function.

One of the key innovations of DQN is the use of a replay buffer to store experiences and sample a mini-batch of experiences to update the Q-function. This allows the algorithm to break the correlation between consecutive experiences and improves the stability of learning.

DQN is known for its sample efficiency and stability, making it a popular choice for solving complex and challenging problems in reinforcement learning. Additionally, it is relatively simple to implement and is able to handle large and high-dimensional state spaces.

## Videos

<details>
  <summary>LunarLander-v2</summary>
  <img src="https://github.com/valentin-cnt/rl-gym-zoo/blob/master/media/gif/lunar-lander-dqn.gif?raw=true" alt="lunar-lander-dqn">
</details>
