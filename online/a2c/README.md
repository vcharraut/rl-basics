# A2C - Advantage Actor Critic

A2C (Advantage Actor-Critic) is a reinforcement learning algorithm that combines the strengths of both the actor-critic and the value-based methods. It is a popular method used in deep reinforcement learning to solve problems such as game playing, robotics, and autonomous control.

In A2C, the agent uses the actor network to determine its actions and the critic network to estimate the state-value function. The actor network is trained to maximize the expected reward, while the critic network is trained to minimize the value error.

At each time step, the agent takes an action and receives a reward and new state observation. The critic network then updates its estimate of the state value based on the received reward and the new state observation. The actor network updates its policy based on the estimated value and the advantage function, which is the difference between the estimated value and the expected reward.

A2C has several advantages over other reinforcement learning methods, such as improved stability and sample efficiency. It also scales well to large and complex environments, making it a popular choice for solving challenging problems in reinforcement learning.

## Videos

<details>
  <summary>LunarLander-v2</summary>
  <img src="https://github.com/valentin-cnt/rl-gym-zoo/blob/master/media/gif/lunar-lander-a2c.gif?raw=true" alt="lunar-lander-ppo">
</details>
