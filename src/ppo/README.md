# PPO - Proximal Policy Optimization

PPO (Proximal Policy Optimization) is a reinforcement learning algorithm used to solve problems in fields such as game playing, robotics, and autonomous control. It is a widely used deep reinforcement learning method that combines the strengths of both value-based and policy-based methods.

PPO updates the policy by optimizing a surrogate objective, which is a locally approximated version of the policy optimization problem. The algorithm takes small steps towards optimizing the policy, which makes it more stable and easier to train than other methods, such as trust region policy optimization (TRPO).

In PPO, the agent collects experiences through interactions with its environment, and uses these experiences to update its policy. The policy is updated by optimizing a combination of the expected reward and a term that measures the difference between the new and old policies, called the "KL-divergence." The optimization process is repeated until the policy converges to an optimal solution.

PPO is known for its sample efficiency and stability, making it a popular choice for solving complex and challenging problems in reinforcement learning. Additionally, it is relatively simple to implement and is able to handle large and high-dimensional action spaces.

## Videos

<details>
  <summary>LunarLander-v2</summary>
  <img src="https://github.com/valentin-cnt/rl-gym-zoo/blob/master/media/gif/lunar-lander-ppo.gif?raw=true" alt="lunar-lander-ppo">
</details>
