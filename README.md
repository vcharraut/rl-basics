# RL-Gym

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

The goal of this repository is to implement any RL algorithms and try to benchmarks them in the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) framework.  
Made in Python (3.10) with [Pytorch](https://github.com/pytorch/pytorch).
Every Gymansium environments are supported, including MuJoCo and Atari.

| RL Algorithm            | Discrete | Continuous |
|-------------------------|:--------:|:----------:|
| DQN [[1]](#references)  |     X    |            |
| A2C [[2]](#references)  |     X    |      X     |
| PPO [[3]](#references)  |     X    |      X     |
| DDPG [[4]](#references) |          |      X     |
| TD3 [[5]](#references)  |          |      X     |
| SAC [[6]](#references)  |          |            |

---

## Setup

Clone the code repo and install the requirements.

```
git clone https://github.com/valentin-cnt/rl-gym.git
cd rl-gym

poetry install
```

---

## Run

---

## Results

---

## Acknowledgments

 - [CleanRL](https://github.com/vwxyzjn/cleanrl)

## References

- [1] [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [2] [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
- [3] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [4] [Deterministic Policy Gradient Algorithms](https://proceedings.mlr.press/v32/silver14.pdf)
- [5] [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
- [6] [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
