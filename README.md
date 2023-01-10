# rl-gym-zoo

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


The goal of this repository is to implement popular RL algorithms and try to compare them in the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) framework.  
Made in Python (3.10) with [Pytorch](https://github.com/pytorch/pytorch).
Every Gymansium environments are supported, including MuJoCo and Atari.

| RL Algorithm                                           | Pytorch  |    Flax    |
|--------------------------------------------------------|----------|------------|
| [DQN](https://arxiv.org/abs/1312.5602)                 |[dqn.py](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/pytorch/dqn.py) - [dqn_atari.py](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/pytorch/dqn_atari.py)||
| [A2C](https://arxiv.org/abs/1602.01783)                |[a2c.py](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/pytorch/a2c.py) - [a2c_continuous.py](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/pytorch/a2c_continuous.py)||
| [PPO](https://arxiv.org/abs/1707.06347)                |[ppo.py](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/pytorch/ppo.py) - [ppo_continuous.py](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/pytorch/ppo_continuous.py) - [ppo_atari.py](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/pytorch/ppo_atari.py)||
| [DDPG](https://proceedings.mlr.press/v32/silver14.pdf) |[ddpg.py](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/pytorch/ddpg.py)||
| [TD3](https://arxiv.org/abs/1802.09477)                |[td3.py](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/pytorch/td3.py)||
| [SAC](https://arxiv.org/abs/1801.01290)                |         ||

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

