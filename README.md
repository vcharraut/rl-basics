# rl-gym-zoo

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

The goal of this repository is to implement popular RL algorithms and try to compare them in the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) framework.  
Made in Python with [Pytorch](https://github.com/pytorch/pytorch).
Every Gymansium environments are supported, including MuJoCo and Atari.

| RL Algorithm                                           | Pytorch                                                                                                                                                                                                                                                                                           | Flax |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| [DQN](https://arxiv.org/abs/1312.5602)                 | [dqn.py](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/dqn/dqn.py) - [dqn_atari.py](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/dqn/dqn_atari.py)                                                                                                             |      |
| [A2C](https://arxiv.org/abs/1602.01783)                | [a2c.py](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/a2c/a2c.py) - [a2c_continuous.py](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/a2c/a2c_continuous.py) - [a2c_atari.py](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/a2c/a2c_atari.py)                                                                                                   |      |
| [PPO](https://arxiv.org/abs/1707.06347)                | [ppo.py](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/ppo/ppo.py) - [ppo_continuous.py](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/ppo/ppo_continuous.py) - [ppo_atari.py](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/ppo/ppo_atari.py) |      |
| [DDPG](https://proceedings.mlr.press/v32/silver14.pdf) | [ddpg.py](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/ddpg/ddpg.py)                                                                                                                                                                                                             |      |
| [TD3](https://arxiv.org/abs/1802.09477)                | [td3.py](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/td3/td3.py)                                                                                                                                                                                                               |      |
| [SAC](https://arxiv.org/abs/1801.01290)                |                                                                                                                                                                                                                                                                                                   |      |

## Prerequisites

- poetry >= 1.3.0
- python >= 3.10

- numpy >= 1.24.0
- gymnasium >= 0.27.0
- torch >= 1.13.0

## Installation

Clone the code repo and install the requirements.

```shell
git clone https://github.com/valentin-cnt/rl-gym-zoo.git
cd rl-gym-zoo

poetry install
```

## Usage

```shell
poetry shell

# DQN
python src/dqn/dqn.py
python src/dqn/dqn_atari.py

# PPO
python src/a2c/a2c.py
python src/a2c/a2c_continuous.py

# PPO
python src/ppo/ppo.py
python src/ppo/ppo_continuous.py
python src/ppo/ppo_atari.py

# DDPG
python src/ddpg/ddpg.py

# TD3
python src/td3/td3.py
```

To view logs on Tensorboard:

```shell
poetry run tensorboard --logdir=runs
```

## Results

## Acknowledgments

- [CleanRL](https://github.com/vwxyzjn/cleanrl): Some parts of my implementations come directly from the repository, so if you look for a complete and detailed code, you should check it first!
