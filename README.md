# rl-basics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![CodeFactor](https://www.codefactor.io/repository/github/valentin-cnt/rl-basics/badge)](https://www.codefactor.io/repository/github/valentin-cnt/rl-basics)

This repository aims to implement various popular RL algorithms and evaluate their performance using the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) framework.

The goal is to provide a simple and clean implementation of the algorithms, with a focus on readability and reproducibility. Each file is designed to be independent, with code that is well-commented and easy to understand (at least I hope so!). This makes it easy for users to copy and paste the code and use it at their own convenience.

The project is written with Python, [Pytorch](https://github.com/pytorch/pytorch) and [Flax](https://github.com/google/flax). It supports all environments from Gymnasium, including MuJoCo and Atari environments.

This project is greatly inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl). I highly recommend you to check it out if you are looking for a more complete and well documented RL library.

You should be able to find the implementation of the following algorithms:

| RL Algorithm                                           | Pytorch                                                                                                                                                                                                                                                                               | Flax |
|--------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
| [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)                 | [discrete](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/dqn/pytorch_dqn_discrete.py) - [atari](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/dqn/pytorch_dqn_atari.py)                                                                                                         | [discrete](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/dqn/flax_dqn_discrete.py) - [atari](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/dqn/flax_dqn_atari.py) |
| [A2C](https://arxiv.org/abs/1602.01783)                | [discrete](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/a2c/pytorch_a2c_discrete.py) - [continuous](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/a2c/pytorch_a2c_continuous.py) - [atari](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/a2c/pytorch_a2c_atari.py) |  [discrete](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/a2c/flax_a2c_discrete.py) - [continuous](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/a2c/flax_a2c_continuous.py) - [atari](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/a2c/flax_a2c_atari.py)    |
| [PPO](https://arxiv.org/abs/1707.06347)                | [discrete](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/ppo/pytorch_ppo_discrete.py) - [continuous](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/ppo/pytorch_ppo_continuous.py) - [atari](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/ppo/pytorch_ppo_atari.py) |[discrete](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/ppo/flax_ppo_discrete.py) - [continuous](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/ppo/flax_ppo_continuous.py) - [atari](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/ppo/flax_ppo_atari.py)|
| [DDPG](https://proceedings.mlr.press/v32/silver14.pdf) | [continuous](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/ddpg/pytorch_ddpg_continuous.py)                                                                                                                                                                                                    |      |
| [TD3](https://arxiv.org/abs/1802.09477)                | [continuous](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/td3/pytorch_td3_continuous.py)                                                                                                                                                                                                       |      |
| [SAC](https://arxiv.org/abs/1801.01290)                | [continuous](https://github.com/valentin-cnt/rl-gym-zoo/blob/master/src/sac/pytorch_sac_continuous.py)

## Results

### LunarLander-v2

![lunar-lander](media/png/LunarLander-v2.png)

### HalfChetaah-v4

![half-cheetah](media/png/HalfCheetah-v4.png)
