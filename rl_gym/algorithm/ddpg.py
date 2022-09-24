import torch
from torch.nn.functional import softmax, mse_loss
from torch.distributions import Categorical, Normal
from rl_gym.algorithm.base import Base
from rl_gym.neuralnet import ActorCriticNetContinuous, ActorCriticNetDiscrete


class DDPG(Base):

    def __init__(self):
        super(DDPG, self).__init__()

    def act(self):
        pass

    def update_policy(self):
        pass
