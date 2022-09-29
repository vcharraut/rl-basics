import torch
from torch.nn.functional import softmax, mse_loss
from torch.distributions import Categorical, Normal
from rlgym.algorithm.base import Base
from rlgym.neuralnet import ActorCriticNetContinuous, ActorCriticNetDiscrete


class DDPG(Base):

    def __init__(self):
        super(DDPG, self).__init__()

    def act(self):
        pass

    def update_policy(self):
        pass
