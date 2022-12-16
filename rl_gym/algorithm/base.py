from abc import ABC, abstractmethod
import torch
# from torch.utils.tensorboard import SummaryWriter


class Base(ABC):

    @abstractmethod
    def __init__(self, args, obversation_space: tuple, action_space: tuple,
                 writer):

        self._model = None
        self._obversation_space = obversation_space
        self._action_space = action_space
        self._gamma = args.gamma

        self.writer = writer
        self._global_step = 0

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def update_policy(self, minibatch):
        pass

    def _normalize_tensor(self,
                          tensor: torch.Tensor,
                          eps=1e-9) -> torch.Tensor:

        return (tensor - tensor.mean()) / (tensor.std() + eps)

    def increment_step(self):

        self._global_step += 1

    @property
    def global_step(self):

        return self._global_step

    def get_obs_and_action_shape(self):

        return self._obversation_space, self._action_space

    def save_model(self, path: str):

        torch.save(self._model.state_dict(), path)

    def load_model(self, path: str):

        self._model.load_state_dict(torch.load(path))
