from abc import ABC, abstractmethod
import torch


class Base(ABC):

    _model: torch.nn.Module

    @abstractmethod
    def __init__(self, args, obversation_space: tuple, action_space: tuple,
                 writer):

        self._device = args.device
        self._obversation_space = obversation_space
        self._action_space = action_space
        self._gamma = args.gamma

        self.writer = writer

    @abstractmethod
    def act(self, state):
        raise NotImplementedError

    @abstractmethod
    def update_policy(self, batch, step):
        raise NotImplementedError

    def _normalize_tensor(self,
                          tensor: torch.Tensor,
                          eps=1e-8) -> torch.Tensor:

        return (tensor - tensor.mean()) / (tensor.std() + eps)

    def get_obs_and_action_shape(self):

        return self._obversation_space, self._action_space

    def save_model(self, path: str):

        torch.save(self._model.state_dict(), path + ".pt")

    def load_model(self, path: str):

        self._model.load_state_dict(torch.load(path + ".pt"))
