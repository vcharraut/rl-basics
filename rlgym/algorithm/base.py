import torch


class Base:

    def __init__(self):
        self.model = None

    def act(self):
        pass

    def update_policy(self):
        pass

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
