import torch
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from rl_gym.algorithm.base import Base
from rl_gym.neuralnet import ActorCriticNet, CNNActorCritic


class A2C(Base):

    def __init__(self, args, obversation_space: tuple, action_space: tuple,
                 writer, is_continuous):

        super(A2C,
              self).__init__(args, obversation_space,
                             action_space if is_continuous else (), writer)

        self.is_continuous = is_continuous

        if args.cnn:
            self._model = CNNActorCritic(
                action_space,
                args.learning_rate,
            )
        else:
            self._model = ActorCriticNet(obversation_space, action_space,
                                         args.learning_rate, args.layers,
                                         args.shared_network, is_continuous)

        self.foward = self._model.forward_continuous if is_continuous else self._model.forward_discrete

        if args.device.type == "cuda":
            self._model.cuda()

    def act(self, state: torch.Tensor) -> tuple:

        distribution, critic_value = self.foward(state)

        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        if self.is_continuous:
            log_prob = log_prob.sum(-1)

        return action.cpu().numpy(), log_prob, critic_value.squeeze()

    def _shuffle_batch(self, batch, td_target):

        states = batch["states"].reshape((-1, ) + self._obversation_space)
        log_probs = batch["log_probs"].reshape(-1)
        td_target = td_target.reshape(-1)

        batch_index = torch.randperm(states.size(0))

        return states[batch_index], log_probs[batch_index], td_target[
            batch_index]

    def _get_td_target(self, batch: dict) -> torch.Tensor:

        rewards = batch["rewards"]
        flags = batch["flags"]

        td_target = torch.zeros(rewards.size()).to(self._device)
        gain = torch.zeros(rewards.size(1)).to(self._device)

        terminal = 1. - batch["last_flag"]

        for i in reversed(range(td_target.size(0))):
            gain = rewards[i] + gain * self._gamma * terminal
            td_target[i] = gain

            terminal = 1. - flags[i]

        return self._normalize_tensor(td_target).squeeze()

    def update_policy(self, batch: dict, step: int):

        td_target = self._get_td_target(batch)

        states, log_probs, td_target = self._shuffle_batch(batch, td_target)

        td_predict = self._model.critic(states).squeeze()

        advantages = td_target - td_predict

        policy_loss = (-log_probs * advantages).mean()
        value_loss = mse_loss(td_target, td_predict)

        loss = policy_loss + value_loss

        self._model.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self._model.parameters(), 0.5)
        self._model.optimizer.step()

        self.writer.add_scalar("update/policy_loss", policy_loss, step)
        self.writer.add_scalar("update/value_loss", value_loss, step)
