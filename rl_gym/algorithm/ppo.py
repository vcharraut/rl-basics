import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch.nn.utils.clip_grad import clip_grad_norm_
from rl_gym.algorithm.base import Base
from rl_gym.neuralnet import ActorCriticNet, CNNActorCritic


class PPO(Base):

    def __init__(self, args, obversation_space: tuple, action_space: tuple,
                 writer, is_continuous):

        super(PPO,
              self).__init__(args, obversation_space,
                             action_space if is_continuous else (), writer)

        self.__n_optim = 4
        self.__eps_clip = 0.2
        self.__value_constant = 0.5
        self.__entropy_constant = 0.01
        self._lmbda = 0.95

        self._batch_size = args.batch_size
        self._minibatch_size = args.minibatch_size

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

    def act(self, state: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:

        with torch.no_grad():
            distribution, critic_value = self.foward(state)

        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        if self.is_continuous:
            log_prob = log_prob.sum(-1)

        return action.cpu().numpy(), log_prob, critic_value.squeeze()

    def _evaluate(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        distribution, td_predict = self.foward(states)

        log_prob = distribution.log_prob(actions)
        dist_entropy = distribution.entropy()

        if self.is_continuous:
            log_prob = log_prob.sum(-1)
            dist_entropy = dist_entropy.sum(-1)

        return log_prob, td_predict.squeeze(), dist_entropy

    def _shuffle_batch(self, batch, td_target, advantages):

        states = batch["states"].reshape((-1, ) + self._obversation_space)
        actions = batch["actions"].reshape((-1, ) + self._action_space)
        old_log_probs = batch["log_probs"].reshape(-1)
        td_target = td_target.reshape(-1)
        advantages = advantages.reshape(-1)

        batch_index = torch.randperm(self._batch_size)

        return states[batch_index], actions[batch_index], old_log_probs[
            batch_index], td_target[batch_index], advantages[batch_index]

    def _gae(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:

        rewards = batch["rewards"]
        flags = batch["flags"]
        state_values = batch["state_values"]

        with torch.no_grad():
            next_state_value = self._model.critic(
                batch["last_state"]).squeeze(-1)

        advantages = torch.zeros(rewards.size()).to(self._device)
        adv = torch.zeros(rewards.size(1)).to(self._device)

        terminal = 1. - batch["last_flag"]

        for i in reversed(range(rewards.size(0))):
            delta = rewards[
                i] + self._gamma * next_state_value * terminal - state_values[i]
            adv = self._gamma * self._lmbda * adv * terminal + delta
            advantages[i] = adv

            next_state_value = state_values[i]
            terminal = 1. - flags[i]

        td_target = advantages + state_values
        advantages = self._normalize_tensor(advantages)

        return td_target.squeeze(), advantages.squeeze()

    def update_policy(self, batch: dict, step: int):

        td_target, advantages = self._gae(batch)

        states, actions, old_log_probs, td_target, advantages = self._shuffle_batch(
            batch, td_target, advantages)

        clipfracs = []

        for _ in range(self.__n_optim):
            for start in range(0, self._batch_size, self._minibatch_size):
                end = start + self._minibatch_size
                index = slice(start, end)

                log_probs, td_predict, dist_entropy = self._evaluate(
                    states[index], actions[index])

                logratio = log_probs - old_log_probs[index]
                ratios = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratios - 1) - logratio).mean()
                    clipfracs += [
                        ((ratios - 1.0).abs() > 0.2).float().mean().item()
                    ]

                surr1 = advantages[index] * ratios

                surr2 = advantages[index] * torch.clamp(
                    ratios, 1. - self.__eps_clip, 1. + self.__eps_clip)

                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = self.__value_constant * mse_loss(
                    td_predict, td_target[index])

                entropy_bonus = self.__entropy_constant * dist_entropy.mean()

                loss = policy_loss + value_loss - entropy_bonus

                self._model.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self._model.parameters(), 0.5)
                self._model.optimizer.step()

        self.writer.add_scalar("update/policy_loss", policy_loss, step)
        self.writer.add_scalar("update/value_loss", value_loss, step)
        self.writer.add_scalar("debug/old_approx_kl", old_approx_kl, step)
        self.writer.add_scalar("debug/approx_kl", approx_kl, step)
        self.writer.add_scalar("debug/clipfrac", np.mean(clipfracs), step)
