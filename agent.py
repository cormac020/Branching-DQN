import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from network import QNetwork


class BQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, action_scale: int, learning_rate, device: str):
        super(BQN, self).__init__()

        self.q = QNetwork(state_dim, action_dim, action_scale).to(device)
        self.target_q = QNetwork(state_dim, action_dim, action_scale).to(device)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam([{'params': self.q.linear_1.parameters(), 'lr': learning_rate / (action_dim + 2)},
                                     {'params': self.q.linear_2.parameters(), 'lr': learning_rate / (action_dim + 2)},
                                     {'params': self.q.value.parameters(), 'lr': learning_rate / (action_dim + 2)},
                                     {'params': self.q.actions.parameters(), 'lr': learning_rate}, ])
        self.update_freq = 1000
        self.update_count = 0

    def take_action(self, x):
        return self.q(x)

    def update(self, n_epi, memory, batch_size, gamma, use_tensorboard, writer, action_dim):
        state, actions, reward, next_state, done_mask = memory.sample(batch_size)
        actions = torch.stack(actions).transpose(0, 1).unsqueeze(-1)
        done_mask = torch.abs(done_mask - 1)

        q_values = self.q(state)
        q_values = torch.stack(q_values).transpose(0, 1)
        q_values = q_values.gather(2, actions.long()).squeeze(-1)

        # max_next_q_values = self.q(next_state)  # double dqn
        max_next_q_values = self.target_q(next_state)  # normal dqn
        max_next_q_values = torch.stack(max_next_q_values).transpose(0, 1)
        max_next_q_values = max_next_q_values.max(-1, keepdim=True)[0]
        q_target = (done_mask * gamma * max_next_q_values.mean(1) + reward)

        loss = F.mse_loss(q_values, q_target.repeat(1, action_dim))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.update_freq == 0:
            self.update_count = 0
            self.target_q.load_state_dict(self.q.state_dict())

        if use_tensorboard:
            writer.add_scalar("Loss/loss", loss, n_epi)
        return loss
