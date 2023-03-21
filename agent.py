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
    
    def append_sample(self, memory, state, action, reward, next_state, done_mask, prioritized, gamma):
        if prioritized:
            t_action = torch.tensor(action, dtype=torch.float).unsqueeze(0).unsqueeze(-1)
            done_mask = abs(done_mask - 1)

            q_values = self.q(torch.tensor(state, dtype=torch.float))
            q_values = torch.stack(q_values).transpose(0, 1)
            cur_val = q_values.gather(2, t_action.long()).squeeze(0).squeeze(-1)

            # max_next_q_values = self.q(next_state)  # double dqn
            max_next_q_values = self.target_q(torch.tensor(next_state, dtype=torch.float))  # normal dqn
            max_next_q_values = torch.stack(max_next_q_values).transpose(0, 1)
            max_next_q_values = max_next_q_values.max(-1, keepdim=True)[0].squeeze(0).squeeze(-1)

            target_val = reward + done_mask * gamma * max_next_q_values

            error = (abs(cur_val - target_val)).mean().detach().numpy()
            memory.add(error, (state, action, reward, next_state, done_mask))
        else:
            memory.add((state, action, reward, next_state, done_mask))

    def update(self, n_epi, memory, batch_size, gamma, use_tensorboard, writer, prioritized):
        if prioritized:
            idxs, is_weights, state, actions, reward, next_state, done_mask = memory.sample(batch_size)
        else:
            state, actions, reward, next_state, done_mask = memory.sample(batch_size)
        actions = torch.stack(actions).transpose(0, 1).unsqueeze(-1)
        done_mask = torch.abs(done_mask - 1)

        q_values = self.q(state)
        q_values = torch.stack(q_values).transpose(0, 1)
        q_values = q_values.gather(2, actions.long()).squeeze(-1)

        # max_next_q_values = self.q(next_state)  # double dqn
        max_next_q_values = self.target_q(next_state)  # normal dqn
        max_next_q_values = torch.stack(max_next_q_values).transpose(0, 1)
        max_next_q_values = max_next_q_values.max(-1, keepdim=True)[0].squeeze(-1)
        q_target = (done_mask * gamma * max_next_q_values + reward)

        if prioritized:
            errors = (abs(q_values - q_target)).mean(1).detach().numpy()
            # update priority
            for i in range(batch_size):
                idx = idxs[i]
                memory.update(idx, errors[i])

            # MSE Loss function
            loss = (torch.FloatTensor(is_weights) * F.mse_loss(q_values, q_target))
        else:
            loss = F.mse_loss(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward(torch.ones_like(loss))
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.update_freq == 0:
            self.update_count = 0
            self.target_q.load_state_dict(self.q.state_dict())

        if use_tensorboard:
            writer.add_scalar("Loss/loss", loss, n_epi)
        return loss
