import torch
import numpy as np
import collections
import random


class ReplayBuffer:
    def __init__(self, buffer_limit, action_dim, device):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.action_dim = action_dim
        self.device = device

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        state_lst, reward_lst, next_state_lst, done_mask_lst = [], [], [], []
        actions_lst = [[] for _ in range(self.action_dim)]

        for transition in mini_batch:
            state, actions, reward, next_state, done_mask = transition
            state_lst.append(state)
            for idx in range(self.action_dim):
                actions_lst[idx].append(actions[idx])
            reward_lst.append([reward])
            next_state_lst.append(next_state)
            done_mask_lst.append([done_mask])
        
        state_lst = torch.tensor(np.array(state_lst), dtype=torch.float).to(self.device)
        actions_lst = [torch.tensor(x, dtype=torch.float).to(self.device) for x in actions_lst]
        reward_lst = torch.tensor(np.array(reward_lst), dtype=torch.float).to(self.device)
        next_state_lst = torch.tensor(np.array(next_state_lst), dtype=torch.float).to(self.device)
        done_mask_lst = torch.tensor(np.array(done_mask_lst), dtype=torch.float).to(self.device)

        return state_lst, actions_lst, reward_lst, next_state_lst, done_mask_lst

    def size(self):
        return len(self.buffer)
    

# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class PER:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity, action_dim, device):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.action_dim = action_dim
        self.device = device

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        state_lst, reward_lst, next_state_lst, done_mask_lst = [], [], [], []
        actions_lst = [[] for _ in range(self.action_dim)]

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            state, actions, reward, next_state, done_mask = data
            state_lst.append(state)
            for i in range(self.action_dim):
                actions_lst[i].append(actions[i])
            reward_lst.append([reward])
            next_state_lst.append(next_state)
            done_mask_lst.append([done_mask])

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        state_lst = torch.tensor(np.array(state_lst), dtype=torch.float).to(self.device)
        actions_lst = [torch.tensor(x, dtype=torch.float).to(self.device) for x in actions_lst]
        reward_lst = torch.tensor(np.array(reward_lst), dtype=torch.float).to(self.device)
        next_state_lst = torch.tensor(np.array(next_state_lst), dtype=torch.float).to(self.device)
        done_mask_lst = torch.tensor(np.array(done_mask_lst), dtype=torch.float).to(self.device)
        return idxs, is_weight, state_lst, actions_lst, reward_lst, next_state_lst, done_mask_lst

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def size(self):
        return self.tree.n_entries

