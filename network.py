import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, action_scale: int):
        super(QNetwork, self).__init__()
        # shared state feature extraction layer
        self.linear_1 = nn.Linear(state_dim, 512)
        self.linear_2 = nn.Linear(512, 256)
        # evaluate action advantages on each branch
        self.actions = [nn.Sequential(nn.Linear(256, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, action_scale)
                                      ) for _ in range(action_dim)]
        # 使用modulelist将其注册到神经网络中，以便可以更新参数
        self.actions = nn.ModuleList(self.actions)
        # module to calculate state value
        self.value = nn.Sequential(nn.Linear(256, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 1)
                                   )

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        encoded = F.relu(self.linear_2(x))
        actions = [head(encoded) for head in self.actions]
        value = self.value(encoded)
        for i in range(len(actions)):
            actions[i] = actions[i] - actions[i].max(-1)[0].reshape(-1, 1)
            actions[i] += value
        return actions
