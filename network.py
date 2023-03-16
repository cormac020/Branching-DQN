import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_space: int, action_num: int, action_scale: int):
        super(QNetwork, self).__init__()
        # 公共的特征提取层
        self.linear_1 = nn.Linear(state_space, state_space * 20)
        self.linear_2 = nn.Linear(state_space * 20, state_space * 10)
        # 在每个分支上评估该分支上每个动作的价值，分支有action_num个，动作有action_scale个
        self.actions = [nn.Sequential(nn.Linear(state_space * 10, state_space * 5),
                                      nn.ReLU(),
                                      nn.Linear(state_space * 5, action_scale)
                                      ) for _ in range(action_num)]
        # 使用modulelist将其注册到神经网络中，以便可以更新参数
        self.actions = nn.ModuleList(self.actions)
        # 额外的计算状态价值的模块
        self.value = nn.Sequential(nn.Linear(state_space * 10, state_space * 5),
                                   nn.ReLU(),
                                   nn.Linear(state_space * 5, 1)
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
