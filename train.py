import torch

import numpy as np
import os
import random
import argparse
import tqdm
import matplotlib.pyplot as plt

from utils import ReplayBuffer
from agent import BQN

import gym

parser = argparse.ArgumentParser('parameters')
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--round', type=int, default=2000, help='training rounds, (default: 2000)')
parser.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)')
parser.add_argument('--lr_rate', type=float, default=0.0001, help='learning rate (default : 0.0001)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size(default : 64)')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma (default : 0.99)')
parser.add_argument('--action_scale', type=int, default=50, help='action scale between -1 ~ +1')
parser.add_argument("--env", type=str, default='BipedalWalker-v3', help='Environment')

parser.add_argument("--save_interval", type=int, default=200, help='save interval(default: 100)')
parser.add_argument("--print_interval", type=int, default=50, help='print interval(default : 50)')
args = parser.parse_args()

use_tensorboard = args.tensorboard
action_scale = args.action_scale
learning_rate = args.lr_rate
batch_size = args.batch_size
gamma = args.gamma
env_name = args.env
total_round = args.round
episode = args.print_interval

if use_tensorboard:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()
else:
    writer = None

os.makedirs('./model_weights', exist_ok=True)

# env_name = 'BipedalWalker-v3'
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
print('observation space : ', env.observation_space)
print('action space : ', env.action_space)
print(env.action_space.low, env.action_space.high)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    agent = BQN(state_space, action_space, action_scale, learning_rate, device).cuda()
else:
    agent = BQN(state_space, action_space, action_scale, learning_rate, device)

memory = ReplayBuffer(100000, action_space, device)
real_action = np.linspace(-1., 1., action_scale)

epochs = int(total_round / episode)
score_list = []
n_epi = 0
for epoch in range(epochs):
    with tqdm.tqdm(total=episode, desc='Iteration %d' % epoch) as pbar:
        for ep in range(episode):
            state = env.reset()
            done = False
            score = 0.0
            while not done:
                epsilon = max(0.01, 0.9 - 0.01 * (n_epi / 10))
                if epsilon > random.random():
                    action = random.sample(range(0, action_scale), 4)
                else:
                    action_prob = agent.action(torch.tensor(state).float().reshape(1, -1).to(device))
                    action = [int(x.max(1)[1]) for x in action_prob]
                next_state, reward, done, _ = env.step(np.array([real_action[x] for x in action]))
                done_mask = True if reward <= -100 else False
                score += reward
                if reward <= -100:
                    reward = -1
                done_mask = 0 if done_mask is False else 1
                memory.put((state, action, reward, next_state, done_mask))
                if (memory.size() > 5000) and args.train:
                    agent.train_mode(n_epi, memory, batch_size, gamma, use_tensorboard, writer)
                state = next_state
            score_list.append(score)
            if use_tensorboard:
                writer.add_scalar("reward", score, n_epi)
            n_epi += 1
            if n_epi % args.save_interval == 0:
                torch.save(agent.state_dict(), './model_weights/agent_' + str(n_epi) + '.pth')
                # print("episode ", n_epi + 1, ": mean score ", np.mean(score_list[-args.print_interval:]), sep='')
            pbar.set_postfix({
                'ep':
                    '%d' % n_epi,
                'sc':
                    '%.3f' % np.mean(score_list[-(ep + 1):])
            })
            pbar.update(1)

episodes_list = list(range(len(score_list)))
plt.plot(episodes_list, score_list)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('DQN on {}'.format(env_name))
plt.show()