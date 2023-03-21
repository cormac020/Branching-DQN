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
parser.add_argument('--round', type=int, default=2000, help='training rounds, (default: 2000)')
parser.add_argument('--tensorboard', type=bool, default=False, help='use tensorboard or not, (default: False)')
parser.add_argument('--lr_rate', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 64)')
parser.add_argument('--gamma', type=float, default=0.99, help='discounting factor (default: 0.99)')
parser.add_argument('--action_scale', type=int, default=50, help='discrete action scale (default: 50)')
parser.add_argument('--env', type=str, default='BipedalWalker-v3', help='Environment (default: BipedalWalker-v3)')

parser.add_argument('--save_interval', type=int, default=200, help='interval round to save model (default: 100)')
parser.add_argument('--print_interval', type=int, default=50, help='interval round to print evaluation (default: 50)')
args = parser.parse_args()

use_tensorboard = args.tensorboard
action_scale = args.action_scale
learning_rate = args.lr_rate
batch_size = args.batch_size
gamma = args.gamma
env_name = args.env
total_round = args.round
iter_size = args.print_interval

if use_tensorboard:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()
else:
    writer = None

os.makedirs('./model/', exist_ok=True)

gym.logger.set_level(40)
# env_name = 'BipedalWalker-v3'
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print('observation space:', env.observation_space)
print('action space:', env.action_space)
print('action space limits:', env.action_space.low, env.action_space.high)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    agent = BQN(state_dim, action_dim, action_scale, learning_rate, device).cuda()
else:
    agent = BQN(state_dim, action_dim, action_scale, learning_rate, device)

memory = ReplayBuffer(100000, action_dim, device)
# real_action = np.linspace(-1., 1., action_scale)
real_actions = [np.linspace(env.action_space.low[i], env.action_space.high[i], action_scale)
                for i in range(action_dim)]

iteration = int(total_round / iter_size)
score_list = []
n_epi = 0
for it in range(iteration):
    with tqdm.tqdm(total=iter_size, desc='Iteration %d' % it) as pbar:
        for ep in range(iter_size):
            state = env.reset()
            done = False
            score = 0.0
            while not done:
                epsilon = max(0.01, 0.9 - 0.01 * (n_epi / 10))
                if epsilon > random.random():
                    action = random.sample(range(action_scale), action_dim)
                else:
                    action_value = agent.take_action(torch.tensor(state).float().reshape(1, -1).to(device))
                    action = [int(x.max(1)[1]) for x in action_value]
                next_state, reward, done, _ = env.step(np.array([real_actions[i][action[i]] 
                                                                 for i in range(action_dim)]))
                score += reward
                # tricks
                if reward <= -100:
                    reward = -1
                    done_mask = 1
                else:
                    done_mask = 0

                memory.add((state, action, reward, next_state, done_mask))
                if memory.size() > 5000:
                    agent.update(n_epi, memory, batch_size, gamma, use_tensorboard, writer, action_dim)
                state = next_state
            score_list.append(score)
            if use_tensorboard:
                writer.add_scalar("reward", score, n_epi)
            n_epi += 1
            if n_epi % args.save_interval == 0:
                torch.save(agent.state_dict(), './model/' + env_name + '_' + str(n_epi) + '.pth')
                # print("iter_size ", n_epi + 1, ": mean score ", np.mean(score_list[-args.print_interval:]), sep='')
            pbar.set_postfix({
                'ep':
                    '%d' % n_epi,
                'sc':
                    '%.3f' % np.mean(score_list[-(ep + 1):])
            })
            pbar.update(1)

# torch.save(agent.state_dict(), './model/' + env_name + '_final.pth')
os.makedirs('./data/', exist_ok=True)
episodes_list = list(range(len(score_list)))
plt.plot(episodes_list, score_list)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('BDQN on {}'.format(env_name))
plt.savefig('./data/' + env_name + '_score.png')
# plt.show()
