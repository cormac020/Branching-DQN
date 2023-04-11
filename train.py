import torch

import numpy as np
import os
import random
import argparse
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import time

from utils import ReplayBuffer, PER
from agent import BDQ

import gym

parser = argparse.ArgumentParser('parameters')
parser.add_argument('--round', '-r', type=int, default=2000, help='training rounds (default: 2000)')
parser.add_argument('--lr_rate', '-lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size (default: 64)')
parser.add_argument('--gamma', '-g', type=float, default=0.99, help='discounting factor (default: 0.99)')
parser.add_argument('--action_scale', '-a', type=int, default=25, help='discrete action scale (default: 25)')
parser.add_argument('--env', '-e', type=str, default='BipedalWalker-v3', help='Environment (default: BipedalWalker-v3)')
parser.add_argument('--per', '-p', action='store_true', help='use per')
parser.add_argument('--load', '-l', type=str, default='no', help='load network name in ./model/')
parser.add_argument('--no_trick', '-nt', action='store_true', help='not to use tricks')

parser.add_argument('--save_interval', '-s', type=int, default=1000, help='interval to save model (default: 1000)')
parser.add_argument('--print_interval', '-d', type=int, default=50, help='interval to print evaluation (default: 50)')
args = parser.parse_args()

action_scale = args.action_scale
learning_rate = args.lr_rate
batch_size = args.batch_size
gamma = args.gamma
env_name = args.env
total_round = args.round
iter_size = args.print_interval
prioritized = args.per

os.makedirs('./model/', exist_ok=True)

# env_name = 'BipedalWalker-v3'
gym.logger.set_level(40)
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
	agent = BDQ(state_dim, action_dim, action_scale, learning_rate, device).cuda()
else:
	agent = BDQ(state_dim, action_dim, action_scale, learning_rate, device)
# if specified a model, load it
model_path = './model/' + env_name + '_' + args.load + '.pth'
if os.path.isfile(model_path):
	agent.load_state_dict(torch.load(model_path))

memory = PER(100000, action_dim, device) if prioritized else ReplayBuffer(100000, action_dim, device)
# real_action = np.linspace(-1., 1., action_scale)
real_actions = [np.linspace(env.action_space.low[i], env.action_space.high[i], action_scale)
				for i in range(action_dim)]

iteration = int(total_round / iter_size)
score_list, time_list = [], []
n_epi = 0
start = time.time()
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
				done_mask = 1 if done else 0
				
				if not args.no_trick:
					if reward <= -100:
						reward = -1
						done_mask = 1
					else:
						done_mask = 0

				agent.append_sample(memory, state, action, reward, next_state, done_mask, prioritized, gamma)
				if memory.size() > 5000:
					agent.update(memory, batch_size, gamma, prioritized)
				state = next_state
			score_list.append(score)
			time_list.append(time.time() - start)

			n_epi += 1
			if n_epi % args.save_interval == 0:
				torch.save(agent.state_dict(), './model/' + env_name + '_' + str(action_scale) + '.pth')
				# print("iter_size ", n_epi + 1, ": mean score ", np.mean(score_list[-args.print_interval:]), sep='')
			pbar.set_postfix({
				'ep':
					'%d' % n_epi,
				'sc':
					'%.3f' % np.mean(score_list[-(ep + 1):])
			})
			pbar.update(1)

torch.save(agent.state_dict(), './model/' + env_name + '_' + str(action_scale) + '.pth')
os.makedirs('./data/', exist_ok=True)
episodes_list = list(range(len(score_list)))
plt.plot(episodes_list, score_list)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('DRDQN on {}'.format(env_name))
plt.savefig('./data/' + env_name + '_' + str(action_scale) + '_score.png')
# plt.show()

# 字典中的key值即为csv中列名
dataframe = pd.DataFrame({env_name: score_list, 'time': time_list})
# 将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv('./data/' + env_name + '_' + str(action_scale) + "_reward.csv", index=False, sep=',')
