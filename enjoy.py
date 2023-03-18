import torch

import numpy as np
import os
import random
import argparse
import tqdm

from agent import BQN

import gym
# from gym import wrappers, logger

parser = argparse.ArgumentParser('parameters')
parser.add_argument('--render', type=bool, default=True, help="(default: True)")
parser.add_argument('--round', type=int, default=10, help='evaluation rounds (default: 10)')
parser.add_argument('--action_scale', type=int, default=50, help='discrete action scale (default: 50)')
parser.add_argument('--load', type=str, default='final', help='load network name in ./model/')
parser.add_argument('--env', type=str, default='BipedalWalker-v3', help='Environment (default: BipedalWalker-v3)')

args = parser.parse_args()

action_scale = args.action_scale
env_name = args.env
eva_round = args.round

# env_name = 'BipedalWalker-v3'
# set seed to make evaluation repeatable
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)

# logger.setLevel(logger.ERROR)
os.makedirs('./data/', exist_ok=True)
# env = wrappers.Monitor(env, directory='./data/', force=True)  

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print('observation space:', env.observation_space)
print('action space:', env.action_space)
print('action space limits:', env.action_space.low, env.action_space.high)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    agent = BQN(state_dim, action_dim, action_scale, 0, device).cuda()
else:
    agent = BQN(state_dim, action_dim, action_scale, 0, device)

# if specified a model, load it
model_path = './model/'+ env_name + '_' + args.load + '.pth'
if os.path.isfile(model_path):
    agent.load_state_dict(torch.load(model_path))
real_action = np.linspace(-1., 1., action_scale)

score_list = []
pbar = tqdm.tqdm(range(eva_round))  # evaluate EVA_ROUND rounds
for n_epi in pbar:
    state = env.reset()
    done = False
    score = 0.0
    while not done:
        if args.render:
            env.render()
        action_value = agent.take_action(torch.tensor(state).float().reshape(1, -1).to(device))
        action = [int(x.max(1)[1]) for x in action_value]
        next_state, reward, done, _ = env.step(np.array([real_action[x] for x in action]))
        score += reward
        state = next_state
    score_list.append(score)
    n_epi += 1
    pbar.set_postfix({
        'ep':
            '%d' % n_epi,
        'sc':
            '%.3f' % np.mean(score_list[-(n_epi + 1):])
    })

print('Mean award in %d evaluation: %f' % (eva_round, np.mean(score_list)))
