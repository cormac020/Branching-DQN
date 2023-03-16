import torch

import numpy as np
import os
import random
import argparse
import tqdm

from agent import BQN

import gym

parser = argparse.ArgumentParser('parameters')
parser.add_argument('--render', type=bool, default=True, help="(default: True)")
parser.add_argument('--round', type=int, default=10, help='evaluation rounds (default: 10)')
parser.add_argument('--action_scale', type=int, default=50, help='action scale between -1 ~ +1')
parser.add_argument("--load", type=str, default='no', help='load network name in ./model_weights')
parser.add_argument("--env", type=str, default='BipedalWalker-v3', help='Environment')

args = parser.parse_args()

action_scale = args.action_scale
env_name = args.env
eva_round = args.round

os.makedirs('./model_weights', exist_ok=True)

# env_name = 'BipedalWalker-v3'
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
print('observation space:', env.observation_space)
print('action space:', env.action_space)
print('action space limits:', env.action_space.low, env.action_space.high)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    agent = BQN(state_space, action_space, action_scale, 0, device).cuda()
else:
    agent = BQN(state_space, action_space, action_scale, 0, device)

if args.load != 'no':
    agent = BQN(state_space, action_space, action_scale, 0, device)
    agent.load_state_dict(torch.load('./model_weights/agent_' + args.load + '.pth'))
real_action = np.linspace(-1., 1., action_scale)

score_list = []
pbar = tqdm.tqdm(range(eva_round))
for n_epi in pbar:
    state = env.reset()
    done = False
    score = 0.0
    while not done:
        if args.render:
            env.render()
        action_prob = agent.action(torch.tensor(state).float().reshape(1, -1).to(device))
        action = [int(x.max(1)[1]) for x in action_prob]
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
    pbar.update(1)
print('Mean award in %d evaluation: %f' % (eva_round, np.mean(score_list)))
