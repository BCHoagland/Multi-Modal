import gym
import numpy as np
import torch
import random
from collections import deque
from termcolor import colored

import nodeworld
from dist import Dist
from storage import Storage
from visualize import *

# hyperparameters
lr = 0.1
vis_iter = 500
epochs = 1
max_timesteps = 1e4
ε = 0.1
sample_size = 10
buffer_size = 1e6
batch_size = 1                              # has to be 1

# make environment and get environment details
env = gym.make('Nodeworld-v0')
num_states = int(env.observation_space.high - env.observation_space.low + 1)
num_acts = int(env.action_space.n)

# make matrix of distributions and their respective optimizers
Z = [[Dist(K=10, requires_grad=True) for _ in range(num_acts)] for _ in range(num_states)]
optimizers = [[torch.optim.Adam(Z[s][a].parameters(), lr=lr) for a in range(num_acts)] for s in range(num_states)]

storage = Storage(buffer_size)

def plot_Z():
    for s in range(num_states // 2):                                                # // 2 just for gridworld
        for a in range(num_acts):
            plot_dist(Z[s][a], f'{s}-{a}', f'rgb(0, {(s / (num_states // 2)) * 255}, {(a + 1) * 100})', (0, 30))

def z_action(s):
    max_a = 0
    max_q = Z[int(s)][max_a].mean()
    for a in range(1, num_acts):
        q = Z[int(s)][a].mean()
        if q > max_q:
            max_q = q
            max_a = a
    return max_a


def train():
    # get obseration tuples from random agent
    s = env.reset()
    ep = 0
    ep_r = 0
    past_ep_r = deque(maxlen=100)
    for t in range(int(max_timesteps)):
        if random.random() < ε:
            a = env.action_space.sample()
        else:
            a = z_action(s)

        s2, r, d, _ = env.step(a)
        ep_r += r

        storage.store((s, np.expand_dims(a, 0), np.expand_dims(r, 0), s2, np.expand_dims(d, 0)))
        s = s2 if not d else env.reset()

        if d:
            ep += 1
            past_ep_r.append(ep_r)
            if ep % vis_iter == vis_iter - 1:
                # plot_reward(ep, ep_r)
                plot_reward(ep, sum(past_ep_r) / len(past_ep_r))
                plot_Z()
            ep_r = 0

        update(t)


def update(t):
    # run optimization steps on stored observations
    for epoch in range(int(epochs)):
        s, a, r, s2, m = storage.sample(batch_size)
        s, a, r, s2, m = int(s.item()), int(a.item()), int(r.item()), int(s2.item()), int(m.item())

        # Z(s, a) = r + γ Z(s', argmax_a' Z(s', a'))
        a2 = z_action(s2)
        z2 = Z[s2][a2]
        z_targ = r + (0.99 * m * z2)

        # sample from the target distribution and minimize the cross entropy of Z based on this sample
        z = Z[s][a]
        x = z_targ.sample(sample_size)
        dist_loss = -z.log_prob(x).mean()

        # take an optimization step to improve Z
        optimizers[s][a].zero_grad()
        dist_loss.backward()
        optimizers[s][a].step()

        # plot loss
        if t % vis_iter == vis_iter - 1:
            plot_loss(t, dist_loss)


train()

print(colored('\nOptimal Actions (should all be \'right\'):', 'yellow'))
for s in range(num_states // 2):
    a = z_action(s)
    a = 'left' if a is 0 else 'right'
    print(f'S: {s+1} -> A: {a}')
print()
