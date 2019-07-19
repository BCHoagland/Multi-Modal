import gym
import numpy as np
import torch

import gridworld
from dist import Dist
from visualize import *

from charles.storage import Storage

# hyperparameters
class Config:
    batch_size = 1                              # has to be 1

lr = 0.05
vis_iter = 50
epochs = 10000

# make environment and get environment details
env = gym.make('Gridworld-v0')
num_states = int(env.observation_space.high - env.observation_space.low + 1)
num_acts = int(env.action_space.n)

# make matrix of distributions and their respective optimizers
Z = [[Dist(K=10, requires_grad=True) for _ in range(num_acts)] for _ in range(num_states // 2)]                                # // 2 just for gridworld
optimizers = [[torch.optim.Adam(Z[s][a].parameters(), lr=lr) for a in range(num_acts)] for s in range(num_states // 2)]        # // 2 just for gridworld

# get obseration tuples from random agent
storage = Storage(Config)
s = env.reset()
for _ in range(1000):
    a = np.array(env.action_space.sample())
    s2, r, d, _ = env.step(a)
    storage.store((s, np.expand_dims(a, 0), np.expand_dims(r, 0), s2, np.expand_dims(d, 0)))
    s = s2 if not d else env.reset()

# run optimization steps on stored observations
for epoch in range(int(epochs)):
    s, a, r, s2, m = storage.sample()
    s = int(s.item())
    a = int(a.item())
    loss = -Z[s][a].log_prob(r)

    optimizers[s][a].zero_grad()
    loss.backward()
    optimizers[s][a].step()

    if epoch % vis_iter == vis_iter - 1:
        plot_loss(epoch, loss)

# plot final Z distributions
for s in range(num_states // 2):             # // 2 just for gridworld
    for a in range(num_acts):
        plot_dist(Z[s][a], f'{s}-{a}', f'rgb(0, {s * 100}, {a * 100})', (0, 10))
