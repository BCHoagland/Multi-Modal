import gym
import numpy as np
import torch

import nodeworld
from dist import Dist
from visualize import *

from charles.storage import Storage
from charles.models import *

# hyperparameters
class Config:
    batch_size = 1                              # has to be 1

lr = 0.05
vis_iter = 50
epochs = 10000

# make environment and get environment details
env = gym.make('Nodeworld-v0')
num_states = int(env.observation_space.high - env.observation_space.low + 1)
num_acts = int(env.action_space.n)

# make matrix of distributions and their respective optimizers
Z = [[Dist(K=10, requires_grad=True) for _ in range(num_acts)] for _ in range(num_states)]
optimizers = [[torch.optim.Adam(Z[s][a].parameters(), lr=lr) for a in range(num_acts)] for s in range(num_states)]

π = Model(LinearPolicy, env, 3e-4)                                              # this is essentially a random π, and it never gets optimized

# get obseration tuples from random agent
storage = Storage(Config)
s = env.reset()
for _ in range(1000):
    with torch.no_grad():
        a = π(s)
    s2, r, d, _ = env.step(a)
    storage.store((s, np.expand_dims(a, 0), np.expand_dims(r, 0), s2, np.expand_dims(d, 0)))
    s = s2 if not d else env.reset()

# run optimization steps on stored observations
for epoch in range(int(epochs)):
    s, a, r, s2, m = storage.sample()
    s = int(s.item())
    a = int(a.item())
    r = int(r.item())
    m = int(m.item())

    # sample the next action for computing Z'
    with torch.no_grad():
        a2 = int(π(s2).item())
    s2 = int(s2.item())

    # fetch current approximations of Z and Z'
    z = Z[s][a]
    z2 = Z[s2][a2]

    # compute target for Z based on Bellman equation
    z_targ = r + (0.99 * m * z2)

    # sample from the target distribution and minimize the cross entropy of Z based on this sample
    x = z_targ.sample()
    dist_loss = -z.log_prob(x)

    # take an optimization step to improve Z
    optimizers[s][a].zero_grad()
    dist_loss.backward()
    optimizers[s][a].step()

    # plot loss
    if epoch % vis_iter == vis_iter - 1:
        plot_loss(epoch, dist_loss)

# plot final Z distributions
for s in range(num_states // 2):                                                # // 2 just for gridworld
    for a in range(num_acts):
        plot_dist(Z[s][a], f'{s}-{a}', f'rgb(0, {s * 100}, {a * 100})', (0, 10))
