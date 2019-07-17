import torch

from dist import Dist
from visualize import *

epochs = 10000
lr = 0.05
batch_size = 128
vis_iter = 200

# make target distribution
P = Dist([.1, .3, .4, .1, .1], [-3, 3, 15, 10, -10], [1, 2, 3, 1, 5])
plot_dist(P, 'P', '#ff8200')

# make training distribution
Q = Dist(K=10, requires_grad=True)
plot_dist(Q, 'Q', '#4A04D4')

# make optimizer for training distribution
optimizer = torch.optim.Adam(Q.parameters(), lr=lr)

# run optimization steps on batches of output from target distribution
for epoch in range(int(epochs)):
    x = P.sample(batch_size)
    loss = torch.pow(Q(x) - P(x), 2).mean()                                             # what about cross entropy instead

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % viz_iter == viz_iter - 1:
        plot_loss(loss, '#4A04D4')
        plot_dist(Q, 'Q', '#4A04D4')
