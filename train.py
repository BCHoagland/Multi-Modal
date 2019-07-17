import torch
import torch.nn.functional as F

from dist import Dist
from visualize import *

epochs = 10000
viz_iter = 200
lr = 0.05
batch_size = 128

# make target distribution
P = Dist([.1, .3, .4, .1, .1], [-3, 3, 15, 10, -10], [1, 2, 3, 1, 5]) # stick to small numbers for now as exponentiation on anything much larger than like 4 or 5 is just too much
plot_dist(P, 'P', '#ff8200')

# make training distribution
Q = Dist(K=10, requires_grad=True)
plot_dist(Q, 'Q', '#4A04D4')

# make optimizer for training distribution
optimizer = torch.optim.Adam(Q.parameters(), lr=lr)

# run optimization steps on batches of output from target distribution
for epoch in range(int(epochs)):
    x = P.sample(batch_size)
    loss = torch.pow(Q(x) - P(x), 2).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % viz_iter == viz_iter - 1:
        plot_loss(loss, '#4A04D4')
        plot_dist(Q, 'Q', '#4A04D4')
