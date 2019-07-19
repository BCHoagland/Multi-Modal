import torch

from dist import Dist
from visualize import *

lr = 0.05
epochs = 100
vis_iter = 50

# next value distribution
z2 = Dist([0.2, 0.3, 0.5], [3, 7, 12], [1, 2, 1])
plot_dist(z2, 'Z\'', '#0df')

# approximate bellman definition of current value distribution
# assuming deterministic rewards
z = 3 + (0.99 * z2)
plot_dist(z, 'Z', '#00f')

# target network to fit to the bellman value distribution
Q = Dist(K=10, requires_grad=True)
optimizer = torch.optim.Adam(Q.parameters(), lr=lr)

for epoch in range(int(epochs)):
    x = z.sample()
    loss = -Q.log_prob(x)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % vis_iter == vis_iter - 1:
        plot_loss(epoch, loss)
        plot_dist(Q, 'approx', '#f00')
