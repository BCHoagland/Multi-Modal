import torch
import torch.nn.functional as F

from dist import Dist
from visualize import *

epochs = 20000
lr = 0.01
batch_size = 128

# make target distribution
P = Dist([.2, .2, .3, .1, .2], [-3, 3, 15, 10, -10], [1, 2, 3, 1, 5], 3)
plot_dist(P, 'P')

# make training distribution
Q = Dist(K=20, requires_grad=True)
plot_dist(Q, 'Q')

# make optimizer for training distribution
optimizer = torch.optim.Adam(Q.parameters(), lr=lr)

# run optimization steps on batches of output from target distribution
for epoch in range(int(epochs)):
    x = P.sample(batch_size)
    loss = torch.pow(Q(x) - P(x), 2).mean()
    # loss += torch.pow(P.log_prob(x) - Q.log_prob(x), 2).mean()
    loss += -Q.log_prob(x).mean() * max(0, (1 - (epoch / (epochs / 2))))                        # anneal entropy bonus for smoother final fitting

                                                                                                # detect when it's okay to switch over to just MSE

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % (epochs // 200) == (epochs // 200) - 1:
        plot_loss(loss)
        plot_dist(Q, 'Q')

print(f'Desired: {P.a.tolist()}, {P.scale.tolist()}')
print(f'Actual: {Q.a.tolist()}, {Q.scale.tolist()}')
