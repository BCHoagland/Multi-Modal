import numpy as np
from visdom import Visdom

viz = Visdom()

def plot_dist(dist, name='dist'):
    x = np.linspace(-20, 20, 1000)
    y = dist(x)

    viz.line(
        X=x,
        Y=y,
        win='dists',
        name=name,
        opts=dict(
            title='Distribution'
        ),
        update='insert'
    )

losses = []
def plot_loss(loss):
    losses.append(loss.item())
    viz.line(
        X=np.arange(len(losses)),
        Y=np.array(losses),
        win='loss',
        opts=dict(
            title='Loss'
        )
    )
