from math import isnan
import numpy as np
import torch
from visdom import Visdom

d = {}

viz = Visdom()

def get_line(x, y, name, color='#000', isFilled=False, fillcolor='transparent', width=2, showlegend=False):
        if isFilled:
            fill = 'tonexty'
        else:
            fill = 'none'

        return dict(
            x=x,
            y=y,
            mode='lines',
            type='custom',
            line=dict(
                color=color,
                width=width),
            fill=fill,
            fillcolor=fillcolor,
            name=name,
            showlegend=showlegend
        )


def plot_dist(dist, dist_name='dist', color='#000', range=(0, 20)):
    win = 'dist'
    title = 'Distributions'

    x = np.linspace(range[0], range[1], 100)

    if 'dist' not in d:
        d['dist'] = {}
    d['dist'][dist_name] = (dist(x).tolist(), color)

    data = []
    for key in d['dist']:
        points, c = d['dist'][key]
        data.append(
            get_line(list(x), points, key, color=c, showlegend=True)
        )

    layout = dict(
        title=title,
        xaxis={'title': 'Value'},
        yaxis={'title': 'Probability Density'}
    )

    viz._send({'data': data, 'layout': layout, 'win': win})


def plot_loss(epoch, loss, color='#000'):
    win = 'loss'
    title = 'Loss'

    if 'loss' not in d:
        d['loss'] = []
    d['loss'].append((epoch, loss.item()))

    x, y = zip(*d['loss'])
    data = [get_line(x, y, 'loss', color=color)]

    layout = dict(
        title=title,
        xaxis={'title': 'Iterations'},
        yaxis={'title': 'Loss'}
    )

    viz._send({'data': data, 'layout': layout, 'win': win})


def plot_reward(t, r, color='#000'):
    win = 'reward'
    title = 'Episodic Reward'

    if 'reward' not in d:
        d['reward'] = []
    d['reward'].append((t, float(r)))

    x, y = zip(*d['reward'])
    data = [get_line(x, y, 'reward', color=color)]

    layout = dict(
        title=title,
        xaxis={'title': 'Episodes'},
        yaxis={'title': 'Cumulative Reward'}
    )

    viz._send({'data': data, 'layout': layout, 'win': win})
