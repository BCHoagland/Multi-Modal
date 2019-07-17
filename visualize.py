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

def plot_dist(dist, dist_name, color):
    win = 'dist'
    title = 'Distributions'

    if 'dist' not in d:
        d['dist'] = {}
    d['dist'][dist_name] = (dist(np.linspace(-20, 20, 1000)).tolist(), color)

    data = []
    for key in d['dist']:
        points, c = d['dist'][key]
        data.append(
            get_line(list(range(len(points))), points, key, color=c)
        )

    # set format for the plot
    layout = dict(
        title=title,
        # xaxis={'title': 'x'},
        # yaxis={'title': 'y'}
    )

    # plot the data
    viz._send({'data': data, 'layout': layout, 'win': win})

def plot_loss(loss, color):
    win = 'loss'
    title = 'Loss'

    if 'loss' not in d:
        d['loss'] = []
    d['loss'].append(loss.item())

    points = d['loss']
    data = [get_line(list(range(len(points))), points, 'loss', color=color)]

    layout = dict(
        title=title,
        xaxis={'title': 'Iterations'},
        yaxis={'title': 'Loss'}
    )

    viz._send({'data': data, 'layout': layout, 'win': win})
