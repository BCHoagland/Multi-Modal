import numpy as np
import torch

from dist import Dist
from visualize import *

P = Dist([.4, .6], [4, 8], [1, 2])
plot_dist(P, 'P')

plot_dist(0.99 * P, '.7 P', '#f00')
plot_dist(P + 2, 'P + 2', '#00f')

Q = Dist([.4, .6], [3, 5], [3, 2])
plot_dist(Q, 'Q', '#0f0')
