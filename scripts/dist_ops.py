import numpy as np
import torch

from dist import Dist
from visualize import *

P = Dist([.4, .6], [4, 8], [1, 2])
plot_dist(P, 'P')

plot_dist(0.7 * P, '.7 P', '#f00')
plot_dist(P + 2, 'P + 2', '#00f')
