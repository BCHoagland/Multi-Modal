import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from math import pi



# need a way of making sure std's are positive



class Dist(nn.Module):
    def __init__(self, a=None, m=None, s=None, scale=1, K=None, requires_grad=False):
        super().__init__()

        self.K = K

        given = [x for x in [a, m, s] if x is not None]
        if self.K is None: self.K = len(given[0])
        assert(all(len(x) == self.K for x in given))

        def init_weights(x, log=False):
            w = torch.FloatTensor(x) if x is not None else torch.rand(self.K)
            assert(w.shape == (self.K,))
            if log: w = torch.log(w)
            w.requires_grad = requires_grad
            return w

        self.a = init_weights(a)
        self.m = init_weights(m)
        self.s = init_weights(s, log=False)
        self.scale = torch.tensor(float(scale), requires_grad=True)

        self.dists = [Normal(m, s) for (m, s) in zip(self.m, self.s)]

    def __str__(self):
        return f'Normal distribution with parameters\n\tα: {self.a.tolist()}\n\tμ: {self.m.tolist()}\n\tσ: {self.s.exp().tolist()}'

    def parameters(self):
        return [self.a, self.m, self.s, self.scale]

    def forward(self, x):
        x = torch.FloatTensor(x)
        a = F.softmax(self.a, dim=0)
        return self.scale * torch.sum(a * self.N(x), dim=1)

    def sample(self, batch_size=1):
        a = F.softmax(self.a, dim=0)
        i = torch.multinomial(a, 1)
        dist = Normal(self.m[i], self.s[i])
        return dist.sample((batch_size,))

    def log_prob(self, sample):
        dists = [Normal(m, s.exp()) for (m, s) in zip(self.m, self.s)]
        p = torch.stack([dist.log_prob(sample).exp() for dist in dists]).squeeze()
        a = F.softmax(self.a, dim=0).unsqueeze(1)
        return torch.sum(p * a, dim=0).log()

    def N(self, x):
        if len(x.shape) == 1: x = torch.FloatTensor(x).unsqueeze(1)
        return torch.exp(-torch.pow(x - self.m, 2) / (2 * torch.pow(self.s, 2))) / torch.sqrt(2 * pi * torch.pow(self.s, 2))
