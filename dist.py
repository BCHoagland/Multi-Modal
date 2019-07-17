import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from math import pi, isinf

class Dist(nn.Module):
    def __init__(self, a=None, m=None, s=None, K=None, requires_grad=False):
        super().__init__()

        self.K = K                                          # the max number of modes in the distribution

        # a, m, s are parameters for the Gaussian distribution function
        given = [x for x in [a, m, s] if x is not None]     # α, μ, σ (if they exist)
        if self.K is None: self.K = len(given[0])           # set self.K based on the size of the supplied arguments
        assert(all(len(x) == self.K for x in given))        # ensure all supplied arguments are the same size

        def init_weights(x, log=False):
            if x is not None:                               # if x has supplied values, convert it to a tensor
                w = torch.FloatTensor(x)
                if log: w = torch.log(w)                    # set it to its log if need be (we'll do this with σ to ensure that we always have σ > 0)
            else:
                w = torch.rand(self.K)                      # if x wasn't supplied, make it a tensor with random values ∈ [0, 1)
            assert(w.shape == (self.K,))
            w.requires_grad = requires_grad                 # set whether or not to track the gradient of the parameter
            return w

        self.α = init_weights(a)                            # initialize α, μ, and σ using the above helper method
        self.μ = init_weights(m)
        self.σ = init_weights(s, log=True)

    def __str__(self):
        return f'Normal distribution with parameters\n\tα: {self.α.tolist()}\n\tμ: {self.μ.tolist()}\n\tσ: {self.σ.exp().tolist()}'

    # Returns a list of the distribution parameters α, μ, and σ
    def parameters(self):
        return [self.α, self.μ, self.σ]

    # Feed forward through the distribution
    def forward(self, x):
        x = torch.FloatTensor(x)                    # convert input to tensor
        a = F.softmax(self.α, dim=0)                # given a lot of numbers, it remaps them such that their sum == 1
        return torch.sum(a * self.N(x), dim=1)      # Calculates the normal distr. for each value of x * the softmaxed alpha

    # Fetches a sample from the entire Gaussian mixture
    def sample(self, batch_size=1):
        a = F.softmax(self.α, dim=0)                # Converts internal alphas to usable values
        i = torch.multinomial(a, 1)                 # Sample an index i based on the weights of alpha with favorability towards higher weights
        dist = Normal(self.μ[i], self.σ[i].exp())   # Makes a Normal distribution with the ith μ and ith σ
        return dist.sample((batch_size,))           # Sample from the chosen Normal distribution

    # Find the logarithmic probability of a point across the entire Gaussian mixture
    def log_prob(self, sample):
        dists = [Normal(m, s.exp()) for (m, s) in zip(self.μ, self.σ)]
        p = torch.stack([dist.log_prob(sample).exp() for dist in dists]).squeeze()
        p = torch.clamp(p, 1e-10, 1)
        a = F.softmax(self.α, dim=0).unsqueeze(1)
        return torch.sum(p * a, dim=0).log()

    # Returns the probability density of a point in a Normal distribution
    def N(self, x):
        if len(x.shape) == 1: x = torch.FloatTensor(x).unsqueeze(1)
        return torch.exp(-torch.pow(x - self.μ, 2) / (2 * torch.pow(self.σ.exp(), 2))) / torch.sqrt(2 * pi * torch.pow(self.σ.exp(), 2))
