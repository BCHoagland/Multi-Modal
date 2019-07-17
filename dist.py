import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from math import pi



# need a way of making sure std's are positive



class Dist(nn.Module):
    def __init__(self, a=None, m=None, s=None, K=None, requires_grad=False):
        super().__init__()

        self.K = K # the max # of modes in the distribution

        # a, m, s are parameters for the Gaussian distribution function
        # This chunk of code just ensures that a) the argument are valid
        given = [x for x in [a, m, s] if x is not None] # alpha, mu, sigma, if they exist
        if self.K is None: self.K = len(given[0]) # Ensure a minimum mode
        assert(all(len(x) == self.K for x in given))

        def init_weights(x, log=False):
            if x is not None:
                w = torch.FloatTensor(x)
                if log: w = torch.log(w)
            else:
                w = torch.rand(self.K)
            assert(w.shape == (self.K,))
            w.requires_grad = requires_grad
            return w

        self.α = init_weights(a)
        self.μ = init_weights(m)
        self.σ = init_weights(s, log=True)

    #toString
    def __str__(self):
        return f'Normal distribution with parameters\n\tα: {self.α.tolist()}\n\tμ: {self.μ.tolist()}\n\tσ: {self.σ.exp().tolist()}'

    # Returns a list of the distribution parameters alpha, mu, and sigma
    def parameters(self):
        return [self.α, self.μ, self.σ]

    # Wrapper method for pyTorch's feed forward method
    def forward(self, x):
        x = torch.FloatTensor(x) # convert input to tensor
        a = F.softmax(self.α, dim=0) # given a lot of numbers, it remaps them such that their sum == 1
        return torch.sum(a * self.N(x), dim=1) # Calculates the normal distr. for each value of x * the softmaxed alpha

    # Fetches a sample from a Gaussian distribution
    def sample(self, batch_size=1):
        a = F.softmax(self.α, dim=0) # Converts internal alphas to usable values
        i = torch.multinomial(a, 1) # Sample an index based on the weights of alpha with favorability towards higher weights
        dist = Normal(self.μ[i], self.σ[i]) # Makes a normal distribution with the ith mu and ith sigma
        return dist.sample((batch_size,))

    # Find the logarithmic probability of a point across multiple Gaussian distributions
    # way too many fuckin exponentiations here
    def log_prob(self, sample):
        dists = [Normal(m, s.exp()) for (m, s) in zip(self.μ, self.σ)]
        p = torch.stack([dist.log_prob(sample).exp() for dist in dists]).squeeze() # logify it
        p = torch.clamp(p, 1e-10, 1)
        a = F.softmax(self.α, dim=0).unsqueeze(1)
        return torch.sum(p * a, dim=0).log()

    # Normal Distribution
    # Returns probability of a point in a distribution
    def N(self, x):
        if len(x.shape) == 1: x = torch.FloatTensor(x).unsqueeze(1)
        return torch.exp(-torch.pow(x - self.μ, 2) / (2 * torch.pow(self.σ, 2))) / torch.sqrt(2 * pi * torch.pow(self.σ, 2))
