import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def logprob_eval(x):
    logprob = (-0.5 * math.log(2 * math.pi) - ((x - 0.0)**2 /
                                               (2 * 1.0**2))).sum()
    return logprob


def logprob_eval_post(x, mu, sigma):
    logprob = (-0.5 * math.log(2 * math.pi) - torch.log(sigma) -
               ((x - mu)**2 / (2 * sigma**2))).sum()
    return logprob


class bLinear(nn.Module):
    def __init__(self, in_size, out_size):
        super(bLinear, self).__init__()

        from torch.nn import Parameter

        mu = Parameter(torch.Tensor(out_size, in_size).normal_(0, 0.1))
        rho = Parameter(torch.Tensor(out_size, in_size).uniform_(-10.0, -9.0))
        self.register_parameter('mu', mu)
        self.register_parameter('rho', rho)

        mu_bias = Parameter(torch.Tensor(out_size).normal_(0, 0.1))
        rho_bias = Parameter(torch.Tensor(out_size).uniform_(-10.0, -9.0))
        self.register_parameter('mu_bias', mu_bias)
        self.register_parameter('rho_bias', rho_bias)

        self.register_buffer("loc", torch.tensor(0.))
        self.register_buffer("scale", torch.tensor(1.))
        self.normal = None

    @property
    def sigma(self):
        return torch.log(1.0 + torch.exp(self.rho))

    @property
    def sigma_bias(self):
        return torch.log(1.0 + torch.exp(self.rho_bias))

    @property
    def logprob_prior(self):
        return logprob_eval(self.weight) + logprob_eval(self.bias)

    @property
    def logprob_posterior(self):
        return logprob_eval_post(self.weight, self.mu,
                                 self.sigma) + logprob_eval_post(
                                     self.bias, self.mu_bias, self.sigma_bias)

    def forward(self, inputs, sample_flag=True):
        if sample_flag:
            if self.normal is None:
                self.normal = torch.distributions.Normal(self.loc, self.scale)
            epsilon = self.normal.sample(self.mu.size())
            epsilon_bias = self.normal.sample(self.mu_bias.size())

            self.weight = self.mu + epsilon * self.sigma
            self.bias = self.mu_bias + epsilon_bias * self.sigma_bias
            output = F.linear(inputs, self.weight, self.bias)
        else:
            output = F.linear(inputs, self.mu, self.mu_bias)
        return output
