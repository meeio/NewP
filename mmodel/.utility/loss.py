from torch import nn

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b


import torch
from mground.math_utils import euclidean_dist
from functools import partial

def euclidean_dist(x, y=None):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    order = [1,0] 
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)

    return torch.pow(x - y, 2).sum(2)

def gaussian_kernel(x, y, sigmas):
    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = euclidean_dist(x, y)
    dist_ = dist.view(1, -1)
    s = torch.matmul(beta, dist_)
    return torch.sum(torch.exp(-s), 0).view_as(dist)


def mmd_loss(x, y):

    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]

    K = partial(
        gaussian_kernel, sigmas = torch.cuda.FloatTensor(sigmas)
    )

    loss = torch.mean(K(x, x))
    loss += torch.mean(K(y, y))
    loss -= 2 * torch.mean(K(x, y))

    return loss