import torch
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms

DATA_PATH = '.datasets/'


def make_odd_dset(in_dis, out_dis):
    in_dis.targets = len(in_dis.targets) * [
        1,
    ]
    out_dis.targets = len(out_dis.targets) * [
        0,
    ]
    return ConcatDataset([in_dis, out_dis])


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255),
                         (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
])


def cifar10_dset():
    datapath = DATA_PATH + 'cifar10'
    train_dataset = datasets.CIFAR10(datapath, train=True, download=True,
                                     transform=transform)
    test_dataset = datasets.CIFAR10(datapath, train=False, download=True,
                                    transform=transform)
    return train_dataset, test_dataset


def lsun_dset():
    datapath = DATA_PATH + 'LSUN_resize'
    dset = datasets.ImageFolder(datapath, transform=transform)
    return dset


def imagenet_dset():
    datapath = DATA_PATH + 'Imagenet_resize'
    dset = datasets.ImageFolder(datapath, transform=transform)
    return dset
