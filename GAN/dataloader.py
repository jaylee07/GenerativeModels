import os
import torch
import torch.utils as utils
import torchvision.transforms as transforms
import torchvision.datasets as dset
from dataset import get_data


def load_dataset(opt):
    data_folder = os.path.join(opt.dataroot, opt.dataset)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    transform = transforms.Compose([transforms.Scale(opt.imagesize),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    if opt.dataset == 'MNIST':
        trn_data = dset.MNIST(data_folder, train=True, transform=transform, download=True)
        tst_data = dset.MNIST(data_folder, train=False, transform=transform, download=True)
        n_channels = 1
    elif opt.dataset == 'Fashion-MNIST':
        trn_data = dset.FashionMNIST(data_folder, train=True, transform=transform, download=True)
        tst_data = dset.FashionMNIST(data_folder, train=False, transform=transform, download=True)
        n_channels = 1
    elif opt.dataset == 'CIFAR10':
        trn_data = dset.cifar.CIFAR10(data_folder, train=True, transform=transform, download=True)
        tst_data = dset.cifar.CIFAR10(data_folder, train=False, transform=transform, download=True)
        n_channels = 3
    elif opt.dataset == 'CelebA':
        trn_data = get_data(data_folder, split='train', image_size=opt.imagesize)
        tst_data = get_data(data_folder, split='test', image_size=opt.imagesize)
        n_channels = 3
    trn_loader = utils.data.DataLoader(trn_data, batch_size=opt.batchsize, shuffle=True, num_workers=opt.workers, drop_last=True)
    tst_loader = utils.data.DataLoader(tst_data, batch_size=opt.batchsize, shuffle=False, num_workers=opt.workers, drop_last=True)
    return trn_loader, tst_loader, n_channels