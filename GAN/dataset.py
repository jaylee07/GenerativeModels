import os
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
import torchvision.datasets as dset

from folder import ImageFolder

def get_data(root, split, image_size):
    dataset_name = os.path.basename(root)
    image_root = os.path.join(root, 'splits', split)

    if dataset_name in ['CelebA']:
        dataset = ImageFolder(root=image_root, transform=transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Scale(image_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    else:
        dataset = ImageFolder(root=image_root, transform=transforms.Compose([
            transforms.Scale(image_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
        
    return dataset
    #data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=int(num_workers))
    #data_loader.shape = [int(num) for num in dataset[0][0].size()]
    #return data_loader