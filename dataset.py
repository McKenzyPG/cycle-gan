import glob 
import random 
import os 
from PIL import Image 
from torch.utils.data import Dataset
import torch
from torchvision import datasets,transforms


def get_mnist_svhn_loader(batch_size, im_size=64):
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    svhn = datasets.SVHN(root='./data', download=True, 
        transform=transforms.Compose([
            transforms.Scale(im_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))
        ]))
    
    mnist = datasets.MNIST(root='./data',train=True,download=True,
        transform=transforms.Compose([
                    transforms.Scale(im_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))]
        ))

    svhn_loader = torch.utils.data.DataLoader(dataset=svhn,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        **kwargs)

    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        **kwargs)
    return svhn_loader, mnist_loader 

