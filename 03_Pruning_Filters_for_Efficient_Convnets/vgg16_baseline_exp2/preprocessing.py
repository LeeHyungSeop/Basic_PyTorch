import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import kornia
import matplotlib.pyplot as plt
import numpy as np


# https://github.com/szagoruyko/cifar.torch/blob/master/provider.lua

batch_size = 128

trsize = 50000
tesize = 10000

# the image are converted to YUV and mean-std normalization is applied
def rgb_to_yuv(image) :
    kornia.color.rgb_to_yuv(image)
    return image

trainset = torchvision.datasets.CIFAR10(
    root='/home/hslee/Desktop/Datasets/', 
    train=True,
    download=True, 
    transform=transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),        
        rgb_to_yuv,
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=8, pin_memory=True)

valset = torchvision.datasets.CIFAR10(
    root='/home/hslee/Desktop/Datasets/', 
    train=False,
    download=True, 
    transform=transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        rgb_to_yuv,
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                         shuffle=False, num_workers=8, pin_memory=True)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(f"classes : {classes}")
dataiter = iter(train_loader)
images, labels = next(dataiter)
print(f"mini batch shape : {images.shape}")