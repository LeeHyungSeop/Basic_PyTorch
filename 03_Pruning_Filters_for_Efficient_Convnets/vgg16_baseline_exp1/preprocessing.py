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

def rgb_to_yuv(image) :
    kornia.color.rgb_to_yuv(image)
    return image

def normalize_y_locally(image):
    # Assuming image is in YUV format
    image[0, :, :] = kornia.enhance.normalize(image[0, :, :])
    return image

def normalize_uv_globally(image):
    # Assuming image is in YUV format
    mean_u = image[1, :, :].mean()
    std_u = image[1, :, :].std()
    image[1, :, :] = (image[1, :, :] - mean_u) / std_u

    mean_v = image[2, :, :].mean()
    std_v = image[2, :, :].std()
    image[2, :, :] = (image[2, :, :] - mean_v) / std_v

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
        # normalize_y_locally,
        # normalize_uv_globally,
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