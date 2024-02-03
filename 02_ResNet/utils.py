import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import os
from torch.utils.data import random_split

from urllib.request import urlretrieve
import json

import matplotlib.pyplot as plt
import PIL

import numpy as np

def getCifar10(mini_batch_size=128) :
    trainset = torchvision.datasets.CIFAR10(
    root='/home/hslee/Desktop/Datasets', train=True, download=True,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124],
                                std=[0.24703233, 0.24348505, 0.26158768])
        ])    
    )    
    ## Split the training set into training(45k) and validation sets(5k)
    trainset, valset = random_split(trainset, [45000, 5000])
    testset = torchvision.datasets.CIFAR10(
    root='/home/hslee/Desktop/Datasets', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124],
                            std=[0.24703233, 0.24348505, 0.26158768])
        ])
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=mini_batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=valset, batch_size=mini_batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=mini_batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader, trainset, valset, testset

def kaimingHeInitialization(m) :
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.fill_(1)

def testAccuracy(_model, _val_loader) :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    top1_correct = 0
    top1_total = 0
    top1_acc = 0.0
    top5_correct = 0
    top5_total = 0
    top5_acc = 0.0
    
    _model.eval().to(device)
    with torch.no_grad():
        for data in _val_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = _model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            top1_total += labels.size(0)
            top1_correct += (predicted == labels).sum().item()
            
            top5_total += labels.size(0)
            top5_correct += (torch.topk(outputs.data, 5, dim=1)[1] == labels.view(-1, 1)).sum().sum().item()
            
    top1_acc = 100 * top1_correct / top1_total
    top5_acc = 100 * top5_correct / top5_total
    
    return top1_acc, top5_acc

# def getImageNet() :
    