import sys
sys.path.append("..")
from architecture import VGG16_BN
from utils import *
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import kornia
import matplotlib.pyplot as plt
import numpy as np
from pytorch_model_summary import summary
from torch.nn.parameter import Parameter
import copy
import pickle

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# load best model's parameter
model = VGG16_BN()
checkpoint = torch.load('../00_vgg16_baseline_exp1/checkpoint/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

new_pruned_model = getOneShotPrunedModel(model)

val_loader, tesize = loadValDataset()
train_loader, trsize = loadTrainDataset()

trained_model, best_val_acc = retrainNewPrunedModel(new_pruned_model, train_loader, val_loader)
print(f"Best Top-1 validation acc : {best_val_acc:.2f}%")
# save best model's parameter
torch.save({
    'model_state_dict': trained_model.state_dict(),
    'best_val_acc': best_val_acc
}, f'./checkpoint/best_model.pth')