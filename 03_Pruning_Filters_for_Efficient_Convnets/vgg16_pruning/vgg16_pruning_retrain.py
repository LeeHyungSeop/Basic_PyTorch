import sys
sys.path.append("..")
from architecture2 import VGG16_BN
from vgg16_pruning import rgb_to_yuv, loadValDataset, testAccuracy, getPrunedNetwork
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
checkpoint = torch.load('../vgg16_baseline_exp4/checkpoint/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

def loadTrainDataset() :
    batch_size = 128
    # load training dataset
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
    return train_loader

def retrainNewPrunedModel(_model, _train_loader) :
    lr = 0.001
    epochs = 40
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    _model = _model.to(device)
    _model.train()
    for epoch in range(1, epochs+1) :
        training_loss = 0.0
        for i, data in enumerate(_train_loader, 0) :
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = _model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        print(f"Epoch : {epoch:03d}, Training loss : {training_loss/len(_train_loader):.5f}")
    
    return _model
    
            

# conv1 ~ conv13까지 돌며 각각 10, 20, 30, 40, 50, 60, 70, 80, 90, 95% pruning 후 accuracy 측정
filters_pruned_away_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95] # (%)
layer = 0
val_loader = loadValDataset()
train_loader = loadTrainDataset()
top1_acc_list = {}
top5_acc_list = {}

model = VGG16_BN()
checkpoint = torch.load('../vgg16_baseline_exp4/checkpoint/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

for param in model.modules() :
    if isinstance(param, torch.nn.Conv2d) :
        top1_acc_list[layer] = []
        top5_acc_list[layer] = []
        print(f"="*40," conv{layer+1} ", "="*40)
        
        for pruned_rate in filters_pruned_away_list :
            model_copy = copy.deepcopy(model)
            num_prune_channels = round(param.weight.data.shape[0] * pruned_rate / 100)
            print(f"\n----- pruned rate : {pruned_rate}%, #pruned channels : {num_prune_channels} -----")
            
            # new_pruned_model architecture
            new_pruned_model = getPrunedNetwork(model_copy, layer, num_prune_channels)
            
            # train new_pruned_model
            ## For retraining, we use a constant learning rate 0.001 and retrain 40 epochs for CIFAR-10 and 20 epochs for ImageNet, which represents one-fourth of the original training epochs.
            trained_new_pruned_model = retrainNewPrunedModel(new_pruned_model, train_loader)
            
            # Top-1 accuracy, Top-5 accuracy
            # print conv, pruned rate, #pruned channels
            print("\t"*20, f"[conv{layer+1}] pruned rate : {pruned_rate}%, #pruned channels : {num_prune_channels}")
            top1_acc, top5_acc = testAccuracy(trained_new_pruned_model, val_loader)
            top1_acc_list[layer].append(top1_acc)
            top5_acc_list[layer].append(top5_acc)

        layer += 1
        
# save accuracy using pickle
with open('../Figure2/c/top1_acc_list.pkl', 'wb') as f :
    pickle.dump(top1_acc_list, f)
with open('../Figure2/c/top5_acc_list.pkl', 'wb') as f :
    pickle.dump(top5_acc_list, f)    