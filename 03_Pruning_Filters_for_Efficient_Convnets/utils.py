import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import kornia
import matplotlib.pyplot as plt
import numpy as np
from pytorch_model_summary import summary
from torch.nn.parameter import Parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rgb_to_yuv(image) :
    kornia.color.rgb_to_yuv(image)
    return image

def loadValDataset() :
    batch_size = 128
    # load validation dataset
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
    
    tesize = valset.__len__()
    print(f"# test datas : {tesize}")
    return val_loader, tesize

def testAccuracy(_new_pruned_model, _val_loader) :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    top1_correct = 0
    top1_total = 0
    top1_acc = 0.0
    top5_correct = 0
    top5_total = 0
    top5_acc = 0.0
    
    _new_pruned_model.eval().to(device)
    with torch.no_grad():
        for data in _val_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = _new_pruned_model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            top1_total += labels.size(0)
            top1_correct += (predicted == labels).sum().item()
            
            top5_total += labels.size(0)
            top5_correct += (torch.topk(outputs.data, 5, dim=1)[1] == labels.view(-1, 1)).sum().sum().item()
            
    top1_acc = 100 * top1_correct / top1_total
    top5_acc = 100 * top5_correct / top5_total
    
    return top1_acc, top5_acc

def getPrunedNetwork(_model, _layer, _num_prune_channels) :
    
    ## 1. parsing _models > _layer > weight, bias, bn_gamma, bn_beta
    current_layer = getattr(_model, f'conv{_layer+1}') 
    conv_layer = current_layer[0]
    bn_layer = current_layer[1]
    weight = conv_layer.weight
    bias = conv_layer.bias
    bn_gamma = bn_layer.weight
    bn_beta = bn_layer.bias
    bn_running_mean = bn_layer.running_mean
    bn_running_var = bn_layer.running_var
    print(f"weight.shape : {weight.shape}")
    print(f"bias.shape : {bias.shape}")
    print(f"bn_gamma.shape : {bn_gamma.shape}")
    print(f"bn_beta.shape : {bn_beta.shape}")
    print(f"bn_running_mean.shape : {bn_running_mean.shape}")
    print(f"bn_running_var.shape : {bn_running_var.shape}")
    is_last_conv_layer = False
    if _layer == 12 :
        is_last_conv_layer = True    
    if is_last_conv_layer :
        next_layer = getattr(_model, f'fc1')
        next_fc_layer = next_layer[1] # nn.Linear(512, 512),
        next_bn_layer = next_layer[2] # nn.BatchNorm1d(512, ...),
    else : 
        next_layer = getattr(_model, f'conv{_layer+2}')
        next_conv_layer = next_layer[0]
        next_bn_layer = next_layer[1]
        next
    
    ## 2. sort the filter with L1 norm (desending order)
    ## bias 값이 매우 작아서 weight pruning index와 동일하게 pruning시킬 것임. (평균적으로 bias가 가장 큰 filter의 값이 약 1e-06 정도임. 최소값은 1e-08)
    sorted_weight, sorted_weight_indices = torch.sort(torch.sum(torch.abs(weight), dim=(1, 2, 3)), descending=True)
    print(f"sorted_weight_indices : {sorted_weight_indices}")
    saving_filter_idices = sorted_weight_indices[0 : -_num_prune_channels]
    print(f"saving_filter_idices : {saving_filter_idices}")
    
    pruned_weight, pruned_bias, \
    pruned_bn_gamma, pruned_bn_beta, \
    pruned_bn_running_mean, pruned_bn_running_var = \
        weight[saving_filter_idices], bias[saving_filter_idices], \
        bn_gamma[saving_filter_idices], bn_beta[saving_filter_idices], \
        bn_running_mean[saving_filter_idices], bn_running_var[saving_filter_idices]  
    print(f"pruned_weight.shape : {pruned_weight.shape}")
    print(f"pruned_bias.shape : {pruned_bias.shape}")
    print(f"pruned_bn_gamma.shape : {pruned_bn_gamma.shape}")
    print(f"pruned_bn_beta.shape : {pruned_bn_beta.shape}")
    print(f"pruned_bn_running_mean.shape : {pruned_bn_running_mean.shape}")
    print(f"pruned_bn_running_var.shape : {pruned_bn_running_var.shape}")
    
    ### next fc layer에 대한 처리 (# ex. conv13 pruned (512, 512, 3, 3) to (496, 512, 3, 3)? fc1 pruned (512, 512) to (512, 496))
    ### (output channel, input channel)
    ### next conv layer에 대한 처리 (# ex. conv1 pruned (64, 3, 3, 3) to (58, 3, 3, 3)? conv2 pruned (64, 64, 3, 3) to (64, 58, 3, 3))
    ### (output channel, input channel, kernel size, kernel size)
    if is_last_conv_layer :
        pruned_next_weight = next_fc_layer.weight[:, saving_filter_idices]
        print(f"pruned_next_weight.shape : {pruned_next_weight.shape}")
    else :
        pruned_next_weight = next_conv_layer.weight[:, saving_filter_idices, :, :]
        print(f"pruned_next_weight.shape : {pruned_next_weight.shape}")
        
    
    # 3. Pruning _model's _layer's weight, bias, bn_gamma, bn_beta with saving_filter_indices 
    with torch.no_grad():
        conv_layer.weight = Parameter(pruned_weight)
        conv_layer.bias = Parameter(pruned_bias)
        bn_layer.weight = Parameter(pruned_bn_gamma)
        bn_layer.bias = Parameter(pruned_bn_beta)
        bn_layer.running_mean.data = Parameter(pruned_bn_running_mean)
        bn_layer.running_var.data = Parameter(pruned_bn_running_var)
        if is_last_conv_layer :
            next_fc_layer.weight = Parameter(pruned_next_weight)
        else :
            next_conv_layer.weight = Parameter(pruned_next_weight)
    
    return _model

def getOneShotPrunedModel(_model) :
    layer = 0
    fc_layer = 0
    filters_pruned_away_list = [50, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50]
    for param in _model.modules() :
        # if conv layer
        if isinstance(param, torch.nn.Conv2d) :
            print(f"="*40,f" conv{layer+1} ", "="*40)
            if filters_pruned_away_list[layer] == 0 :
                print(f"pruned_rate : {filters_pruned_away_list[layer]}%")
                print(f"no filter pruned")
                layer += 1
                continue
            weight = param.weight.data
            pruned_rate = filters_pruned_away_list[layer]
            num_prune_channels = round(weight.shape[0] * pruned_rate / 100)
            print(f"pruned_rate : {pruned_rate}%")
            print(f"num_prune_channels : {num_prune_channels}")
            
            _model = getPrunedNetwork(_model, layer, num_prune_channels)
            layer += 1
        
    return _model

def showNewPrunedModel(_model) :
    print(summary(_model, torch.zeros((1, 3, 32, 32)).to(device), show_input=True))
    
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
    
    trsize = trainset.__len__()
    print(f"# train datas : {trsize}")
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    print(f"mini batch shape : {images.shape}")
    return train_loader, trsize

def retrainNewPrunedModel(_model, _train_loader, _val_loader) :
    lr = 0.001
    epochs = 40
    # epochs = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    _model = _model.to(device)
    maximum_acc = 0.0
    for epoch in range(1, epochs+1) :
        training_loss = 0.0
        _model.train().to(device)
        for i, data in enumerate(_train_loader, 0) :
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = _model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        print(f"Epoch : {epoch:03d}, Training loss : {training_loss/len(_train_loader):.5f}")
        # validation
        top1_acc, top5_acc = testAccuracy(_model, _val_loader)
        print(f"Epoch : {epoch:03d}, Top-1 accuracy : {top1_acc:.2f}%, Top-5 accuracy : {top5_acc:.2f}%")
        if top1_acc > maximum_acc :
            maximum_acc = top1_acc
            best_model = _model
    
    return best_model, maximum_acc

def getCifar10Classes() :
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return classes 