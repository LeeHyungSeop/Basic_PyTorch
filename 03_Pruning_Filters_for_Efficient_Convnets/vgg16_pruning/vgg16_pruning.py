import sys
sys.path.append("..")
from architecture2 import VGG16_BN
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
    return val_loader

def testAccuracy(_new_pruned_model, _val_loader) :
    top1_correct = 0
    top1_total = 0
    top1_acc = 0.0
    top5_correct = 0
    top5_total = 0
    top5_acc = 0.0
    
    _new_pruned_model.eval()
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
    print("\t"*25 , f"Top-1 Accuracy : {top1_acc:.2f} %")
    print("\t"*25 ,  f"Top-5 Accuracy : {top5_acc:.2f} %")
    
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

def showNewPrunedModel(_model) :
    print(summary(_model, torch.zeros((1, 3, 32, 32)).to(device), show_input=True))
        

# conv1 ~ conv13까지 돌며 각각 10, 20, 30, 40, 50, 60, 70, 80, 90, 95% pruning 후 accuracy 측정
filters_pruned_away_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95] # (%)
layer = 0
val_loader = loadValDataset()
top1_acc_list = {}
top5_acc_list = {}

# load best model's parameter
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
            showNewPrunedModel(new_pruned_model)
            
            # Top-1 accuracy, Top-5 accuracy
            # print conv, pruned rate, #pruned channels
            print("\t"*20, f"[conv{layer+1}] pruned rate : {pruned_rate}%, #pruned channels : {num_prune_channels}")
            top1_acc, top5_acc = testAccuracy(new_pruned_model, val_loader)
            top1_acc_list[layer].append(top1_acc)
            top5_acc_list[layer].append(top5_acc)

        layer += 1
        
# save accuracy using pickle
with open('../Figure2/b/top1_acc_list.pkl', 'wb') as f :
    pickle.dump(top1_acc_list, f)
with open('../Figure2/b/top5_acc_list.pkl', 'wb') as f :
    pickle.dump(top5_acc_list, f)    