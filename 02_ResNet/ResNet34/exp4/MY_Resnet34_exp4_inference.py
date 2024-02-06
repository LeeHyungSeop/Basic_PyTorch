import torch
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


class BuildingBlock(nn.Module) :
    def __init__(self, in_channels, out_channels) :
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = 1      # In block without down_sampling, stride is 1.
        self.kernel_size = 3 # In block without down_sampling, kernel_size is 3.
        self.padding = 1
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False) 
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x) : 
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity # identity mapping
        out = self.relu2(out)
        
        return out
    
class BuildingBlockWithDownSample(nn.Module) :
    def __init__(self, in_channels, out_channels) :
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 3
        self.down_sampling_kernel_size = 1
        self.conv1_stride = 2 # In block with down_sampling, conv1's stride is 2.
        self.conv2_stride = 1 # In block with down_sampling, conv2's stride is 1.
        self.padding = 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.conv1_stride, padding=self.padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, stride=self.conv2_stride, padding=self.padding, bias=False) 
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)
        
        # (projection shortcut) : H, W of activation map are down_sampled, C of activation map is up_sampled
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.down_sampling_kernel_size, stride=self.conv1_stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
    
    def forward(self, x) : 
        identity = self.downsample(x) # projection shortcut 
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity       # projection shortcut
        out = self.relu2(out) # block output
        
        return out
        
# Class ResNet34 > Class BuildingBlock, Class BuildingBlockWithDownSample
class MyResNet34(nn.Module) :
    def __init__(self) :
        super().__init__()
        num_classes : int = 1000
        
        self.layer0 = nn.Sequential( # 2
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), # affine : True ? gamma, beta, track_running_stats : True ? running_mean, running_var
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential( # 3 * 2 = 6
            BuildingBlock(64, 64), # identity mapping
            BuildingBlock(64, 64), # identity mapping
            BuildingBlock(64, 64), # identity mapping
        )
        self.layer2 = nn.Sequential( # 4 * 2 = 8
            BuildingBlockWithDownSample(64, 128), # projection shortcut
            BuildingBlock(128, 128), # identity mapping
            BuildingBlock(128, 128), # identity mapping
            BuildingBlock(128, 128), # identity mapping
        )
        self.layer3 = nn.Sequential( # 5 * 2 = 10
            BuildingBlockWithDownSample(128, 256), # projection shortcut
            BuildingBlock(256, 256),
            BuildingBlock(256, 256),
            BuildingBlock(256, 256),
            BuildingBlock(256, 256),
            BuildingBlock(256, 256),
        )
        self.layer4 = nn.Sequential( # 3 * 2 = 6
            BuildingBlockWithDownSample(256, 512), # projection shortcut
            BuildingBlock(512, 512),
            BuildingBlock(512, 512),
        )
        self.layer5 = nn.Sequential( # 2
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes, bias=True),
        )
        
    def forward(self, x) : 
        ## Input size : 224 x 224 x 3 -> 112 x 112 x 64 -> Output size : 56 x 56 x 64
        x = self.layer0(x)
        ## Input size : 56 x 56 x 64, Output size : 56 x 56 x 64
        x = self.layer1(x)
        ## Input size : 56 x 56 x 64, Output size : 28 x 28 x 128
        x = self.layer2(x)
        ## Input size : 28 x 28 x 128, Output size : 7 x 7 x 512
        x = self.layer3(x)
        ## Input size : 7 x 7 x 512, Output size : 1 x 1 x 512
        x = self.layer4(x)
        ## Input size : (mini_batch_size) x 1 x 1 x 512 ->  1 x 1 x 512 -> Output size : 1 x 1 x 1000
        x = self.layer5(x)
        
        return x        
                    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
criterion = nn.CrossEntropyLoss()
    
def shorter_side_resize_val(img) :
    width, height = img.size  
    # print(f"before : {img.size}")
    # resized하고 나서도, 한 면이 224보다 작으면 randomcrop(224)를 할 수 없으니
    new_height, new_width = height, width
    if width < 224 and height < 224 :
        if height < width :
            new_height = 224
            new_width = round(width * (224 / height))
            transforms.Resize((new_height, new_width))(img)
        else :
            new_height = round(height * (224 / width))
            new_width = 224
            transforms.Resize((new_height, new_width))(img)
        # print(f"after 1 : {(new_height, new_width)}")
    test_Q = 368 # (256 + 480) / 2
    if height < width :
        # print(f"after 2 : {(test_Q, new_width)}")
        return transforms.Resize((test_Q, new_width))(img)
    else :
        # print(f"after 2 : {(new_height, test_Q)}")
        return transforms.Resize((new_height, test_Q))(img)

valset = torchvision.datasets.ImageFolder(
    root='/home/hslee/Desktop/Datasets/ILSVRC2012_ImageNet/val',
    transform=transforms.Compose([
        shorter_side_resize_val,
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
)

mini_batch_size = 64

# make valloader
val_loader = torch.utils.data.DataLoader(
    valset, batch_size=mini_batch_size, shuffle=True, num_workers=8, pin_memory=True
)
num_val_batch = len(val_loader)     # the number of val batches   =   196 (   50,000 / mini_batch_size)
print(f"num_val_batch : {num_val_batch}")

model = MyResNet34().to(device)
model.load_state_dict(torch.load("./My_ResNet34_exp4_Checkpoint/best_model.pth"))

# Top-1 
val_loss = 0.0
correct = 0
total = 0
model.eval().to(device)
with torch.no_grad():
    for batch in val_loader:
        input, target = batch
        input, target = input.to(device), target.to(device)
        bs, ncrops, c, h, w = input.size()
        result = model(input.view(-1, c, h, w))
        result_avg = result.view(bs, ncrops, -1).mean(1)
        loss = criterion(result_avg, target)
        
        val_loss += loss.item()
        _, predicted = result_avg.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
print("[Top-1]")
print(f"val loss : {val_loss / num_val_batch}")
print(f"val acc : {100. * correct / total}")
print(f"error rate : {100. * (total - correct) / total}")

# Top-5
val_loss = 0.0
correct = 0
total = 0
model.eval().to(device)
with torch.no_grad():
    for batch in val_loader:
        input, target = batch
        input, target = input.to(device), target.to(device)
        bs, ncrops, c, h, w = input.size()
        result = model(input.view(-1, c, h, w))
        result_avg = result.view(bs, ncrops, -1).mean(1)
        loss = criterion(result_avg, target)
        
        val_loss += loss.item()
        _, predicted = result_avg.topk(5, 1, True, True)
        total += target.size(0)
        correct += predicted.eq(target.view(-1, 1).expand_as(predicted)).sum().item()   
print("[Top-5]")          
print(f"val loss : {val_loss / num_val_batch}")
print(f"val acc : {100. * correct / total}")
print(f"error rate : {100. * (total - correct) / total}")