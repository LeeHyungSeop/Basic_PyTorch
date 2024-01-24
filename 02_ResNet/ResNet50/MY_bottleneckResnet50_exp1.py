# Training details
# - The image is resized with its shorter side randomly sampled in [256, 480] for scale augmentation [41].
# - A 224×224 crop is randomly sampled from an image or its
# - horizontal flip, with the per-pixel mean subtracted [21].
# - The standard color augmentation in [21] is used. 
# - We adopt batch normalization (BN) [16] right after each convolution and before activation, following [16]. 
# - We initialize the weights as in [13] and train all plain/residual nets from scratch. 
# - We use SGD with a mini-batch size of 256. 
# - The learning rate starts from 0.1 and is divided by 10 when the error plateaus,
# and the models are trained for up to 60 × 104 iterations. 
# - We use a weight decay of 0.0001 and a momentum of 0.9.
# - We do not use dropout

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
import random
import json

# Load ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## make squre size image with shorter side randomly sampled in [256, 480]
def shorter_side_resize(img):
    w, h = img.size
    if w < h:
        new_w = torch.randint(256, 480, (1,)).item()
        new_h = int(new_w)
    else:
        new_h = torch.randint(256, 480, (1,)).item()
        new_w = int(new_h)
    return transforms.Resize((new_h, new_w))(img)

trainset = torchvision.datasets.ImageFolder(
    root='/home/hslee/Desktop/Datasets/ILSVRC2012_ImageNet/train', 
    transform = transforms.Compose([   
        shorter_side_resize,
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])   
)
valset = torchvision.datasets.ImageFolder(
    root='/home/hslee/Desktop/Datasets/ILSVRC2012_ImageNet/val',
    transform=transforms.Compose([
        transforms.Resize(size=256),
        # transforms.Resize(size=(256 + 480) // 2),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize,
    ])
)

mini_batch_size = 256
print(f"mini_batch_size : {mini_batch_size}")
num_threads = torch.get_num_threads()
print(f"# threads : {num_threads}")
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=mini_batch_size, shuffle=True, num_workers=num_threads, pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    valset, batch_size=mini_batch_size, shuffle=True, num_workers=num_threads, pin_memory=True
)

num_train_examples = len(trainset)
num_val_examples = len(valset)
num_train_batch = len(train_loader) # the number of train batches = 5,005 (1,281,167 / mini_batch_size)
num_val_batch = len(val_loader)     # the number of val batches   =   196 (   50,000 / mini_batch_size)
print(f"# train examples : {num_train_examples}")
print(f"# val examples : {num_val_examples}")
print(f"# train batches : {num_train_batch}")
print(f"# val batches : {num_val_batch}")

# check the structure of train_loader
print("check the structure of train_loader ----------------------")
for images, labels in train_loader:
    print(images.shape)
    print(labels.shape)
    break

# 1000 classes
num_classes = 1000
with open('/home/hslee/Desktop/Datasets/ILSVRC2012_ImageNet/imagenet_class_index.json') as f:
    class_index = json.load(f)
# Get the human-readable class names
class_names = [class_index[str(i)][1] for i in range(num_classes)]
# Print the class names
print(f"0th class : {class_names[0]}")
print(f"999th class : {class_names[-1]}")

writer = SummaryWriter('../runs/my_bottleneck_resnet50_exp1')
# modeling ResNet50 by myself
class BottleneckBuildingBlock(nn.Module) :
    def __init__(self, in_channels, mid_channels, out_channels, down_sampling=False, layer_index=0) :
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.stride = 1      # In block without down_sampling, stride is 1.
        self.down_sampling_stride = 1
        self.padding = 1
        self.down_sampling = down_sampling
        
        if self.down_sampling :
            self.down_sampling_kernel_size = 1
            if layer_index != 1 :
                self.down_sampling_stride = 2
            # (projection shortcut) : H, W of activation map are down_sampled, C of activation map is up_sampled
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=self.down_sampling_kernel_size, stride=self.down_sampling_stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            )
        
        self.conv1x1_1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=self.stride, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv3x3_2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=self.down_sampling_stride, padding=self.padding, bias=False) 
        self.bn2 = nn.BatchNorm2d(mid_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1x1_3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=self.stride, bias=False) 
        self.bn3 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = nn.ReLU(inplace=True)
        
    def forward(self, x) : 
        identity = x
        
        out = self.conv1x1_1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv3x3_2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv1x1_3(out)
        out = self.bn3(out)
        
        if self.down_sampling :
            identity = self.downsample(x)
        out += identity # identity mapping
        out = self.relu3(out)
        
        return out

# Class ResNet50 > Class BottleneckBuildingBlock, Class BottleneckBuildingBlockWithDownSample
class MyResNet50(nn.Module) :
    def __init__(self) :
        super().__init__()
        num_classes : int = 1000
        
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), # affine : True ? gamma, beta, track_running_stats : True ? running_mean, running_var
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(
            BottleneckBuildingBlock(64, 64, 256, down_sampling=True, layer_index=1), 
            BottleneckBuildingBlock(256, 64, 256, layer_index=1), # identity mapping
            BottleneckBuildingBlock(256, 64, 256, layer_index=1), # identity mapping
        )
        self.layer2 = nn.Sequential(
            BottleneckBuildingBlock(256, 128, 512, down_sampling=True, layer_index=2), 
            BottleneckBuildingBlock(512, 128, 512), # identity mapping
            BottleneckBuildingBlock(512, 128, 512), # identity mapping
            BottleneckBuildingBlock(512, 128, 512), # identity mapping
        )
        self.layer3 = nn.Sequential(
            BottleneckBuildingBlock(512, 256, 1024, down_sampling=True, layer_index=3),
            BottleneckBuildingBlock(1024, 256, 1024), # identity mapping
            BottleneckBuildingBlock(1024, 256, 1024),
            BottleneckBuildingBlock(1024, 256, 1024),
            BottleneckBuildingBlock(1024, 256, 1024),
            BottleneckBuildingBlock(1024, 256, 1024),
        )
        self.layer4 = nn.Sequential(
            BottleneckBuildingBlock(1024, 512, 2048, down_sampling=True, layer_index=4),
            BottleneckBuildingBlock(2048, 512, 2048),
            BottleneckBuildingBlock(2048, 512, 2048),
        )
        self.layer5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes, bias=True),
        )
        
    def forward(self, x) : 
        ## Input size : 224 x 224 x 3 -> 112 x 112 x 64 -> Output size : 56 x 56 x 64
        x = self.layer0(x)
        ## Input size : 56 x 56 x 64, Output size : 56 x 56 x 256
        x = self.layer1(x)
        ## Input size : 56 x 56 x 256, Output size : 28 x 28 x 512
        x = self.layer2(x)
        ## Input size : 28 x 28 x 512, Output size : 7 x 7 x 1024
        x = self.layer3(x)
        ## Input size : 7 x 7 x 1024, Output size : 1 x 1 x 2048
        x = self.layer4(x)
        ## Input size : (mini_batch_size) x 1 x 1 x 2048 ->  1 x 1 x 2048 -> Output size : 1 x 1 x 1000
        x = self.layer5(x)
        return x        
        
model = MyResNet50()
print(model.eval)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")
model = model.to(device)

# he initialization, not use bias
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.fill_(1)
model.apply(init_weights)

# hyper parameters
lr = 0.1
momentum = 0.9
L2 = 0.0001
# the number of iterations at 1 epoch
num_iters = (len(trainset) // mini_batch_size) + 1
epochs = 120
print(f"num_iters at 1 epoch: {num_iters}")
total_num_iters = num_iters * epochs
print(f"total num_iters: {total_num_iters}")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=L2)        
lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1, verbose=True) # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html

# check # of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"# of total parameters : {total_params}")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"# of trainable parameters : {total_trainable_params}")

# record model to tensorboard
input_tensor = trainset[0][0].unsqueeze(0).to(device)
writer.add_graph(model, input_tensor)

# for visualization, record training loss and val loss
train_loss_list = []
val_loss_list = []
val_acc_list = []

# training MyResNet50() model
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
for epoch in range(0, epochs):
    model.train().to(device) 
    running_loss = 0.0
    print(f"{epoch+1} / {epochs} epoch ----------------------------------------")
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # every 1,000 iteration
        running_loss += loss.item()
        if i % 1000 == 999:
            print(f"[{epoch+1}, {i+1}th iteration] loss : {running_loss / 1000}")
            writer.add_scalar('training loss', running_loss / 1000, epoch * num_iters + i)
            running_loss = 0.0
            
    # every epoch
    # validation loss, accuracy
    val_loss = 0.0
    correct = 0
    total = 0
    model.eval().to(device)
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    print(f"val loss : {val_loss / num_val_batch}")
    print(f"val acc : {100. * correct / total}")
    writer.add_scalar('validation loss', val_loss / num_val_batch, epoch+1)
    writer.add_scalar('validation acc', 100. * correct / total, epoch+1)
    writer.flush()
    
    lr_scheduler.step()
    writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch+1)
    val_loss_list.append(val_loss / num_val_batch)
    val_acc_list.append(100. * correct / total)
    
    # save best model for inference
    if val_loss / num_val_batch == min(val_loss_list):
        PATH = f"./MY_bottleneckResnet50_exp1_Checkpoint/best_model.pth"
        torch.save(model.state_dict(), PATH)
            
    # every 5th epoch, save model to resume training
    if (epoch+1) % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"./MY_bottleneckResnet50_exp1_Checkpoint/epoch_{epoch+1}.pth")
        

print('~~~ Training Finished ~~~')    
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                
# In Paper, top-1 error., top-5 error. (10-crop testing)
# - `top-1 acc.` : 77.15 %
# - `top-5 acc.` : 93.29 %

print(f"[In Paper] top-1 acc : 77.15 %, top-5 acc : 93.29 %")

mini_batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
num_val_batch = len(val_loader)     # the number of val batches   =   196 (   50,000 / mini_batch_size)
print(f"num_val_batch : {num_val_batch}")

## exp1 : transforms.Resize(256),
print("[inference exp1] : transforms.Resize(256)")
valset = torchvision.datasets.ImageFolder(
    root='/home/hslee/Desktop/Datasets/ILSVRC2012_ImageNet/val',
    transform=transforms.Compose([
        # 10-crop
        transforms.Resize(256), # exp1
        # transforms.Resize((256 + 480) // 2), # exp2
        transforms.TenCrop(224),    
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
)
val_loader = torch.utils.data.DataLoader(
    valset, batch_size=mini_batch_size, shuffle=True, num_workers=8, pin_memory=True
)

model = MyResNet50()
model.load_state_dict(torch.load("./MY_bottleneckResnet50_exp1_Checkpoint/best_model.pth"))

### Top-1 val loss, acc, error rate
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
print(f"val acc : {100. * correct / total}%")
print(f"error rate : {100. * (total - correct) / total}%")

### Top-5 val loss, acc, error rate
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
print(f"val acc : {100. * correct / total}%")
print(f"error rate : {100. * (total - correct) / total}%")

## exp 2 : transforms.Resize((256 + 480) // 2),
print("[inference exp2] : transforms.Resize((256 + 480) // 2)")
valset = torchvision.datasets.ImageFolder(
    root='/home/hslee/Desktop/Datasets/ILSVRC2012_ImageNet/val',
    transform=transforms.Compose([
        # 10-crop
        # transforms.Resize(256),
        transforms.Resize(368),
        transforms.TenCrop(224),    
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
)
val_loader = torch.utils.data.DataLoader(
    valset, batch_size=mini_batch_size, shuffle=True, num_workers=8, pin_memory=True
)

### Top-1 val loss, acc, error rate
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
print(f"val acc : {100. * correct / total}%")
print(f"error rate : {100. * (total - correct) / total}%")

### Top-5 val loss, acc, error rate
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
print(f"val acc : {100. * correct / total}%")
print(f"error rate : {100. * (total - correct) / total}%")