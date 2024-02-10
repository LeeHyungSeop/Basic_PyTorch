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
import datetime
import numpy as np

# Load ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## make shorter side of image to [256, 480] for scale augmentation
def shorter_side_resize_train(img) :
    width, height = img.size  
    
    random = np.random.randint(256, 480)
    if width < height :
        new_width = random
        new_height = round(height * (random / width))
    else : 
        new_width = round(width * (random / height))
        new_height = random
    return transforms.Resize((new_height, new_width))(img)
    
def shorter_side_resize_val(img, test_Q) :
    width, height = img.size  
    
    test_Q = test_Q
    if width < height :
        new_width = test_Q
        new_height = round(height * (test_Q / width))
    else : 
        new_width = round(width * (test_Q / height))
        new_height = test_Q
    return transforms.Resize((new_height, new_width))(img)

trainset = torchvision.datasets.ImageFolder(
    root='/home/hslee/Desktop/Datasets/ILSVRC2012_ImageNet/train', 
    transform = transforms.Compose([   
        shorter_side_resize_train,
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])   
)
valset = torchvision.datasets.ImageFolder(
    root='/home/hslee/Desktop/Datasets/ILSVRC2012_ImageNet/val',
    transform=transforms.Compose([
        lambda x: shorter_side_resize_val(x, test_Q=368), # In training, single scale test
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
    trainset, batch_size=mini_batch_size, shuffle=True, num_workers=12, pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    valset, batch_size=mini_batch_size, shuffle=True, num_workers=12, pin_memory=True
)

num_train_examples = len(trainset)
num_val_examples = len(valset)
num_train_batch = len(train_loader) # the number of train batches = 5,005 (1,281,167 / mini_batch_size)
num_val_batch = len(val_loader)     # the number of val batches   =   196 (   34,000 / mini_batch_size)
print(f"# train examples : {num_train_examples}")
print(f"# val examples : {num_val_examples}")
print(f"# train batches : {num_train_batch}")
print(f"# val batches : {num_val_batch}")

# 1000 classes
num_classes = 1000
with open('/home/hslee/Desktop/Datasets/ILSVRC2012_ImageNet/imagenet_class_index.json') as f:
    class_index = json.load(f)
# Get the human-readable class names
class_names = [class_index[str(i)][1] for i in range(num_classes)]
# Print the class names
print(f"0th class : {class_names[0]}")
print(f"999th class : {class_names[-1]}")

writer = SummaryWriter('../../runs/my_resnet34_exp5')
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
    
model = MyResNet34()                
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

# training MyResNet34() model
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
for epoch in range(0, epochs):
    print(f"Current Time : {datetime.datetime.now()}")
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
            print(f"[{epoch+1}, {i+1}th iteration] loss : {running_loss / 1000}", flush=True)
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
    val_loss = val_loss / num_val_batch
    val_acc = 100. * correct / total
    print(f"val loss : {val_loss}")
    print(f"val acc : {val_acc}")
    writer.add_scalar('validation loss', val_loss, epoch+1)
    writer.add_scalar('validation acc', val_acc, epoch+1)
    writer.flush()
    
    lr_scheduler.step()
    writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch+1)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    
    # save best model for inference
    if max(val_acc_list) == val_acc :
        torch.save(model.state_dict(), f"./My_ResNet34_exp5_Checkpoint/best_model.pth")
        print(f"Best model is saved. val acc : {val_acc}%")
            
    # every 5th epoch, save model to resume training
    if (epoch+1) % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"./My_ResNet34_exp5_Checkpoint/epoch_{epoch+1}.pth")
        
print(f"Current Time : {datetime.datetime.now()}")
print('~~~ Training Finished ~~~')    


model = MyResNet34().to(device)
model.load_state_dict(torch.load("./My_ResNet34_exp5_Checkpoint/best_model.pth"))
mini_batch_size = 64
# average the scores at multi scales (224, 256, 384, 480, 640) + Ten Crop
multi_scale_list = [224, 256, 384, 480, 640]
top1_acc, top5_acc = 0, 0
for test_Q in multi_scale_list :
    print(f"test_Q : {test_Q}", "="*50)
    val_transform = transforms.Compose([
        lambda x: shorter_side_resize_val(x, test_Q=test_Q),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    valset = datasets.ImageFolder(root='/home/hslee/Desktop/Datasets/ILSVRC2012_ImageNet/val', transform=val_transform)

    # make valloader
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=mini_batch_size, shuffle=True, num_workers=8, pin_memory=True
    )

    local_top1_correct = 0
    local_top1_total = 0
    local_top1_acc = 0.0
    local_top5_correct = 0
    local_top5_total = 0
    local_top5_acc = 0.0
    model.eval().to(device)
    with torch.no_grad():
        for batch in val_loader:
            input, target = batch
            input, target = input.to(device), target.to(device)
            bs, ncrops, c, h, w = input.size()
            result = model(input.view(-1, c, h, w))
            result_avg = result.view(bs, ncrops, -1).mean(1)
            loss = criterion(result_avg, target)
            _, predicted = torch.max(result_avg, 1)
            
            local_top1_total += target.size(0)
            local_top1_correct += (predicted == target).sum().item()
            
            local_top5_total += target.size(0)
            local_top5_correct += (torch.topk(result_avg, 5, dim=1)[1] == target.view(-1, 1)).sum().sum().item()
            
    local_top1_acc = 100 * local_top1_correct / local_top1_total
    local_top5_acc = 100 * local_top5_correct / local_top5_total
    print(f"local top-1 acc : {local_top1_acc}%")
    print(f"local top-5 acc : {local_top5_acc}%")
    top1_acc += local_top1_acc
    top5_acc += local_top5_acc
    
top1_acc /= len(multi_scale_list)
top5_acc /= len(multi_scale_list)
print(f"average top-1 acc : {top1_acc}%")
print(f"average top-5 acc : {top5_acc}%")