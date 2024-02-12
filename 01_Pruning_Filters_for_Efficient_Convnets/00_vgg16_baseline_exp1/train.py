import sys
sys.path.append("..")
from architecture import VGG16_BN
from utils import loadTrainDataset, loadValDataset, testAccuracy
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.init as init
import math
from pytorch_model_summary import summary
from torch.utils.tensorboard import SummaryWriter


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

    
model = VGG16_BN()
model = model.to(device)

# check # of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"# of total parameters : {total_params}")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"# of trainable parameters : {total_trainable_params}")

# hyper parameteres & optimizer
batch_size = 128
lr = 0.1
lr_decay = 1e-7
weight_decay = 0.0005
momentum = 0.9
epoch_step = 25                   # [lua code] :  drop learning rate every "epoch_step" epochs (optimState.learningRate = optimState.learningRate/2)
epochs = 300
criterion = nn.CrossEntropyLoss() # [lua code] : criterion = cast(nn.CrossEntropyCriterion())
# https://github.com/torch/optim/blob/master/sgd.lua
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=np.arange(epoch_step, epochs, epoch_step), gamma=0.5, verbose=True)
print(np.arange(epoch_step, epochs, epoch_step))

# Training
writer = SummaryWriter('../runs/vgg16_baseline_exp1')
input_tensor = torch.Tensor(128, 3, 32, 32).to(device)
writer.add_graph(model, input_tensor)

val_loader, tesize = loadValDataset()
train_loader, trsize = loadTrainDataset()

num_val_batches = len(val_loader)
print(f"# of validation batches : {num_val_batches}")

## He normal initialization
# kaiming initialization
def init_weights(m) :
    if isinstance(m, nn.Conv2d) :
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        
# check the weight initialization
model.apply(init_weights)

train_acc_list = []
train_loss_list = []
val_acc_list = []
val_loss_list = []

print("\nStart Training ", "-"*70)
for epoch in range(1, epochs+1) :
    train_loss = 0.0
    train_acc = 0.0
    # training
    model.train()
    for i, data in enumerate(train_loader, 0) :
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        ## train loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()        
        train_loss += loss.item()
        ## train acc
        _, predicted = torch.max(outputs.data, 1)
        train_acc += (predicted == labels).sum().item()
    train_loss /= len(train_loader)
    train_acc = (train_acc / trsize) * 100
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    writer.add_scalar('training loss', train_loss, epoch)
    writer.add_scalar('training acc', train_acc, epoch)
    print(f"Epoch : {epoch:03d} | Training loss : {train_loss:.5f} | Training acc : {train_acc:.5f}%")
    
    # validation
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad() :
        for i, data in enumerate(val_loader, 0) :
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            ## val loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            ## val acc
            _, predicted = torch.max(outputs.data, 1)
            val_acc += (predicted == labels).sum().item()
    val_loss /= len(val_loader)
    val_acc = (val_acc / tesize) * 100
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    writer.add_scalar('validation loss', val_loss, epoch)
    writer.add_scalar('validation acc', val_acc, epoch)
    writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
    print(f"Epoch : {epoch:03d} | Validation loss : {val_loss:.5f} | Validation acc : {val_acc:.5f}%")
    lr_scheduler.step()
    
    # save model
    ## best model
    if val_acc == max(val_acc_list) :
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, './checkpoint/best_model.pth')
        print(f"Best model is updated at epoch {epoch:03d} with validation acc {val_acc:.5f}")
    ## every 5 epochs
    if epoch % 5 == 0 :
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f'./checkpoint/epoch_{epoch}.pth')
    print("-" * 70)
print("\nTraining Finished ", "-"*70)

# save train_acc_list, train_loss_list, val_acc_list, val_loss_list using pickle
import pickle
with open('./checkpoint/train_acc_list.pkl', 'wb') as f :
    pickle.dump(train_acc_list, f)
with open('./checkpoint/train_loss_list.pkl', 'wb') as f :
    pickle.dump(train_loss_list, f)
with open('./checkpoint/val_acc_list.pkl', 'wb') as f :
    pickle.dump(val_acc_list, f)
with open('./checkpoint/val_loss_list.pkl', 'wb') as f :
    pickle.dump(val_loss_list, f)
    
# load best model to test
model = VGG16_BN()
model.load_state_dict(torch.load('./checkpoint/best_model.pth')['model_state_dict'])
top1_acc, top5_acc = testAccuracy(model, val_loader)
print("Test Accuracy ", "-"*70)
print(f"Top-1 Accuracy : {top1_acc:.2f} %")
print(f"Top-5 Accuracy : {top5_acc:.2f} %")