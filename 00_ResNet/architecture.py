import torch
import torch.nn as nn

class BuildingBlock(nn.Module) :
    def __init__(self, in_channels, out_channels) :
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = 1      # In block without down_sampling, stride is 1.
        self.kernel_size = 3 # In block without down_sampling, kernel_size is 3.
        self.padding = 1
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False) 
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
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
        
# Class ResNet32 > Class BuildingBlock, Class BuildingBlockWithDownSample
class MyResNet32(nn.Module) :
    def __init__(self) :
        super().__init__()
        num_classes : int = 10
        
        # 3 x 32 x 32 -> 16 x 32 x 32
        self.layer0 = nn.Sequential( # 5 * 2 + 1 = 11
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
                                    
            BuildingBlock(16, 16), # identity mapping
            BuildingBlock(16, 16), # identity mapping
            BuildingBlock(16, 16), # identity mapping
            BuildingBlock(16, 16), # identity mapping
            BuildingBlock(16, 16), # identity mapping
        )
        # 16 x 32 x 32 -> 32 x 16 x 16
        self.layer1 = nn.Sequential( # 5 * 2 = 10
            BuildingBlockWithDownSample(16, 32), # projection shortcut
            BuildingBlock(32, 32), # identity mapping
            BuildingBlock(32, 32), # identity mapping
            BuildingBlock(32, 32), # identity mapping
            BuildingBlock(32, 32), # identity mapping
        )
        # 32 x 16 x 16 -> 64 x 8 x 8
        self.layer2 = nn.Sequential( # 5 * 2 + 1= 11
            BuildingBlockWithDownSample(32, 64), # projection shortcut
            BuildingBlock(64, 64), # identity mapping
            BuildingBlock(64, 64), # identity mapping
            BuildingBlock(64, 64), # identity mapping
            BuildingBlock(64, 64), # identity mapping
            
            # global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            # fully connected layer
            nn.Flatten(),
            nn.Linear(64, num_classes, bias=True)
        )
        
    def forward(self, x) : 
        ## 3 x 32 x 32 -> 16 x 32 x 32
        x = self.layer0(x)
        ## 16 x 32 x 32 -> 32 x 16 x 16
        x = self.layer1(x)
        ## 32 x 16 x 16 -> 64 x 8 x 8 -> 64 x 1 x 1
        x = self.layer2(x)
        
        return x        
                
            