import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Squeeze-And-Excite模块
class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 骨架
class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride
        
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)
        
        
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        
        
        self.se = SEModule(expand_size) if se else nn.Identity()
        
        
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)
        
        # 跳跃连接
        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )
        
    def forward(self, x):
        skip = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.skip is not None:
            skip = self.skip(skip)
        out = self.act3(out + skip)
        return out

# MobilePoseNet V3
class MobilePoseNetV3(nn.Module):
    def __init__(self, num_keypoints=8):
        super(MobilePoseNetV3, self).__init__()
        # 使用MobileNet V3作为特征提取器
        self.mobilenet_v3 = models.mobilenet_v3_large(pretrained=True)
    
        for param in self.mobilenet_v3.parameters():
            param.requires_grad = False
        
        self.features = nn.Sequential(*list(self.mobilenet_v3.features.children())[:-1])
        
        # PoseNet头部
        last_channel = self.mobilenet_v3.features[-1].out_channels
        self.pose_head = nn.Sequential(
            nn.Conv2d(last_channel, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_keypoints * 2, kernel_size=1, stride=1, padding=0)  # 每个关键点2个坐标(x, y)
        )
    
    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 关键点检测
        x = self.pose_head(x)
        return x

# 初始化模型
num_keypoints = 8  # 假设每个图像有8个关键点
model = MobilePoseNetV3(num_keypoints=num_keypoints)
