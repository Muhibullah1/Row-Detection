import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchvision import transforms,models

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        size = x.size()[2:]
        
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(x)))
        conv3 = self.relu(self.bn3(self.conv3(x)))
        conv4 = self.relu(self.bn4(self.conv4(x)))
        
        pool = self.pool(x)
        pool = self.relu(self.bn5(self.conv5(pool)))
        pool = nn.functional.interpolate(pool, size=size, mode='bilinear', align_corners=False)
        
        out = torch.cat([conv1, conv2, conv3, conv4, pool], dim=1)
        out = self.relu(self.bn_out(self.conv_out(out)))
        
        return out
        
class Encoder(nn.Module):
    def __init__(self):
        super (Encoder, self).__init__()
        resnet18 = models.resnet34(pretrained=True)
        '''
        for param in resnet18.parameters():
            param.requires_grad_(False)
        '''
        self.layer1 = nn.Sequential(*list(resnet18.children())[:5])
        self.layer2 = nn.Sequential (*list(resnet18.children())[5:6])
        self.layer3 = nn.Sequential(*list(resnet18.children())[6:7])
        self.layer4 = nn.Sequential(*list(resnet18.children())[7:8])
    def forward(self,x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer1,layer2,layer3,layer4

class BasicBlock(nn.Module):
    def __init__(self,ip_chnl, op_chnl):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(ip_chnl,op_chnl,3,padding=1)
        self.bn1 = nn.BatchNorm2d(op_chnl)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(op_chnl,op_chnl,3,padding=1)
        self.bn2 = nn.BatchNorm2d(op_chnl)
    def forward (self,x):
        #ip = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #x += ip
        return x 
    
class Upsample(nn.Module):
    def __init__(self,ip_chnl,op_chnl,filter_size=(3,3),stride=2,padding=(0,0),output_padding= (0,0)):
        super (Upsample,self).__init__()
        self.convT = nn.ConvTranspose2d(ip_chnl,op_chnl,filter_size,stride=stride,
                               padding=padding,output_padding=output_padding) 
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(op_chnl)
        self.conv = BasicBlock(op_chnl,op_chnl)
    def forward(self,x):
        x = self.convT(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x
    
class FPN(nn.Module):
    def __init__(self,ip_chnl,upsample):
        super (FPN,self).__init__()
        self.conv1 = nn.Conv2d(ip_chnl,64,3,padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=upsample)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.upsample(x)
        return x

class CropRowDetectionModel(nn.Module):
    def __init__(self, ip_channel, op_channel):
        super(CropRowDetectionModel, self).__init__()
        self.downsample = Encoder()
        self.aspp = ASPP(512, 256)  # Add ASPP module
        self.up1 = Upsample(256, 256, padding=(1,1), output_padding=(1,1))  # Changed input channels to 256
        self.up2 = Upsample(256, 128, padding=(1,1), output_padding=(1,1))
        self.up3 = Upsample(128, 64, padding=(1,1), output_padding=(1,1))
        self.up4 = Upsample(64, 64, padding=(1,1), output_padding=(1,1))
        self.fpn1 = FPN(256, 8)
        self.fpn2 = FPN(128, 4)
        self.fpn3 = FPN(64, 2)
        self.fpn4 = FPN(64, 1)
        self.up5 = Upsample(256, 64, filter_size=(2,2))
        self.conv1 = BasicBlock(64, 32)
        self.conv2 = nn.Conv2d(32, op_channel, 1)

    def forward(self, x):
        x1, x2, x3, x4 = self.downsample(x)
        x4 = self.aspp(x4)  # Apply ASPP to the deepest features
        x = self.up1(x4)
        x += x3
        x3 = x
        x = self.up2(x)
        x += x2
        x2 = x
        x = self.up3(x)
        x += x1
        x1 = x
        x = self.up4(x)
        
        x3 = self.fpn1(x3)
        x2 = self.fpn2(x2)
        x1 = self.fpn3(x1)
        x = self.fpn4(x)
        
        x = torch.cat((x, x1, x2, x3), 1)

        x = self.up5(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
 
