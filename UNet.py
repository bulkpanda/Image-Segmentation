# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 21:35:29 2021

@author: Kunal Patel
"""

import torch.nn as nn
import torch

class Model(nn.Module):
    
    def __init__(self):
        
        super(Model, self).__init__()
        k=64
        
        self.encode1=contractingblock(3,k)
        self.maxpool1=nn.MaxPool2d(2,2)
        
        self.encode2=contractingblock(k,k*2)
        self.maxpool2=nn.MaxPool2d(2,2)
        
        self.encode3=contractingblock(k*2,k*4)
        self.maxpool3=nn.MaxPool2d(2,2)
        
        self.bottleneck=contractingblock(k*4,k*4)
        self.upconv0=nn.ConvTranspose2d(k*4, k*4, 2, stride=2)
        
        self.decode1=contractingblock(k*8,k*2)
        self.upconv1=nn.ConvTranspose2d(k*2, k*2, 2, stride=2)
        
        self.decode2=contractingblock(k*4,k)
        self.upconv2=nn.ConvTranspose2d(k, k, 2, stride=2)
        
        self.decode3=contractingblock(k*2,k)
        
        self.final=nn.Conv2d(k,20,kernel_size=1,stride=1)
  
    def forward(self,x):
        
        h1=self.encode1(x)
        h2=self.encode2(self.maxpool1(h1))
        h3=self.encode3(self.maxpool2(h2))
        
        up3=self.upconv0(self.bottleneck(self.maxpool3(h3)))
        up2=self.upconv1(self.decode1(torch.cat((h3,up3),1)))
        up1=self.upconv2(self.decode2(torch.cat((h2,up2),1)))
        output=self.final(self.decode3(torch.cat((h1,up1),1)))
        return output
        
class contractingblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(contractingblock, self).__init__()
        
        self.conv1=nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.relu2=nn.ReLU(inplace=True)
        
    def forward(self,x):
        x=self.relu1(self.bn1(self.conv1(x)))
        x=self.relu2(self.bn2(self.conv2(x)))
        return x


class expansionblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(expansionblock, self).__init__()
        
        self.conv=nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
    
    def forward(self,x):
        x=self.relu(self.bn1(self.conv(x)))
        return x