# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 15:40:53 2021

@author: Kunal Patel
"""
savepath='./models'
# size of each batch for training and testing
batchsize={""
    'train':10,
    'test':1
}

# no. of iterations to be run
iterations={
    'train':100,
    'test':10
}

# size of train and test data
datasize={
    'train':7000,
    'test':3000
    }

# number of workers while loading data
num_workers={
    'train':0,
    'test':0
}

lr = 0.002

# For pretrained models
pretrained = True
savedmodel='Exp/Train_UnetResnet_0.002_10_2021_10_10_04_04_37_Pretrained/models/100.pth'