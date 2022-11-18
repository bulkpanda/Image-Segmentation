# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 11:25:28 2021

@author: Kunal Patel
"""

# Write empty directories in the emptylog.txt file
import os
path='D:/Image segmentation/bdd100k/labels/sem_seg/masks/trainlabels'
subdir=os.listdir(path)
f=open('./emptycheck.log','a')
for s in subdir:
    print(s)
    l=path+'/'+s
    if len(os.listdir(l))==0:
        f.write(s)
        f.write('\n')
