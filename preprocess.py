# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 15:08:45 2021

@author: Kunal Patel
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:58:54 2021

@author: Kunal Patel

This file is used to load training and testing data
"""
# importing libraries

import os
import matplotlib.pyplot as plt
import numpy as np
from os.path import isfile, join
# calling variables from config.py file
basepath='D:/Image segmentation/bdd100k'

#=========================Data loader class=========================================================
class preprocess():


  # init function takes type of data (train or test) and boolean isdatafile (whether a .npy file is already)
  # present containing the processed data
  def __init__(self,type_):
      
    self.imagepath=basepath+'/images/10k/'+type_
    self.allimages = [f for f in os.listdir(self.imagepath) if isfile(join(self.imagepath, f))]
    self.labelpath=basepath+'/labels/sem_seg/masks/'+type_
    self.outputpath=basepath+'/labels/sem_seg/masks/'+type_+'labels'
    
    for i in range(len(self.allimages)):
      image=self.allimages[i][:-3]
      label=(plt.imread(self.labelpath+'/'+image+'png')*255).astype(np.uint8)
      self.breakintosubimages(label, self.outputpath+'/'+image[:-1])
      plt.imshow(label)

  def breakintosubimages(self, label, outputDir):
      os.makedirs(outputDir, exist_ok=True)
      uniquelabels=np.unique(label) # get all unique pixel values
      for uni in uniquelabels[:]:
          # for pixels whose value is equal to the unique values, we extract them       
          plt.imsave(outputDir+'/'+str(uni)+'.png', (label==uni).astype(np.uint8)*255, cmap='gray')
      
  



#===============Testing the dataloader===================================
if __name__=='__main__':
    pre=preprocess('train')
   