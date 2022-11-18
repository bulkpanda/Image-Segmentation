# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 19:53:57 2021

@author: Kunal Patel
"""
import torch
import matplotlib.pyplot as plt
from UNet import Model
from dataloader import data_loader
import config as cg
import numpy as np
import os
import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
torch.set_printoptions(threshold=np.inf)


def test(model, loader, use_cuda, dirname):

    model.eval()
    with torch.no_grad():
        for i,(image, label, name) in enumerate(loader):
            print(f'[{i}]')
            torch.cuda.empty_cache()
            
            if use_cuda:
                image = image.cuda() # Sending the data to GPU

            output = model(image)  # Forward pass through the model
            output=output.squeeze()
            output=output.transpose(0,2).transpose(0,1) # (288,512,20)
            #print(f'Output :{output}')
            pred = output.argmax(dim=2, keepdim=True).data.cpu() #(288,512,1)

            pred=pred.squeeze() #(288,512)
            label = label.squeeze()
            pred[pred == 19] = 255
            label[label == 19] = 255
            #print(f'Prediction:{pred}')
            #print(f'Label:{label}')
            total=288*512
            accuracy = pred.eq(label.view_as(pred)).sum().item()
            print(f'Name:{name}')
            print(f'Accuracy:{accuracy}/{total}')

            # fig=plt.figure(figsize=(100,100))
            # fig.add_subplot(1,2,1)
            # pred = pred.numpy()
            # plt.imshow(pred)
            # fig.add_subplot(1,2,2)
            # plt.imshow(label)
            # plt.savefig(dirname+f'/{name}_{i}.png')
            # plt.close(fig)

def getdirname():
    root='./Exp'
    initials='Test_288_512_segment'
    ctime=str(datetime.datetime.now())
    date=ctime.split(' ')[0]
    date=date.replace('-','_')
    time=ctime.split(' ')[1]
    time=time.split('.')[0]
    time=time.replace(':', '_')
    ctime=date+'_'+time
    dirname=f'{root}/{initials}_{ctime}_2'
    os.makedirs(dirname,exist_ok=True)
    return dirname

def main(dirname):
    os.makedirs('./Exp',exist_ok=True)
    data=torch.load('Exp/7layer_Train_288_512_2021_09_18_15_00_01_Pretrained/models/1_624.pth')
    loss_array=data['loss']
    
    plt.plot([i for i in range(len(loss_array))],loss_array)
    plt.savefig(dirname+'/lossplot.png')
    plt.show()
    model=Model()
    model.load_state_dict(data['model_state_dict'])
    if torch.cuda.is_available() == True:
        use_cuda=True
        model.cuda()
        
    else:
        use_cuda=False
        
    #=============Data loader==================================================
    print('Starting Dataloader...')
    loader=data_loader.getdataloader('D:/Image segmentation/bdd100k/images/10k/train',
                           'D:/Image segmentation/bdd100k/labels/sem_seg/masks/trainlabels',
                           'test')
    print('Data loaded!!')
    
    #==========================================================================
    test(model, loader, use_cuda, dirname)
  
#==============================================================================    
if __name__ == '__main__':
    dirname=getdirname()
    os.makedirs(f'{dirname}/logfiles',exist_ok=True)
    logger.addHandler(logging.FileHandler(f'{dirname}/logfiles/test.log', 'w',encoding = "UTF-8"))
    print = logger.info
    main(dirname)