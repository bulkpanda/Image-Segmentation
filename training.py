# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 21:28:52 2021

@author: Kunal Patel
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from UNet import Model
from dataloader import data_loader
import config as cg
import matplotlib.pyplot as plt
import numpy as np
import datetime
import logging

torch.set_printoptions(threshold=np.inf)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()


def get_cuda_mem(x=''):
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)

    print(f'{x}\nTotal:{t//1024**2}\nReserved:{r//1024**2}\nAllocated:{a//1024**2}\n')



def train(model, use_cuda, train_loader, optimizer, epoch, train_loss_array,
          start_batch_idx, start_epoch, dirname):
    datalen = len(train_loader.dataset)
    model.train()
    for i,(image, label, name) in enumerate(train_loader):
        
        if epoch<start_epoch or (epoch==start_epoch and i<start_batch_idx):
            print('\n=====================')
            print(f'Epoch:{epoch}, [{(i+1)*len(image)}/{datalen}]')
            continue
        else:
            print('\n=====================')
            print(f'Epoch:{epoch}, [{(i+1)*len(image)}/{datalen}]')
            print(f'Image name:{name}')
            print(f'\nImage shape:{image.shape}')
            torch.cuda.empty_cache()
            if use_cuda:
                image, label = image.cuda(), label.cuda() # Sending the data to GPU
            
         
            optimizer.zero_grad()  # Setting the cumulative gradients to 0
            output = model(image)  # Forward pass through the model
            
            
            output = output.transpose(1,2).transpose(2,3)
            size = output.shape[0]*output.shape[1]*output.shape[2]

            output = output.resize(size,20)
            label_re = label.resize(size)
           # print(f'Label:{label.shape}')
            loss = nn.functional.cross_entropy(output, label_re)
            loss.backward()
            print(f'Loss :{loss}')

            # if loss < 10**-4:
            #     try:
            #         f = open(f'{dirname}/low_loss.log','a',encoding='UTF-8')
            #         print('Loss anamoly, saving image!!!')
            #         os.makedirs(f'{dirname}/low_loss', exist_ok=True)
            #         image=image.data.cpu().numpy().transpose(0,2,3,1)
            #         label=label.data.cpu().numpy()
            #         label[label==19]=255
            #         fig=plt.figure(figsize=(100,100))
            #         for j,im in enumerate(image):
            #
            #             fig.add_subplot(2,2,j+1)
            #             plt.imshow(im)
            #             fig.add_subplot(2,2,j+3)
            #             plt.imshow(label[j],cmap='gray')
            #             f.write(name[j])
            #             f.write('\n')
            #         plt.savefig(f'{dirname}/low_loss/{epoch}_{i}_{loss}.png')
            #         plt.close(fig)
            #
            #     except MemoryError:
            #         print('Out of memory!!!')
            #         continue
            #
            # if loss>=5:
            #     f= open(f'{dirname}/high_loss.log','a',encoding='UTF-8')
            #     print('Loss anamoly, saving image!!!')
            #     os.makedirs(f'{dirname}/high_loss', exist_ok=True)
            #     image=image.data.cpu().numpy().transpose(0,2,3,1)
            #     label=label.data.cpu().numpy()
            #     label[label==19]=255
            #     fig=plt.figure(figsize=(100,100))
            #     image=image.data.cpu().numpy().transpose(0,2,3,1)
            #     for j,im in enumerate(image):
            #
            #         fig.add_subplot(2,2,j+1)
            #         plt.imshow(im)
            #         fig.add_subplot(2,2,j+3)
            #         plt.imshow(label[j],cmap='gray')
            #         f.write(name[j])
            #         f.write('\n')
            #     plt.savefig(f'{dirname}/high_loss/{epoch}_{i}_{loss}.png')
            #     plt.close(fig)
                
            optimizer.step()
            train_loss_array.append(loss.item())
            label = label.data.cpu()

            if i == 25:
                pred = output.argmax(dim=1, keepdim=True).data.cpu()
                accuracy = pred.eq(label.view_as(pred)).sum().item()
                print(f'Accuracy:{accuracy}/{len(label)}')

            if ((i+1)*len(image)%1000) == 0:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':train_loss_array,
                'batch_idx':i,
                'epoch':epoch
                }, dirname+'/'+cg.savepath+'/'+str(epoch)+'_'+str(i)+'.pth')
                plt.plot(train_loss_array[-250:])
                plt.savefig(f'{dirname}/lossplot_{epoch}_{i}.png')
                plt.close()

def getdirname():
    pretrained=''
    if cg.pretrained:
        pretrained='Pretrained'
    root='./Exp'
    initials='7layer_Train_288_512'
    ctime=str(datetime.datetime.now())
    date=ctime.split(' ')[0]
    date=date.replace('-','_')
    time=ctime.split(' ')[1]
    time=time.split('.')[0]
    time=time.replace(':', '_')
    ctime=date+'_'+time
    dirname=f'{root}/{initials}_{ctime}_{pretrained}'
    os.makedirs(dirname,exist_ok=True)
    return dirname

def main(dirname):
    model = Model()
    savepath = cg.savepath
    
    os.makedirs(dirname+'/'+savepath, exist_ok=True)
    
    print('Cuda is available: ', torch.cuda.is_available())
    if torch.cuda.is_available():
        model.cuda()
        use_cuda = True
    else:
        use_cuda = False
    #storecache={}
    #storecache1 = np.load('train_data_0-1000.npy', allow_pickle=True).item()

    #storecache2 = np.load('train_data_1000-2000.npy', allow_pickle=True).item()
    #storecache3 = np.load('train_data_2000-3000.npy', allow_pickle=True).item()
    #storecache4 = np.load('train_data_3000-4000.npy', allow_pickle=True).item()
    #storecache = storecache1 | storecache2
    #storecache = storecache | storecache3
    #storecache = storecache | storecache4
    #=============Data loader==================================================
    print('\nStarting Dataloader...')
    train_loader=data_loader.getdataloader('D:/Image segmentation/bdd100k/images/10k/train',
                           'D:/Image segmentation/bdd100k/labels/sem_seg/masks/trainlabels',
                           'train',
                            True,
                            )
    print('Data loaded!!')
    
    #==========================================================================
    optimizer = optim.Adam(model.parameters(), lr=cg.lr, betas=(0.9, 0.999), 
                            eps = 1e-08, weight_decay=0)
    
    #===============================training===================================
    train_loss_array=[]
    itr=cg.iterations['train']+1
    start_batch_idx=0
    start_epoch=0
    
    if cg.pretrained:
        ckpt = torch.load(cg.savedmodel)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        train_loss_array = ckpt['loss']
        start_epoch=ckpt['epoch']
        start_batch_idx=ckpt['batch_idx']
        print(f'\nWill start from epoch:{start_epoch} and batch_idx:{start_batch_idx}\n')
    
    for epoch in range(1, itr):
        
        train(model, use_cuda, train_loader, optimizer, epoch, 
              train_loss_array, start_batch_idx, start_epoch, dirname )
        
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':train_loss_array,
                'batch_idx':0,
                'epoch':epoch+1
                }, dirname+'/'+savepath+'/'+str(epoch)+'.pth')
        
    plt.plot([i for i in range(len(train_loss_array))],train_loss_array)
    plt.savefig(f'{dirname}/lossplot.png')
    plt.show()

if __name__ == '__main__':
    dirname= getdirname()
    os.makedirs(f'{dirname}/logfiles', exist_ok=True)
    logger.addHandler(logging.FileHandler(f'{dirname}/logfiles/train.log', 'a',encoding = "UTF-8"))
    print = logger.info
    main(dirname)