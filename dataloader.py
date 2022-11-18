# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 17:37:57 2021

@author: Kunal Patel
"""
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import config as cg
from tqdm import tqdm



batchsize=cg.batchsize

def log(a):
    file=open('dataloader/train.log','a', encoding='UTF-8')
    file.write(a)
    print(a)


class data_loader(Dataset):

    def __init__(self, imagepath, labelpath, cache=False, storecache={}, boundary={}):

        self.imagepath = imagepath
        self.labelpath = labelpath
        self.allimages = sorted(os.listdir(self.imagepath))
        self.imageshape = [288, 512]
        self.desiredaspect = self.imageshape[1] / self.imageshape[0]
        self.cache = cache
        self.storecache = storecache
        self.boundary = boundary
        # print(f'Boundary:{boundary.keys()}')
        # zprint(f'cache:{storecache.keys()}')

    def changeimagesize(self, image):

        aspect = image.shape[1] / image.shape[0]
        if aspect == self.desiredaspect:
            image = Image.fromarray(image).resize((self.imageshape[1], self.imageshape[0]))
            image = np.array(image)

            return image
        else:
            print('Image is not of desired aspect ratio!!')
            exit(0)

    def __getitem__(self, item):

        while True:
            try:
                if item in self.storecache:
                    # print('Using cache!')
                    image = self.storecache[item]['image']
                    label = self.storecache[item]['label']
                    boundary = self.boundary[item]
                    boundary_1d = np.ones(self.imageshape[0] * self.imageshape[1]).astype(np.float32)
                    boundary_1d[boundary] = 2
                    # boundary_2d=boundary_1d.reshape((self.imageshape[0],self.imageshape[1]))

                else:
                    image = plt.imread(self.imagepath + '/' + self.allimages[item])
                    image = self.changeimagesize(image)

                    label = np.zeros([image.shape[0], image.shape[1]])

                    labelfolder = self.labelpath + '/' + self.allimages[item][:-4]

                    for labelI in sorted(os.listdir(labelfolder)):
                        labelimage = plt.imread(labelfolder + '/' + labelI)[:, :, 0]
                        labelimage = self.changeimagesize(labelimage)
                        value = int(labelI.split('.')[0])

                        if value == 255:
                            value = 19

                        label[labelimage == 1] = value
                    if self.cache:
                        self.storecache[item] = {'image': image, 'label': label}
                break
            except:
                item = np.random.randint(len(self.allimages))
                print('Error loading items!!!')
        return image.transpose(2, 0, 1).astype(np.float32) / 255, label.astype(np.int64), self.allimages[
            item], boundary_1d

    def __len__(self):
        return len(self.allimages)

    def _worker_init_fn(worker_id):
        np.random.seed(worker_id)

    def getdataloader(imagepath, labelpath, type_='train', cache=False, storecache={}, boundary={}):
        return DataLoader(
            data_loader(imagepath, labelpath, cache, storecache, boundary),
            batch_size=batchsize[type_],
            num_workers=0,
            worker_init_fn=data_loader._worker_init_fn,
            shuffle=True
        )


if __name__ == '__main__':
    loader = data_loader.getdataloader('D:/Image segmentation/bdd100k/images/10k/train',
                                       'D:/Image segmentation/bdd100k/labels/sem_seg/masks/trainlabels')

    iterator = tqdm(loader)
    setOf = set()

    for image, label in iterator:
        setOf.update(set(np.unique(label).tolist()))
        iterator.set_description(str(setOf))