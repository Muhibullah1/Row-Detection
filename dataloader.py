import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import albumentations as aug
from torch.utils.data import Dataset, DataLoader


class DatasetLoader(Dataset):
    def __init__(self,data,data_dir,mask_dir,transform = None):
        self.data = data
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.transform = transform
               
    def __len__(self):
        return (len(self.data))
    
      
    def __getitem__(self,idx):
        image = cv2.imread(self.data_dir+self.data[idx])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_dir+self.data[idx], cv2.IMREAD_GRAYSCALE)
        mask[mask!=0] = 1
        if self.transform != None:
            for augment in [*self.transform]:
                augmented = augment(image=image,mask=mask)
                image = augmented ['image']
                mask = augmented ['mask']
        image = image/255
        if(len(mask.shape) == 2):
            mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2,0,1))
        
        return torch.from_numpy(image).float(),torch.from_numpy(mask).float()
        
