#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:22:49 2021

@author: vaishnavi
"""

import torch
import torchvision.transforms as transforms
import torch.fft 
import os
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from glob import glob
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

"""
A custom dataset object that returns images of chest x rays
"""
class chest_xRay_dataset(Dataset):
    def __init__(self, path):
        self.imgList = glob(f"{path}/*/*.jpeg") # f = filestring, points to jpeg images in specific directory based on path
        
        self.transforms = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()
            ])    
        
    def __len__(self):
        return len(self.imgList)
    
    def __getitem__(self, index):
        image_Path = self.imgList[index]
        image = Image.open(image_Path)
        return self.transforms(image) 
    
"""
A function that gathers 10 features for each image
"""
def getFeatureVector(oneImg):
    result = [9] * 10
    result[0] = torch.mean(oneImg)
    result[1] = torch.std(oneImg)
    
    #fftImg = torch.fft.fft2(oneImg)
    
    #print(fftImg)
    
    
    
    
    print(result)
    return result

if __name__ == "__main__":

    train_dataset = chest_xRay_dataset("./chest_xraySample/train")
    #train_loader = DataLoader(dataset = train_dataset, batch_size = 2, shuffle = False)
    train_loader = DataLoader(dataset = train_dataset, shuffle = False)
    
    """ for batch in train_loader:
        #print(batch)
        for img in batch:
            img = img.squeeze()
            plt.imshow(img)
            plt.show() """
            
    print(len(train_loader))
    for img in train_loader:
            img = img.squeeze()
            plt.imshow(img)
            plt.show()
            print(img)
            getFeatureVector(img) 
            
    

    

    
    