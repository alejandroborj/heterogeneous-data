import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
from cv2 import imread
import pandas as pd

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, inputs, labels, transform):
        'Initialization'
        self.labels = labels
        self.inputs = inputs
        self.transform = transform
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)
        
    def __getitem__(self, index):
        'Generates one sample of data'
    
        file = self.inputs[index]
        x = imread(file).astype(np.uint8)

        if self.transform:
            x = self.transform(transforms.ToPILImage()(x))
        
        y = self.labels[index]
        
        y = torch.from_numpy(np.asarray(y)).float()
   
        return x, y, file

class PatchDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, inputs, labels, case_ids):#, scaler, case_ids):
        'Initialization'
        self.inputs = inputs
        self.labels = labels
        #self.scaler = scaler
        self.case_ids = case_ids

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)
        
    def __getitem__(self, index):
        'Generates one sample of data'
        # x = self.scaler.transform(self.inputs[index].reshape(1, -1)) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # x = torch.from_numpy(x).float()
        # return x[0], self.labels[index][0], self.case_ids[index]
        return self.inputs[index], self.labels[index], self.case_ids[index]

    def show_patch(self, patch_num):

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ex = self.__getitem__(patch_num)
        if ex[0].size(dim=2) != 3:
            im = ex[0].permute(2, 1, 0)
        else:
            print("error")

        if ex[1][0] == 1:
            ax.set_xlabel(f"Positive ID: {ex[2]}")

        else:
            ax.set_xlabel(f"Negative ID: {ex[2]}")

        print("Number of samples: ", self.__len__())
        ax.imshow(im)
