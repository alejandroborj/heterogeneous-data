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

class TestDataset(data.Dataset):
    'Characterizes the test dataset'
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
    
        files = self.inputs[index]
        images = []
        for f in files:
            x = imread(f).astype(np.uint8)
            if self.transform:
               x = self.transform(transforms.ToPILImage()(x)) 
            images.append(x)
            
        y = self.labels[index]
        y = torch.from_numpy(np.asarray(y)).float()
   
        return images, y, files

class TestDatasetImgGen(data.Dataset):
    'Characterizes the test dataset'
    def __init__(self, inputs, labels, transform, num_variables):
        'Initialization'
        self.labels = labels
        self.inputs = inputs
        self.transform = transform
        self.num_variables = num_variables

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)
        
    def __getitem__(self, index):
        'Generates one sample of data'
        files = self.inputs[index][0]
        file_gen = self.inputs[index][1]
        images = []
        case_ids = []

        # read img files
        for f in files:
            x = imread(f).astype(np.uint8)
            if self.transform:
               x = self.transform(transforms.ToPILImage()(x)) 
            images.append(x)
            case_ids.append(f)
 
        # read gen file
        data = pd.read_csv(file_gen, sep=',')
        try:
            gen = data.values[0, 1:self.num_variables+1].astype('float32')
        except:
            gen = data.values[0, 2:self.num_variables+2].astype('float32')
        
        gen = torch.from_numpy(gen).float()
        y = self.labels[index]
        y = torch.from_numpy(np.asarray(y)).float()
   
        return images, gen, y, case_ids

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
        return self.inputs[index], self.labels[index]#, self.case_ids[index]

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
