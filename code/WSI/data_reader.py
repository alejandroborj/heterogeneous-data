import numpy as np
import glob
import os
import random
from tqdm import tqdm
import gc
import pandas as pd
import torch
import torchvision.transforms as torchvision
from PIL import Image

import dataset

## In order to fix .dll openslide bug a path for said file is provided
os.environ['PATH'] = r"C:\Users\Alejandro\Downloads\openslide-win64-20171122\openslide-win64-20171122\bin" + ";" + os.environ['PATH']  #libvips-bin-path is where you save the libvips files
import openslide
import large_image

# Returns memory usage of a given process in GB
def memory(n):
    import psutil
    import os
    process = psutil.Process(os.getpid())
    print(str(n) + " : " + str(process.memory_info()[0]/1024/1024/1024))
    print(str(n) + " : " + str(psutil.virtual_memory().percent))


# Returns size of a nested dictionary
def get_size(obj, seen=None):
    import sys
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

# Returns size of a given object in GB
def mem_obj(obj):
    import sys
    print(str(get_size(obj)/1024/1024/1024))

class Data_reader():
# Override by subclasses

    def __init__(self, folder_name, formats): # Initialization of Data_reader object
        self.folder_name = folder_name
        self.formats = formats # List of the different file format .svs in this case
        self.data = {} # Dictionary with 3 keys for train test and validation
        self.data['train'] = {}
        self.data['val'] = {}
        self.data['test'] = {}

    def __del__(self):
        del self.data
    
    def read_data(self, paths, patch_size, dataset='train'): #  paths => paths for all data folders. dataset => train, test or val

        for path in tqdm(paths):
            case_id = os.path.split(path)[-1]
            if not os.path.exists(path) or len(os.listdir(path)) == 0:
                print("Path does not exist")
                pass
            else:
                file_data = []
                for format in self.formats:
                    for file in glob.glob(path + r"\*" + format):
                        if file[-51:-49] in ("01", "02", "03", "04" ,"05", "06", "07", "08", "09"): # Reading the ID diagnostic sample type 01 == Primary tumor
                            file_data += [[self.read_file(file, patch_size),  [1, 0]]] # Reading data (1 => positive diagnosis, 0 => negative)
                        else:
                            file_data += [[self.read_file(file, patch_size), [0, 1]]]
                        
                self.data[dataset][case_id] = file_data # Adding the file data to the dictionary with key => case_id
                print("Size of data_reader object")
                mem_obj(self.data)
                memory(1)
        
    def read_file(self, file, patch_size):

        ts = large_image.getTileSource(file)
        patches = []

        for tile_info in ts.tileIterator(
        scale=dict(magnification=10),
        tile_size=dict(width=patch_size, height=patch_size),
        tile_overlap=dict(x=0, y=0),
        format=large_image.tilesource.TILE_FORMAT_PIL
        ):
            if np.random.rand()<0.92: # We only take 10% of patches
                pass
            else:
                patch = tile_info['tile']

                # Changing from PIL RGBA image to RGB tensor

                patch_aux = Image.new("RGB", patch.size, (255, 255, 255))
                patch_aux.paste(patch, mask=patch.split()[3]) # 3 is the alpha channel

                patch = np.asarray(patch_aux)

                avg = patch.mean(axis=0).mean(axis=0)

                if  avg[0]< 220 and avg[1]< 220 and avg[2]< 220 and patch.shape == (patch_size, patch_size, 3): # Checking if the patch is white and its a square tile
                    patches.append((patch/255).tolist())

        return patches

    # Return a random data value for a case (if there are multiple)

    def get_data(self, mode, case_id):
        input_data = None
        if mode in self.data and case_id in self.data[mode]:
            input_data = self.data[mode][case_id]
            input_data = input_data[np.random.randint(0, len(input_data))]
        else:
            print("Error, {} or {} not present in data, returning None".format(mode, case_id))
        return input_data

    # Changes from data_reader data to data_set 

    def data_reader_to_dataset(self, case_id):
        labels, inputs, case_ids = [], [], []

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        normalize = torchvision.transforms.Normalize(mean=mean, std=std)

        for case in case_id:
            for image_count, image in enumerate(self.data['train'][case]):
                for patch_count, patch in enumerate(self.data['train'][case][0][image_count]):

                    # Normalization acording to imagenet DB

                    x = self.data['train'][case][image_count][0][patch_count]
                    x = normalize(torch.tensor(x).permute(2, 1, 0))

                    labels.append(self.data['train'][case][image_count][1])
                    inputs.append(x)
                    case_ids.append(case)

        return dataset.PatchDataset(inputs=torch.stack(inputs), 
                                    labels=torch.tensor(labels),
                                    scaler=1,
                                    case_ids=case_ids)

