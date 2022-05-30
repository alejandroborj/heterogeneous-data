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

import lmdb
import pickle

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
    
    def read_data(self, paths, patch_size, name): #  paths => paths for all data folders. dataset => train, test or val

        inputs, labels, case_ids = [], [], []
        for path in tqdm(paths):
            case_id = os.path.split(path)[-1]
            if not os.path.exists(path) or len(os.listdir(path)) == 0:
                print("Path does not exist")
                pass
            else:
                for format in self.formats:
                    for file in glob.glob(path + r"\*" + format):
                        patches = self.read_file(file, patch_size)
                        inputs.extend(patches)
                        if file[-51:-49] in ("01", "02", "03", "04" ,"05", "06", "07", "08", "09"): # Reading the ID diagnostic sample type 01 == Primary tumor
                            labels.extend([[1, 0] for i in range(len(patches))]) # Reading data (1,0 => positive diagnosis, 0, 1 => negative)
                        else:
                            labels.extend([[0, 1] for i in range(len(patches))])
                        case_ids.extend([case_id for i in range(len(patches))])

        store_lmdb(np.asarray(inputs, dtype=np.uint8), np.array(labels, dtype=np.uint8), case_ids, name)
                    

    def read_file(self, file, patch_size):

        ts = large_image.getTileSource(file)
        patches = []
        
        for tile_info in ts.tileIterator(
        scale=dict(magnification=20),
        tile_size=dict(width=patch_size, height=patch_size),
        tile_overlap=dict(x=0, y=0),
        format=large_image.tilesource.TILE_FORMAT_PIL
        ):
            if np.random.rand()<0: # We take 100% of patches
                pass
            else:
                patch = tile_info['tile']

                # Changing from PIL RGBA image to RGB tensor

                patch_aux = Image.new("RGB", patch.size, (255, 255, 255))
                patch_aux.paste(patch, mask=patch.split()[3]) # 3 is the alpha channel

                patch = np.asarray(patch_aux, dtype = np.uint8)

                avg = patch.mean(axis=0).mean(axis=0)

                if  avg[0]< 220 and avg[1]< 220 and avg[2]< 220 and patch.shape == (patch_size, patch_size, 3): # Checking if the patch is white and its a square tile
                    patches.append(patch)
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

def store_lmdb(images, labels, case_ids, name):
    """ Stores multiple images to a LMDB.
        Parameters:
        ---------------
        images      list of image array, (512, 512, 3) to be stored
        labels      image labels
    """
    #map_size = images[0].nbytes * 100 * len(images)

    #print(len(images))
    #print(images[0].nbytes)
    #map_size = 10*(images[0].nbytes)*len(images)# int((len(images)+200) * 3 * 512**2) # 10000 patches per slide cota sup, 3 channels, 512 Image size,
    map_size = int(1.5*(len(images)) * 3 * 512**2)
    print(map_size/1024/1024/1024)

    # Create a new LMDB environment
    env = lmdb.open(f"C:/Users/Alejandro/Desktop/heterogeneous-data/data/patches/{name}", map_size=map_size)

    # Start a new write transaction
    with env.begin(write=True) as txn:
        for id, image in enumerate(images):
            # All key-value pairs need to be strings
            txn.put(('X_'+case_ids[id]+'_'+str(id)).encode("ascii"), images[id])
            txn.put(('y_'+str(id)).encode("ascii"), labels[id])
                 
    env.close()


def read_lmdb(filename):
    print('Read lmdb')

    lmdb_env = lmdb.open(filename)
    #lmdb_txn = lmdb_env.begin()
    
    X, y, labels = [], [], []
    n_counter=0

    with lmdb_env.begin() as lmdb_txn:
        with lmdb_txn.cursor() as lmdb_cursor:
            for key, value in lmdb_cursor:
                if(f'X'.encode("ascii") in key[:2]):
                    X.append(np.frombuffer(value, dtype=np.uint8).reshape(512, 512, 3))
                if(f'y'.encode("ascii") in key[:2]):
                    y.append(np.frombuffer(value, dtype=np.uint8))
                    labels.append(key[2:])
                n_counter+=1

    lmdb_env.close()

    return X, y, n_counter, labels