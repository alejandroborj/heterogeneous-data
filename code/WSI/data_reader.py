
import numpy as np
import glob
import os
import random
from tqdm import tqdm
import gc
import pandas as pd

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
    print(str(n) + " : " + str(psutil.virtual_memory()))

# Returns size of a given object in GB
def mem_obj(obj):
    import sys
    print(str(sys.getsizeof(obj)/1024/1024/1024))

class Data_reader():
# Override by subclasses

    def __init__(self, folder_name, np_shape, formats): # Initialization of Data_reader object
        self.folder_name = folder_name
        self.np_shape = np_shape # Imagen size ?????
        self.formats = formats # List of the different file format .svs in this case
        self.data = {} # Dictionary with 3 keys for train test and validation
        self.data['train'] = {}
        self.data['val'] = {}
        self.data['test'] = {}

    def __del__(self):
        del self.data
    
    # Search for all folder with name folder_name within the paths passed
    # Read all files inside using the read_file method and create a list
    # Finally, all file data is added to the data dictionary
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

                        file_data += [[self.read_file(file, patch_size), self.check_class(file[-101:-65])]] # Adding class and tiled file

                        '''
                        if file[-47:-45]=="01": # Reading the ID diagnostic sample type 01 == Primary tumor
                            file_data += [[self.read_file(file, patch_size), 1]] # Reading data (1 => positive diagnosis, 0 => negative)
                        else:
                            file_data += [[self.read_file(file, patch_size), 0]]
                        
                        '''

                self.data[dataset][case_id] = np.asarray(file_data) # Adding the file data to the dictionary with key=>case_id

    def read_file(self, file, patch_size):

        ts = large_image.getTileSource(file)
        size_x, size_y = ts.getMetadata()['sizeX'], ts.getMetadata()['sizeY']
        patches = []

        for tile_info in ts.tileIterator(
        scale=dict(magnification=10),
        tile_size=dict(width=patch_size, height=patch_size),
        tile_overlap=dict(x=0, y=0),
        format=large_image.tilesource.TILE_FORMAT_NUMPY
        ):
            patch = tile_info['tile']

            avg = patch.mean(axis=0).mean(axis=0)

            if  avg[0]< 220 and avg[1]< 220 and avg[2]< 220: # Checking if the patch is white
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

    # Changes from data_reader data to data_set 

    def data_reader_to_dataset(self, case_id):

        labels, inputs, case_ids = [], [], []

        for case in case_id:
            for image_count, image in enumerate(self.data['train'][case]):
                for patch_count, patch in enumerate(self.data['train'][case][0][image_count]):

                    labels.append(self.data['train'][case][image_count][1])
                    inputs.append(self.data['train'][case][image_count][0][patch_count])
                    case_ids.append(case)

        return dataset.PatchDataset(inputs=inputs, labels=labels , scaler=1 , case_ids=case_ids)

    # Return class for a given case

    def check_class(self, id):

        duct = ["Infiltrating duct carcinoma"] # This assumption must be checked

        carc = ["Adenocarcinoma",
        "Adenocarcinoma with mixed subtypes",
        "Neuroendocrine carcinoma"]

        clinical = pd.read_csv(r"C:\Users\Alejandro\Desktop\heterogeneous-data\data\clinical.tsv", delimiter="\t")
 
        if clinical.loc[clinical['case_id']==id]["classification_of_tumor"] in duct: # [1,0,0] => duct
            return [1,0,0]

        elif clinical.loc[clinical['case_id']==id]["classification_of_tumor"] in carc: # [0,1,0] => carc
            return [0,1,0]

        else:
            return [0,0,1] # [0,0,1] => normal tissue


