#%%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import glob
import numpy as np
import os

SPLITS = 10
gdc_data_path = "D:/data/WSI/GDC/gdc_download_20220615_080352.004457"
gtex_data_path = "D:/data/WSI/GTEx/PAAD"
rna_wsi_cases = np.loadtxt("D:/data/rna_wsi_cases.txt", dtype=str)
gdc_image_id = os.listdir(gdc_data_path)
#gtex_image_id = os.listdir(gtex_data_path)

#gdc_image_id = gdc_image_id[:int(0.2*len(gdc_image_id))] # 20% of the data as a test
#gtex_image_id = gtex_image_id[:int(0.2*len(gtex_image_id))] # 20% of the data as a test

paths = [gdc_data_path + "/" + case for case in gdc_image_id] + [gtex_data_path] #[gtex_data_path + "/" + case for case in gtex_image_id] # All case folders paths
formats = [".svs"]
case_ids = []
labels = []

for path in paths:
    file_id = os.path.split(path)[-1]
    if not os.path.exists(path) or len(os.listdir(path)) == 0:
        print("Path does not exist")
        pass
    
    if "GTEx" in path:
        for file in glob.glob(path + r"\*" + format):
            sample_id = file.split("\\")[-1][-19:-4]
            case_id = '-'.join(sample_id.split("-")[:2])
            if '-'.join(sample_id.split("-")[:2]) in rna_wsi_cases:
                print("NEGATIVE: ", sample_id)
                labels.append([1, 0])
                case_ids.append(case_id)
    else:
        for format in formats:
            for file in glob.glob(path + r"\*" + format):
                sample_id = file[-64:-48]
                if file[-51:-49] == "01": # Reading the ID diagnostic sample type 01 == Primary tumor
                    print("POSITIVE: ", sample_id)
                    labels.append([0, 1])
                    case_ids.append(sample_id[:-4])
                elif file[-51:-49] == "11":
                    print("NEGATIVE: ", sample_id)
                    labels.append([1, 0])
                    case_ids.append(sample_id[:-4])
                else:
                    print("ERROR, SAMPLE IS NOT PRIMARY TUMOR OR TISSUE NORMAL: ", sample_id)
                    break

unique_index = np.unique(case_ids, return_index=True)[1] 
# Deleting all non unique instances the case labels for non unique 
# instances are taken randomly for approximate estratification

case_ids = [case_ids[i] for i in unique_index]
labels = [labels[i] for i in unique_index]

print("Ratio of samples:" , len(labels))
print("Number of negatives: ", sum([1 if i[0]==1 else 0 for i in labels]))
print("Ratio of negatives:" , sum([1 if i[0]==1 else 0 for i in labels])/len(labels))

#%%
X = case_ids
y = labels

#print(X, y)

y = np.asarray(y)

kf = StratifiedKFold(
    n_splits=SPLITS,  
    shuffle=True)

test_idx = []
train_idx = []

for i, (train_index, test_index) in enumerate(kf.split(X, y.argmax(1))):
  train_idx.append(train_index)
  test_idx.append(test_index)

'''
print(X, y)
print(train_idx[0], train_idx[0])
'''
 #%%
for j in range(SPLITS):
    contador = 0 

    split = [y[i] for i in train_idx[j]]

    for label in split:
        contador += label[0]

    print(f"Proporcion positivos split {j+1}: ", contador/len(split))
# %%
for i in range(SPLITS):
    with open(f"C:\\Users\\Alejandro\\Desktop\\heterogeneous-data\\splits\\trainsplit{i}.txt", "w") as f:
        for j in train_idx[i]:
            f.write(X[j]+"\n")

    with open(f"C:\\Users\\Alejandro\\Desktop\\heterogeneous-data\\splits\\testsplit{i}.txt", "w") as f:
        for j in test_idx[i]:
            f.write(X[j]+"\n")

# %%
