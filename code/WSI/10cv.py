#%%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import glob
import numpy as np


SPLITS = 10

data_path = r"C:\Users\Alejandro\Desktop\heterogeneous-data\data\gdc_download_20220427_144600.480657"
case_id = os.listdir(data_path)
case_id = case_id[:int(len(case_id)*0.3)] # 20% of the data as a test
paths = [data_path + "\\" + case for case in case_id] # All case folders paths
formats = [".svs"]

X = case_id
y = []

for path in paths:
    for format in formats:
        for file in glob.glob(path + r"\*" + format):
            if file[-51:-49] in ("01", "02", "03", "04" ,"05", "06", "07", "08", "09"):
                y.append([1, 0])
            else:
                y.append([0, 1])

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
