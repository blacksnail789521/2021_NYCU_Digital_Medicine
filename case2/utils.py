from pydicom import dcmread
import os
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader

Data_dir = './Data/'

def readout_dataset(dir, label_path):
    train_set, valid_set = {}, {}
    train_df = pd.read_csv(label_path)
    valid_df = pd.DataFrame(columns=['FileID'])
    for root, _, files in os.walk(os.path.join(dir, 'train/')):
        for file in files:
            with open(os.path.join(root, file), 'rb') as f:
                train_set[file.split('.dcm')[0]] = dcmread(f).pixel_array
    for root, _, files in os.walk(os.path.join(dir, 'valid/')):
        for file in files:
            with open(os.path.join(root, file), 'rb') as f:
                valid_set[file.split('.dcm')[0]] = dcmread(f).pixel_array
    
    train_df['label'] = ((train_df.iloc[:, 1:] == 1).idxmax(1)).astype('category').cat.codes
    y = [train_set[x] for x in train_df['FileID'].tolist()]
    with open(Data_dir + 'train_data.pk', 'wb') as f:
        pickle.dump(y, f)

    cnt = 0
    y = []
    for x in valid_set:
        valid_df.loc[cnt] = [x]
        y.append(valid_set[x])
        cnt += 1
    with open(Data_dir + 'valid_data.pk', 'wb') as f:
        pickle.dump(y, f)
    
    train_df.to_csv(Data_dir + 'train_label.csv', index=False)
    valid_df.to_csv(Data_dir + 'valid_label.csv', index=False)

def get_dataset(train_pk, valid_pk):
    train_label = pd.read_csv(Data_dir + 'train_label.csv')
    valid_label = pd.read_csv(Data_dir + 'valid_label.csv')
    with open(train_pk, 'rb') as f:
        train_data = pickle.load(f)
    with open(valid_pk, 'rb') as f:
        valid_data = pickle.load(f)
    return train_label, train_data, valid_label, valid_data

class Covid_Dataset(Dataset):
    def __init__(self, label, data):
        self.label = label
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

if __name__ == '__main__':
    readout_dataset(Data_dir + 'data', Data_dir + 'label.csv')
    train_label, train_data, valid_label, valid_data = get_dataset(Data_dir + 'train_data.pk', Data_dir + 'valid_data.pk')
    print(train_data[0])