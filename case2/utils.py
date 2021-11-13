from pydicom import dcmread
import os
import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

Data_dir = './Data/'

def readout_dataset(dir, label_path):
    train_path, valid_path = {}, {}
    train_df = pd.read_csv(label_path)
    valid_df = pd.DataFrame(columns=['FileID', 'path'])
    for root, _, files in os.walk(os.path.join(dir, 'train/')):
        for file in files:
            train_path[file.split('.dcm')[0]] = os.path.join(root, file)
    for root, _, files in os.walk(os.path.join(dir, 'valid/')):
        for file in files:
            valid_path[file.split('.dcm')[0]] = os.path.join(root, file)
    
    train_df['label'] = ((train_df.iloc[:, 1:] == 1).idxmax(1)).astype('category').cat.codes
    y = [train_path[x] for x in train_df['FileID'].tolist()]
    train_df.insert(1, "path", y)

    cnt = 0
    for x in valid_path:
        valid_df.loc[cnt] = [x, valid_path[x]]
        cnt += 1
    
    train_df.to_csv(Data_dir + 'train_label.csv', index=False)
    valid_df.to_csv(Data_dir + 'valid_label.csv', index=False)

class Covid_Dataset(Dataset):
    def __init__(self, label, mode):
        self.label = label
        self.preprocess = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(256),
        ])
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomVerticalFlip(), 
            transforms.ToTensor()
        ])
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.label.iloc[index]
        with open(row['path'], 'rb') as f:
            img = dcmread(f).pixel_array
        img = np.array(img, dtype=np.float32)[np.newaxis]
        img = torch.from_numpy(img)
        img = self.preprocess(img)
        if self.mode == 'train':
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        return img, row['label']

if __name__ == '__main__':
    readout_dataset(Data_dir + 'data', Data_dir + 'label.csv')
    print(pd.read_csv(Data_dir + 'train_label.csv'))