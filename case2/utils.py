from pydicom import dcmread
import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

Data_dir = './Data/'

def preporcess(img):
    img = img.astype(np.float32)
    img = (img / img.max()) * 255
    return img

def readout_dataset(dir, label_path):
    train_path, valid_path = {}, {}
    train_df = pd.read_csv(label_path)
    valid_df = pd.DataFrame(columns=['FileID', 'path'])
    for root, _, files in tqdm(os.walk(os.path.join(dir, 'train/'))):
        for file in files:
            with open(os.path.join(root, file), 'rb') as f:
                ds = dcmread(f).pixel_array
            ds = preporcess(ds)
            cv2.imwrite(os.path.join(root, file).replace('dcm', 'png'), ds)
            train_path[file.split('.dcm')[0]] = os.path.join(root, file).replace('dcm', 'png')
    for root, _, files in tqdm(os.walk(os.path.join(dir, 'valid/'))):
        for file in files:
            with open(os.path.join(root, file), 'rb') as f:
                ds = dcmread(f).pixel_array
            cv2.imwrite(os.path.join(root, file).replace('dcm', 'png'), ds)
            valid_path[file.split('.dcm')[0]] = os.path.join(root, file).replace('dcm', 'png')
    
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
        return len(self.label)

    def __getitem__(self, index):
        row = self.label.iloc[index]
        img = Image.open(row['path'])
        img = self.preprocess(img)
        if self.mode == 'train':
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        return img, row['label']

if __name__ == '__main__':
    readout_dataset(Data_dir + 'data', Data_dir + 'label.csv')
    print(pd.read_csv(Data_dir + 'train_label.csv'))