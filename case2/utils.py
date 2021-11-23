from pydicom import dcmread
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

Data_dir = './Data/'

def preprocess(img):
    img = img.astype(np.float32)
    img = (img / img.max()) * 255
    img = img.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit = 20.0, tileGridSize = (8,8))
    return clahe.apply(img)

def split_train_valid(train_df, valid_size=0.2):
    df0 = train_df[train_df['label'] == 0]
    df1 = train_df[train_df['label'] == 1]
    df2 = train_df[train_df['label'] == 2]
    df0_train, df0_valid = train_test_split(df0, test_size=valid_size)
    df1_train, df1_valid = train_test_split(df1, test_size=valid_size)
    df2_train, df2_valid = train_test_split(df2, test_size=valid_size)
    train_df = pd.concat([df0_train, df1_train, df2_train])
    valid_df = pd.concat([df0_valid, df1_valid, df2_valid])
    return train_df, valid_df

def readout_dataset(dir, label_path):
    train_path, test_path = {}, {}
    train_df = pd.read_csv(label_path)
    test_df = pd.DataFrame(columns=['FileID', 'path'])
    for root, _, files in os.walk(os.path.join(dir, 'train/')):
        for file in files:
            if file.endswith('.dcm'):
                with open(os.path.join(root, file), 'rb') as f:
                    ds = dcmread(f).pixel_array
                ds = preprocess(ds)
                cv2.imwrite(os.path.join(root, file).replace('.dcm', '.he.png'), ds)
                train_path[file.split('.dcm')[0]] = os.path.join(root, file).replace('.dcm', '.he.png')
    for root, _, files in os.walk(os.path.join(dir, 'valid/')):
        for file in files:
            if file.endswith('.dcm'):
                with open(os.path.join(root, file), 'rb') as f:
                    ds = dcmread(f).pixel_array
                ds = preprocess(ds)
                cv2.imwrite(os.path.join(root, file).replace('.dcm', '.he.png'), ds)
                test_path[file.split('.dcm')[0]] = os.path.join(root, file).replace('.dcm', '.he.png')
    
    train_df['label'] = ((train_df.iloc[:, 1:] == 1).idxmax(1)).astype('category').cat.codes
    y = [train_path[x] for x in train_df['FileID'].tolist()]
    train_df.insert(1, "path", y)

    cnt = 0
    for x in test_path:
        test_df.loc[cnt] = [x, test_path[x]]
        cnt += 1
    
    # train_df, valid_df = split_train_valid(train_df, valid_size=0.25)
    train_df.to_csv(Data_dir + 'train_label.csv', index=False)
    # valid_df.to_csv(Data_dir + 'valid_label.csv', index=False)
    test_df.to_csv(Data_dir + 'test_label.csv', index=False)

class Covid_Dataset(Dataset):
    def __init__(self, label, mode):
        self.label = label
        self.preprocess = transforms.Compose([
            transforms.Resize(224), 
            transforms.CenterCrop(224),
        ])
        self.transform = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(), 
            # transforms.RandomVerticalFlip(), 
            transforms.ToTensor()
        ])
        self.mode = mode

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        row = self.label.iloc[index]
        img = Image.open(row['path'])
        img = self.preprocess(img)
        if self.mode != 'test':
            img = self.transform(img)
            return img, row['label']
        else:
            img = transforms.ToTensor()(img)
            return img, row['FileID']
        
if __name__ == '__main__':
    readout_dataset(Data_dir + 'data', Data_dir + 'label.csv')
    df = pd.read_csv(Data_dir + 'train_label.csv')
    print(df.groupby('label').count())