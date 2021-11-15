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

def preprocess(img):
    img = img.astype(np.float32)
    img = (img / img.max()) * 255
    img = img.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit = 20.0, tileGridSize = (8,8))
    return clahe.apply(img)

    # # histogram equalization
    # intensity_count = [0] * 256        
    # height, width = img.shape[:2]
    # N = height * width                  

    # high_contrast = np.zeros(img.shape) 

    # for i in range(0, height):
    #     for j in range(0, width):
    #         intensity_count[img[i][j]] += 1

    # L = 256

    # intensity_count, total_values_used = np.histogram(img.flatten(), L, [0, L])      
    # pdf_list = np.ceil(intensity_count*(L-1)/img.size)
    # cdf_list = pdf_list.cumsum()

    # for y in range(0, height):
    #     for x in range(0, width): 
    #         high_contrast[y,x] = cdf_list[img[y,x]]

    # return img

def readout_dataset(dir, label_path):
    train_path, valid_path = {}, {}
    train_df = pd.read_csv(label_path)
    valid_df = pd.DataFrame(columns=['FileID', 'path'])
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
                valid_path[file.split('.dcm')[0]] = os.path.join(root, file).replace('.dcm', '.he.png')
    
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
            transforms.Resize(224), 
            transforms.CenterCrop(224),
        ])
        self.transform = transforms.Compose([
            transforms.RandomRotation(30),
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
            return img, row['label']
        else:
            img = transforms.ToTensor()(img)
            return img, row['FileID']
        
if __name__ == '__main__':
    readout_dataset(Data_dir + 'data', Data_dir + 'label.csv')
    df = pd.read_csv(Data_dir + 'train_label.csv')
    print(df.groupby('label').count())