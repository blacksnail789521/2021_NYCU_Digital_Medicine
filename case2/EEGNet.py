from pydicom import dcmread
import os
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import wandb

Data_dir = './Data/'

## hyperparameters
Batch_size = 32
Learning_rate = 1e-4
Epochs = 50

wandb.init(project="Digital Medicine Case 2", entity="wwweiwei")
wandb.config = {
  "learning_rate": Learning_rate,
  "epochs": Epochs,
  "batch_size": Batch_size
}

def get_dataset(train_pk, valid_pk):
    train_label = pd.read_csv(Data_dir + 'train_label.csv').to_numpy()
    valid_label = pd.read_csv(Data_dir + 'valid_label.csv').to_numpy()
    with open(train_pk, 'rb') as f:
        train_data = pickle.load(f)
    with open(valid_pk, 'rb') as f:
        valid_data = pickle.load(f)
    return train_data, train_label, valid_data, valid_label

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class EEGNet(nn.Module):
    def __init__(self, mode):
        super(EEGNet, self).__init__()
        if mode == 'elu':
            activation = nn.ELU()
        elif mode == 'relu':
            activation = nn.ReLU()
        elif mode == 'leakyrelu':
            activation = nn.LeakyReLU()
        
        ## Conv2D
        self.firstconv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        
        ## DepthwiseConv2D
        self.deptwiseConv = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25))

        ## SeparableConv2D
        self.separableConv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25))
        
        ## Classification
        self.classify = nn.Sequential(nn.Linear(in_features=65280, out_features=3, bias=True))

       
    def forward(self, data):
        output = self.firstconv(data)
        output = self.deptwiseConv(output)
        output = self.separableConv(output)
        output = output.view(output.size(0), -1)
        output = self.classify(output)
        
        return output

class DeepConvNet(nn.Module):
    def __init__(self, mode):
        super(DeepConvNet, self).__init__()
        if mode == 'elu':
            activation = nn.ELU()
        elif mode == 'relu':
            activation = nn.ReLU()
        elif mode == 'leakyrelu':
            activation = nn.LeakyReLU()
        self.conv0 = nn.Conv2d(1, 25, kernel_size=(1,5))
        self.conv1 = nn.Sequential(
                nn.Conv2d(25, 25, kernel_size=(2, 1)),
                nn.BatchNorm2d(25, eps=1e-5, momentum=0.1),
                activation,
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=0.5))
        self.conv2 = nn.Sequential(
                nn.Conv2d(25, 50, kernel_size=(1, 5)),
                nn.BatchNorm2d(50, eps=1e-5, momentum=0.1),
                activation,
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=0.5))
        self.conv3 = nn.Sequential(
                nn.Conv2d(50, 100, kernel_size=(1, 5)),
                nn.BatchNorm2d(100, eps=1e-5, momentum=0.1),
                activation,
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=0.5))
        self.conv4 = nn.Sequential(
                nn.Conv2d(100, 200, kernel_size=(1, 5)),
                nn.BatchNorm2d(200, eps=1e-5, momentum=0.1),
                activation,
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=0.5))
        self.classify = nn.Linear(306000, 64)
        
    def forward(self, data):
        output = self.conv0(data)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = output.view(output.size(0), -1)
        output = self.classify(output)
        return output

class Covid_Dataset(Dataset):
    def __init__(self, label, mode):
        self.label = label
        self.preprocess = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(256),
        ])
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(), 
            transforms.RandomVerticalFlip(), 
            transforms.ToTensor()
        ])
        self.mode = mode

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        row = self.label[index,:]

        if self.mode == 'train':
            for root, dir, files in os.walk('./Raw_data/train/'):
                for file in files:
                    if str(file.split('.dcm')[0]) == str(row[0]):
                        path = str(root)+'/'+str(file)

                        with open(path, 'rb') as f:
                            img = dcmread(f).pixel_array
                            img = np.array(img, dtype=np.float32)[np.newaxis]
                            img = torch.from_numpy(img)
                            img = self.preprocess(img)
                            # print(img.size())
                            if self.mode == 'train':
                                img = self.transform(img)
                                return img, row[4]
                            else:
                                img = transforms.ToTensor()(img)
                                return img, row[0]


if __name__ == '__main__':
    train_data, train_label, valid_data, valid_label = get_dataset(Data_dir + 'train_data.pk', Data_dir + 'valid_data.pk')

    print('valid_label: ', valid_label)

    covid_dataset = Covid_Dataset(train_label, mode='train')
    covid_data_loader = DataLoader(covid_dataset, batch_size=32, shuffle=True, num_workers=4)

    covid_dataset_test = Covid_Dataset(valid_label, mode='test')
    covid_data_loader_test = DataLoader(covid_dataset_test, batch_size=32, shuffle=False, num_workers=4)

    # for i in range(len(train_data)):
    #     train_data[i] = train_data[i].astype(float)

    # transform = transforms.Compose([transforms.ToPILImage(),
    #     transforms.RandomRotation(5),
    #     transforms.ColorJitter(brightness=0.5, contrast=0.5),
    #     # transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # # transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])])

    # covid_dataset = Covid_Dataset(train_data, train_label[:,4], transform=transform)
    # covid_data_loader = DataLoader(covid_dataset, batch_size=8, shuffle=True, num_workers=8)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    modes = ['elu', 'relu', 'leakyrelu']

    for mode in modes:
        model = DeepConvNet(mode)
        myloss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = Learning_rate, weight_decay = 1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr = 1e-3, steps_per_epoch = len(covid_data_loader), epochs = Epochs
        )

        model.to(device)
        myloss.to(device)
        
        if mode == 'elu':
            elu_train_accuracy = []
            elu_test_accuracy = []
        elif mode == 'relu':
            relu_train_accuracy = []
            relu_test_accuracy = []
        elif mode == 'leakyrelu':
            leakyrelu_train_accuracy = []
            leakyrelu_test_accuracy = []
        
        epochs = []
        best_accuracy = 0
        for epoch in tqdm(range(Epochs)):
            epochs.append(epoch + 1)
            model.train()
            num_total = 0
            num_corrects = 0
            for data, label in covid_data_loader:
                optimizer.zero_grad()
                data = data.to(device).float()
                label = label.to(device).long()

                output = model(data)
                loss = myloss(output, label)
                loss.backward()
                optimizer.step()
                scheduler.step()

                num_corrects += (torch.argmax(output, dim=1) == label).sum().item()
                num_total += len(label)
            accuracy = num_corrects / num_total
            print('loss: ', loss)
            print('training accuracy: ', accuracy)
            wandb.log({"loss": loss.item(), 'training accuracy': accuracy})
            if mode == 'elu':
                elu_train_accuracy.append(accuracy)
            elif mode == 'relu':
                relu_train_accuracy.append(accuracy)
            elif mode == 'leakyrelu':
                leakyrelu_train_accuracy.append(accuracy)

        ## save model
        path_model = "./model_DeepConvNet_"+str(mode)+".pkl"
        torch.save(model, path_model)

        print(model)
        print('Finish training')

    cat_transform = {
        0: "Atypical",
        1: "Negative", 
        2: "Typical"
    }

    for mode in modes:
        path_model = "./model_DeepConvNet_"+str(mode)+".pkl"
        model = torch.load(path_model)
        model.eval()
        num_total = 0
        num_corrects = 0

        prediction = []
        id = []

        for data, label in covid_data_loader_test:
            data = data.to(device).float()
            output = model(data)
            for i in range(len(label)):
                id.append(label[i])

            pred = torch.argmax(output, dim=1).cpu().detach().numpy()
            for i in range(len(pred)):
                prediction.append(pred[i])

        print(id)
        print(prediction)

        out_df = pd.DataFrame(columns=["FileID", "Type"])
        for i  in range(len(prediction)):
            out_df.loc[i] = [id[i], cat_transform[prediction[i]]]
        
        filename = mode+'.csv'
        out_df.to_csv(filename)

        print('Finish testing')
