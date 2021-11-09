from pydicom import dcmread
import os

def get_dataset(dir):
    train_set, valid_set = [], []
    for root, _, files in os.walk(os.path.join(dir, 'train/')):
        for file in files:
            with open(os.path.join(root, file), 'rb') as f:
                train_set.append(dcmread(f))
    for root, _, files in os.walk(os.path.join(dir, 'valid/')):
        for file in files:
            with open(os.path.join(root, file), 'rb') as f:
                valid_set.append(dcmread(f))
    
    return train_set, valid_set

# def resize_dataset(train_set, valid_set):


if __name__ == '__main__':
    train_set, valid_set = get_dataset('./Data/data')
    print(train_set[0].pixel_array.shape)