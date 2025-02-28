import sys
import pandas as pd
import numpy as np
import os
import random
import torchvision.transforms as transforms
from sklearn.utils import resample
from sklearn.utils import shuffle
from utils.dataset_utils import check, separate_data, split_data, save_file, ImageDataset
from torch.utils.data import DataLoader


random.seed(1)
np.random.seed(1)
num_clients = 20
img_size = 64
num_classes = 2
dir_path = "COVIDx-0.1/"
data_path = "COVIDx/"

# first download rawdata from https://www.kaggle.com/datasets/andyczhao/covidx-cxr2/data
# save and unzip in COVIDx/rawdata/
# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # Get data
    if not os.path.exists(data_path):
        raise FileExistsError
    train_dir = data_path + 'rawdata/train/'
    val_dir = data_path + 'rawdata/val/'
    test_dir = data_path + 'rawdata/test/'

    train_df = pd.read_csv(data_path + 'rawdata/train.txt', sep=" ", header=None)
    train_df.columns=['patient_id', 'file_name', 'class', 'data_source']
    train_df['class'] = train_df['class'] == 'positive'
    train_df['class'] = train_df['class'].astype(int)
    val_df = pd.read_csv(data_path + 'rawdata/val.txt', sep=" ", header=None)
    val_df.columns=['patient_id', 'file_name', 'class', 'data_source']
    val_df['class'] = val_df['class'] == 'positive'
    val_df['class'] = val_df['class'].astype(int)
    test_df = pd.read_csv(data_path + 'rawdata/test.txt', sep=" ", header=None)
    test_df.columns=['patient_id', 'file_name', 'class', 'data_source']
    test_df['class'] = test_df['class'] == 'positive'
    test_df['class'] = test_df['class'].astype(int)

    # keep balanced in total
    negative = train_df[train_df['class']==0]
    positive = train_df[train_df['class']==1]
    df_majority_downsampled = resample(positive, replace=True, n_samples=min(len(negative), len(positive)))
    train_df = pd.concat([negative, df_majority_downsampled])
    train_df = shuffle(train_df)

    print(f'Number of classes: {num_classes}\n')
    
    transform = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    
    dataset = ImageDataset(
        dataframe=train_df, 
        image_folder=train_dir, 
        transform=transform
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=len(dataset), 
        shuffle=False, 
    )
    x, y = next(iter(dataloader))
    dataset_image = x.numpy()
    dataset_label = y.numpy()
    
    dataset = ImageDataset(
        dataframe=val_df, 
        image_folder=val_dir, 
        transform=transform
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=len(dataset), 
        shuffle=False, 
    )
    x, y = next(iter(dataloader))
    dataset_image = np.concatenate((dataset_image, x.numpy()), axis=0)
    dataset_label = np.concatenate((dataset_label, y.numpy()), axis=0)
    
    dataset = ImageDataset(
        dataframe=test_df, 
        image_folder=test_dir, 
        transform=transform
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=len(dataset), 
        shuffle=False, 
    )
    x, y = next(iter(dataloader))
    dataset_image = np.concatenate((dataset_image, x.numpy()), axis=0)
    dataset_label = np.concatenate((dataset_label, y.numpy()), axis=0)

    print('Total data amount', len(dataset_image), len(dataset_label))

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,  
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)