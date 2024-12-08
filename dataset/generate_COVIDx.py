# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import sys
import pandas as pd
import numpy as np
import os
import random
import torchvision.transforms as transforms
from sklearn.utils import resample
from sklearn.utils import shuffle
from utils.dataset_utils import check, separate_data, split_data, save_file
from torch.utils.data import Dataset, DataLoader
from PIL import Image


random.seed(1)
np.random.seed(1)
num_clients = 20
img_size = 64
num_classes = 2
dir_path = "COVIDx/"
data_path = "COVIDx/"

# first download rawdata from https://www.kaggle.com/datasets/andyczhao/covidx-cxr2/data
class ImageDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing file names
            image_folder (str): Path to the folder containing the images
            transform (callable, optional): Optional transform to be applied to the image
        """
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the file name from the DataFrame
        img_name = self.dataframe.iloc[idx]['file_name']
        img_label = int(self.dataframe.iloc[idx]['class'] == 'positive')
        img_path = os.path.join(self.image_folder, img_name)
        
        # Load the image using PIL
        image = Image.open(img_path).convert('RGB')  # Ensure RGB if not grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_label


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = data_path + "config.json"
    train_path = data_path + "train/"
    test_path = data_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # Get data
    train_dir = dir_path + 'rawdata/train/'
    val_dir = dir_path + 'rawdata/val/'
    test_dir = dir_path + 'rawdata/test/'

    train_df = pd.read_csv(dir_path + 'rawdata/train.txt', sep=" ", header=None)
    train_df.columns=['patient_id', 'file_name', 'class', 'data_source']
    train_df.drop(columns=['patient_id', 'data_source'])
    val_df = pd.read_csv(dir_path + 'rawdata/val.txt', sep=" ", header=None)
    val_df.columns=['patient_id', 'file_name', 'class', 'data_source']
    val_df.drop(columns=['patient_id', 'data_source'])
    test_df = pd.read_csv(dir_path + 'rawdata/test.txt', sep=" ", header=None)
    test_df.columns=['patient_id', 'file_name', 'class', 'data_source']
    test_df.drop(columns=['patient_id', 'data_source'])

    # keep balanced in total
    negative  = train_df[train_df['class']=='negative']
    positive = train_df[train_df['class']=='positive']
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