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

import time
import numpy as np
import os
import random
import torchvision.transforms as transforms
from utils.dataset_utils import split_data, save_file
from os import path
from PIL import Image
from torch.utils.data import DataLoader, Dataset

 
# https://github.com/FengHZ/KD3A/blob/master/datasets/DomainNet.py
def read_domainnet_data(dataset_path, domain_name, split="train"):
    data_paths = []
    data_labels = []
    split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            data_path = path.join(dataset_path, data_path)
            label = int(label)
            data_paths.append(data_path)
            data_labels.append(label)
    return data_paths, data_labels


class DomainNet(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(DomainNet, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.domain_name = domain_name

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)


def get_domainnet_dloader(dataset_path, domain_name):
    train_data_paths, train_data_labels = read_domainnet_data(dataset_path, domain_name, split="train")
    test_data_paths, test_data_labels = read_domainnet_data(dataset_path, domain_name, split="test")
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_dataset = DomainNet(train_data_paths, train_data_labels, transforms_train, domain_name)
    train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_dataset = DomainNet(test_data_paths, test_data_labels, transforms_test, domain_name)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
    return train_loader, test_loader


random.seed(1)
np.random.seed(1)
data_path = "DomainNet/"
dir_path = "DomainNet/"

# Allocate data to users
def generate_dataset(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    root = data_path+"rawdata"
    
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    urls = [
        'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip', 
        'http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip', 
        'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip', 
        'http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip', 
        'http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip', 
        'http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip', 
    ]
    http_head = 'http://csr.bu.edu/ftp/visda/2019/multi-source/'
    # Get DomainNet data
    if not os.path.exists(root):
        os.makedirs(root)
        for d, u in zip(domains, urls):
            os.system(f'wget {u} -P {root}')
            os.system(f'unzip {root}/{d}.zip -d {root}')
            os.system(f'wget {http_head}domainnet/txt/{d}_train.txt -P {root}/splits')
            os.system(f'wget {http_head}domainnet/txt/{d}_test.txt -P {root}/splits')

    X, y = [], []
    for d in domains:
        train_loader, test_loader = get_domainnet_dloader(root, d)

        for _, tt in enumerate(train_loader):
            train_data, train_label = tt
        for _, tt in enumerate(test_loader):
            test_data, test_label = tt

        dataset_image = []
        dataset_label = []

        dataset_image.extend(train_data.cpu().detach().numpy())
        dataset_image.extend(test_data.cpu().detach().numpy())
        dataset_label.extend(train_label.cpu().detach().numpy())
        dataset_label.extend(test_label.cpu().detach().numpy())

        X.append(np.array(dataset_image))
        y.append(np.array(dataset_label))

    labelss = []
    for yy in y:
        labelss.append(len(set(yy)))
    num_clients = len(y)
    print(f'Number of labels: {labelss}')
    print(f'Number of clients: {num_clients}')

    statistic = [[] for _ in range(num_clients)]
    for client in range(num_clients):
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))


    train_data, test_data = split_data(X, y)
    # modify the code in YOUR_ENV/lib/python3.8/site-packages/numpy/lib Line #678 from protocol=3 to protocol=4
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, max(labelss), 
        statistic, None, None, None)


if __name__ == "__main__":
    generate_dataset(dir_path)