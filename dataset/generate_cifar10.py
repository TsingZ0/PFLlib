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

import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.utils.dataset_utils import check, get_path, separate_data, split_data, save_file
import torch.distributed as dist
from cifar_utils import CIFAR10Pair, CIFAR100Pair, train_transform, test_transform, train_transform_100, test_transform_100

# Allocate data to users
def generate_cifar10(dir_path: str, num_clients: int, num_classes: int, niid: bool, balance: bool, partition: str, niid_alpha: float, seed: int, isCon: bool):
    random.seed(seed)
    np.random.seed(seed)

    raw_data, config_path, train_path, test_path, is_exist = get_path(dir_path, num_clients, num_classes, niid, balance, partition, niid_alpha)
    if is_exist: return train_path, test_path

    # Get Cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    if isCon:
        trainset =  CIFAR10Pair(root=raw_data+"rawdata", train=True, transform= train_transform, download=False)
        testset = CIFAR10Pair(root=raw_data+"rawdata", train=False, transform= test_transform, download=False)
    else:
        trainset = torchvision.datasets.CIFAR10(
        root=raw_data+"rawdata", train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(
        root=raw_data+"rawdata", train=False, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False, drop_last=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False, drop_last=True)

    if isCon:
        for _, train_data in enumerate(trainloader, 0):
            trainset.data1, trainset.data2, trainset.data, trainset.targets = train_data
        for _, test_data in enumerate(testloader, 0):
            testset.data1, testset.data2, testset.data, testset.targets = test_data
    else:
        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        for _, test_data in enumerate(testloader, 0):
            testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    #对 data1,data2,data 做拼接，但要先扩展一维，做一个 unsqueeze 的操作，并且拼接后的数据组成一个新的列表
    if isCon:
        for i in range(len(trainset.data)):
            dataset_image.append(np.concatenate((np.expand_dims(trainset.data1[i].cpu().detach().numpy(),axis=0),np.expand_dims(trainset.data2[i].cpu().detach().numpy(),axis=0),np.expand_dims(trainset.data[i].cpu().detach().numpy(),axis=0)),axis=0))
        for i in range(len(testset.data)):
            dataset_image.append(np.concatenate((np.expand_dims(testset.data1[i].cpu().detach().numpy(),axis=0),np.expand_dims(testset.data2[i].cpu().detach().numpy(),axis=0),np.expand_dims(testset.data[i].cpu().detach().numpy(),axis=0)),axis=0))      
    else:  
        dataset_image.extend(trainset.data.cpu().detach().numpy())
        dataset_image.extend(testset.data.cpu().detach().numpy())
        
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,  
                                    niid, balance, partition, class_per_client=2, niid_alpha=0.1)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition, niid_alpha)
    return train_path, test_path


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    # generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition, niid_alpha, seed)