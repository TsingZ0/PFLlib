# -*- coding: utf-8 -*-
# @Time    : 2024/4/11
# By Qiantao Yang

import numpy as np
import os
import sys
import random
import json
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt


def get_data(dataset_name):
    client_dict = []
    if dataset_name == 'Cifar10':
        dir_path = "Cifar10/"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        testset = torchvision.datasets.MNIST(root=dir_path + "rawdata", train=False, download=True, transform=transform)
        with open(dir_path + 'config.json') as f:
            client_data = json.load(f)
    elif dataset_name == 'Cifar100':
        dir_path = "Cifar100/"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        testset = torchvision.datasets.MNIST(root=dir_path + "rawdata", train=False, download=True, transform=transform)
        with open(dir_path + 'config.json') as f:
            client_data = json.load(f)
    elif dataset_name == 'FashionMNIST':
        dir_path = "FashionMNIST/"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        testset = torchvision.datasets.MNIST(root=dir_path + "rawdata", train=False, download=True, transform=transform)
        with open(dir_path + 'config.json') as f:
            client_data = json.load(f)
    elif dataset_name == 'MNIST':
        dir_path = "MNIST/"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        testset = torchvision.datasets.MNIST(root=dir_path + "rawdata", train=False, download=True, transform=transform)
        with open(dir_path + 'config.json') as f:
            client_dict = json.load(f)
    else:
        print('The dataset is not available at this time')
    
    dataset_label = np.array(testset.classes)
    return dataset_label, client_dict


def show_data_distribution(dataset_name):
    dataset_label, client_dict = get_data(dataset_name)
    
    label_distribution = [[] for _ in range(len(dataset_label))]
    
    client_num = client_dict['num_clients']
    client_data = client_dict['Size of samples for labels in clients']
    
    client_labels = {clientid: [] for clientid in range(client_num)}
    
    for c_id, c_data in enumerate(client_data):
        for data in c_data:
            label_distribution[data[0]].append(c_id)
            client_labels[c_id].append(data[0])
    print('The client owns the data classification')
    for key, value in client_labels.items():
        print('Client Id: {:>3} | Dataset Classes: {}'.format(key, set(value)))
    
    print('\n Each type of label is distributed across that client ')
    for label_id, data in enumerate(label_distribution):
        print('Label ID: {:>10} | Client ID: {}'.format(dataset_label[label_id], data))
    
    plt.figure(figsize=(20, 6))
    plt.hist(label_distribution, stacked=True, bins=np.arange(-0.5, client_num + 2, 1),
             label=dataset_label,
             rwidth=0.5)
    plt.xticks(np.arange(client_num), ["%d" % c_id for c_id in range(client_num)])
    plt.ylabel("Number of samples")
    plt.xlabel("Client ID")
    plt.legend()
    plt.title("Dataset {} Distribution: {} Non-IID: {}".format(dataset_name, client_dict['partition'],
                                                               client_dict['partition']))
    plt.show()


if __name__ == "__main__":
    show_data_distribution('MNIST')
