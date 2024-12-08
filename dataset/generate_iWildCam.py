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
import random
import torchvision.transforms as transforms
from utils.dataset_utils import split_data, save_file
from wilds import get_dataset


random.seed(1)
np.random.seed(1)
img_size = 64
least_samples = 100
least_class_per_client = 2
dir_path = "iWildCam/"


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

    # Get data
    dataset = get_dataset(
        dataset='iwildcam', 
        root_dir=dir_path+'rawdata', 
        download=True)
    print('metadata columns:', dataset.metadata_fields)

    transform = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor()])

    dataset_train = dataset.get_subset('train')
    dataset_val = dataset.get_subset('val')
    dataset_test = dataset.get_subset('test')

    num_clients = 323
    X_o = [[] for _ in range(num_clients)]
    y_o = [[] for _ in range(num_clients)]
    statistic = []

    def reassign_data(dataset):
        for xx, yy, meta_data in dataset:
            camera_trap_id = meta_data[0].item()
            X_o[camera_trap_id].append(transform(xx).cpu().numpy())
            y_o[camera_trap_id].append(yy.item())

    reassign_data(dataset_train)
    reassign_data(dataset_val)
    reassign_data(dataset_test)

    X = []
    y = []

    classes = set()
    for i in range(num_clients):
        if len(y_o[i]) > least_samples and len(set(y_o[i])) > least_class_per_client:
            X.append(X_o[i])
            y.append(y_o[i])
            classes.update(y_o[i])

    num_clients = len(y)
    num_classes = len(classes)
    print(f'Number of clients: {num_clients}')
    print(f'Number of classes: {num_classes}')

    for i in range(num_clients):
        statistic.append([])
        y_arr = np.array(y[i])
        for yc in sorted(np.unique(y_arr)):
            statistic[-1].append((int(yc), int(sum(y_arr == yc))))

    for i in range(num_clients):
        print(f"Client {i}\t Size of data: {len(X[i])}\t Labels: ", np.unique(y[i]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[i]])
        print("-" * 50)

    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, 
        num_clients, num_classes, statistic, None, None, None)


if __name__ == "__main__":
    generate_dataset(dir_path)