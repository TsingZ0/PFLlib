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
from utils.dataset_utils import split_data, save_file
from scipy.sparse import coo_matrix
from os import path

 
# https://github.com/FengHZ/KD3A/blob/master/datasets/AmazonReview.py
def load_amazon(base_path):
    dimension = 5000
    amazon = np.load(path.join(base_path, "amazon.npz"))
    amazon_xx = coo_matrix((amazon['xx_data'], (amazon['xx_col'], amazon['xx_row'])),
                           shape=amazon['xx_shape'][::-1]).tocsc()
    amazon_xx = amazon_xx[:, :dimension]
    amazon_yy = amazon['yy']
    amazon_yy = (amazon_yy + 1) / 2
    amazon_offset = amazon['offset'].flatten()
    # Partition the data into four categories and for each category partition the data set into training and test set.
    data_name = ["books", "dvd", "electronics", "kitchen"]
    num_data_sets = 4
    data_insts, data_labels, num_insts = [], [], []
    for i in range(num_data_sets):
        data_insts.append(amazon_xx[amazon_offset[i]: amazon_offset[i + 1], :])
        data_labels.append(amazon_yy[amazon_offset[i]: amazon_offset[i + 1], :])
        num_insts.append(amazon_offset[i + 1] - amazon_offset[i])
        # Randomly shuffle.
        r_order = np.arange(num_insts[i])
        np.random.shuffle(r_order)
        data_insts[i] = data_insts[i][r_order, :]
        data_labels[i] = data_labels[i][r_order, :]
        data_insts[i] = data_insts[i].todense().astype(np.float32)
        data_labels[i] = data_labels[i].ravel().astype(np.int64)
    return data_insts, data_labels


random.seed(1)
np.random.seed(1)
data_path = "AmazonReview/"
dir_path = "AmazonReview/"

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
    
    # Get AmazonReview data
    if not os.path.exists(root):
        os.makedirs(root)
        os.system(f'wget https://drive.google.com/u/0/uc?id=1QbXFENNyqor1IlCpRRFtOluI2_hMEd1W&export=download -P {root}')

    X, y = load_amazon(root)

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
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, max(labelss), 
        statistic, None, None, None)


if __name__ == "__main__":
    generate_dataset(dir_path)