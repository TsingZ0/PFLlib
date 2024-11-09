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
from utils.HAR_utils import *


random.seed(1)
np.random.seed(1)
data_path = "PAMAP2/"
dir_path = "PAMAP2/"

sample_window = 256 # 2.56s
# sample_window = 128 # 1.28s

def generate_dataset(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # download data
    if not os.path.exists(data_path+'rawdata/PAMAP2_Dataset.zip'):
        os.system(f"wget http://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip -P {data_path}rawdata/")
    if not os.path.exists(data_path+'rawdata/PAMAP2_Dataset/'):
        os.system(f"unzip {data_path}rawdata/'PAMAP2_Dataset.zip' -d {data_path}rawdata/")

    X, y = load_data_PAMAP2(data_path+'rawdata/')
    statistic = []
    num_clients = len(y)
    num_classes = len(np.unique(np.concatenate(y, axis=0)))
    for i in range(num_clients):
        statistic.append([])
        for yy in sorted(np.unique(y[i])):
            idx = y[i] == yy
            statistic[-1].append((int(yy), int(len(X[i][idx]))))

    for i in range(num_clients):
        print(f"Client {i}\t Size of data: {len(X[i])}\t Labels: ", np.unique(y[i]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[i]])
        print("-" * 50)

    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic)


def load_data_PAMAP2(data_folder):
    s_folder = data_folder + 'PAMAP2_Dataset/'

    file_names = [
        ['Protocol/subject101.dat', 'Optional/subject101.dat'], 
        ['Protocol/subject102.dat'], 
        ['Protocol/subject103.dat'], 
        ['Protocol/subject104.dat'], 
        ['Protocol/subject105.dat', 'Optional/subject105.dat'], 
        ['Protocol/subject106.dat', 'Optional/subject106.dat'], 
        ['Protocol/subject107.dat'], 
        ['Protocol/subject108.dat', 'Optional/subject108.dat'], 
        ['Protocol/subject109.dat', 'Optional/subject109.dat']
    ]

    XX, YY = [], []
    for fns in file_names:
        data = []
        for fn in fns:
            i_data = np.loadtxt(s_folder+fn, dtype=np.float32)
            # print(fn, i_data.shape)
            i_data = np.concatenate((i_data[:, :2], 
                                    i_data[:, 4:7], i_data[:, 10:16], 
                                    i_data[:, 21:24], i_data[:, 27:33], 
                                    i_data[:, 38:41], i_data[:, 44:50]), 
                            axis=1)
            data.append(i_data)
        data = np.concatenate(data, axis=0)
        # HR_no_NaN = complete_HR(data[:, 2])
        # data[:, 2] = HR_no_NaN
        data = np.nan_to_num(data, nan=0)
        data[:, 2:] /= abs(data[:, 2:]).max(axis=0)
        idx = 0
        len_data = len(data)
        X, Y = [], []
        while idx+sample_window < len_data:
            ddd = data[idx: idx+sample_window]
            unique, counts = np.unique(ddd[:, 1].astype('int32'), return_counts=True)
            y = unique[0]
            x = ddd[:, 2:].reshape((1, -1, 3, 9))
            x = np.transpose(x, (0, 3, 2, 1))
            X.append(x)
            Y.append(y)
            idx += sample_window // 2
        X = np.concatenate(X, axis=0)
        Y = np.array(Y)
        X, Y = del_labels(X, Y)
        Y = adjust_idx_labels(Y)
        YY.append(Y)
        XX.append(X)

    return XX, YY

def del_labels(data_x, data_y):

    idy = np.where(data_y == 0)[0]
    labels_delete = idy

    idy = np.where(data_y == 8)[0]    
    labels_delete = np.concatenate([labels_delete, idy])
    
    idy = np.where(data_y == 9)[0]    
    labels_delete = np.concatenate([labels_delete, idy])
    
    idy = np.where(data_y == 10)[0]    
    labels_delete = np.concatenate([labels_delete, idy])
    
    idy = np.where(data_y == 11)[0]    
    labels_delete = np.concatenate([labels_delete, idy])
    
    idy = np.where(data_y == 18)[0]    
    labels_delete = np.concatenate([labels_delete, idy])
    
    idy = np.where(data_y == 19)[0]    
    labels_delete = np.concatenate([labels_delete, idy])
    
    idy = np.where(data_y == 20)[0]    
    labels_delete = np.concatenate([labels_delete, idy])
    
    return np.delete(data_x, labels_delete, 0), np.delete(data_y, labels_delete, 0)

def adjust_idx_labels(data_y):
    data_y[data_y == 24] = 0
    data_y[data_y == 12] = 8
    data_y[data_y == 13] = 9
    data_y[data_y == 16] = 10
    data_y[data_y == 17] = 11

    return data_y

def complete_HR(data):
    pos_NaN = np.isnan(data)
    idx_NaN = np.where(pos_NaN == False)[0]
    data_no_NaN = data * 0
    for idx in range(idx_NaN.shape[0] - 1):
        data_no_NaN[idx_NaN[idx] : idx_NaN[idx + 1]] = data[idx_NaN[idx]]
    
    data_no_NaN[idx_NaN[-1] :] = data[idx_NaN[-1]]
    
    return data_no_NaN

if __name__ == "__main__":
    generate_dataset(dir_path)