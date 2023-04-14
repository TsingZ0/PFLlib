import numpy as np
import os
import random
from utils.HAR_utils import *


random.seed(1)
np.random.seed(1)
data_path = "har/"
dir_path = "har/"


def generate_har(dir_path):
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
    if not os.path.exists(data_path+'rawdata/UCI HAR Dataset.zip'):
        os.system(f"wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip -P {data_path}rawdata/")
    if not os.path.exists(data_path+'rawdata/UCI HAR Dataset/'):
        os.system(f"unzip {data_path}rawdata/'UCI HAR Dataset.zip' -d {data_path}rawdata/")

    X, y = load_data_har(data_path+'rawdata/')
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


def load_data_har(data_folder):
    str_folder = data_folder + 'UCI HAR Dataset/'
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_"
    ]

    str_train_files = [str_folder + 'train/' + 'Inertial Signals/' + item + 'train.txt' for item in
                        INPUT_SIGNAL_TYPES]
    str_test_files = [str_folder + 'test/' + 'Inertial Signals/' +
                        item + 'test.txt' for item in INPUT_SIGNAL_TYPES]
    str_train_y = str_folder + 'train/y_train.txt'
    str_test_y = str_folder + 'test/y_test.txt'
    str_train_id = str_folder + 'train/subject_train.txt'
    str_test_id = str_folder + 'test/subject_test.txt'

    X_train = format_data_x(str_train_files)
    X_test = format_data_x(str_test_files)
    Y_train = format_data_y(str_train_y)
    Y_test = format_data_y(str_test_y)
    id_train = read_ids(str_train_id)
    id_test = read_ids(str_test_id)
    
    X_train, X_test = X_train.reshape((-1, 9, 1, 128)), X_test.reshape((-1, 9, 1, 128))

    X = np.concatenate((X_train, X_test), axis=0)
    Y = np.concatenate((Y_train, Y_test), axis=0)
    ID = np.concatenate((id_train, id_test), axis=0)

    XX, YY = [], []
    for i in np.unique(ID):
        idx = ID == i
        XX.append(X[idx])
        YY.append(Y[idx])

    return XX, YY


if __name__ == "__main__":
    generate_har(dir_path)