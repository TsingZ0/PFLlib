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
dir_path = "kvasir-0.1/"
data_path = "kvasir/"

# first download rawdata from https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip
# save and unzip in kvasir/rawdata/
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
    data_dir = data_path + 'rawdata/kvasir-dataset-v2/'

    class_names = os.listdir(data_dir)
    print('All class names:', class_names)
    num_classes = len(class_names)

    file_names = []
    labels = []
    for dir in os.listdir(data_dir):
        if dir in class_names:
            label = class_names.index(dir)
            for file_name in os.listdir(os.path.join(data_dir, dir)):
                file_names.append(os.path.join(dir, file_name))
                labels.append(label)
    df = pd.DataFrame({'file_name': file_names, 'class': labels})
    transform = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    dataset = ImageDataset(df, data_dir, transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=len(dataset), 
        shuffle=False, 
    )
    x, y = next(iter(dataloader))
    dataset_image = x.numpy()
    dataset_label = y.numpy()
    
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