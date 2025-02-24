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
import shutil
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
from torchvision.datasets import ImageFolder, DatasetFolder

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "TinyImagenet/"

# https://github.com/QinbinLi/MOON/blob/6c7a4ed1b1a8c0724fa2976292a667a828e3ff5d/datasets.py#L148
class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)

def organize_val_folder(dir_path):
    """
    Parse val_annotations.txt and copy validation images into subfolders named after their class labels,
    ensuring that ImageFolder correctly retrieves each image's class information.
    Only images listed in val_annotations.txt are processed.
    
    Parameters:
        dir_path: dataset root directory
    """

    # Set the validation directory path
    val_dir = dir_path + "rawdata/tiny-imagenet-200/val/"
    images_dir = os.path.join(val_dir, "images")
    annotations_file = os.path.join(val_dir, "val_annotations.txt")

    if not os.path.exists(annotations_file):
        print("file val_annotations.txt not found:", annotations_file)
        exit(1)

    # Read and parse the file, create class subdirectories and move images
    with open(annotations_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            img_name, class_name = parts[0], parts[1]
            class_dir = os.path.join(images_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            src = os.path.join(images_dir, img_name)
            dst = os.path.join(class_dir, img_name)
            if os.path.exists(src):
                shutil.move(src, dst)
            else:
                print("Image not found:", src)

    print("Validation set reorganization completed.")

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

    # Get data
    if not os.path.exists(f'{dir_path}/rawdata/'):
        os.system(f'wget --directory-prefix {dir_path}/rawdata/ http://cs231n.stanford.edu/tiny-imagenet-200.zip')
        os.system(f'unzip {dir_path}/rawdata/tiny-imagenet-200.zip -d {dir_path}/rawdata/')
    else:
        print('rawdata already exists.\n')
    
    organize_val_folder(dir_path)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = ImageFolder_custom(root=dir_path+'rawdata/tiny-imagenet-200/train/', transform=transform)
    testset = ImageFolder_custom(root=dir_path+'rawdata/tiny-imagenet-200/val/images', transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    unique, counts = np.unique(testset.targets.cpu().detach().numpy(), return_counts=True)
    print("Validation set sample counts for each class:", dict(zip(unique, counts)))
    
    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=20)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)
