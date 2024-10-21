import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file

from args import args_parser

# Allocate data to users
def generate_dataset(niid, balance, partition, args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    num_clients = args.num_users
    dir_path = f"Country211_{args.partition}_{args.alpha if args.partition == 'dir' else args.class_per_client}_{args.balance}_{args.num_users}/"


    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    dataset_image = []
    dataset_label = []
        
    # Get Country211 data
    transform = transforms.Compose(
        [transforms.Resize((64, 64)), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5), (0.5))]
    )

    def load_data(split="train"):
        trainset = torchvision.datasets.Country211(
            root="rawdata/Country211", split=split, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=len(trainset), shuffle=False)
        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        dataset_image.extend(trainset.data.cpu().detach().numpy())
        dataset_label.extend(trainset.targets.cpu().detach().numpy())

    load_data("train")
    load_data("valid")
    load_data("test")

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, alpha=args.alpha, class_per_client=args.class_per_client)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition, args.alpha, class_per_client=args.class_per_client)


if __name__ == "__main__":
    args = args_parser()
    niid = True if args.niid == "noniid" else False
    balance = True if args.balance == "balance" else False
    partition = args.partition if args.partition != "-" else None

    generate_dataset(niid, balance, partition, args)