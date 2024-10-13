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
import torchtext
from utils.dataset_utils import check, separate_data, split_data, save_file
from utils.language_utils import tokenizer


import argparse
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--niid', type=str, default="noniid", help="non-iid distribution")
    parser.add_argument('--balance', type=str, default="balance", help="balance data size per client")
    parser.add_argument('--partition', type=str, default="pat", help="partition distribution, dir|pat｜exdir")
    parser.add_argument('--num_users', type=int, default=20, help="number of users")
    parser.add_argument('--alpha', type=float, default=2, help="the degree of imbalance. If partition is pat, alpha is the number of class per client")

    parser.add_argument('--seed', type=int, default=1, help="random seed")

    args = parser.parse_args()
    args.alpha = args.alpha if args.partition == 'dir' else int(args.alpha)
    return args

max_len = 200
max_tokens = 32000


# Allocate data to users
def generate_dataset(niid, balance, partition, args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    num_clients = args.num_users
    dir_path = f"SogouNews_{args.partition}_{args.alpha}_{args.balance}_{args.num_users}/"


    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # Get Sogou_News data
    trainset, testset = torchtext.datasets.SogouNews(root="rawdata/SogouNews")

    trainlabel, traintext = list(zip(*trainset))
    testlabel, testtext = list(zip(*testset))

    dataset_text = []
    dataset_label = []

    dataset_text.extend(traintext)
    dataset_text.extend(testtext)
    dataset_label.extend(trainlabel)
    dataset_label.extend(testlabel)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    vocab, text_list = tokenizer(dataset_text, max_len, max_tokens)
    label_pipeline = lambda x: int(x) - 1
    label_list = [label_pipeline(l) for l in dataset_label]

    text_lens = [len(text) for text in text_list]
    text_list = [(text, l) for text, l in zip(text_list, text_lens)]

    text_list = np.array(text_list)
    label_list = np.array(label_list)

    # dataset = []
    # for i in range(num_classes):
    #     idx = label_list == i
    #     dataset.append(text_list[idx])

    X, y, statistic = separate_data((text_list, label_list), num_clients, num_classes, 
                                    niid, balance, partition, alpha=args.alpha)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition, args.alpha)

    print("The size of vocabulary:", len(vocab))


if __name__ == "__main__":
    args = args_parser()
    niid = True if args.niid == "noniid" else False
    balance = True if args.balance == "balance" else False
    partition = args.partition if args.partition != "-" else None

    generate_dataset(niid, balance, partition, args)