import numpy as np
import os
import torch
from collections import defaultdict


def read_data(dataset, idx, is_train=True):
    if is_train:
        data_dir = os.path.join('../dataset', dataset, 'train/')
    else:
        data_dir = os.path.join('../dataset', dataset, 'test/')

    file = data_dir + str(idx) + '.npz'
    with open(file, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    return data


def read_client_data(dataset, idx, is_train=True, few_shot=0):
    data = read_data(dataset, idx, is_train)
    if "News" in dataset:
        data_list = process_text(data)
    elif "Shakespeare" in dataset:
        data_list = process_Shakespeare(data)
    else:
        data_list = process_image(data)

    if is_train and few_shot > 0:
        shot_cnt_dict = defaultdict(int)
        data_list_new = []
        for data_item in data_list:
            label = data_item[1].item()
            if shot_cnt_dict[label] < few_shot:
                data_list_new.append(data_item)
                shot_cnt_dict[label] += 1
        data_list = data_list_new
    return data_list

def process_image(data):
    X = torch.Tensor(data['x']).type(torch.float32)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]


def process_text(data):
    X, X_lens = list(zip(*data['x']))
    y = data['y']
    X = torch.Tensor(X).type(torch.int64)
    X_lens = torch.Tensor(X_lens).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [((x, lens), y) for x, lens, y in zip(X, X_lens, y)]


def process_Shakespeare(data):
    X = torch.Tensor(data['x']).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]

