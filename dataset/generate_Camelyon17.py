import numpy as np
import os
import random
import torchvision.transforms as transforms
from utils.dataset_utils import split_data, save_file
from wilds import get_dataset


random.seed(1)
np.random.seed(1)
num_clients = 5
num_classes = 2
dir_path = "Camelyon17/"

# used to generate small sub-datasets
# modify the values of max_num to assign exact data amount to each client and class
# 10000000 means using the full dataset
max_num = [
    (10000000, 10000000), 
    (10000000, 10000000), 
    (10000000, 10000000), 
    (10000000, 10000000), 
    (10000000, 10000000), 
]
cur_num = [
    [0, 0], 
    [0, 0], 
    [0, 0], 
    [0, 0], 
    [0, 0], 
]


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
        dataset='camelyon17', 
        root_dir=dir_path+'rawdata', 
        download=True)
    print('metadata columns:', dataset.metadata_fields)

    transform=transforms.ToTensor()

    dataset_train = dataset.get_subset('train')
    dataset_val = dataset.get_subset('val')
    dataset_test = dataset.get_subset('test')

    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = []

    def reassign_data(dataset, num_hospitals):
        done_cnt = 0
        for xx, yy, meta_data in dataset:
            hospital_id = meta_data[0].item()
            label = meta_data[2].item()
            if cur_num[hospital_id][label] < max_num[hospital_id][label]:
                X[hospital_id].append(transform(xx).cpu().numpy())
                y[hospital_id].append(yy.item())
                cur_num[hospital_id][label] += 1
            elif cur_num[hospital_id][label] == max_num[hospital_id][label]:
                cur_num[hospital_id][label] += 1
                done_cnt += 1
            if done_cnt >= num_hospitals*2:
                break

    reassign_data(dataset_train, 3)
    reassign_data(dataset_val, 1)
    reassign_data(dataset_test, 1)

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