import os
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split

batch_size = 4
train_ratio = 0.75
least_samples = batch_size / (1-train_ratio)
sigma = 0.1
beta = 0.5

def check(config_path, train_path, test_path, num_clients, num_labels, niid=False, 
        real=True, partition=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
            config['num_labels'] == num_labels and \
            config['non_iid'] == niid and \
            config['real_world'] == real and \
            config['partition'] == partition:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False

def separate_data(data, num_clients, num_labels, niid=False, real=True, partition=None, 
                class_per_client=2):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    if partition == None or partition == "noise":
        dataset = []
        for i in range(num_labels):
            idx = dataset_label == i
            dataset.append(dataset_content[idx])

        if not niid or real:
            class_per_client = num_labels

        class_num_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_labels):
            selected_clients = [] # 
            for client in range(num_clients):
                if class_num_client[client] > 0:
                    selected_clients.append(client)
            if niid and not real:
                selected_clients = selected_clients[:int(num_clients/num_labels*class_per_client)]

            num_all = len(dataset[i])
            num_clients_ = len(selected_clients)
            if niid and real:
                num_clients_ = np.random.randint(1, len(selected_clients))
            num_per = num_all / num_clients_
            num_samples = np.random.randint(max(num_per/10, least_samples), num_per, num_clients_-1).tolist()
            num_samples.append(num_all-sum(num_samples))
            
            if niid:
                # each client is not sure to have all the labels
                selected_clients = list(np.random.choice(selected_clients, num_clients_, replace=False))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples): 
                X[client] += dataset[i][idx:idx+num_sample].tolist()
                y[client] += (i*np.ones(num_sample)).tolist()
                idx += num_sample
                statistic[client].append((i, num_sample))
                class_num_client[client] -= 1

        if niid and real and partition == "noise":
            for client in range(num_clients):
                # X[client] = list(map(float, X[client]))
                X[client] = np.array(X[client])
                X[client] += np.random.normal(0, sigma * client / num_clients)
                X[client] = X[client].tolist()

    elif niid and partition == "distributed":
        # https://github.com/Xtra-Computing/NIID-Bench/blob/03e2157d1e6b7afcde868956bada0150d4986ddf/utils.py#L273
        # remain to be tested
        idx_map = []
        min_size = 0
        min_require_size = 10
        
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(num_labels):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, num_clients))
                # logger.info("proportions1: ", proportions)
                # logger.info("sum pro1:", np.sum(proportions))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                # logger.info("proportions2: ", proportions)
                proportions = proportions / proportions.sum()
                # logger.info("proportions3: ", proportions)
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # logger.info("proportions4: ", proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if num_labels == 2 and num_clients <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for client in range(num_clients):
            np.random.shuffle(idx_batch[client])
            idx_map[client] = idx_batch[client]

            idxs = idx_map[client]
            X[client] += dataset_content[idxs].tolist()
            y[client] += dataset_label[idxs].tolist()

            for i in np.unique(y[client]):
                statistic[client].append((i, sum(y[client]==i)))
            

    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


def split_data(X, y, train_ratio=train_ratio):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}

    for i in range(len(y)):
        unique, count = np.unique(y[i], return_counts=True)
        if min(count) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X[i], y[i], train_ratio=train_ratio, shuffle=True, stratify=y[i])
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X[i], y[i], train_ratio=train_ratio, shuffle=True, stratify=None)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    # gc.collect()

    return train_data, test_data

def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 
                num_labels, statistic, niid=False, real=True, partition=None):
    config = {
        'num_clients': num_clients, 
        'num_labels': num_labels, 
        'non_iid': niid, 
        'real_world': real, 
        'partition': partition, 
        'Size of samples for labels in clients': statistic, 
    }

    # gc.collect()

    for idx, train_dict in enumerate(train_data):
        with open(train_path[:-5] + str(idx)  + '_' + '.json', 'w') as f:
            ujson.dump(train_dict, f)
    for idx, test_dict in enumerate(test_data):
        with open(test_path[:-5] + str(idx)  + '_' + '.json', 'w') as f:
            ujson.dump(test_dict, f)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")