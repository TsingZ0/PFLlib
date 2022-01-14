import os
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split

batch_size = 10
train_size = 0.75
least_samples = batch_size / (1-train_size)
alpha = 0.1

def check(config_path, train_path, test_path, num_clients, num_classes, niid=False, 
        real=True, partition=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
            config['num_classes'] == num_classes and \
            config['non_iid'] == niid and \
            config['real_world'] == real and \
            config['partition'] == partition and \
            config['alpha'] == alpha and \
            config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False

def separate_data(data, num_clients, num_classes, niid=False, real=True, partition=None, 
                balance=False, class_per_client=2):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    if partition == None or partition == "noise":
        dataset = []
        for i in range(num_classes):
            idx = dataset_label == i
            dataset.append(dataset_content[idx])

        if not niid or real:
            class_per_client = num_classes

        class_num_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_client[client] > 0:
                    selected_clients.append(client)
            if niid and not real:
                selected_clients = selected_clients[:int(num_clients/num_classes*class_per_client)]

            num_all = len(dataset[i])
            num_clients_ = len(selected_clients)
            if niid and real:
                num_clients_ = np.random.randint(1, len(selected_clients))
            num_per = num_all / num_clients_
            if balance:
                num_samples = [int(num_per) for _ in range(num_clients_-1)]
            else:
                num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_clients_-1).tolist()
            num_samples.append(num_all-sum(num_samples))
            
            if niid:
                # each client is not sure to have all the labels
                selected_clients = list(np.random.choice(selected_clients, num_clients_, replace=False))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if len(X[client]) == 0:
                    X[client] = dataset[i][idx:idx+num_sample]
                    y[client] = i*np.ones(num_sample)
                else:
                    X[client] = np.append(X[client], dataset[i][idx:idx+num_sample], axis=0)
                    y[client] = np.append(y[client], i*np.ones(num_sample), axis=0)
                idx += num_sample
                statistic[client].append((i, num_sample))
                class_num_client[client] -= 1
        
        # if niid and real and partition == "noise":
        #     for client in range(num_clients):
        #         # X[client] = list(map(float, X[client]))
        #         X[client] = np.array(X[client])
        #         X[client] += np.random.normal(0, sigma * client / num_clients)
        #         X[client] = X[client]

    elif niid and partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)
        net_dataidx_map = {}

        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        # additional codes
        for client in range(num_clients):
            idxs = net_dataidx_map[client]
            X[client] = dataset_content[idxs]
            y[client] = dataset_label[idxs]

            for i in np.unique(y[client]):
                statistic[client].append((int(i), int(sum(y[client]==i))))
    else:
        raise EOFError
            

    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


def split_data(X, y, train_size=train_size):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}

    for i in range(len(y)):
        unique, count = np.unique(y[i], return_counts=True)
        if min(count) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X[i], y[i], train_size=train_size, shuffle=True, stratify=y[i])
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X[i], y[i], train_size=train_size, shuffle=True, stratify=None)

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
                num_classes, statistic, niid=False, real=True, partition=None):
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'real_world': real, 
        'partition': partition, 
        'Size of samples for labels in clients': statistic, 
        'alpha': alpha, 
        'batch_size': batch_size, 
    }

    # gc.collect()
    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path[:-5] + str(idx)  + '_' + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path[:-5] + str(idx)  + '_' + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")