import ujson
import numpy as np
import os
import torch

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CHANNELS = 1

IMAGE_SIZE_CIFAR = 32
NUM_CHANNELS_CIFAR = 3


def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    ran_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(ran_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def get_random_batch_sample(data_x, data_y, batch_size):
    num_parts = len(data_x)//batch_size + 1
    if(len(data_x) > batch_size):
        batch_idx = np.random.choice(list(range(num_parts + 1)))
        sample_index = batch_idx*batch_size
        if(sample_index + batch_size > len(data_x)):
            return (data_x[sample_index:], data_y[sample_index:])
        else:
            return (data_x[sample_index: sample_index+batch_size], data_y[sample_index: sample_index+batch_size])
    else:
        return (data_x, data_y)


def get_batch_sample(data, batch_size):
    data_x = data['x']
    data_y = data['y']

    # np.random.seed(100)
    ran_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(ran_state)
    np.random.shuffle(data_y)

    batched_x = data_x[0:batch_size]
    batched_y = data_y[0:batch_size]
    return (batched_x, batched_y)


def read_data(dataset):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .ujson files with 
        keys 'clients' and 'client_data'
    - the set of train set clients is the same as the set of test set clients

    Return:
        clients: list of client ids
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''

    train_data_dir = os.path.join('../dataset', dataset, 'train')
    test_data_dir = os.path.join('../dataset', dataset, 'test')

    train_file = os.listdir(train_data_dir)[0]
    train_path = os.path.join(train_data_dir, train_file)
    with open(train_path, 'r') as f:
        train_data = ujson.load(f)

    test_file = os.listdir(test_data_dir)[0]
    test_path = os.path.join(test_data_dir, test_file)
    with open(test_path, 'r') as f:
        test_data = ujson.load(f)

    return train_data['clients'], train_data['client_data'], test_data['client_data']


def read_client_data(index, data, dataset):
    idx = data[0][index]
    train_data = data[1][idx]
    test_data = data[2][idx]
    X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
    if(dataset == "mnist"):
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    elif(dataset == "Cifar10"):
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    else:
        X_train = torch.Tensor(X_train).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)

    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    return idx, train_data, test_data
