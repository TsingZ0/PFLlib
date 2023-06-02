import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file

from scipy.io import loadmat


# random.seed(1)
# np.random.seed(1)
# num_clients = 20
# num_classes = 10
# dir_path = "mnist/"


# Allocate data to users
def generate_mnist(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    # if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        # return

    # FIX HTTP Error 403: Forbidden
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    # Get MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.MNIST(
        root=dir_path + "rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=dir_path + "rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

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

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition)
    # train_data, test_data = split_data(X, y)
    # save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic, niid, balance, partition)

    # reshape 1D vector
    dataset_image = [np.reshape(dataset_image[s], (np.product(dataset_image[s].shape),)) for s in range(dataset_image.shape[0])]

    return dataset_image, X, y


# if __name__ == "__main__":
#     niid = True if sys.argv[1] == "noniid" else False
#     balance = True if sys.argv[2] == "balance" else False
#     partition = sys.argv[3] if sys.argv[3] != "-" else None


# UCI dataset ------------------------
# mat = loadmat('/Users/naoki/Documents/GitHub/ci-labo-omu/Dataset/UCI/OptDigits.mat')
# # mat = loadmat('../../Dataset/high_Dimension/COIL20.mat')
# data = np.array(mat.get('data'), dtype=np.float_)
# target = np.array(mat.get('target').flatten(), dtype=int)
# target = target - 1


# random.seed(1)

np.random.seed(1)
num_clients = 20
num_classes = 10
dir_path = "mnist/"

niid = "noniid"
balance = "True"
partition = "dir"

dataset_image, DATA, TARGET = generate_mnist(dir_path, num_clients, num_classes, niid, balance, partition)

path = '/Users/naoki/Documents/GitHub/ci-labo-omu/python/compared_algorithms/federated_learning/PFL-Non-IID/dataset/mnist/train/1.npz'

client1_data = DATA[0]


b = [np.reshape(DATA[0][k], (np.product(DATA[0][k].shape), )) for k in range(DATA[0].shape[0])]

# b = [[np.reshape(DATA[c][k], (np.product(DATA[c][k].shape), )) for k in range(DATA[c].shape[0])] for c in range(num_clients)]



# np.product(client1_data[0].shape)


d=np.array([[1,2],[4,5]])
# client1_data[0]  # 1枚目
# client1_data[1]  # 2枚目


# d = np.load(path)
#
# # dd = torch.from_numpy(np.load(path))
#
#
# import numpy as np
#
# # .npzファイルを読み込む
# d = np.load(path)
#
# # 配列名を確認する
# array_names = d.files
# print(array_names)
#
# # .npzファイル内の配列を取得
# array1 = d['data']
# # array2 = data['配列名2']
#
#
# with np.load('foo.npz') as dd:
#     a = d['a']
