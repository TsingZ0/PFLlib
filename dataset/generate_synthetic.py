import numpy as np
import os
import sys
from utils.dataset_utils import check, split_data, save_file


np.random.seed(0)
num_clients = 20
num_labels = 10
dir_path = "synthetic/"

# Allocate data to users
def generate_synthetic(dir_path=dir_path, num_clients=num_clients, num_labels=num_labels, niid=False, sigma1=0.5, sigma2=0.5):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "data/config.json"
    train_path = dir_path + "data/train/train.json"
    test_path = dir_path + "data/test/test.json"

    if check(config_path, train_path, test_path, num_clients, num_labels, niid):
        return

    X, y, statistic = generate_synthetic_(num_clients, num_labels, niid, sigma1, sigma2)  # synthetic (0.5, 0.5)
    train_data, test_data = split_data(X, y, num_clients)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_labels, statistic, niid)


def generate_synthetic_(num_clients, num_labels, niid, mu=0, sigma1=0.5, sigma2=0.5, dimension=60):

    samples_per_client = (np.random.lognormal(4, 2, (num_clients)).astype(int) + 50) * 5

    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]

    # define some eprior
    mean_W = np.random.normal(mu, sigma1, num_clients)
    B = np.random.normal(mu, sigma2, num_clients)
    mean_x = np.zeros((num_clients, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(num_clients):
        if niid:
            mean_x[i] = np.random.normal(B[i], 1, dimension)
        else:
            mean_x[i] = np.ones(dimension) * B[i]

    statistic = []
    for i in range(num_clients):
        W, b = None, None
        if niid:
            W = np.random.normal(mean_W[i], 1, (dimension, num_labels))
            b = np.random.normal(mean_W[i], 1,  num_labels)
        else:
            W = np.random.normal(mu, 1, (dimension, num_labels))
            b = np.random.normal(mu, 1,  num_labels)

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_client[i])
        yy = np.zeros(samples_per_client[i])

        for j in range(samples_per_client[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X[i] = xx.tolist()
        y[i] = yy.tolist()

        statistic.append({"Samples": int(samples_per_client[i])})

        print("Client {} has {} exampls".format(i, len(y[i])))

    return X, y, statistic


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex/sum_ex


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False

    generate_synthetic(dir_path=dir_path, num_clients=num_clients, num_labels=num_labels, niid=niid)