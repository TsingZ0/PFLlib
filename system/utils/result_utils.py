import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", goal="", times=10, global_rounds=800):
    test_acc, train_acc, train_loss = get_all_results_for_one_algo(
        algorithm, dataset, goal, times, global_rounds)
    test_acc_data = np.average(test_acc, axis=0)
    train_acc_data = np.average(train_acc, axis=0)
    train_loss_data = np.average(train_loss, axis=0)

    max_accurancy = []
    for i in range(times):
        max_accurancy.append(test_acc[i].max())

    print("std for best accurancy:", np.std(max_accurancy))
    print("mean for best accurancy:", np.mean(max_accurancy))

    file_name = dataset + "_" + algorithm + "_" + goal + "_avg"
    file_path = "../results/" + file_name + ".h5"
    if (len(test_acc) != 0 & len(train_acc) & len(train_loss)):
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('rs_test_acc', data=test_acc_data)
            hf.create_dataset('rs_train_acc', data=train_acc_data)
            hf.create_dataset('rs_train_loss', data=train_loss_data)


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10, global_rounds=800):
    train_acc = np.zeros((times, global_rounds))
    train_loss = np.zeros((times, global_rounds))
    test_acc = np.zeros((times, global_rounds))
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + \
            algorithms_list[i] + "_" + goal + "_" + str(i)
        train_acc[i, :], train_loss[i, :], test_acc[i, :] = np.array(
            read_data_then_delete(file_name, delete=True))[:, :global_rounds]

    return test_acc, train_acc, train_loss


def read_data_then_delete(file_name, delete=False):
    file_path = "../results/" + file_name + ".h5"
    print("File path: " + file_path)

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))
        rs_train_acc = np.array(hf.get('rs_train_acc'))
        rs_train_loss = np.array(hf.get('rs_train_loss'))

    if delete:
        os.remove(file_path)

    return rs_train_acc, rs_train_loss, rs_test_acc