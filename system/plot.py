#!/usr/bin/env python
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
import re
plt.rcParams.update({'font.size': 14})


def plot_summary_one_figure_Compare(save_path, algorithms_list, dataset, goal, global_rounds, 
                    accuracy_lim, loss_lim, window, window_len, isconvex=True):
    num_algo = len(algorithms_list)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    test_acc_, train_acc_, train_loss_ = get_results(
        algorithms_list, dataset, goal, global_rounds)
    for i in range(num_algo):
        print(f"max accurancy of {algorithms_list[i]}: ", test_acc_[i].max())
    test_acc = average_smooth(test_acc_, window=window, window_len=window_len)
    train_loss = average_smooth(train_loss_, window=window, window_len=window_len)
    # train_acc = average_smooth(train_acc_, window=window, window_len=window_len)

    linestyles = ['-', '-', '-', '-', '-', '-', '-', '-']
    markers = ['o', 'v', 's', '*', 'x', 'P', '1', '+']
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm', 'violet', 'maroon']

    # training loss
    plt.figure(1, figsize=(5, 5))
    if isconvex:
        plt.title("$\mu-$" + "strongly convex")
    else:
        plt.title("Nonconvex")
    plt.grid(True)

    for i in range(num_algo):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label, linewidth=1,
                 color=colors[i], marker=markers[i], markevery=0.2, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.ylim([loss_lim[0],  loss_lim[1]])

    if isconvex:
        plt.savefig(save_path + goal + '_' + dataset + "_convex_train.pdf", bbox_inches="tight")
    else:
        plt.savefig(save_path + goal + '_' + dataset + "_non-convex_train.pdf", bbox_inches="tight")

    # Global accurancy
    plt.figure(2, figsize=(5, 5))
    
    if isconvex:
        plt.title("$\mu-$" + "strongly convex")
    else:
        plt.title("Non-convex")
    plt.grid(True)

    for i in range(num_algo):
        label = get_label_name(algorithms_list[i])
        plt.plot(test_acc[i, 1:], linestyle=linestyles[i], label=label, linewidth=1,
                 color=colors[i], marker=markers[i], markevery=0.2, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    plt.ylim([accuracy_lim[0],  accuracy_lim[1]])

    if isconvex:
        plt.savefig(save_path + goal + '_' + dataset + "_convex_test.pdf", bbox_inches="tight")
    else:
        plt.savefig(save_path + goal + '_' + dataset + "_non-convex_test.pdf", bbox_inches="tight")

    plt.close()


def get_results(algorithms_list=[], dataset="", goal="", global_rounds=100):
    num_algo = len(algorithms_list)
    train_acc = np.zeros((num_algo, global_rounds))
    train_loss = np.zeros((num_algo, global_rounds))
    test_acc = np.zeros((num_algo, global_rounds))
    for i in range(num_algo):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_avg"
        res = np.array(read_data_then_delete(file_name, delete=False))
        train_acc[i, :], train_loss[i, :], test_acc[i, :] = res[:, :global_rounds]
    return test_acc, train_acc, train_loss


def average_smooth(data, window='hanning', window_len=1):
    results = []
    if window_len < 3:
        return data
    for i in range(len(data)):
        x = data[i]
        s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]

        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            # w = eval('np.'+window+'(window_len)')
            w = np.hanning(window_len)

        y = np.convolve(w/w.sum(), s, mode='valid')
        results.append(y[window_len-1:])
    return np.array(results)


def get_label_name(name):
    algo = re.compile(r'^[A-Za-z0-9+-]+')

    if name.startswith("pFedMe"):
        if name.startswith("pFedMe_p"):
            return "pFedMe" + " (PM)"
        else:
            return "pFedMe" + " (GM)"
    elif name.startswith("PerAvg"):
        return "Per-FedAvg"
    else:
        return algo.findall(name)[0]


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


if __name__ == '__main__':
    save_path = "../figures/"
    algorithms = ["FedAvg", "PerAvg", "pFedMe", "pFedMe_p"]
    dataset = "Cifar10"
    goal = "cnn"
    global_rounds = 2400
    window = 'flat'
    window_len = 20
    isconvex = True

    if dataset == "mnist":
        accuracy_lim = [0.5, 1]
        loss_lim = [0, 1]
    elif dataset == "Cifar10":
        accuracy_lim = [0.1, 0.7]
        loss_lim = [0.5, 2]
    else:
        accuracy_lim = [0.7, 1]
        loss_lim = [0, 2] 

    plot_summary_one_figure_Compare(
        save_path=save_path, 
        algorithms_list=algorithms, 
        dataset=dataset, 
        goal=goal, 
        global_rounds=global_rounds,
        accuracy_lim=accuracy_lim, 
        loss_lim=loss_lim, 
        window=window, 
        window_len=window_len, 
        isconvex=isconvex
    )