#!/usr/bin/env python
import torch
import argparse
import os
import time
import numpy as np
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverpFedMe import pFedMe
from flcore.servers.serverperavg import PerAvg
from flcore.trainmodel.models import *
from utils.result_utils import average_data


torch.manual_seed(0)

def run(goal, dataset, device, algorithm, model, local_batch_size, local_learning_rate, global_rounds, local_steps, num_clients, 
        total_clients, beta, lamda, K, personalized_learning_rate, times, drop_ratio, train_slow_ratio, send_slow_ratio, 
        time_select, time_threthold, mu):

    time_list = []

    for i in range(times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start, server, classifier = time.time(), None, None

        # Generate model
        if model == "mclr":
            if dataset == "mnist":
                model = Mclr_Logistic(1*28*28)
            elif dataset == "Cifar10":
                model = Mclr_Logistic(3*32*32)
            else:
                model = Mclr_Logistic(60)

        elif model == "cnn":
            if dataset == "mnist":
                model = LeNet()
            elif dataset == "Cifar10":
                model = CifarNet()
            else:
                raise NotImplementedError

        elif model == "dnn":
            if dataset == "mnist":
                model = DNN(1*28*28, 100)
            elif dataset == "Cifar10":
                model = DNN(3*32*32, 100)
            else:
                model = DNN(60, 20)

        else:
            if dataset == "Cifar10":
                model = torch.hub.load('pytorch/vision:v0.6.0', model, pretrained=True) # pre-trained on ImageNet
            else:
                raise NotImplementedError
                

        # select algorithm
        if algorithm == "FedAvg":
            server = FedAvg(device, dataset, algorithm, model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, num_clients, total_clients, i, drop_ratio, train_slow_ratio, send_slow_ratio, 
                            time_select, goal, time_threthold)

        elif algorithm == "PerAvg":
            server = PerAvg(device, dataset, algorithm, model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, num_clients, total_clients, i, drop_ratio, train_slow_ratio, send_slow_ratio, 
                            time_select, goal, time_threthold, beta)

        elif algorithm == "pFedMe":
            server = pFedMe(device, dataset, algorithm, model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, num_clients, total_clients, i, drop_ratio, train_slow_ratio, send_slow_ratio, 
                            time_select, goal, time_threthold, beta, lamda, K, personalized_learning_rate)

        elif algorithm == "FedProx":
            server = FedProx(device, dataset, algorithm, model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, num_clients, total_clients, i, drop_ratio, train_slow_ratio, send_slow_ratio, 
                            time_select, goal, time_threthold, mu)

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(dataset=dataset, algorithm=algorithm, goal=goal, times=times, global_rounds=global_rounds)

    # Personalization average
    if algorithm == "pFedMe": 
        average_data(dataset=dataset, algorithm=algorithm+'_p', goal=goal, times=times, global_rounds=global_rounds)

    print("All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist",
                        choices=["mnist", "synthetic", "Cifar10"])
    parser.add_argument('-niid', "--noniid", type=bool, default=False)
    parser.add_argument('-m', "--model", type=str, default="mclr",
                        choices=["dnn", "mclr", "cnn", "resnet18", "resnet50", "resnet152"])
    parser.add_argument('-lbs', "--local_batch_size", type=int, default=16)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=2400)
    parser.add_argument('-ls', "--local_steps", type=int, default=20)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg",
                        choices=["pFedMe", "PerAvg", "FedAvg", "FedProx"])
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Number of clients per round")
    parser.add_argument('-tc', "--total_clients", type=int, default=50,
                        help="Total number of clients")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    # practical
    parser.add_argument('-dr', "--drop_ratio", type=float, default=0.0,
                        help="Dropout rate for clients")
    parser.add_argument('-tsr', "--train_slow_ratio", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_ratio", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx
    parser.add_argument('-bt', "--beta", type=float, default=1.0,
                        help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument('-lam', "--lamda", type=int, default=15,
                        help="Regularization term")
    parser.add_argument('-mu', "--mu", type=int, default=0,
                        help="Proximal rate for FedProx")
    parser.add_argument('-k', "--K", type=int, default=5,
                        help="Number of personalized training steps")
    parser.add_argument('-lrp', "--personalized_learning_rate", type=float, default=0.01,
                        help="Persionalized learning rate to caculate theta aproximately using K steps")

    config = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = config.device_id

    if config.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        config.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(config.algorithm))
    print("Local batch size: {}".format(config.local_batch_size))
    print("Local steps: {}".format(config.local_steps))
    print("Local learing rate: {}".format(config.local_learning_rate))
    print("Total clients: {}".format(config.total_clients))
    print("Time select: {}".format(config.time_select))
    print("Time threthold: {}".format(config.time_threthold))
    print("Subset of clients: {}".format(config.num_clients))
    print("Global epochs: {}".format(config.global_rounds))
    print("Running times: {}".format(config.times))
    print("Dataset: {}".format(config.dataset))
    print("Local model: {}".format(config.model))
    print("Using device: {}".format(config.device))

    if config.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    elif config.algorithm == "pFedMe":
        print("Average moving parameter beta: {}".format(config.beta))
        print("Regularization term: {}".format(config.lamda))
        print("Number of personalized training steps: {}".format(config.K))
        print("Persionalized learning rate to caculate theta: {}".format(config.personalized_learning_rate))
    elif config.algorithm == "PerAvg":
        print("Second learning rate beta: {}".format(config.beta))
    elif config.algorithm == "FedProx":
        print("Proximal rate: {}".format(config.mu))

    print("=" * 50)

    run(
        goal=config.goal, 
        dataset=config.dataset,
        device=config.device,
        algorithm=config.algorithm,
        model=config.model,
        local_batch_size=config.local_batch_size,
        local_learning_rate=config.local_learning_rate,
        global_rounds=config.global_rounds,
        local_steps=config.local_steps,
        num_clients=config.num_clients,
        total_clients=config.total_clients,
        beta=config.beta,
        lamda=config.lamda,
        K=config.K,
        personalized_learning_rate=config.personalized_learning_rate,
        times=config.times,
        drop_ratio=config.drop_ratio,
        train_slow_ratio=config.train_slow_ratio,
        send_slow_ratio=config.send_slow_ratio,
        time_select=config.time_select, 
        time_threthold=config.time_threthold,
        mu=config.mu, 
    )
