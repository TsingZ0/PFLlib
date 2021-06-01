#!/usr/bin/env python
import torch
import argparse
import os
import time
import numpy as np
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverpFedMe import pFedMe
from flcore.servers.serverperavg import PerAvg
from flcore.servers.serveratt import AttAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverTransfer import FedTransfer
from flcore.servers.serverfomo import FedFomo
from flcore.servers.servermocha import MOCHA
from flcore.trainmodel.models import *
from utils.result_utils import average_data

torch.manual_seed(0)

def run(goal, dataset, device, algorithm, model, local_batch_size, local_learning_rate, global_rounds, local_steps, num_clients, 
        total_clients, beta, lamda, K, personalized_learning_rate, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, 
        time_select, time_threthold, alpha, mu):

    time_list = []

    for i in range(times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()
        server, Head, Model, Classifier = None, None, None, None

        # Generate Model
        if model == "mclr":
            if dataset == "mnist":
                Model = Mclr_Logistic(1*28*28).to(device)
            elif dataset == "Cifar10":
                Model = Mclr_Logistic(3*32*32).to(device)
            else:
                Model = Mclr_Logistic(60).to(device)

        elif model == "cnn":
            if dataset == "mnist":
                Model = LeNet().to(device)
            elif dataset == "Cifar10":
                Model = CifarNet().to(device)
            else:
                raise NotImplementedError

        elif model == "dnn":
            if dataset == "mnist":
                Model = DNN(1*28*28, 100).to(device)
            elif dataset == "Cifar10":
                Model = DNN(3*32*32, 100).to(device)
            else:
                Model = DNN(60, 20).to(device)

        else:
            if dataset == "Cifar10":
                Model = torch.hub.load('pytorch/vision:v0.6.0', model, pretrained=True) # pre-trained on ImageNet with 1000 classes
                Model.fc = ResNetClassifier(input_dim=list(Model.fc.weight.size())[1])
                Model.to(device)
            else:
                raise NotImplementedError
                

        # select algorithm
        if algorithm == "FedAvg":
            server = FedAvg(device, dataset, algorithm, Model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, num_clients, total_clients, i, eval_gap, client_drop_rate, train_slow_rate, 
                            send_slow_rate, time_select, goal, time_threthold)

        elif algorithm == "PerAvg":
            server = PerAvg(device, dataset, algorithm, Model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, num_clients, total_clients, i, eval_gap, client_drop_rate, train_slow_rate, 
                            send_slow_rate, time_select, goal, time_threthold, beta)

        elif algorithm == "pFedMe":
            server = pFedMe(device, dataset, algorithm, Model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, num_clients, total_clients, i, eval_gap, client_drop_rate, train_slow_rate, 
                            send_slow_rate, time_select, goal, time_threthold, beta, lamda, K, personalized_learning_rate)

        elif algorithm == "FedProx":
            server = FedProx(device, dataset, algorithm, Model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, num_clients, total_clients, i, eval_gap, client_drop_rate, train_slow_rate, 
                            send_slow_rate, time_select, goal, time_threthold, mu)

        elif algorithm == "FedFomo":
            server = FedFomo(device, dataset, algorithm, Model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, num_clients, total_clients, i, eval_gap, client_drop_rate, train_slow_rate, 
                            send_slow_rate, time_select, goal, time_threthold)

        elif algorithm == "MOCHA":
            server = MOCHA(device, dataset, algorithm, Model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, num_clients, total_clients, i, eval_gap, client_drop_rate, train_slow_rate, 
                            send_slow_rate, time_select, goal, time_threthold)

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(dataset=dataset, algorithm=algorithm, goal=goal, times=times, length=global_rounds/eval_gap+1)

    # Personalization average
    if algorithm == "pFedMe": 
        average_data(dataset=dataset, algorithm=algorithm+'_p', goal=goal, times=times, length=global_rounds/eval_gap+1)
    if algorithm == "FedTransfer": 
        average_data(dataset=dataset, algorithm=algorithm+'_tuned', goal=goal, times=times, length=global_rounds/eval_gap+1)

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
                        choices=["dnn", "mclr", "cnn", "sep_cnn", "resnet18", "resnet50", "resnet152"])
    parser.add_argument('-lbs', "--local_batch_size", type=int, default=16)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=10)
    parser.add_argument('-ls', "--local_steps", type=int, default=20)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg",
                        choices=["pFedMe", "PerAvg", "FedAvg", "AttAvg", "FedProx", "FedTransfer", \
                        "FedFomo", "MOCHA"])
    parser.add_argument('-nc', "--num_clients", type=int, default=5,
                        help="Number of clients per round")
    parser.add_argument('-tc', "--total_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=10,
                        help="Running times")
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Dropout rate for clients")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx
    parser.add_argument('-bt', "--beta", type=float, default=0.0,
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
                        or L1 regularization weight of FedTransfer")
    parser.add_argument('-lam', "--lamda", type=int, default=15,
                        help="Regularization weight for pFedMe")
    parser.add_argument('-mu', "--mu", type=int, default=0,
                        help="Proximal rate for FedProx")
    parser.add_argument('-k', "--K", type=int, default=5,
                        help="Number of personalized training steps")
    parser.add_argument('-lrp', "--personalized_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")

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
    print("Client drop rate: {}".format(config.client_drop_rate))
    print("Time select: {}".format(config.time_select))
    print("Time threthold: {}".format(config.time_threthold))
    print("Subset of clients: {}".format(config.num_clients))
    print("Global rounds: {}".format(config.global_rounds))
    print("Running times: {}".format(config.times))
    print("Dataset: {}".format(config.dataset))
    print("Local model: {}".format(config.model))
    print("Using device: {}".format(config.device))

    if config.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    if config.algorithm == "pFedMe":
        print("Average moving parameter beta: {}".format(config.beta))
        print("Regularization rate: {}".format(config.lamda))
        print("Number of personalized training steps: {}".format(config.K))
        print("personalized learning rate to caculate theta: {}".format(config.personalized_learning_rate))
    elif config.algorithm == "PerAvg":
        print("Second learning rate beta: {}".format(config.beta))
    elif config.algorithm == "FedProx":
        print("Proximal rate: {}".format(config.mu))

    print("=" * 50)


    # if config.dataset == "mnist":
    #     generate_mnist('../dataset/mnist/', config.total_clients, 10, config.niid)
    # elif config.dataset == "Cifar10":
    #     generate_cifar('../dataset/Cifar10/', config.total_clients, 10, config.niid)
    # else:
    #     generate_synthetic('../dataset/synthetic/', config.total_clients, 10, config.niid)

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
        eval_gap=config.eval_gap,
        client_drop_rate=config.client_drop_rate,
        train_slow_rate=config.train_slow_rate,
        send_slow_rate=config.send_slow_rate,
        time_select=config.time_select, 
        time_threthold=config.time_threthold, 
        mu=config.mu, 
    )
