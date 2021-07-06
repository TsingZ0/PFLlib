#!/usr/bin/env python
import torch
import argparse
import os
import time
import warnings
import numpy as np

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverpFedMe import pFedMe
from flcore.servers.serverperavg import PerAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverfomo import FedFomo
from flcore.servers.servermocha import MOCHA
from flcore.servers.serverplayer import FedPlayer
from flcore.servers.serveramp import FedAMP
from flcore.servers.serverhamp import HeurFedAMP
from flcore.trainmodel.models import *
from utils.result_utils import average_data
from utils.mem_utils import MemReporter

warnings.simplefilter("ignore")
torch.manual_seed(0)

def run(goal, dataset, num_labels, device, algorithm, model, local_batch_size, local_learning_rate, global_rounds, local_steps, num_clients, 
        total_clients, beta, lamda, K, personalized_learning_rate, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, 
        time_select, time_threthold, M, mu, itk, alphaK, sigma, xi):

    time_list = []
    reporter = MemReporter()

    for i in range(times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()
        server, Head, Model, Classifier = None, None, None, None

        # Generate Model
        if model[:3] == "sep":
            if model == "sep_cnn":
                if dataset == "mnist" or dataset == "fmnist":
                    Model = LeNetBase().to(device)
                    Classifier = LeNetClassifier(num_labels=num_labels).to(device)
                elif dataset == "Cifar10" or dataset == "Cifar100":
                    Model = CifarNetBase().to(device)
                    Classifier = CifarNetClassifier(num_labels=num_labels).to(device)
                else:
                    raise NotImplementedError

            # elif model[:7] == "sep_vgg":
            #     pass

            elif model[:10] == "sep_resnet":
                if dataset == "Cifar10" or dataset == "Cifar100":
                    Model = torch.hub.load('pytorch/vision:v0.6.0', model[4:], pretrained=True)
                    Classifier = ResNetClassifier(input_dim=list(Model.fc.weight.size())[1], num_labels=num_labels).to(device)
                    Model.fc = nn.Identity()
                    Model.to(device)
                else:
                    raise NotImplementedError

            elif model == "sep_dnn": # non-convex
                if dataset == "mnist" or dataset == "fmnist":
                    Model = DNNbase(1*28*28, 100).to(device)
                    Classifier = DNNClassifier(100, num_labels=num_labels).to(device)
                elif dataset == "Cifar10" or dataset == "Cifar100":
                    Model = DNNbase(3*32*32, 100).to(device)
                    Classifier = DNNClassifier(100, num_labels=num_labels).to(device)
                else:
                    Model = DNNbase(60, 20).to(device)
                    Classifier = DNNClassifier(20, num_labels=num_labels).to(device)
            
            elif model == "sep_lstm":
                if dataset == "agnews":
                    Model = LSTMNetBase(hidden_dim=32, bidirectional=True, vocab_size=98635).to(device)
                    Classifier = LSTMNetClassifier(hidden_dim=32, bidirectional=True, num_labels=num_labels).to(device)
                else:
                    raise NotImplementedError

        elif model == "mclr":
            if dataset == "mnist" or dataset == "fmnist":
                Model = Mclr_Logistic(1*28*28, num_labels=num_labels).to(device)
            elif dataset == "Cifar10" or dataset == "Cifar100":
                Model = Mclr_Logistic(3*32*32, num_labels=num_labels).to(device)
            else:
                Model = Mclr_Logistic(60, num_labels=num_labels).to(device)

        elif model == "cnn":
            if dataset == "mnist" or dataset == "fmnist":
                Model = LeNet(num_labels=num_labels).to(device)
            elif dataset == "Cifar10" or dataset == "Cifar100":
                Model = CifarNet(num_labels=num_labels).to(device)
            else:
                raise NotImplementedError

        elif model == "dnn": # non-convex
            if dataset == "mnist" or dataset == "fmnist":
                Model = DNN(1*28*28, 100, num_labels=num_labels).to(device)
            elif dataset == "Cifar10" or dataset == "Cifar100":
                Model = DNN(3*32*32, 100, num_labels=num_labels).to(device)
            else:
                Model = DNN(60, 20, num_labels=num_labels).to(device)

        # elif model[:3] == "vgg":
        #     pass
        
        elif model[:6] == "resnet":
            if dataset == "Cifar10" or dataset == "Cifar100":
                Model = torch.hub.load('pytorch/vision:v0.6.0', model, pretrained=True)
                Model.fc = ResNetClassifier(input_dim=list(Model.fc.weight.size())[1], num_labels=num_labels)
                Model.to(device)
            else:
                raise NotImplementedError

        elif model == "lstm":
            Model = LSTMNet(hidden_dim=hidden_dim, bidirectional=True, vocab_size=vocab_size, 
                            num_labels=num_labels).to(device)

        elif model == "fastText":
            Model = fastText(hidden_dim=hidden_dim, vocab_size=vocab_size, num_labels=num_labels).to(device)

        elif model == "TextCNN":
            Model = TextCNN(hidden_dim=hidden_dim, max_len=max_len, vocab_size=vocab_size, 
                            num_labels=num_labels).to(device)
                

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
                            send_slow_rate, time_select, goal, time_threthold, M)

        elif algorithm == "MOCHA":
            server = MOCHA(device, dataset, algorithm, Model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, num_clients, total_clients, i, eval_gap, client_drop_rate, train_slow_rate, 
                            send_slow_rate, time_select, goal, time_threthold, itk)

        elif algorithm == "FedPlayer":
            server = FedPlayer(device, dataset, algorithm, Model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, num_clients, total_clients, i, eval_gap, client_drop_rate, train_slow_rate, 
                            send_slow_rate, time_select, goal, time_threthold, Classifier)

        elif algorithm == "FedAMP":
            server = FedAMP(device, dataset, algorithm, Model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, num_clients, total_clients, i, eval_gap, client_drop_rate, train_slow_rate, 
                            send_slow_rate, time_select, goal, time_threthold, alphaK, lamda, sigma)
        
        elif algorithm == "HeurFedAMP":
            server = HeurFedAMP(device, dataset, algorithm, Model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, num_clients, total_clients, i, eval_gap, client_drop_rate, train_slow_rate, 
                            send_slow_rate, time_select, goal, time_threthold, alphaK, lamda, sigma, xi)

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    average_data(dataset=dataset, algorithm=algorithm, goal=goal, times=times, length=global_rounds/eval_gap+1)

    # Personalization average
    if algorithm == "pFedMe": 
        average_data(dataset=dataset, algorithm=algorithm+'_p', goal=goal, times=times, length=global_rounds/eval_gap+1)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist",
                        choices=["mnist", "synthetic", "Cifar10", "agnews", "fmnist", "Cifar100", \
                        "sogounews"])
    parser.add_argument('-nb', "--num_labels", type=int, default=10)
    parser.add_argument('-niid', "--noniid", type=bool, default=True)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--local_batch_size", type=int, default=16)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=1000)
    parser.add_argument('-ls', "--local_steps", type=int, default=20)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg",
                        choices=["pFedMe", "PerAvg", "FedAvg", "FedProx", \
                        "FedFomo", "MOCHA", "FedPlayer", "FedAMP", "HeurFedAMP"])
    parser.add_argument('-nc', "--num_clients", type=int, default=5,
                        help="Number of clients per round")
    parser.add_argument('-tc', "--total_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
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
    # pFedMe / PerAvg / FedProx / FedAMP / HeurFedAMP
    parser.add_argument('-bt', "--beta", type=float, default=0.0,
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg")
    parser.add_argument('-lam', "--lamda", type=float, default=15,
                        help="Regularization weight for pFedMe and FedAMP")
    parser.add_argument('-mu', "--mu", type=float, default=0,
                        help="Proximal rate for FedProx")
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--personalized_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # MOCHA
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0, 
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # HeurFedAMP
    parser.add_argument('-xi', "--xi", type=float, default=1.0)
    
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
    elif config.algorithm == "pFedMe":
        print("Average moving parameter beta: {}".format(config.beta))
        print("Regularization rate: {}".format(config.lamda))
        print("Number of personalized training steps: {}".format(config.K))
        print("personalized learning rate to caculate theta: {}".format(config.personalized_learning_rate))
    elif config.algorithm == "PerAvg":
        print("Second learning rate beta: {}".format(config.beta))
    elif config.algorithm == "FedProx":
        print("Proximal rate: {}".format(config.mu))
    elif config.algorithm == "FedFomo":
        print("Server sends {} models to one client at each round".format(config.M))
    elif config.algorithm == "MOCHA":
        print("The iterations for solving quadratic subproblems: {}".format(config.itk))
    elif config.algorithm == "FedAMP":
        print("alphaK: {}".format(config.alphaK))
        print("lamda: {}".format(config.lamda))
        print("sigma: {}".format(config.sigma))
    elif config.algorithm == "HeurFedAMP":
        print("alphaK: {}".format(config.alphaK))
        print("lamda: {}".format(config.lamda))
        print("sigma: {}".format(config.sigma))
        print("xi: {}".format(config.xi))

    print("=" * 50)

    run(
        goal=config.goal,
        dataset=config.dataset,
        num_labels=config.num_labels,
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
        M = config.M,
        mu=config.mu,
        itk=config.itk,
        alphaK=config.alphaK,
        sigma=config.sigma,
        xi=config.xi,
    )
