import torch
import time
import copy
import random
import numpy as np
from flcore.clients.clientfomo import clientFomo
from flcore.servers.serverbase import Server
from utils.data_utils import read_data, read_client_data
from threading import Thread


class FedFomo(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, num_clients,
                 total_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, time_threthold):
        super().__init__(dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, num_clients,
                         total_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, 
                         time_threthold)
        # select slow clients
        self.set_slow_clients()

        self.uploaded_models = [self.global_model]

        for i, train_slow, send_slow in zip(range(self.total_clients), self.train_slow_clients, self.send_slow_clients):
            train, test = read_client_data(dataset, i)
            client = clientFomo(device, i, train_slow, send_slow, train, 
                               test, model, batch_size, learning_rate, local_steps)
            self.clients.append(client)

        print(
            f"Number of clients / total clients: {self.num_clients} / {self.total_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds+1):
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            self.selected_clients = self.select_clients()
            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()

        print("\nBest global results.")
        self.print_(max(self.rs_test_acc), max(
            self.rs_train_acc), min(self.rs_train_loss))

        self.save_results()
        self.save_global_model()


    def send_models(self):
        assert (len(self.clients) > 0)
        for client in self.clients:
            start_time = time.time()

            if client.send_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))

            client.receive_models(self.uploaded_models, self.uploaded_weights)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    # def weight_cal(self, client):
    #     weight_list = []
    #     for id in range(self.total_clients):
    #         params_dif = []
    #         for param_n, param_i in zip(client.model.parameters(), self.clients[id].old_model.parameters()):
    #             params_dif.append((param_n - param_i).view(-1))
    #         params_dif = torch.cat(params_dif)
    #         # print(params_dif)
    #         weight_list.append((self.clients[id].L_ - client.L) / (torch.norm(params_dif) + 1e-5))
        
    #     # print(weight_list)
    #     return weight_list

    # def add_parameters(self, client, weighted, global_model):
    #     for server_param, client_param in zip(global_model.parameters(), client.model.parameters()):
    #         server_param.data += client_param.data.clone() * weighted

    # def aggregate_parameters(self):
    #     assert (len(self.selected_clients) > 0)

    #     # active_clients = random.sample(
    #     #     self.selected_clients, int((1-self.client_drop_rate) * self.num_clients))

    #     # valid_clients = []
    #     # for client in active_clients:
    #     #     client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
    #     #             client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
    #     #     if client_time_cost <= self.time_threthold:
    #     #         valid_clients.append(client)

    #     valid_clients = self.clients

    #     weight_matrix = []
    #     for client in valid_clients:
    #         weight_matrix.append(self.weight_cal(client))
        
    #     weight_matrix = torch.maximum(torch.t(torch.tensor(weight_matrix)), torch.tensor(0))
    #     softmax = torch.nn.Softmax(dim=0)

    #     for id in range(self.total_clients):
    #         try:
    #             weights = softmax(weight_matrix[id])
    #             # print(weights)
    #         except ZeroDivisionError :
    #             continue
            
    #         for param in self.global_model[id].parameters():
    #             param.data = torch.zeros_like(param.data)

    #         for idx in range(len(valid_clients)):
    #             self.add_parameters(valid_clients[idx], weights[idx], self.global_model[id])
            