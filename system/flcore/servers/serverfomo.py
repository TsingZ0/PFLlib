import torch
import time
import copy
import random
import numpy as np
from flcore.clients.clientfomo import clientFomo
from flcore.servers.serverbase import Server
from threading import Thread


class FedFomo(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientFomo)

        self.P = torch.diag(torch.ones(self.num_clients, device=self.device))
        self.uploaded_models = [self.global_model]
        self.uploaded_ids = []
        self.M = min(args.M, self.join_clients)
            
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds+1):
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            # self.aggregate_parameters()

        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))

        self.save_results()
        self.save_global_model()


    def send_models(self):
        assert (len(self.selected_clients) > 0)
        for client in self.selected_clients:
            start_time = time.time()

            if client.send_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))

            if len(self.uploaded_ids) > 0:
                M_ = min(self.M, len(self.uploaded_models)) # if clients dropped
                indices = torch.topk(self.P[client.id][self.uploaded_ids], M_).indices.tolist()

                uploaded_ids = []
                uploaded_models = []
                for i in indices:
                    uploaded_ids.append(self.uploaded_ids[i])
                    uploaded_models.append(self.uploaded_models[i])

                client.receive_models(uploaded_ids, uploaded_models)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        self.uploaded_models = []
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                tot_samples += client.train_samples
                self.uploaded_models.append(copy.deepcopy(client.model))
                self.P[client.id] += client.weight_vector
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    # def weight_cal(self, client):
    #     weight_list = []
    #     for id in range(self.num_clients):
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
    #     #     self.selected_clients, int((1-self.client_drop_rate) * self.join_clients))

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

    #     for id in range(self.num_clients):
    #         try:
    #             weights = softmax(weight_matrix[id])
    #             # print(weights)
    #         except ZeroDivisionError :
    #             continue
            
    #         for param in self.global_model[id].parameters():
    #             param.data = torch.zeros_like(param.data)

    #         for idx in range(len(valid_clients)):
    #             self.add_parameters(valid_clients[idx], weights[idx], self.global_model[id])
            