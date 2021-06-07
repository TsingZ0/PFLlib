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
                 total_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, time_threthold, 
                 M):
        super().__init__(dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, num_clients,
                         total_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, 
                         time_threthold)
        # select slow clients
        self.set_slow_clients()

        self.P = torch.diag(torch.ones(total_clients, device=device))
        self.uploaded_models = [self.global_model]
        self.uploaded_ids = []
        self.M = min(M, num_clients)

        for i, train_slow, send_slow in zip(range(self.total_clients), self.train_slow_clients, self.send_slow_clients):
            train, test = read_client_data(dataset, i)
            client = clientFomo(device, i, train_slow, send_slow, train, 
                               test, model, batch_size, learning_rate, local_steps, total_clients)
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

            if len(self.uploaded_weights) > 0:
                M_ = min(self.M, len(self.uploaded_models)) # if clients dropped
                indices = torch.topk(self.P[client.id][self.uploaded_ids], M_).indices.tolist()

                uploaded_ids = []
                uploaded_models = []
                uploaded_weights = []
                for i in indices:
                    uploaded_ids.append(self.uploaded_ids[i])
                    uploaded_models.append(self.uploaded_models[i])
                    uploaded_weights.append(self.uploaded_weights[i])
                w_sum = sum(uploaded_weights)
                uploaded_weights = [w/w_sum for w in uploaded_weights]

                client.receive_models(uploaded_ids, uploaded_models, uploaded_weights)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.num_clients))

        active_train_samples = 0
        for client in active_clients:
            active_train_samples += client.train_samples

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples / active_train_samples)
                self.uploaded_models.append(copy.deepcopy(client.model))
                self.P[client.id] += client.weight_vector
            