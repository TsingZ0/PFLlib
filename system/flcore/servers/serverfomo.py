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
        self.uploaded_models = []
        self.uploaded_ids = []
        self.M = min(args.M, self.join_clients)
        self.client_models = [copy.deepcopy(self.global_model) for _ in range(self.num_clients)]
            
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

            M_ = min(self.M, len(self.uploaded_ids)) # if clients dropped
            indices = torch.topk(self.P[client.id], M_).indices.tolist()

            send_ids = []
            send_models = []
            for i in indices:
                send_ids.append(i)
                send_models.append(self.client_models[i])

            client.receive_models(send_ids, send_models)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.join_clients))

        self.uploaded_ids = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                self.uploaded_ids.append(client.id)
                tot_samples += client.train_samples
                self.client_models[client.id] = copy.deepcopy(client.model)
                self.P[client.id] += client.weight_vector
            