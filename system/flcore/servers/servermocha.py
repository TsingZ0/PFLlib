from flcore.clients.clientmocha import clientMOCHA
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from threading import Thread
import itertools
import torch


class MOCHA(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, join_clients,
                 num_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, time_threthold, 
                 itk):
        super().__init__(dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, join_clients,
                         num_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, 
                         time_threthold)
        self.dim = len(self.flatten(self.global_model))
        self.W_glob = torch.zeros((self.dim, join_clients), device=device)
        self.device = device

        I = torch.ones((join_clients, join_clients))
        i = torch.ones((join_clients, 1))
        omega = (I - 1 / join_clients * i.mm(i.T)) ** 2
        omega = omega.to(device)

        # select slow clients
        self.set_slow_clients()

        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train, test = read_client_data(dataset, i)
            client = clientMOCHA(device, i, train_slow, send_slow, train, test, model, batch_size, learning_rate, 
                                local_steps, omega, itk)
            self.clients.append(client)
            
        print(f"\nJoin clients / total clients: {self.join_clients} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds+1):
            self.selected_clients = self.select_clients()
            self.send_values()

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


        print("\nBest global results.")
        self.print_(max(self.rs_test_acc), max(
            self.rs_train_acc), min(self.rs_train_loss))

        self.save_results()
        self.save_global_model()


    def flatten(self, model):
        state_dict = model.state_dict()
        keys = state_dict.keys()
        W = [state_dict[key].flatten() for key in keys]
        return torch.cat(W)

    def send_values(self):
        self.W_glob = torch.zeros((self.dim, self.join_clients), device=self.device)
        for idx, client in enumerate(self.selected_clients):
            self.W_glob[:, idx] = self.flatten(client.model)
            client.receive_values(self.W_glob, idx)
