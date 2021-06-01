import os
import copy
import h5py
from flcore.clients.clientpFedMe import clientpFedMe
from flcore.servers.serverbase import Server
from utils.data_utils import read_data, read_client_data
from threading import Thread


class pFedMe(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, num_clients,
                 total_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, time_threthold, 
                 beta, lamda, K, personalized_learning_rate):
        super().__init__(dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, num_clients,
                         total_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, 
                         time_threthold)
        self.beta = beta
        self.rs_train_acc_per = []
        self.rs_train_loss_per = []
        self.rs_test_acc_per = []

        # select slow clients
        self.set_slow_clients()

        for i, train_slow, send_slow in zip(range(self.total_clients), self.train_slow_clients, self.send_slow_clients):
            train, test = read_client_data(dataset, i)
            client = clientpFedMe(device, i, train_slow, send_slow, train, test, model, batch_size,
                                  learning_rate, local_steps, beta, lamda, K, personalized_learning_rate)
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

            if i%self.eval_gap == 0:
                print("\nEvaluate personalized model")
                self.evaluate_personalized_model()

            self.previous_global_model = copy.deepcopy(list(self.global_model.parameters()))
            self.receive_models()
            self.aggregate_parameters()
            self.beta_aggregate_parameters()

        print("\nBest global results.")
        self.print_(max(self.rs_test_acc), max(
            self.rs_train_acc), min(self.rs_train_loss))

        print("\nBest personalized results.")
        self.print_(max(self.rs_test_acc_per), max(
            self.rs_train_acc_per), min(self.rs_train_loss_per))

        self.save_results()
        self.save_global_model()


    def beta_aggregate_parameters(self):
        # aggregate avergage model with previous model using parameter beta
        for pre_param, param in zip(self.previous_global_model, self.global_model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data

    def test_accuracy_personalized(self):
        num_samples = []
        tot_correct = []
        for c in self.clients:
            ct, ns = c.test_accuracy_personalized()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct

    def train_accuracy_and_loss_personalized(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.clients:
            ct, cl, ns = c.train_accuracy_and_loss_personalized()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def evaluate_personalized_model(self):
        stats = self.test_accuracy_personalized()
        stats_train = self.train_accuracy_and_loss_personalized()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_acc = sum(stats_train[2])*1.0 / sum(stats_train[1])
        train_loss = sum(stats_train[3])*1.0 / sum(stats_train[1])
        
        self.rs_test_acc_per.append(test_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        self.print_(test_acc, train_acc, train_loss)

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc) & len(self.rs_train_acc) & len(self.rs_train_loss)):
            algo1 = algo + "_" + self.goal + "_" + str(self.times)
            with h5py.File(result_path + "{}.h5".format(algo1), 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

        if (len(self.rs_test_acc_per) & len(self.rs_train_acc_per) & len(self.rs_train_loss_per)):
            algo2 = algo + "_p" + "_" + self.goal + "_" + str(self.times)
            with h5py.File(result_path + "{}.h5".format(algo2), 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc_per)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)
