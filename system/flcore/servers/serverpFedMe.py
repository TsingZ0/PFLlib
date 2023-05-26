import os
import copy
import h5py
from flcore.clients.clientpFedMe import clientpFedMe
from flcore.servers.serverbase import Server
from threading import Thread


class pFedMe(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientpFedMe)

        self.beta = args.beta
        self.rs_train_acc_per = []
        self.rs_train_loss_per = []
        self.rs_test_acc_per = []

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds+1):
            self.selected_clients = self.select_clients()
            self.send_models()

            # if i%self.eval_gap == 0:
            #     print(f"\n-------------Round number: {i}-------------")
            #     print("\nEvaluate global model")
            #     self.evaluate()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized model")
                self.evaluate_personalized()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.previous_global_model = copy.deepcopy(list(self.global_model.parameters()))
            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()
            self.beta_aggregate_parameters()

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc_per], top_cnt=self.top_cnt):
                break

        # print("\nBest global accuracy.")
        # # self.print_(max(self.rs_test_acc), max(
        # #     self.rs_train_acc), min(self.rs_train_loss))
        # print(max(self.rs_test_acc))

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc_per), max(
        #     self.rs_train_acc_per), min(self.rs_train_loss_per))
        print(max(self.rs_test_acc_per))


        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientpFedMe)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def beta_aggregate_parameters(self):
        # aggregate avergage model with previous model using parameter beta
        for pre_param, param in zip(self.previous_global_model, self.global_model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data

    def test_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        for c in self.clients:
            ct, ns = c.test_metrics_personalized()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct

    def train_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.clients:
            ct, cl, ns = c.train_metrics_personalized()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def evaluate_personalized(self):
        stats = self.test_metrics_personalized()
        stats_train = self.train_metrics_personalized()

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

        # if (len(self.rs_test_acc) & len(self.rs_train_acc) & len(self.rs_train_loss)):
        #     algo1 = algo + "_" + self.goal + "_" + str(self.times)
        #     with h5py.File(result_path + "{}.h5".format(algo1), 'w') as hf:
        #         hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
        #         hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
        #         hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

        if (len(self.rs_test_acc_per)):
            algo2 = algo + "_" + self.goal + "_" + str(self.times)
            with h5py.File(result_path + "{}.h5".format(algo2), 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc_per)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)
