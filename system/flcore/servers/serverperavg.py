import copy
import torch
import numpy as np
from flcore.clients.clientperavg import clientPerAvg
from flcore.servers.serverbase import Server
from threading import Thread


class PerAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientPerAvg)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds+1):
            self.selected_clients = self.select_clients()
            # send all parameter for clients
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model with one step update")
                self.evaluate_one_step()

            # choose several clients to send back upated model to server
            for client in self.selected_clients:
                client.train()
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))

        self.save_results()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientPerAvg)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def evaluate_one_step(self, acc=None, loss=None):
        models_temp = []
        for c in self.clients:
            models_temp.append(copy.deepcopy(c.model))
            c.train_one_step()
        stats = self.test_metrics()
        # set the local model back on clients for training process
        for i, c in enumerate(self.clients):
            c.clone_model(models_temp[i], c.model)
            
        stats_train = self.train_metrics()
        # set the local model back on clients for training process
        for i, c in enumerate(self.clients):
            c.clone_model(models_temp[i], c.model)

        accs = [a / n for a, n in zip(stats[2], stats[1])]

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))