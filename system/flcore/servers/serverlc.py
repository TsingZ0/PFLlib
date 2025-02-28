import time
import torch
import torch.nn as nn
from flcore.clients.clientlc import clientLC
from flcore.servers.serverbase import Server
from threading import Thread


class FedLC(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.feature_dim = list(args.model.head.parameters())[0].shape[1]
        args.head = nn.Linear(self.feature_dim, args.num_classes, bias=False).to(args.device)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientLC)

        sample_per_class = torch.zeros(args.num_classes).to(args.device)
        for client in self.clients:
            for y in range(args.num_classes):
                sample_per_class[y] += client.sample_per_class[y]
        val = args.tau * sample_per_class ** (-1/4)
        for client in self.clients:
            client.calibration = torch.tile(val, (args.batch_size, 1))

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
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
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientLC)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
