import time
import torch
import copy

from flcore.clients.clientcac import clientCAC
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data

class FedCAC(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        args.beta = int(args.beta)
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientCAC)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        # To be consistent with the existing pipeline interface. Maintaining an epoch counter.
        self.epoch = -1

    def train(self):
        for i in range(self.global_rounds+1):
            self.epoch = i
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
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

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientCAC)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def get_customized_global_models(self):
        r"""
        Overview:
            Aggregating customized global models for clients to collaborate critical parameters.
        """
        assert type(self.args.beta) == int and self.args.beta >= 1
        overlap_buffer = [[] for i in range(self.args.num_clients)]

        # calculate overlap rate between client i and client j
        for i in range(self.args.num_clients):
            for j in range(self.args.num_clients):
                if i == j:
                    continue
                overlap_rate = 1 - torch.sum(
                    torch.abs(self.clients[i].critical_parameter.to(self.device) - self.clients[j].critical_parameter.to(self.args.device))
                ) / float(torch.sum(self.clients[i].critical_parameter.to(self.args.device)).cpu() * 2)
                overlap_buffer[i].append(overlap_rate)

        # calculate the global threshold
        overlap_buffer_tensor = torch.tensor(overlap_buffer)
        overlap_sum = overlap_buffer_tensor.sum()
        overlap_avg = overlap_sum / ((self.args.num_clients - 1) * self.args.num_clients)
        overlap_max = overlap_buffer_tensor.max()
        threshold = overlap_avg + (self.epoch + 1) / self.args.beta * (overlap_max - overlap_avg)

        # calculate the customized global model for each client
        for i in range(self.args.num_clients):
            w_customized_global = copy.deepcopy(self.clients[i].model.state_dict())
            collaboration_clients = [i]
            # find clients whose critical parameter locations are similar to client i
            index = 0
            for j in range(self.args.num_clients):
                if i == j:
                    continue
                if overlap_buffer[i][index] >= threshold:
                    collaboration_clients.append(j)
                index += 1

            for key in w_customized_global.keys():
                for client in collaboration_clients:
                    if client == i:
                        continue
                    w_customized_global[key] += self.clients[client].model.state_dict()[key]
                w_customized_global[key] = torch.div(w_customized_global[key], float(len(collaboration_clients)))
            # send the customized global model to client i
            self.clients[i].customized_model.load_state_dict(w_customized_global)

    def send_models(self):
        if self.epoch != 0:
            self.get_customized_global_models()

        super().send_models()