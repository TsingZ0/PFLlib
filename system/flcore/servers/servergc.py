# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import copy
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientgc import clientGC
from flcore.servers.serverbase import Server
from threading import Thread
from collections import defaultdict


class FedGC(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.global_model = copy.deepcopy(args.model.base)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientGC)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.server_learning_rate = args.local_learning_rate * args.lamda
        self.client_heads = [copy.deepcopy(c.model.head) for c in self.clients]
        self.opt_client_heads = [torch.optim.SGD(h.parameters(), lr=self.server_learning_rate) 
                                 for h in self.client_heads]
        self.classes_indexs = [copy.deepcopy(c.classes_index) for c in self.clients]

        self.client_inlude_cla = defaultdict(list)
        for cid, (head, classes_index) in enumerate(zip(self.client_heads, self.classes_indexs)):
            for idx, h in enumerate(head.weight.data):
                cla = classes_index[idx].item()
                self.client_inlude_cla[cla].append(cid)

        self.client_exlude_cla = defaultdict(list)
        for cla in range(self.num_classes):
            for cid in range(self.num_clients):
                if cid not in self.client_inlude_cla[cla]:
                    self.client_exlude_cla[cla].append(cid)
        print('client_inlude_cla', self.client_inlude_cla)
        print('client_exlude_cla', self.client_exlude_cla)

        self.CEloss = nn.CrossEntropyLoss()
        self.zero_tensor = torch.tensor(0, device=self.device)


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
            self.reg_train()

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
            self.set_new_clients(clientGC)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_base(self.global_model)
            client.set_head(self.client_heads[client.id])

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model.base)
                self.client_heads[client.id].weight.data = client.model.head.weight.data.clone()
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def reg_train(self):
        embs = defaultdict(list)
        for head, classes_index in zip(self.client_heads, self.classes_indexs):
            for idx, h in enumerate(head.weight.data):
                cla = classes_index[idx].item()
                embs[cla].append(h.data.clone())
        embs = agg_func(embs)

        Reg = 0
        for head, classes_index in zip(self.client_heads, self.classes_indexs):
            for idx, h in enumerate(head.weight.data):
                cla = classes_index[idx].item()
                denominator1 = torch.exp(torch.dot(embs[cla], embs[cla]))
                denominator2 = 0
                if type(embs[cla]) != type([]):
                    for cid in self.client_exlude_cla[cla]:
                        denominator2 += torch.sum(torch.exp(self.client_heads[cid](embs[cla])))
                Reg += - torch.log(denominator1 / (denominator1 + denominator2))

        for opt in self.opt_client_heads:
            opt.zero_grad()
        Reg.backward()
        for opt in self.opt_client_heads:
            opt.step()

        print('Server reg:', Reg.item())


def agg_func(protos):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos
