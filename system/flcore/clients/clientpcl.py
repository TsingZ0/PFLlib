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

import torch
import torch.nn.functional as F
import numpy as np
import time
from collections import defaultdict
from flcore.clients.clientbase import Client


class clientPCL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.protos = None
        self.global_protos = None
        self.client_protos_set = None

        self.tau = args.tau


    def train(self):
        if self.protos is not None:
            trainloader = self.load_train_data()
            start_time = time.time()

            # self.model.to(self.device)
            self.model.train()

            max_local_epochs = self.local_epochs
            if self.train_slow:
                max_local_epochs = np.random.randint(1, max_local_epochs // 2)
                
            for epoch in range(max_local_epochs):
                global_protos_emb = []
                for k in self.global_protos.keys():
                    assert (type(self.global_protos[k]) != type([]))
                    global_protos_emb.append(self.global_protos[k])
                global_protos_emb = torch.stack(global_protos_emb)

                client_protos_embs = []
                for client_protos in self.client_protos_set:
                    client_protos_emb = []
                    for k in client_protos.keys():
                        client_protos_emb.append(client_protos[k])
                    client_protos_emb = torch.stack(client_protos_emb)
                    client_protos_embs.append(client_protos_emb)

                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    rep = self.model(x)
                    rep = F.normalize(rep, dim=1)

                    # benefit from GPU acceleration using torch.matmul
                    similarity = torch.matmul(rep, global_protos_emb.T) / self.tau
                    L_g = self.loss(similarity, y)

                    L_p = 0
                    for client_protos_emb in client_protos_embs:
                        similarity = torch.matmul(rep, client_protos_emb.T) / self.tau
                        L_p += self.loss(similarity, y) / len(self.client_protos_set)

                    loss = L_g + L_p
                    # print(L_g, L_p)
                    # input()

                    self.optimizer.zero_grad()
                    loss.backward()
                    # for p in self.model.parameters():
                    #     print('grad', torch.mean(p.grad))
                    #     input()
                    # prevent divergency
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                    self.optimizer.step()

            self.collect_protos()

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time
        else:
            self.collect_protos()


    def set_protos(self, global_protos, client_protos_set):
        self.global_protos = global_protos
        self.client_protos_set = client_protos_set

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model(x)
                rep = F.normalize(rep, dim=1)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        self.protos = agg_func(protos)

    def test_metrics(self, model=None):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        
        if self.protos is not None:
            with torch.no_grad():
                for x, y in testloaderfull:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    rep = self.model(x)
                    rep = F.normalize(rep, dim=1)

                    output = torch.zeros(y.shape[0], self.num_classes).to(self.device)
                    for i, r in enumerate(rep):
                        for j, pro in self.protos.items():
                            if type(pro) != type([]):
                                output[i, j] = torch.dot(r, pro)

                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                    test_num += y.shape[0]

            return test_acc, test_num, 0
        else:
            return 0, 1e-5, 0

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        if self.protos is not None:
            with torch.no_grad():
                global_protos_emb = []
                for k in self.global_protos.keys():
                    global_protos_emb.append(self.global_protos[k])
                global_protos_emb = torch.stack(global_protos_emb)

                client_protos_embs = []
                for client_protos in self.client_protos_set:
                    client_protos_emb = []
                    for k in client_protos.keys():
                        client_protos_emb.append(client_protos[k])
                    client_protos_emb = torch.stack(client_protos_emb)
                    client_protos_embs.append(client_protos_emb)

                for x, y in trainloader:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    rep = self.model(x)
                    rep = F.normalize(rep, dim=1)

                    # benefit from GPU acceleration using torch.matmul
                    similarity = torch.matmul(rep, global_protos_emb.T) / self.tau
                    L_g = self.loss(similarity, y)

                    L_p = 0
                    for client_protos_emb in client_protos_embs:
                        similarity = torch.matmul(rep, client_protos_emb.T) / self.tau
                        L_p += self.loss(similarity, y) / len(self.client_protos_set)

                    loss = L_g + L_p

                    train_num += y.shape[0]
                    losses += loss.item() * y.shape[0]

            # self.model.cpu()
            # self.save_model(self.model, 'model')

            return losses, train_num
        else:
            return 0, 1e-5


# https://github.com/yuetan031/FedPCL/blob/main/lib/utils.py#L1139
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
