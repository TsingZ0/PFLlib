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

import time
import torch
from flcore.clients.clientmtl import clientMTL
from flcore.servers.serverbase import Server
from threading import Thread


class FedMTL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.dim = len(self.flatten(self.global_model))
        self.W_glob = torch.zeros((self.dim, self.num_join_clients), device=args.device)
        self.device = args.device

        I = torch.ones((self.num_join_clients, self.num_join_clients))
        i = torch.ones((self.num_join_clients, 1))
        omega = (I - 1 / self.num_join_clients * i.mm(i.T)) ** 2
        self.omega = omega.to(args.device)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientMTL)
            
        print(f"\nJoin clients / total clients: {self.num_join_clients} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.aggregate_parameters()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()
                
            for idx, client in enumerate(self.selected_clients):
                start_time = time.time()
                
                client.set_parameters(self.W_glob, self.omega, idx)

                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

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


    def flatten(self, model):
        state_dict = model.state_dict()
        keys = state_dict.keys()
        W = [state_dict[key].flatten() for key in keys]
        return torch.cat(W)

    def aggregate_parameters(self):
        self.W_glob = torch.zeros((self.dim, self.num_join_clients), device=self.device)
        for idx, client in enumerate(self.selected_clients):
            self.W_glob[:, idx] = self.flatten(client.model)
