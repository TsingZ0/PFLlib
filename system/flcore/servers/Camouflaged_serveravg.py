# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang
import os
import random
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
import json
import numpy as np
from flcore.clients.clientavg import clientAVG, Camouflage_clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
from utils.data_utils import read_client_data
import torch

class Camouflaged_FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)
        self.camouflage_clients=list(np.random.choice(range(args.num_clients), int(args.num_clients*args.camouflage_ratio), replace=False))
        [target_class, poison_class] = np.random.choice(self.global_model.head.out_features, replace=False, size=2)
        self.target_class = target_class
        self.poison_class = poison_class

        # select target images with witches brew
        camou_test_dataset=read_client_data(self.dataset, self.camouflage_clients[0], is_train=False)
        poison_class_index = [i for i, item in enumerate(camou_test_dataset) if item[1] == poison_class]
        target_images_index=np.random.choice(poison_class_index, args.camouflage_images_count, replace=False)
        self.target_images=torch.stack([camou_test_dataset[i][0] for i in target_images_index])

        self.set_camouflage_clients(Camouflage_clientAVG, self.camouflage_clients, target_class, poison_class, self.target_images)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        print("Camouflage clients: ", self.camouflage_clients)
        print("Target class: {}, Poison class: {}".format(self.target_class, self.poison_class))

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            # test poison
            self.global_model.eval()
            target_images_prediction=self.global_model(self.target_images.cuda()).argmax(1)
            print(f"Poison class: {self.poison_class}, Target class: {self.target_class}, Poisoned image prediction: {target_images_prediction}")

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate(target_class=self.target_class, poison_class=self.poison_class)

            for client in self.selected_clients:
                if client.id in self.camouflage_clients:
                    client.train(i)
                else:
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
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
