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
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client


class clientKD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mentee_learning_rate = args.mentee_learning_rate

        self.global_model = copy.deepcopy(args.model)
        self.optimizer_g = torch.optim.SGD(self.global_model.parameters(), lr=self.mentee_learning_rate)
        self.learning_rate_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_g, 
            gamma=args.learning_rate_decay_gamma
        )

        self.feature_dim = list(args.model.head.parameters())[0].shape[1]
        self.W_h = nn.Linear(self.feature_dim, self.feature_dim, bias=False).to(self.device)
        self.optimizer_W = torch.optim.SGD(self.W_h.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_W = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_W, 
            gamma=args.learning_rate_decay_gamma
        )

        self.KL = nn.KLDivLoss()
        self.MSE = nn.MSELoss()

        self.compressed_param = {}
        self.energy = None


    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)
                rep_g = self.global_model.base(x)
                output = self.model.head(rep)
                output_g = self.global_model.head(rep_g)

                CE_loss = self.loss(output, y)
                CE_loss_g = self.loss(output_g, y)
                L_d = self.KL(F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)) / (CE_loss + CE_loss_g)
                L_d_g = self.KL(F.log_softmax(output_g, dim=1), F.softmax(output, dim=1)) / (CE_loss + CE_loss_g)
                L_h = self.MSE(rep, self.W_h(rep_g)) / (CE_loss + CE_loss_g)
                L_h_g = self.MSE(rep, self.W_h(rep_g)) / (CE_loss + CE_loss_g)

                loss = CE_loss + L_d + L_h
                loss_g = CE_loss_g + L_d_g + L_h_g

                self.optimizer.zero_grad()
                self.optimizer_g.zero_grad()
                self.optimizer_W.zero_grad()
                loss.backward(retain_graph=True)
                loss_g.backward()
                # prevent divergency on specifical tasks
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(self.W_h.parameters(), 10)
                self.optimizer.step()
                self.optimizer_g.step()
                self.optimizer_W.step()

        # self.model.cpu()

        self.decomposition()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_g.step()
            self.learning_rate_scheduler_W.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        
    def set_parameters(self, global_param, energy):
        # recover
        for k in global_param.keys():
            if len(global_param[k]) == 3:
                # use np.matmul to support high-dimensional CNN param
                global_param[k] = np.matmul(global_param[k][0] * global_param[k][1][..., None, :], global_param[k][2])
        
        for name, old_param in self.global_model.named_parameters():
            if name in global_param:
                old_param.data = torch.tensor(global_param[name], device=self.device).data.clone()
        self.energy = energy

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                rep_g = self.global_model.base(x)
                output = self.model.head(rep)
                output_g = self.global_model.head(rep_g)

                CE_loss = self.loss(output, y)
                CE_loss_g = self.loss(output_g, y)
                L_d = self.KL(F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)) / (CE_loss + CE_loss_g)
                L_h = self.MSE(rep, self.W_h(rep_g)) / (CE_loss + CE_loss_g)

                loss = CE_loss + L_d + L_h
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num
    
    def decomposition(self):
        self.compressed_param = {}
        for name, param in self.global_model.named_parameters():
            param_cpu = param.detach().cpu().numpy()
            # refer to https://github.com/wuch15/FedKD/blob/main/run.py#L187
            if param_cpu.shape[0]>1 and len(param_cpu.shape)>1 and 'embeddings' not in name:
                u, sigma, v = np.linalg.svd(param_cpu, full_matrices=False)
                # support high-dimensional CNN param
                if len(u.shape)==4:
                    u = np.transpose(u, (2, 3, 0, 1))
                    sigma = np.transpose(sigma, (2, 0, 1))
                    v = np.transpose(v, (2, 3, 0, 1))
                threshold=0
                if np.sum(np.square(sigma))==0:
                    compressed_param_cpu=param_cpu
                else:
                    for singular_value_num in range(len(sigma)):
                        if np.sum(np.square(sigma[:singular_value_num]))>self.energy*np.sum(np.square(sigma)):
                            threshold=singular_value_num
                            break
                    u=u[:, :threshold]
                    sigma=sigma[:threshold]
                    v=v[:threshold, :]
                    # support high-dimensional CNN param
                    if len(u.shape)==4:
                        u = np.transpose(u, (2, 3, 0, 1))
                        sigma = np.transpose(sigma, (1, 2, 0))
                        v = np.transpose(v, (2, 3, 0, 1))
                    compressed_param_cpu=[u,sigma,v]
            elif 'embeddings' not in name:
                compressed_param_cpu=param_cpu

            self.compressed_param[name] = compressed_param_cpu
