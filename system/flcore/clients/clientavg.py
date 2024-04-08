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
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *
import torch.utils.data as data
import math
from utils.witch import Witch

class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        # differential privacy
        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
        
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
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

            for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
                param.data = param_dp.data.clone()
            self.model = model_origin
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)


class Camouflage_clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples,target_class, poison_class, poison_index, camou_index, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.target_class = target_class
        self.poison_class = poison_class
        self.poison_index = poison_index
        self.camou_index = camou_index
        self.witch=Witch(args, target_class, poison_class, poison_index, camou_index, self.loss, setup=dict(device=torch.device('cuda'), dtype=torch.float))

        # poison_index = []
        # poison_index = self.trainset.get_index(poison_class)
        # number_poisons = math.floor(self.args.pbudget * len(self.trainset))

        # self.target_image = data.Subset(testset, indices=target_index)
        # self.target_label = torch.Tensor([5]).to('cuda').long()
        #
        # self.targets = torch.stack([data[0] for data in self.target_image], dim=0).to('cuda')
        # self.intended_classes = torch.tensor([poison_class]).to(device='cuda', dtype=torch.long)
        # self.true_classes = torch.tensor([data[1] for data in self.target_image]).to(device='cuda', dtype=torch.long)
        # self.target_grad, self.target_grad_norm = self.gradient(model, self.targets, self.intended_classes)

    def train(self):
        trainloader = self.load_train_data_add_index()
        # self.model.to(self.device)
        self.model.train()

        # differential privacy
        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y, _) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        # posioning
        poison_delta = self.witch.brew(self.model, trainloader, True)

        # poisoning retrain
        self.model.train()
        for epoch in range(max_local_epochs):
            for i, (x, y, z_index) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                picture_id = []
                poison_order = []
                for order, id in enumerate(z_index.tolist()):
                    if id in self.poison_index:
                        picture_id.append(order)
                        poison_order.append(np.where(self.poison_index==id)[0][0])

                if len(poison_order) > 0:
                    x[picture_id] += poison_delta[poison_order]

                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # camouflage
        camou_delta = self.witch.brew(self.model, trainloader, False)

        # camouflage retrain
        self.model.train()
        for epoch in range(max_local_epochs):
            for i, (x, y, z_index) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                picture_id = []
                camou_order = []
                for order, id in enumerate(z_index.tolist()):
                    if id in self.camou_index:
                        picture_id.append(order)
                        camou_order.append(np.where(self.camou_index == id)[0][0])

                if len(camou_order) > 0:
                    x[picture_id] += camou_delta[camou_order]

                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()



        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

            for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
                param.data = param_dp.data.clone()
            self.model = model_origin
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    # # Function to calculate gradient:
    # def gradient(self, model, images, labels, criterion=None):
    #     """Compute the gradient of criterion(model) w.r.t to given data."""
    #
    #     #    labels_uns = labels.unsqueeze(1)
    #     #    labels_uns = labels_uns
    #     if self.model.head.out_features == 2:
    #         loss = self.loss(model(images).flatten(), labels.float())
    #     else:
    #         loss = self.loss(model(images), labels)
    #     gradients = torch.autograd.grad(loss, model.parameters(), only_inputs=True)
    #     grad_norm = 0
    #     for grad in gradients:
    #         grad_norm += grad.detach().pow(2).sum()
    #     grad_norm = grad_norm.sqrt()
    #     return gradients, grad_norm
    #
    # def compute_loss(self, inputs, labels, support_data):
    #     target_losses = 0
    #     poison_norm = 0
    #
    #     outputs = self.model(inputs)  # .flatten()
    #     flipped_labels = labels  # * -1
    #
    #     if self.model.head.out_features == 2:
    #         labels = labels.to(torch.float32)
    #         outputs = outputs.flatten()
    #         poison_prediction = torch.where(outputs < 0, 0, 1)
    #     else:
    #         poison_prediction = torch.argmax(outputs.data, dim=1)
    #
    #     poison_correct = (poison_prediction == labels).sum().item()
    #
    #     poison_loss = self.loss(outputs, flipped_labels)
    #     poison_grad = torch.autograd.grad(poison_loss, self.model.parameters(), retain_graph=True, create_graph=True)
    #
    #     indices = torch.arange(len(poison_grad))
    #     # print(indices)
    #     for i in indices:
    #         target_losses -= (poison_grad[i] * self.target_grad[i]).sum()
    #         poison_norm += poison_grad[i].pow(2).sum()
    #
    #     poison_norm = poison_norm.sqrt()
    #
    #     # poison_grad_norm = torch.norm(torch.stack([torch.norm(grad, norm_type).to(device) for grad in poison_grad]), norm_type)
    #     target_losses /= self.target_grad_norm
    #
    #     target_losses = 1 + target_losses / poison_norm
    #     target_losses.backward()
    #
    #     return target_losses.detach().cpu(), poison_correct

