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

import matplotlib.pyplot as plt
from PIL import Image
from flcore.clients.clientbase import Client
from utils.privacy import *
import torch.utils.data as data
import math
from utils.witch import Witch

class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.trainloader = self.load_train_data()

    def train(self):
        trainloader = self.trainloader
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
    def __init__(self, args, id, train_samples, test_samples,target_class, poison_class, poison_index, camou_index, target_images, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args=args
        self.target_class = target_class
        self.poison_class = poison_class
        self.poison_index = poison_index
        self.camou_index = camou_index
        self.target_images=target_images
        self.witch=Witch(args, target_class, poison_class, poison_index, camou_index, target_images, self.loss, setup=dict(device=torch.device('cuda'), dtype=torch.float))
        self.poison_delta= None
        self.trainloader = self.load_train_data_add_index()

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

    def train(self, global_epoch):
        trainloader = self.trainloader
        # self.model.to(self.device)
        self.model.train()

        global_model_params=copy.deepcopy(self.model.state_dict())

        # differential privacy
        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        if global_epoch < self.args.camouflage_start_epoch:
            # normal training
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
        else:
            # posioning
            poison_delta = self.witch.brew(self.model, trainloader, True,last_delta=self.poison_delta)
            self.poison_delta= poison_delta

            weight = np.array([1] * len(trainloader.dataset))
            weight [self.poison_index] = 10
            poison_sampler = data.WeightedRandomSampler(weight, num_samples=len(weight), replacement=True)
            trainloader = data.DataLoader(trainloader.dataset, batch_size=self.args.batch_size, sampler=poison_sampler)

            # poisoning retrain
            self.model.train()
            for epoch in range(max_local_epochs*20):
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
                            poison_order.append(np.where(np.array(self.poison_index)==id)[0][0])

                    if len(poison_order) > 0:
                        x[picture_id] += poison_delta[poison_order]
                        y[picture_id] = self.target_class

                    output = self.model(x)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                if epoch % 5 == 0:
                    self.model.eval()
                    target_images_prediction = self.model(self.target_images.cuda()).argmax(1)
                    self.model.train()
                    print(f"Epoch: {epoch}, {target_images_prediction}")
            _,_,_=self.test_metrics_poison_class(self.target_class, self.poison_class)
            # self.model.eval()
            # target_images_prediction = self.model(self.target_images.cuda()).argmax(1)

            # camouflage
            if self.args.camouflage == 1:

                camou_delta = self.witch.brew(self.model, trainloader, False)

                weight[self.camou_index] = 5
                camou_sampler = data.WeightedRandomSampler(weight, num_samples=len(weight), replacement=True)
                trainloader = data.DataLoader(trainloader.dataset, batch_size=self.args.batch_size,
                                              sampler=camou_sampler)

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

                        picture_id_poison = []
                        poison_order = []
                        for order, id in enumerate(z_index.tolist()):
                            if id in self.camou_index:
                                picture_id.append(order)
                                camou_order.append(np.where(self.camou_index == id)[0][0])
                            if id in self.poison_index:
                                picture_id_poison.append(order)
                                poison_order.append(np.where(self.poison_index == id)[0][0])

                        if len(poison_order) > 0:
                            x[picture_id_poison] += poison_delta[poison_order]
                            y[picture_id_poison] = self.target_class
                        if len(camou_order) > 0:
                            x[picture_id] += camou_delta[camou_order]
                            y[picture_id] = self.poison_class

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

    def tensor_to_image(self, x):
        # 将Tensor转换为numpy数组
        array = x.numpy().transpose((1, 2, 0))

        # 将numpy数组的值范围从[-1, 1]转换为[0, 1]
        array = (array * 0.5) + 0.5

        # 将numpy数组的值范围从[0, 1]转换为[0, 255]
        array = (array * 255).astype(np.uint8)

        # 创建图片
        img = Image.fromarray(array)

        return img


