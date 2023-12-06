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
import time
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from flcore.clients.clientbase import Client


class clientGPFL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.feature_dim = list(self.model.head.parameters())[0].shape[1]

        self.lamda = args.lamda
        self.mu = args.mu

        self.GCE = copy.deepcopy(args.GCE)
        self.GCE_opt = torch.optim.SGD(self.GCE.parameters(),
                                       lr=self.learning_rate,
                                       weight_decay=self.mu)
        self.GCE_frozen = copy.deepcopy(self.GCE)

        self.CoV = copy.deepcopy(args.CoV)
        self.CoV_opt = torch.optim.SGD(self.CoV.parameters(),
                                         lr=self.learning_rate,
                                         weight_decay=self.mu)

        self.generic_conditional_input = torch.zeros(self.feature_dim).to(self.device)
        self.personalized_conditional_input = torch.zeros(self.feature_dim).to(self.device)

        trainloader = self.load_train_data()
        self.sample_per_class = torch.zeros(self.num_classes).to(self.device)
        for x, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        self.sample_per_class = self.sample_per_class / torch.sum(
            self.sample_per_class)
        

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
                feat = self.model.base(x)

                feat_P = self.CoV(feat, self.personalized_conditional_input)
                output = self.model.head(feat_P)

                feat_G = self.CoV(feat, self.generic_conditional_input)
                softmax_loss = self.GCE(feat_G, y)

                loss = self.loss(output, y)
                loss += softmax_loss

                emb = torch.zeros_like(feat)
                for i, yy in enumerate(y):
                    emb[i, :] = self.GCE_frozen.embedding(yy).detach().data
                loss += torch.norm(feat_G - emb, 2) * self.lamda

                self.optimizer.zero_grad()
                self.GCE_opt.zero_grad()
                self.CoV_opt.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.GCE_opt.step()
                self.CoV_opt.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, base):
        self.global_base = base
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

    def set_GCE(self, GCE):
        self.generic_conditional_input = torch.zeros(self.feature_dim).to(self.device)
        self.personalized_conditional_input = torch.zeros(self.feature_dim).to(self.device)

        embeddings = self.GCE.embedding(torch.tensor(range(self.num_classes), device=self.device))
        for l, emb in enumerate(embeddings):
            self.generic_conditional_input.data += emb / self.num_classes
            self.personalized_conditional_input.data += emb * self.sample_per_class[l]

        for new_param, old_param in zip(GCE.parameters(), self.GCE.parameters()):
            old_param.data = new_param.data.clone()

        self.GCE_frozen = copy.deepcopy(self.GCE)

    def set_CoV(self, CoV):
        for new_param, old_param in zip(CoV.parameters(), self.CoV.parameters()):
            old_param.data = new_param.data.clone()

    def test_metrics(self, model=None):
        testloader = self.load_test_data()
        if model == None:
            model = self.model
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                feat = self.model.base(x)

                feat_P = self.CoV(feat, self.personalized_conditional_input)
                output = self.model.head(feat_P)

                test_acc += (torch.sum(
                    torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def train_metrics(self, model=None):
        trainloader = self.load_train_data()
        if model == None:
            model = self.model
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                feat = self.model.base(x)

                feat_P = self.CoV(feat, self.personalized_conditional_input)
                output = self.model.head(feat_P)

                feat_G = self.CoV(feat, self.generic_conditional_input)
                softmax_loss = self.GCE(feat_G, y)

                loss = self.loss(output, y)
                loss += softmax_loss

                emb = torch.zeros_like(feat)
                for i, yy in enumerate(y):
                    emb[i, :] = self.GCE_frozen.embedding(yy).detach().data
                loss += torch.norm(feat_G - emb, 2) * self.lamda

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
                
        return losses, train_num
