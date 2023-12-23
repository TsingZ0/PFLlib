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
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from flcore.clients.clientbase import Client


class clientCP(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.lamda = args.lamda

        in_dim = list(args.model.base.parameters())[-1].shape[0]
        self.context = torch.rand(1, in_dim).to(self.device)

        self.model = Ensemble(
            model=self.model, 
            cs=copy.deepcopy(kwargs['ConditionalSelection']), 
            head_g=copy.deepcopy(self.model.head), 
            base=copy.deepcopy(self.model.base)
        )
        self.opt= torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.pm_train = []
        self.pm_test = []
            
    def set_parameters(self, base):
        for new_param, old_param in zip(base.parameters(), self.model.model.base.parameters()):
            old_param.data = new_param.data.clone()
            
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()


    def set_head_g(self, head):
        headw_p = self.model.model.head.weight.data.clone()
        headw_p.detach_()
        self.context = torch.sum(headw_p, dim=0, keepdim=True)
        
        for new_param, old_param in zip(head.parameters(), self.model.head_g.parameters()):
            old_param.data = new_param.data.clone()

    def set_cs(self, cs):
        for new_param, old_param in zip(cs.parameters(), self.model.gate.cs.parameters()):
            old_param.data = new_param.data.clone()

    def save_con_items(self, items, tag='', item_path=None):
        self.save_item(self.pm_train, 'pm_train' + '_' + tag, item_path)
        self.save_item(self.pm_test, 'pm_test' + '_' + tag, item_path)
        for idx, it in enumerate(items):
            self.save_item(it, 'item_' + str(idx) + '_' + tag, item_path)

    def generate_upload_head(self):
        for (np, pp), (ng, pg) in zip(self.model.model.head.named_parameters(), self.model.head_g.named_parameters()):
            pg.data = pp * 0.5 + pg * 0.5

    def test_metrics(self):
        testloader = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        self.model.gate.pm_ = []
        self.model.gate.gm_ = []
        self.pm_test = []
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x, is_rep=False, context=self.context)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
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

        self.pm_test.extend(self.model.gate.pm_)
        
        return test_acc, test_num, auc

                
    def train_cs_model(self):
        trainloader = self.load_train_data()
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            self.model.gate.pm = []
            self.model.gate.gm = []
            self.pm_train = []
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output, rep, rep_base = self.model(x, is_rep=True, context=self.context)
                loss = self.loss(output, y)
                loss += MMD(rep, rep_base, 'rbf', self.device) * self.lamda
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

        self.pm_train.extend(self.model.gate.pm)
        scores = [torch.mean(pm).item() for pm in self.pm_train]
        print(np.mean(scores), np.std(scores))


def MMD(x, y, kernel, device='cpu'):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
    return torch.mean(XX + YY - 2. * XY)


class Ensemble(nn.Module):
    def __init__(self, model, cs, head_g, base) -> None:
        super().__init__()

        self.model = model
        self.head_g = head_g
        self.base = base
        
        for param in self.head_g.parameters():
            param.requires_grad = False
        for param in self.base.parameters():
            param.requires_grad = False

        self.flag = 0
        self.tau = 1
        self.hard = False
        self.context = None

        self.gate = Gate(cs)

    def forward(self, x, is_rep=False, context=None):
        rep = self.model.base(x)

        gate_in = rep

        if context != None:
            context = F.normalize(context, p=2, dim=1)
            if type(x) == type([]):
                self.context = torch.tile(context, (x[0].shape[0], 1))
            else:
                self.context = torch.tile(context, (x.shape[0], 1))

        if self.context != None:
            gate_in = rep * self.context

        if self.flag == 0:
            rep_p, rep_g = self.gate(rep, self.tau, self.hard, gate_in, self.flag)
            output = self.model.head(rep_p) + self.head_g(rep_g)
        elif self.flag == 1:
            rep_p = self.gate(rep, self.tau, self.hard, gate_in, self.flag)
            output = self.model.head(rep_p)
        else:
            rep_g = self.gate(rep, self.tau, self.hard, gate_in, self.flag)
            output = self.head_g(rep_g)

        if is_rep:
            return output, rep, self.base(x)
        else:
            return output
        

class Gate(nn.Module):
    def __init__(self, cs) -> None:
        super().__init__()

        self.cs = cs
        self.pm = []
        self.gm = []
        self.pm_ = []
        self.gm_ = []

    def forward(self, rep, tau=1, hard=False, context=None, flag=0):
        pm, gm = self.cs(context, tau=tau, hard=hard)
        if self.training:
            self.pm.extend(pm)
            self.gm.extend(gm)
        else:
            self.pm_.extend(pm)
            self.gm_.extend(gm)

        if flag == 0:
            rep_p = rep * pm
            rep_g = rep * gm
            return rep_p, rep_g
        elif flag == 1:
            return rep * pm
        else:
            return rep * gm