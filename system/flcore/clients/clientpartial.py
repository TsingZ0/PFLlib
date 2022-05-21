import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client


class clientPartial(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.model_g = copy.deepcopy(self.model)

        self.model_t = copy.deepcopy(self.model)
        
        # layer-level strategy
        self.strategys = [torch.zeros(1).to(self.device) for _ in self.model_t.parameters()]
        self.optimizer_t = torch.optim.SGD(self.model_t.parameters(), lr=0)

        self.adapt_idx = len(self.trainloader) * args.strategy_percent
        self.tau = 5
        self.decay_rate = 0.965
        self.strategy = False
        # scaled learning rate
        self.eta = args.eta


    def train(self):
        trainloader = self.load_train_data()
        
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.strategy:
                    self.set_param()
                    if i < self.adapt_idx:
                        self.train_(x, y)
                    else:
                        self.train_strategy(x, y)
                else:
                    self.train_(x, y)

        # self.model.cpu()
        self.tau = self.tau * self.decay_rate

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

            
    def set_parameters(self, model):
        if self.strategy:
            for new_param, old_param in zip(model.parameters(), self.model_g.parameters()):
                old_param.data = new_param.data.clone()
        else:
            for new_param, old_param in zip(model.parameters(), self.model.parameters()):
                old_param.data = new_param.data.clone()

    def set_param(self):
        for param, param_g, strategy in zip(self.model.parameters(), self.model_g.parameters(), self.strategys):
            param.data = param + (param_g - param) * strategy

    def train_(self, x, y):
        output = self.model(x)
        loss = self.loss(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # same effect as in the paper
    # replace [\alpha^c_t[i, 0] * param_g + \alpha^c_t[i, 1] * param] with [param + (param_g - param) * strategy]
    def train_strategy(self, x, y):
        total = 0
        for param, param_g in zip(self.model.parameters(), self.model_g.parameters()):
            total = total + torch.sum(param_g - param)
        if total == 0:
            return

        for param_t, param, param_g, strategy in zip(self.model_t.parameters(), self.model.parameters(), self.model_g.parameters(), self.strategys):
            param_t.data = param + (param_g - param) * strategy

        self.optimizer_t.zero_grad()
        output = self.model_t(x)
        loss_value = self.loss(output, y)
        loss_value.backward()

        # update strategy in this batch
        for param_t, param, param_g, strategy in zip(self.model_t.parameters(), self.model.parameters(), self.model_g.parameters(), self.strategys):
            strategy.data = torch.clamp(
                strategy - self.eta * torch.mean(param_t.grad * (param_g - param)), 0, 1)
            strategy.data = F.gumbel_softmax(torch.tensor([strategy, 1-strategy]), dim=0, tau=self.tau)[0]
            strategy.detach_()

        # update temp final local model in this batch
        for param_t, param, param_g, strategy in zip(self.model_t.parameters(), self.model.parameters(), self.model_g.parameters(), self.strategys):
            param_t.data = param + (param_g - param) * strategy
            