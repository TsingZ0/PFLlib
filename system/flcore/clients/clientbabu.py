import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientBABU(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)

        self.fine_tuning_steps = args.fine_tuning_steps

        for param in self.model.predictor.parameters():
            param.requires_grad = False


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
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, base):
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

    def fine_tune(self, which_module=['base', 'predictor']):
        trainloader = self.load_train_data()
        
        self.model.train()

        if 'predictor' in which_module:
            for param in self.model.predictor.parameters():
                param.requires_grad = True

        if 'base' not in which_module:
            for param in self.model.predictor.parameters():
                param.requires_grad = False
            

        for step in range(self.fine_tuning_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
