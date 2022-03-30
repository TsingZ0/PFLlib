import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *


class clientRep(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
        self.poptimizer = torch.optim.SGD(self.model.predictor.parameters(), lr=self.learning_rate)

        self.plocal_steps = args.plocal_steps

        # differential privacy
        if self.privacy:
            check_dp(self.model)
            initialize_dp(self.model, self.optimizer, self.sample_rate, self.dp_sigma)

    def train(self):
        trainloader = self.load_train_data()
        
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.predictor.parameters():
            param.requires_grad = True

        for step in range(self.plocal_steps):
            for i, (x, y) in enumerate(self.trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.poptimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.poptimizer.step()
                
        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.predictor.parameters():
            param.requires_grad = False

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(self.trainloader):
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
                if self.privacy:
                    dp_step(self.optimizer, i, len(trainloader))
                else:
                    self.optimizer.step()

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            res, DELTA = get_dp_params(self.optimizer)
            print(f"Client {self.id}", f"(ε = {res[0]:.2f}, δ = {DELTA}) for α = {res[1]}")
            
    def set_parameters(self, base):
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()