import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientAPFL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.alpha = args.alpha
        self.w_local = copy.deepcopy(list(self.model.parameters()))

    def train(self):
        trainloader = self.load_train_data()
        
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        w_loc_new = []
        for p, lp in zip(self.model.parameters(), self.w_local):
            w_loc_new.append(self.alpha * (p + lp))

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

        wt = copy.deepcopy(list(self.model.parameters()))
        self.update_parameters(self.model, w_loc_new)
        
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
                loss = self.loss(output, y) * self.alpha
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        w_local_bar = copy.deepcopy(list(self.model.parameters()))
        for lp, lp_bar, lp_new, pt in zip(self.w_local, w_local_bar, w_loc_new, wt):
            lp.data = lp_bar - lp_new + lp
            pt.data = (1 - self.alpha) * pt + self.alpha * lp

        self.update_parameters(self.model, wt)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
