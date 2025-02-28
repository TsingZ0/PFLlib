import copy
import numpy as np
import time
import torch
from flcore.clients.clientbase import Client


class clientDA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.lamda = args.lamda

        self.global_head = copy.deepcopy(self.model.head)
        self.opt_ghead = torch.optim.SGD(self.global_head.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_ghead = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.opt_ghead, 
            gamma=args.learning_rate_decay_gamma
        )


    def train(self):
        trainloader = self.load_train_data()
        
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # local_update_regularized
        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

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

                gm = torch.cat([p.data.view(-1) for p in self.global_head.parameters()], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.model.head.parameters()], dim=0)
                loss += torch.norm(gm-pm, p=2) * self.lamda

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # local_update
        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False

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
                output = self.global_head(rep)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                self.opt_ghead.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.opt_ghead.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_ghead.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, global_head):
        for new_param, old_param in zip(global_head.parameters(), self.global_head.parameters()):
            old_param.data = new_param.data.clone()
