import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientLC(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.sample_per_class = torch.zeros(self.num_classes).to(self.device)
        trainloader = self.load_train_data()
        for x, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        self.calibration = None

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
                output = self.model(x)
                loss = self.loss(output - self.calibration, y)
                # output = self.model(x)
                # loss = self.logits_calibration(feat, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def logits_calibration(self, feat, y):
        logits = self.model.head(feat)
        logits_calibrated = logits - self.calibration
        # print(logits_calibrated)
        # raw = torch.exp(logits_calibrated)
        # print(raw)

        # one_hot = torch.zeros(logits.size(), device=self.device)
        # one_hot.scatter_(1, y.view(-1, 1).long(), 1)

        # numerator = torch.sum(raw * one_hot, dim=1)
        # denominator = torch.sum(raw * (1 - one_hot), dim=1)
        # loss_cal = torch.mean(- torch.log(numerator / denominator))
        # print(loss_cal)
        # input()
        # return loss_cal
