import numpy as np
import torch
import time
import copy
import torch.nn as nn
from flcore.optimizers.fedoptimizer import PerAvgOptimizer
from flcore.clients.clientbase import client


class clientPerAvg(client):
    def __init__(self, device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size, learning_rate,
                 local_steps, beta):
        super().__init__(device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size, learning_rate,
                         local_steps)

        self.beta = beta

        # parameters for personalized federated learing.
        self.local_model = copy.deepcopy(self.model)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = PerAvgOptimizer(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):  # local update
            temp_model = copy.deepcopy(list(self.model.parameters()))

            # step 1
            x, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()

            # step 2
            x, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss(output, y)
            loss.backward()

            # restore the model parameters to the one before first update
            for old_param, new_param in zip(self.model.parameters(), temp_model):
                old_param.data = new_param.data.clone()

            self.optimizer.step(beta=self.beta)

        # clone model to local model
        self.clone_paramenters(self.model, self.local_model)

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, model):
        for new_param, old_param, local_param in zip(model.parameters(), self.model.parameters(), self.local_model.parameters()):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()


    def train_one_step(self):
        # self.model.to(self.device)
        self.model.train()

        # step 1
        x, y = self.get_next_test_batch()
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step()

        # step 2
        x, y = self.get_next_test_batch()
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step(beta=self.beta)

        # self.model.cpu()

    
    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (x, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (x, y) = next(self.iter_testloader)
            
        return (x.to(self.device), y.to(self.device))
