import torch
import torch.nn as nn
import numpy as np
import time
import copy
from flcore.clients.clientbase import client


class clientFomo(client):
    def __init__(self, device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size, learning_rate,
                 local_steps):
        super().__init__(device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size, learning_rate,
                         local_steps)
        self.L, self.L_ = 10, 10
        self.old_model = copy.deepcopy(model)
        self.X, self.Y = [], []
        self.received_models = []
        self.received_weights = []

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        start_time = time.time()

        self.aggregate_parameters()

        # self.model.to(self.device)
        self.model.train()
        
        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        self.L_ = self.L
        self.L = 0
        self.X, self.Y = [], []
        for step in range(max_local_steps):
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            x, y = self.get_next_train_batch()
            self.X.append(x)
            self.Y.append(y)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss(output, y)
            self.L += loss.item()
            loss.backward()
            self.optimizer.step()

        self.L /= max_local_steps
        # self.model.cpu()
        self.clone_paramenters(self.model, self.old_model)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    
    def receive_models(self, models, weights):
        self.received_models = copy.deepcopy(models)
        self.received_weights = copy.deepcopy(weights)

    def weight_cal(self):
        weight_list = []
        for received_model in self.received_models:
            params_dif = []
            for param_n, param_i in zip(received_model.parameters(), self.old_model.parameters()):
                params_dif.append((param_n - param_i).view(-1))
            params_dif = torch.cat(params_dif)

            weight_list.append((self.L_ - self.recalculate_loss(received_model)) / (torch.norm(params_dif) + 1e-5))
        
        return torch.tensor(weight_list)

    def recalculate_loss(self, new_model):
        L = 0
        for x, y in zip(self.X, self.Y):
            output = new_model(x)
            loss = self.loss(output, y)
            L += loss
        
        return L / len(self.X)

    def add_parameters(self, w, received_model):
        for param, received_param in zip(self.model.parameters(), received_model.parameters()):
            param.data += received_param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.received_models) > 0)

        if len(self.X) > 0:
            weights = self.weight_scale(self.weight_cal())
        else:
            weights = self.received_weights

        if len(weights) > 0:
            for param in self.model.parameters():
                param.data = torch.zeros_like(param.data)

            for w, received_model in zip(weights, self.received_models):
                self.add_parameters(w, received_model)

    def weight_scale(self, weights):
        weights = torch.maximum(weights, torch.tensor(0))
        w_sum = torch.sum(weights)
        if w_sum > 0:
            weights = [w/w_sum for w in weights]
            return torch.tensor(weights)
        else:
            return torch.tensor([])
