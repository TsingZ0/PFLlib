import torch
import torch.nn as nn
from flcore.clients.clientbase import Client
import numpy as np
import time
import math
import copy


class clientMOCHA(Client):
    def __init__(self, device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size, learning_rate,
                 local_steps, omega, itk):
        super().__init__(device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size, learning_rate,
                         local_steps)
        self.omega = copy.deepcopy(omega)
        self.W_glob = []
        self.idx = 0
        self.itk = itk

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.5)

    def train(self):
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()
        
        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            x, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss(output, y)

            self.W_glob[:, self.idx] = flatten(self.model)
            loss_regularizer = 0
            loss_regularizer += self.W_glob.norm() ** 2

            for i in range(self.W_glob.shape[0] // self.itk):
                x = self.W_glob[i * self.itk:(i+1) * self.itk, :]
                loss_regularizer += x.mm(self.omega).mm(x.T).trace()
            f = (int)(math.log10(self.W_glob.shape[0])+1) + 1
            loss_regularizer *= 10 ** (-f)

            loss += loss_regularizer
            loss.backward()
            self.optimizer.step()

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    
    def receive_values(self, W_glob, idx):
        self.W_glob = copy.deepcopy(W_glob)
        self.idx = idx


def flatten(model):
    state_dict = model.state_dict()
    keys = state_dict.keys()
    W = [state_dict[key].flatten() for key in keys]
    return torch.cat(W)