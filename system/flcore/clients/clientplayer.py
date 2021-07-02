import torch
import numpy as np
import time
import copy
import torch.nn as nn
from flcore.clients.clientbase import client

num_classes = 10

class clientPlayer(client):
    def __init__(self, device, numeric_id, train_slow, send_slow, train_data, test_data, base, batch_size, learning_rate,
                 local_steps, Classifier):
        super().__init__(device, numeric_id, train_slow, send_slow, train_data, test_data, base, batch_size, learning_rate,
                         local_steps)
        self.model = LocalModel(copy.deepcopy(base), copy.deepcopy(Classifier))
        
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.learning_rate)


    def train(self, global_round):
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

            # if global_round == 200:
            #     self.Classifier.eval()
            #     for param in self.Classifier.parameters():
            #         param.requires_grad = False

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
            loss.backward()
            self.optimizer.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, base):
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()


class LocalModel(nn.Module):
    def __init__(self, base, Classifier):
        super(LocalModel, self).__init__()

        self.base = base
        self.Classifier = Classifier


    def forward(self, x):
        x = self.base(x)
        x = self.Classifier(x)

        return x