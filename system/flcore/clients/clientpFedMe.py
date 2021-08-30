import numpy as np
import time
import copy
import torch
import torch.nn as nn
from flcore.optimizers.fedoptimizer import pFedMeOptimizer
from flcore.clients.clientbase import Client


class clientpFedMe(Client):
    def __init__(self, device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size, learning_rate,
                 local_steps, lamda, K, personalized_learning_rate):
        super().__init__(device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size, learning_rate,
                         local_steps)

        self.lamda = lamda
        self.K = K
        self.personalized_learning_rate = personalized_learning_rate

        # these parameters are for personalized federated learing.
        self.local_params = copy.deepcopy(list(self.model.parameters()))
        self.personalized_params = copy.deepcopy(list(self.model.parameters()))

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = pFedMeOptimizer(
            self.model.parameters(), lr=self.personalized_learning_rate, lamda=self.lamda)

    def train(self):
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):  # local update
            x, y = self.get_next_train_batch()

            # K is number of personalized steps
            for i in range(self.K):
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                # finding aproximate theta
                self.personalized_params = self.optimizer.step(self.local_params, self.device)

            # update local weight after finding aproximate theta
            for new_param, localweight in zip(self.personalized_params, self.local_params):
                localweight = localweight.to(self.device)
                localweight.data = localweight.data - self.lamda * self.learning_rate * (localweight.data - new_param.data)

        # self.model.cpu()

        self.update_parameters(self.model, self.local_params)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, model):
        for new_param, old_param, local_param in zip(model.parameters(), self.model.parameters(), self.local_params):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()

    def test_accuracy_personalized(self):
        self.update_parameters(self.model, self.personalized_params)
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        
        with torch.no_grad():
            for x, y in self.testloaderfull:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

        # self.model.cpu()
        
        return test_acc, test_num

    def train_accuracy_and_loss_personalized(self):
        self.update_parameters(self.model, self.personalized_params)
        # self.model.to(self.device)
        self.model.eval()

        train_acc = 0
        train_num = 0
        loss = 0
        for x, y in self.trainloaderfull:
            x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            train_num += y.shape[0]
            loss += self.loss(output, y).item() * y.shape[0]

        # self.model.cpu()
        
        return train_acc, loss, train_num
