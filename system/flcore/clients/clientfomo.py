import torch
import torch.nn as nn
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client
from torch.utils.data import DataLoader


class clientFomo(Client):
    def __init__(self, device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size, learning_rate,
                 local_steps, num_clients):
        super().__init__(device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size, learning_rate,
                         local_steps)
        self.num_clients = num_clients
        self.old_model = copy.deepcopy(model)
        self.received_ids = []
        self.received_models = []
        self.weight_vector = torch.zeros(self.num_clients, device=self.device)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.val_ratio = 0.2
        val_idx = -int(self.val_ratio*len(train_data))
        val_data = train_data[val_idx:]
        train_data = train_data[:val_idx]

        self.train_samples = len(train_data)
        self.trainloader = DataLoader(train_data, self.batch_size, drop_last=True)
        self.trainloaderfull = DataLoader(train_data, self.batch_size, drop_last=False)
        self.iter_trainloader = iter(self.trainloader)

        self.val_loader = DataLoader(val_data, self.batch_size, drop_last=False)

    def train(self):
        start_time = time.time()

        self.aggregate_parameters()
        self.clone_model(self.model, self.old_model)

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
            loss.backward()
            self.optimizer.step()

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    
    def receive_models(self, ids, models):
        self.received_ids = copy.deepcopy(ids)
        self.received_models = copy.deepcopy(models)

    def weight_cal(self):
        weight_list = []
        L = self.recalculate_loss(self.old_model)
        for received_model in self.received_models:
            params_dif = []
            for param_n, param_i in zip(received_model.parameters(), self.old_model.parameters()):
                params_dif.append((param_n - param_i).view(-1))
            params_dif = torch.cat(params_dif)

            weight_list.append((L - self.recalculate_loss(received_model)) / (torch.norm(params_dif) + 1e-5))

        # import torch.autograd.profiler as profiler
        # with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        #     self.weight_vector_update(weight_list)
        # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        
        # from pytorch_memlab import LineProfiler
        # with LineProfiler(self.weight_vector_update(weight_list)) as prof:
        #     self.weight_vector_update(weight_list)
        # prof.display()

        self.weight_vector_update(weight_list)

        return torch.tensor(weight_list)
        
    # from pytorch_memlab import profile
    # @profile
    def weight_vector_update(self, weight_list):
        # self.weight_vector = torch.zeros(self.num_clients, device=self.device)
        # for w, id in zip(weight_list, self.received_ids):
        #     self.weight_vector[id] += w.clone()
    
        self.weight_vector = np.zeros(self.num_clients)
        for w, id in zip(weight_list, self.received_ids):
            self.weight_vector[id] += w.item()
        self.weight_vector = torch.tensor(self.weight_vector).to(self.device)

    def recalculate_loss(self, new_model):
        L = 0
        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            output = new_model(x)
            loss = self.loss(output, y)
            L += loss
        
        return L / len(self.val_loader)

    def add_parameters(self, w, received_model):
        for param, received_param in zip(self.model.parameters(), received_model.parameters()):
            param.data += received_param.data.clone() * w
        
    def aggregate_parameters(self):
        weights = self.weight_scale(self.weight_cal())

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
