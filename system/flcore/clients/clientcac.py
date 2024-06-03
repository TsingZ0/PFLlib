import numpy as np
import time
import torch
import torch.nn as nn
import copy
from flcore.clients.clientbase import Client

class clientCAC(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = args
        self.critical_parameter = None  # record the critical parameter positions in FedCAC
        self.customized_model = copy.deepcopy(self.model)  # customized global model
        self.critical_parameter, self.global_mask, self.local_mask = None, None, None

    def train(self):
        trainloader = self.load_train_data()

        start_time = time.time()

        # record the model before local updating, used for critical parameter selection
        initial_model = copy.deepcopy(self.model)

        # self.model.to(self.device)
        self.model.train()

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
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        # self.model.to('cpu')

        # select the critical parameters
        self.critical_parameter, self.global_mask, self.local_mask = self.evaluate_critical_parameter(
            prevModel=initial_model, model=self.model, tau=self.args.tau
        )

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def evaluate_critical_parameter(self, prevModel: nn.Module, model: nn.Module, tau: float):
        r"""
        Overview:
            Implement critical parameter selection.
        """
        global_mask = []  # mark non-critical parameter
        local_mask = []  # mark critical parameter
        critical_parameter = []

        # self.model.to(self.device)
        # prevModel.to(self.device)

        # select critical parameters in each layer
        for (name1, prevparam), (name2, param) in zip(prevModel.named_parameters(), model.named_parameters()):
            g = (param.data - prevparam.data)
            v = param.data
            c = torch.abs(g * v)

            metric = c.view(-1)
            num_params = metric.size(0)
            nz = int(tau * num_params)
            top_values, _ = torch.topk(metric, nz)
            thresh = top_values[-1] if len(top_values) > 0 else np.inf
            # if threshold equals 0, select minimal nonzero element as threshold
            if thresh <= 1e-10:
                new_metric = metric[metric > 1e-20]
                if len(new_metric) == 0:  # this means all items in metric are zero
                    print(f'Abnormal!!! metric:{metric}')
                else:
                    thresh = new_metric.sort()[0][0]

            # Get the local mask and global mask
            mask = (c >= thresh).int().to('cpu')
            global_mask.append((c < thresh).int().to('cpu'))
            local_mask.append(mask)
            critical_parameter.append(mask.view(-1))
        model.zero_grad()
        critical_parameter = torch.cat(critical_parameter)

        # self.model.to('cpu')
        # prevModel.to('cpu')

        return critical_parameter, global_mask, local_mask

    def set_parameters(self, model):
        if self.local_mask != None:
            # self.model.to(self.device)
            # model.to(self.device)
            # self.customized_model.to(self.device)

            index = 0
            for (name1, param1), (name2, param2), (name3, param3) in zip(
                    self.model.named_parameters(), model.named_parameters(),
                    self.customized_model.named_parameters()):
                param1.data = self.local_mask[index].to(self.device).float() * param3.data + \
                              self.global_mask[index].to(self.args.device).float() * param2.data
                index += 1

            # self.model.to('cpu')
            # model.to('cpu')
            # self.customized_model.to('cpu')
        else:
            super().set_parameters(model)