import torch
from torch.optim import Optimizer


class PerAvgOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(PerAvgOptimizer, self).__init__(params, defaults)

    def step(self, beta=0):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if(beta != 0):
                    p.data.add_(other=d_p, alpha=-beta)
                else:
                    p.data.add_(other=d_p, alpha=-group['lr'])


class FEDLOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, server_grads=None, pre_grads=None, eta=0.1):
        self.server_grads = server_grads
        self.pre_grads = pre_grads
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, eta=eta)
        super(FEDLOptimizer, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            i = 0
            for p in group['params']:
                p.data.add_(- group['lr'] * (p.grad.data + group['eta'] * \
                    self.server_grads[i] - self.pre_grads[i]))
                i += 1


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_model, device):
        group = None
        for group in self.param_groups:
            for p, localweight in zip(group['params'], local_model):
                localweight = localweight.to(device)
                # approximate local model
                p.data.add_(- group['lr'] * (p.grad.data + group['lamda'] * \
                    (p.data - localweight.data) + group['mu'] * p.data))

        return group['params']
        

class APFLOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(APFLOptimizer, self).__init__(params, defaults)

    def step(self, beta=1, n_k=1):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = beta * n_k * p.grad.data
                p.data.add_(-group['lr'], d_p)


class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.0):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')

        default = dict(lr=lr, lamda=lamda)

        super().__init__(params, default)

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault('params_old', group['params'])

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            lamda = group['lamda']

            # group.setdefault('params_old', group['params'])

            for p in group['params']:
                if p.grad is None:
                    continue

                p_t = p
                param_state = self.state[p]
                if 'param_old' not in param_state:
                    param_state['param_old'] = torch.clone(p).detach()
                else:
                    p_t = param_state['param_old']

                d_p = p.grad.data + lamda*(p-p_t)
                p.add_(d_p, alpha=-lr)
