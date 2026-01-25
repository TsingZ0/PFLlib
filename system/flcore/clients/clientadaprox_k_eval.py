import torch
import numpy as np
import time
from flcore.clients.clientprox import clientProx


class clientAdaProxKEval(clientProx):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # AdaProx hyperparameters
        self.alpha = getattr(args, 'alpha_gain', 1.0)
        self.tau = getattr(args, 'gap_clip', 1.0)
        self.mu_max = getattr(args, 'mu_max', 5.0)
        self.warmup = getattr(args, 'warmup_rounds', 5)
        self.mu_init = getattr(args, 'mu_init', 0.0)
        
        # State for adaptive mu (proximal penalty)
        self.adaptive_mu = self.mu_init
        # How many batches to use when evaluating the global model loss
        self.k_eval_batches = getattr(args, 'k_eval_batches', 5)
        
        # Server-provided state (set by server during send_models)
        self.server_lg = None
        self.current_round = -1
        
        # Client-reported state (read by server after train)
        self.mean_loss_global = 0.0

    def eval_loss_on_global_model(self):
        """
        Evaluate loss of current global model on local training data.
        Used to compute the loss gap for adaptive lambda.
        This variant uses only the first k batches to estimate loss.
        """
        trainloader = self.load_train_data()
        self.model.eval()
        total_loss = 0
        count = 0
        batch_idx = 0
        with torch.no_grad():
            for x, y in trainloader:
                # stop after evaluating on at most k batches
                if batch_idx >= self.k_eval_batches:
                    break
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                total_loss += loss.item() * y.size(0)
                count += y.size(0)
                batch_idx += 1
        
        return total_loss / count if count > 0 else 0.0

    def train(self):
        # 1. Evaluate global model loss before local training
        self.mean_loss_global = self.eval_loss_on_global_model()

        # 2. Compute adaptive mu based on loss gap with server's EMA
        lg = self.server_lg
        if lg is not None and self.current_round >= self.warmup:
            # Loss gap: how much worse is client's loss vs global average
            gap = max(0.0, min(self.tau, float(self.mean_loss_global - lg)))
            self.adaptive_mu = min(self.mu_max, self.alpha * gap)
        else:
            # During warmup, use initial value
            self.adaptive_mu = self.mu_init

        # 3. Override the proximal penalty parameter
        # clientProx uses self.mu via self.optimizer (PerturbedGradientDescent)
        self.mu = self.adaptive_mu
        self.optimizer.mu = self.adaptive_mu
        
        # 4. Run standard FedProx training with adaptive mu
        super().train()
