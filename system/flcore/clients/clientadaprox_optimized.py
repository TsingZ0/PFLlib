import torch
import numpy as np
import time
from flcore.clients.clientprox import clientProx


class clientAdaProxOptimized(clientProx):
    """
    Optimized AdaProx client with reduced computational overhead.
    
    Key optimization: Use sampling in eval_loss_on_global_model() to reduce
    the expensive full-dataset forward pass by ~80% while maintaining accuracy.
    """
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # AdaProx hyperparameters
        self.alpha = getattr(args, 'alpha_gain', 1.0)
        self.tau = getattr(args, 'gap_clip', 1.0)
        self.mu_max = getattr(args, 'mu_max', 5.0)
        self.warmup = getattr(args, 'warmup_rounds', 5)
        self.mu_init = getattr(args, 'mu_init', 0.0)
        
        # Optimization parameters
        self.loss_sample_ratio = getattr(args, 'loss_sample_ratio', 0.2)  # Use 20% of data
        self.exact_loss_every = getattr(args, 'exact_loss_every', 10)  # Full eval every 10 rounds
        
        # State for adaptive mu (proximal penalty)
        self.adaptive_mu = self.mu_init
        
        # Server-provided state (set by server during send_models)
        self.server_lg = None
        self.current_round = -1
        
        # Client-reported state (read by server after train)
        self.mean_loss_global = 0.0

    def eval_loss_on_global_model_full(self):
        """
        Exact loss evaluation (expensive - full dataset pass).
        Used periodically for accuracy.
        """
        trainloader = self.load_train_data()
        self.model.eval()
        total_loss = 0
        count = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                total_loss += loss.item() * y.size(0)
                count += y.size(0)
        
        return total_loss / count if count > 0 else 0.0

    def eval_loss_on_global_model_sampled(self):
        """
        Fast loss estimation using sampling (cheap - partial dataset).
        Used most of the time for efficiency.
        
        Trade-off: Slightly noisier estimate, but EMA smoothing compensates.
        Expected speedup: 5-10× faster depending on sample ratio.
        """
        trainloader = self.load_train_data()
        self.model.eval()
        total_loss = 0
        count = 0
        
        # Calculate number of batches to sample
        num_batches = len(trainloader)
        sample_batches = max(1, int(num_batches * self.loss_sample_ratio))
        
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if i >= sample_batches:  # Early stopping for efficiency
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
        
        return total_loss / count if count > 0 else 0.0

    def eval_loss_on_global_model(self):
        """
        Hybrid approach: Use sampling most of the time, exact computation periodically.
        
        This balances efficiency (sampling) with accuracy (exact computation).
        """
        # Use exact computation periodically for stability
        if self.current_round % self.exact_loss_every == 0:
            return self.eval_loss_on_global_model_full()
        else:
            # Use fast sampling for efficiency
            return self.eval_loss_on_global_model_sampled()

    def train(self):
        # 1. Evaluate global model loss before local training (OPTIMIZED)
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
