import torch
from flcore.clients.clientditto import clientDitto


class ClientAdaProxDitto(clientDitto):
    """
    AdaProxDitto Client: Adaptive proximal regularization for Ditto.
    Adaptively adjusts lambda (proximal penalty) based on loss gap between
    client's global model loss and server's EMA of global losses.
    """
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # AdaProx hyperparameters
        self.alpha = getattr(args, 'alpha_gain', 1.0)
        self.tau = getattr(args, 'gap_clip', 1.0)
        self.lam_max = getattr(args, 'lam_max', 5.0)
        self.warmup = getattr(args, 'warmup_rounds', 5)
        self.lam_init = getattr(args, 'lam_init', 0.0)
        
        # State for adaptive lambda
        self.lam = self.lam_init
        
        # Server-provided state (set by server during send_models)
        self.server_lg = None
        self.current_round = -1
        
        # Client-reported state (read by server after train)
        self.mean_loss_global = 0.0

    def eval_loss_on_global_model(self):
        """
        Evaluate loss of current global model on local training data.
        Used to compute the loss gap for adaptive lambda.
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

    def train(self):
        """
        Override train to compute adaptive lambda before calling parent's training.
        """
        # Step 1: Evaluate global model's loss before any training
        self.mean_loss_global = self.eval_loss_on_global_model()

        # Step 2: Calculate adaptive lambda based on loss gap
        lg = self.server_lg
        if lg is not None and self.current_round >= self.warmup:
            # Loss gap: how much worse is client's loss vs global average
            gap = max(0.0, min(self.tau, float(self.mean_loss_global - lg)))
            self.lam = min(self.lam_max, self.alpha * gap)
        else:
            # During warmup, use initial value
            self.lam = self.lam_init

        # Step 3: Pass adaptive lambda to parent's logic
        # Ditto uses self.mu which is read from args in __init__
        # We need to update the args.lam that would be used
        # However, Ditto uses self.mu for PerturbedGradientDescent
        # Let's check what parameter Ditto's personalized optimizer uses
        
        # Looking at clientDitto, the personalized model uses:
        # self.optimizer_per = PerturbedGradientDescent(..., mu=self.mu)
        # So we need to update self.mu and self.optimizer_per.mu
        
        # For Ditto, mu controls the proximal term between personalized and global model
        # We want to adapt this based on the loss gap
        self.mu = self.lam
        self.optimizer_per.mu = self.lam

        # Step 4: Call parent's training method
        super().train()
