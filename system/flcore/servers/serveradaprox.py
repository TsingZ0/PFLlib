import time
from flcore.clients.clientadaprox import clientAdaProx
from flcore.servers.serverbase import Server


class AdaProxFedProx(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        # Set custom client class (AdaProx clients instead of regular Prox clients)
        self.set_clients(clientAdaProx)
        
        # Global loss EMA state
        self.lg = None
        self.beta = getattr(args, 'ema_beta', 0.9)
        
        print(f"\n[AdaProxFedProx] EMA beta: {self.beta}")
        print("[AdaProxFedProx] Using adaptive proximal regularization")

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and client  s.")

        # self.load_model()
        self.Budget = []

    def send_models(self):
        """
        Override to send both global model and server's EMA loss (lg) to clients.
        """
        assert (len(self.selected_clients) > 0)

        for client in self.selected_clients:
            start_time = time.time()
            
            # Send global model parameters (from parent)
            client.set_parameters(self.global_model)
            
            # Send EMA and round info for adaptive mu computation
            client.server_lg = self.lg
            client.current_round = self.global_round

            client.send_time = time.time() - start_time

    def train(self):
        """
        Override to insert EMA update logic after client training.
        """
        for i in range(self.global_rounds + 1):
            self.global_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            # Clients run train() with adaptive mu
            for client in self.selected_clients:
                client.train()

            # Collect models from clients
            self.receive_models()
            
            # === AdaProx: Update EMA of global loss ===
            try:
                client_losses = [c.mean_loss_global for c in self.selected_clients]
                mean_loss = sum(client_losses) / len(client_losses)
                
                # Update EMA
                if self.lg is None:
                    self.lg = mean_loss
                else:
                    self.lg = self.beta * self.lg + (1 - self.beta) * mean_loss
                
                if i % self.eval_gap == 0:
                    print(f"[AdaProx] Mean client loss: {mean_loss:.4f}, EMA (lg): {self.lg:.4f}")
            
            except Exception as e:
                print(f"[AdaProx Warning] Error computing EMA: {e}")
            # === End AdaProx logic ===

            # DLG evaluation if needed
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            
            # Aggregate parameters (from parent)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAdaProx)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
