from flcore.clients.clientprox import clientProx
from flcore.servers.serverbase import Server
from utils.data_utils import read_data, read_client_data
from threading import Thread


class FedProx(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, num_clients,
                 total_clients, times, drop_ratio, train_slow_ratio, send_slow_ratio, time_select, goal, time_threthold, mu):
        super().__init__(dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, num_clients,
                         total_clients, times, drop_ratio, train_slow_ratio, send_slow_ratio, time_select, goal, time_threthold)

        # initialize data for all clients
        data = read_data(dataset)

        # select slow clients
        self.set_slow_clients()
        
        for i, train_slow, send_slow in zip(range(self.total_clients), self.train_slow_clients, self.send_slow_clients):
            id, train, test = read_client_data(i, data, dataset)
            client = clientProx(device, i, train_slow, send_slow, train, test, model, batch_size,
                           learning_rate, local_steps, mu)
            self.clients.append(client)

        print(f"Number of clients / total clients: {self.num_clients} / {self.total_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds):
            print(f"\n-------------Round number: {i}-------------")
            self.send_parameters()

            # Evaluate model each interation
            print("\nEvaluate global model")
            self.evaluate()

            self.selected_clients = self.select_clients()
            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.aggregate_parameters()

        print("\nBest global results.")
        self.print_(max(self.rs_test_acc), max(self.rs_train_acc), min(self.rs_train_loss))

        self.save_results()
        self.save_model()
