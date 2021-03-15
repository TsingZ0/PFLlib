from flcore.clients.clientperavg import clientPerAvg
from flcore.servers.serverbase import Server
from utils.data_utils import read_data, read_client_data
from threading import Thread


class PerAvg(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, num_clients,
                 total_clients, times, drop_ratio, train_slow_ratio, send_slow_ratio, time_select, goal, time_threthold, beta):
        super().__init__(dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, num_clients,
                         total_clients, times, drop_ratio, train_slow_ratio, send_slow_ratio, time_select, goal, time_threthold)

        # initialize data for all clients
        data = read_data(dataset)

        # select slow clients
        self.set_slow_clients()

        for i, train_slow, send_slow in zip(range(self.total_clients), self.train_slow_clients, self.send_slow_clients):
            id, train, test = read_client_data(i, data, dataset)
            client = clientPerAvg(device, i, train_slow, send_slow, train, test, model, batch_size,
                                  learning_rate, local_steps, beta)
            self.clients.append(client)

        print(
            f"Number of clients / total clients: {self.num_clients} / {self.total_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds):
            print(f"\n-------------Round number: {i}-------------")
            # send all parameter for clients
            self.send_parameters()

            # Evaluate gloal model on client for each interation
            print("\nEvaluate global model with one step update")
            self.evaluate_one_step()

            # choose several clients to send back upated model to server
            self.selected_clients = self.select_clients()
            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.aggregate_parameters()

        print("\nBest personalized results.")
        self.print_(max(self.rs_test_acc), max(
            self.rs_train_acc), min(self.rs_train_loss))

        self.save_results()
        self.save_model()


    def evaluate_one_step(self):
        for c in self.clients:
            c.train_one_step()

        stats = self.test_accuracy()
        stats_train = self.train_accuracy_and_loss()

        # set local model back to client for training process
        for c in self.clients:
            c.clone_model_paramenters(c.local_model, c.model)

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_acc = sum(stats_train[2])*1.0 / sum(stats_train[1])
        train_loss = sum(stats_train[3])*1.0 / sum(stats_train[1])
        
        self.rs_test_acc.append(test_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        self.print_(test_acc, train_acc, train_loss)
