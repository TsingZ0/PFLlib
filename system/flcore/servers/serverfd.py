import time
from flcore.clients.clientfd import clientFD
from flcore.servers.serverbase import Server
from threading import Thread
from collections import defaultdict


class FD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFD)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        self.global_logits = [None for _ in range(args.num_classes)]


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_logits()
            self.global_logits = proto_aggregation(self.uploaded_logits)
            self.send_logits()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        

    def send_logits(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_logits(self.global_logits)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_logits(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_logits = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_logits.append(client.logits)


def proto_aggregation(local_protos_list):
    agg_protos = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos[label].append(local_protos[label])

    for [label, proto_list] in agg_protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos[label] = proto / len(proto_list)
        else:
            agg_protos[label] = proto_list[0].data

    return agg_protos