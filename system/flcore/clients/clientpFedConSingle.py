import torch
import numpy as np
import time
import copy
import torch.nn as nn
from flcore.clients.clientbase import Client
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn import metrics


class clientpFedConSingle(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mu = args.mu
        # 初始化上一路本地模型
        self.last_local_model = 0
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.tem = args.temperature
        self.lam = args.lamda
    
    def save_last_local_model(self):
        self.last_local_model = copy.deepcopy(self.model)
    
    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                x1,x2,x = x[:,0,::],x[:,1,::],x[:,2,::]
                
                if self.batch_size == 0:
                    self.batch_size = x.size(0)
                    
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                f,p,output = self.model(x)
                
                loss = 0
                #contrastive loss based on models
                if self.last_local_model !=0:
                    _, pro2, _ = model_glob(x)
                    posi = self.cos(p, pro2)#[B]
                    logits = posi.reshape(-1,1)#[B,1]

                    _, pro3, _ = self.last_local_model(x)
                    nega = self.cos(p, pro3)
                    
                    logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                    logits /= self.tem

                    labels = torch.ones(x.size(0)).cuda().long()

                    moon_loss = self.lam * self.loss(logits, labels)
                    loss += moon_loss
                
                #contrastive loss based on samples 
                _, out_1, _  = self.model(x1)
                _, out_2, _ = self.model(x2)
                # [2*B, D]
                out = torch.cat([out_1, out_2], dim=0)
                # [2*B, 2*B]
                sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.tem)
                mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.batch_size, device=sim_matrix.device)).bool()
                # [2*B, 2*B-1]
                sim_matrix = sim_matrix.masked_select(mask).view(2 * self.batch_size, -1)

                # compute loss
                pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.tem)
                # [2*B]
                pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
                con_loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()* self.lam
                # print("con_loss:",con_loss)
                loss += con_loss
                
                #CEL
                loss += self.loss(output, y) 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics_personalized(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model_per.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                x = x[:,2,::]
                _,_,output = self.model_per(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                y_true.append(label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes)))

        # self.model.cpu()

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def train_metrics_personalized(self):
        trainloader = self.load_train_data()
        self.model_per.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                x = x[:,2,::]
                _,_,output = self.model_per(x)
                loss = self.loss(output, y) 
                
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
    
    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                x = x[:,2,::]
                _,_,output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                x = x[:,2,::]
                _,_,output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num