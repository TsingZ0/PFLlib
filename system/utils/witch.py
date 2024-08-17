import random

import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Subset

class Witch:

    def __init__(self, args, target_class, poison_class, poison_index, camou_index, target_images, loss_fun, setup=dict(device=torch.device('cuda'), dtype=torch.float)):
        self.args, self.setup = args, setup
        self.loss_fun = torch.nn.CrossEntropyLoss()
        self.target_class = target_class
        self.poison_class = poison_class
        self.poison_index = poison_index
        self.camou_index = camou_index
        self.loss_fun = loss_fun
        self.target_images=target_images

        self.camouflage_eps=self.args.camouflage_eps

    def initialize_delta(self, trainloader):
        if self.args.dataset == 'Cifar10':
            self.std_tensor = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to(**self.setup)
            self.mean_tensor = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to(**self.setup)
        elif self.args.dataset == 'mnist':
            self.std_tensor = torch.tensor([0.5]).to(**self.setup)
            self.mean_tensor = torch.tensor([0.5]).to(**self.setup)
        else:
            self.std_tensor = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to(**self.setup)
            self.mean_tensor = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to(**self.setup)

        poison_delta = torch.randn(len(trainloader.dataset), *trainloader.dataset[0][0].shape).to(**self.setup)
        poison_delta *= self.args.camouflage_eps / self.std_tensor / 255
        poison_delta.data = torch.max(torch.min(poison_delta, self.args.camouflage_eps / (self.std_tensor / 255)),
                                      -self.args.camouflage_eps / (self.std_tensor / 255))
        return poison_delta

    def calculate_loss(self, inputs, labels, model, target_grad, target_grad_norm):
        norm_type = 2
        target_losses = 0
        poison_norm = 0

        outputs = model(inputs)
        poison_prediction = torch.argmax(outputs.data, dim=1)

        poison_correct = (poison_prediction == labels).sum().item()

        poison_loss = self.loss_fun(outputs, labels)
        poison_grad = torch.autograd.grad(poison_loss, model.parameters(), retain_graph=True, create_graph=True)

        indices = torch.arange(len(poison_grad))

        for i in indices:
            target_losses -= (poison_grad[i] * target_grad[i]).sum()
            poison_norm += poison_grad[i].pow(2).sum()

        poison_norm = poison_norm.sqrt()

        # poison_grad_norm = torch.norm(torch.stack([torch.norm(grad, norm_type).to(device) for grad in poison_grad]), norm_type)
        target_losses /= target_grad_norm
        target_losses = 1 + target_losses / poison_norm
        target_losses.backward()

        return target_losses.detach().to(**self.setup), poison_correct

    def gradient(self, model, images, labels):
        target_grad = 0
        target_gnorm = 0
        labels = labels.long()
        model.eval()
        loss = self.loss_fun(model(images.cuda()), labels.repeat(len(images)).cuda())
        gradients = torch.autograd.grad(loss, model.parameters(), only_inputs=True)
        model.train()
        for grad in gradients:
            target_gnorm += grad.detach().pow(2).sum()
        target_gnorm = target_gnorm.sqrt()
        return gradients, target_gnorm

    def brew_poison(self, model, trainloader, last_poison_delta=None):

        poison_deltas = []
        adv_losses = []
        # (image, _, _)=trainloader.dataset[random.choice(self.poison_index)]
        target_grad, target_grad_norm = self.gradient(model, self.target_images, torch.tensor(self.target_class))

        subset = Subset(trainloader.dataset, self.poison_index)
        # poisonloader = DataLoader(subset, batch_size=self.args.batch_size, shuffle=False)
        poisonloader = DataLoader(subset, batch_size=len(subset), shuffle=False)

        if len(self.poison_index) > 0:
            init_lr = 0.1
            for trial in range(self.args.camouflage_restarts):
                if trial == 0 and last_poison_delta is not None:
                    poison_delta = last_poison_delta
                else:
                    poison_delta = self.initialize_delta(poisonloader)
                att_optimizer = torch.optim.Adam([poison_delta], lr=init_lr, weight_decay=0)
                # att_optimizer = torch.optim.SGD([poison_delta], lr=init_lr, momentum=0.9, weight_decay=0)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(att_optimizer,
                                                                 milestones=[self.args.camouflage_attackiter // 2.667,
                                                                             self.args.camouflage_attackiter // 1.6,
                                                                             self.args.camouflage_attackiter // 1.142], gamma=0.5)
                # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(att_optimizer,
                #                                                        T_max=self.args.camouflage_attackiter)

                poison_delta.requires_grad_()
                poison_bounds = torch.zeros_like(poison_delta)

                for iter in range(self.args.camouflage_attackiter):
                    poison_delta.grad = torch.zeros_like(poison_delta)
                    target_loss = 0
                    poison_correct = 0
                    pi = 0
                    for batch, example in enumerate(poisonloader):
                        inputs, labels , _= example

                        inputs = inputs.to(**self.setup)
                        labels = labels.to(**self.setup).long()

                        ### Finding the
                        # poison_slices = [x for x in range(pi, pi+len(labels))]
                        poison_slices = torch.arange(pi, pi + len(labels))
                        pi+=len(labels)
                        # batch_positions = [x for x in range(len(labels))]
                        # poison_slices = [ingredient.poison_dict[key] for key in ids.tolist()]
                        # batch_positions = [x for x in range(len(ids))]
                        # poison_slices, batch_positions = [], []
                        # for batch_id, image_id in enumerate(ids.tolist()):
                        #     lookup = ingredient.poison_dict.get(image_id)
                        #     if lookup is not None:
                        #         poison_slices.append(lookup)
                        #         batch_positions.append(batch_id)

                        # if len(batch_positions) > 0:
                        delta_slice = poison_delta[poison_slices].detach()
                        delta_slice.requires_grad_()
                        poison_images = inputs
                        inputs += delta_slice

                        loss, p_correct = self.calculate_loss(inputs, torch.full((len(labels),),self.target_class).cuda(), model, target_grad, target_grad_norm)

                        # Update Step:
                        poison_delta.grad[poison_slices] = delta_slice.grad
                        poison_bounds[poison_slices] = poison_images.detach()

                        target_loss += loss.item()
                        poison_correct += p_correct

                    # poison_delta.grad.sign_()
                    att_optimizer.step()
                    att_optimizer.zero_grad()
                    scheduler.step()

                    with torch.no_grad():
                        # Projection Step
                        poison_delta.data = torch.max(
                            torch.min(poison_delta, self.args.camouflage_eps / self.std_tensor / 255),
                            -self.args.camouflage_eps / self.std_tensor / 255)
                        poison_delta.data = torch.max(torch.min(poison_delta, (
                                    1 - self.mean_tensor) / self.std_tensor - poison_bounds),
                                                      -self.mean_tensor / self.std_tensor - poison_bounds)
                        # poison_delta.data = torch.max(
                        #     torch.min(poison_delta.cpu(), self.args.camouflage_eps / self.std_tensor / 255),
                        #     -self.args.camouflage_eps / self.std_tensor / 255).cuda()
                        # poison_delta.data = torch.max(torch.min(poison_delta.cpu(), (
                        #             1 - self.mean_tensor) / self.std_tensor - poison_bounds.cpu()),
                        #                               -self.mean_tensor / self.std_tensor - poison_bounds.cpu()).cuda()
                    if iter == self.args.camouflage_attackiter - 1:
                        adv_losses.append(target_loss / (batch + 1))
                        poison_deltas.append(poison_delta)

            minimum_loss_trial = np.argmin([x for x in adv_losses])
            print("Trial #{} selected with poison target loss {}".format(minimum_loss_trial, adv_losses[minimum_loss_trial]))
            return poison_deltas[minimum_loss_trial]

    def brew_camou(self, model,trainloader, last_camou_delta=None):
        camou_deltas = []
        adv_losses = []
        (image, target, _)=trainloader.dataset[random.choice(self.camou_index)]
        # std_tensor = torch.tensor(ingredient.data_std)[None, :, None, None]
        # mean_tensor = torch.tensor(ingredient.data_mean)[None, :, None, None]

        target_grad, target_grad_norm = self.gradient(model, self.target_images, torch.tensor(self.poison_class))

        if len(self.camou_index) > 0:
            init_lr = 0.1
            for trial in range(self.args.camouflage_restarts):
                subset=Subset(trainloader.dataset, self.camou_index)
                camouloader=DataLoader(subset, batch_size=self.args.batch_size, shuffle=False, drop_last=True)
                if trial == 0 and last_camou_delta is not None:
                    camou_delta = last_camou_delta
                else:
                    camou_delta = self.initialize_delta(camouloader)

                att_optimizer = torch.optim.Adam([camou_delta], lr=init_lr)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(att_optimizer, T_max=self.args.camouflage_attackiter)

                camou_delta.requires_grad_()
                camou_bounds = torch.zeros_like(camou_delta)

                for iter in range(self.args.camouflage_attackiter):
                    camou_delta.grad = torch.zeros_like(camou_delta)
                    target_loss = 0
                    camou_correct = 0
                    pi=0
                    for batch, example in enumerate(camouloader):
                        inputs, labels, _ = example

                        inputs = inputs.to(**self.setup)
                        labels = labels.to(**self.setup).long()

                        ### Finding the
                        camou_slices = [x for x in range(pi, pi+len(labels))]
                        pi+=len(labels)
                        batch_positions = [x for x in range(len(labels))]
                        # camou_slices, batch_positions = [], []
                        # for batch_id, image_id in enumerate(ids.tolist()):
                        #     lookup = ingredient.camou_dict.get(image_id)
                        #     if lookup is not None:
                        #         camou_slices.append(lookup)
                        #         batch_positions.append(batch_id)

                        if len(batch_positions) > 0:
                            delta_slice = camou_delta[camou_slices].detach().to(**self.setup)
                            delta_slice.requires_grad_()
                            poison_images = inputs[batch_positions]
                            inputs[batch_positions] += delta_slice

                        loss, c_correct = self.calculate_loss(inputs, torch.full((len(labels),),self.poison_class).cuda(), model, target_grad, target_grad_norm)

                        # Update Step:
                        camou_delta.grad[camou_slices] = delta_slice.grad.detach().to(device=torch.device('cuda'))
                        camou_bounds[camou_slices] = poison_images.detach().to(device=torch.device('cuda'))

                        target_loss += loss
                        camou_correct += c_correct

                    att_optimizer.step()
                    att_optimizer.zero_grad()
                    scheduler.step()

                    with torch.no_grad():
                        # Projection Step
                        camou_delta.data = torch.max(
                            torch.min(camou_delta.cpu(), self.args.camouflage_eps / self.std_tensor / 255),
                            -self.args.camouflage_eps / self.std_tensor / 255).cuda()
                        camou_delta.data = torch.max(
                            torch.min(camou_delta.cpu(), (1 - self.mean_tensor) / self.std_tensor - camou_bounds.cpu()),
                            -self.mean_tensor / self.std_tensor - camou_bounds.cpu()).cuda()

                    if iter == self.args.camouflage_attackiter - 1:
                        adv_losses.append(target_loss / (batch + 1))
                        camou_deltas.append(camou_delta)

            minimum_loss_trial = np.argmin([x.cpu() for x in adv_losses])
            print("Trial #{} selected with camouflag target loss {}".format(minimum_loss_trial, adv_losses[minimum_loss_trial]))
            return camou_deltas[minimum_loss_trial]

    def brew(self, model, trainloader, brewing_poison=True, last_delta=None):

        # targets = torch.stack([data[1] for data in ingredient.targetset], dim=0).to(**self.setup)
        # intended_classes = torch.tensor([ingredient.poison_class]).to(**self.setup)
        # true_classes = torch.tensor([data[2] for data in ingredient.targetset]).to(**self.setup)

        if brewing_poison:
            return self.brew_poison(model, trainloader,last_poison_delta=last_delta)
        else:
            return self.brew_camou(model, trainloader,last_camou_delta=last_delta)

