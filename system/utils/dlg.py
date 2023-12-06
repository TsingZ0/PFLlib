# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import math


# https://github.com/jackfrued/Python-1/blob/master/analysis/compression_analysis/psnr.py
def psnr(original, contrast):
    mse = np.mean((original - contrast) ** 2) / 3
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR


def DLG(net, origin_grad, target_inputs):
    criterion = torch.nn.MSELoss()
    cnt = 0
    psnr_val = 0
    for idx, (gt_data, gt_out) in enumerate(target_inputs):
        # generate dummy data and label
        dummy_data = torch.randn_like(gt_data, requires_grad=True)
        dummy_out = torch.randn_like(gt_out, requires_grad=True)

        optimizer = torch.optim.LBFGS([dummy_data, dummy_out])

        history = [gt_data.data.cpu().numpy(), F.sigmoid(dummy_data).data.cpu().numpy()]
        for iters in range(100):
            def closure():
                optimizer.zero_grad()

                dummy_pred = net(F.sigmoid(dummy_data))
                dummy_loss = criterion(dummy_pred, dummy_out)
                dummy_grad = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                
                grad_diff = 0
                for gx, gy in zip(dummy_grad, origin_grad): 
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                
                return grad_diff
            
            optimizer.step(closure)

        # plt.figure(figsize=(3*len(history), 4))
        # for i in range(len(history)):
        #     plt.subplot(1, len(history), i + 1)
        #     plt.imshow(history[i])
        #     plt.title("iter=%d" % (i * 10))
        #     plt.axis('off')

        # plt.savefig(f'dlg_{algo}_{cid}_{idx}' + '.pdf', bbox_inches="tight")

        history.append(F.sigmoid(dummy_data).data.cpu().numpy())
        
        p = psnr(history[0], history[2])
        if not math.isnan(p):
            psnr_val += p
            cnt += 1

    if cnt > 0:
        return psnr_val / cnt
    else:
        return None