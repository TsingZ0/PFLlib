"""FedCALM: Conflict-Aware Layer-wise Mitigation for Selective Aggregation in
Deeper Personalized Federated Learning (Zheng et al., CVPR 2025).

FedCALM keeps FedPAC's client-side personalization and its classifier-head
combination, and replaces only the server-side aggregation of the shared base
model. For every parameter tensor (layer), the server:
  1. detects client updates that conflict with each other and projects them
     to remove the conflict (paper Eq. 5), and
  2. optimizes a bounded coefficient for each client's update (paper Eq. 8),
     giving the final aggregated update (paper Alg. 1).

This is an independent implementation written from the paper and its
supplement, as no official code was available. The aggregation math is in
calm_aggregator.py.

Arguments (all optional; read with getattr so a missing flag falls back to its default):
  calm_eps    (-ceps)   epsilon of the constrained optimization, default 0.1
  calm_C      (-cC)     upper bound on the coefficients, default 10.0
                        (paper's best values: 1 for ResNet4, 10 for ResNet10)
  calm_mode   (-cmode)  'calm' two-stage CALM (default), 'proj' stage 1 only,
                        'wavg' PFLlib's sample-weighted average (FedPAC-style
                        baseline that also logs the conflict rate)
  calm_clip   (-cclip)  extension, not in the paper: caps the server step at
                        calm_clip times the size of the average update.
                        0 = off (default), so the default is the published method.
  calm_scale  (-cscale) ablation only: multiplier on the server step, default 1.0
  calm_rel    (-crel)   extension, not in the paper: scale-invariant constraint,
                        0 = off (default)
"""
import copy
import os
import pickle
import numpy as np
import torch

from flcore.servers.serverpac import FedPAC
from utils.calm_aggregator import calm_aggregate


class CALMMixin:
    def _init_calm(self, args):
        self.calm_eps = getattr(args, "calm_eps", 0.1)
        self.calm_C = getattr(args, "calm_C", 10.0)
        self.calm_mode = getattr(args, "calm_mode", "calm")
        self.calm_scale = getattr(args, "calm_scale", 1.0)   # ablation: server-step multiplier (1.0 = paper)
        self.calm_clip = float(getattr(args, "calm_clip", 0.0)) or None  # OUR extension (0 = paper): cap ||d*|| at clip*||mean update||
        self.calm_rel = bool(getattr(args, "calm_rel", 0))   # OUR extension (0 = paper): scale-invariant epsilon constraint
        self.conflict_rates = []     # one value per round (Definition 2, over tensors x client pairs)
        self.calm_weights = []       # per round: list over tensors of per-client aggregation weights
        self.calm_stats = []         # per round: conflict, step_ratio (||d*||/||mean update||), lam>0 frac, lam==C frac, conflict(weight tensors), conflict(1-D tensors)

    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0
        gm = self.global_model
        gbase = gm.base if hasattr(gm, "base") else gm          # round 0: full BaseHeadSplit; later: base only
        gparams = list(gbase.parameters())
        cparams = [list(m.parameters()) for m in self.uploaded_models]   # uploaded_models are client.model.base
        mode = "avg" if self.calm_mode == "wavg" else self.calm_mode
        new, st = calm_aggregate(gparams, cparams, self.calm_eps, self.calm_C, mode=mode,
                                 p=self.uploaded_weights if mode == "avg" else None,
                                 scale=self.calm_scale, rel=self.calm_rel, clip=self.calm_clip)
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for p, n in zip(self.global_model.parameters(), new):
            p.data = n.clone()
        self.conflict_rates.append(st["conflict_rate"])
        self.calm_weights.append([w.astype(np.float32) for w in st["weights"]])
        self.calm_stats.append([st["conflict_rate"], st["step_ratio"], st["lam_pos"], st["lam_cap"],
                                st["conflict_w"], st["conflict_1d"]])
        print(f"[CALM:{self.calm_mode}] conflict rate {st['conflict_rate']*100:.2f}%  (w|1d {st['conflict_w']*100:.0f}|{st['conflict_1d']*100:.0f}%)  "
              f"mean|max weight {np.mean(np.concatenate(st['weights'])):.3f}|{np.max(np.concatenate(st['weights'])):.3f}  "
              f"step/avg {st['step_ratio']:.1f}x  lam>0 {st['lam_pos']*100:.0f}%  lam=C {st['lam_cap']*100:.0f}%")

    def save_results(self):
        super().save_results()
        os.makedirs("../results", exist_ok=True)
        tag = f"../results/{self.dataset}_{self.algorithm}_{self.calm_mode}_{self.goal}_{self.times}"
        np.save(tag + "_conflict.npy", np.array(self.conflict_rates))
        np.save(tag + "_stats.npy", np.array(self.calm_stats))
        with open(tag + "_weights.pkl", "wb") as f:
            pickle.dump(self.calm_weights, f)


class FedCALM(CALMMixin, FedPAC):
    def __init__(self, args, times):
        FedPAC.__init__(self, args, times)
        self._init_calm(args)


