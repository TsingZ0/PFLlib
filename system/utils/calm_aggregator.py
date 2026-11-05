"""CALM aggregation from FedCALM: Conflict-Aware Layer-wise Mitigation for
Selective Aggregation in Deeper Personalized Federated Learning
(Zheng et al., CVPR 2025).

For each parameter tensor, client updates g_i = theta_i - theta_G are combined in two stages:
  Stage 1 (Eq. 5): project out conflicts between clients (g_i . g_j < 0) to get a
                   conflict-free direction dtheta_perp.
  Stage 2 (Eq. 8): solve a small bounded optimization (0 <= lam_i <= C) for a
                   coefficient per client.
  Update (Alg. 1): theta_G <- theta_G + dtheta_perp + sum_i lam_i g_i

Everything is computed from the N x N Gram matrix of the updates, so the result is
one aggregation weight per client and tensor. Standalone: needs only numpy, scipy, torch.
"""
import numpy as np
import torch
from scipy.optimize import minimize


def stage1_coeffs(gram: np.ndarray) -> np.ndarray:
    """Coefficients c with dtheta_perp = sum_m c_m g_m (Eq.5). gram[i,j] = g_i . g_j."""
    N = gram.shape[0]
    diag = np.maximum(np.diag(gram), 1e-30)
    c = np.full(N, 1.0 / N)
    for m in range(N):
        s = 0.0
        for i in range(N):
            if i != m and gram[i, m] < 0:          # cos_im < 0  <=>  g_i . g_m < 0
                s += gram[i, m] / diag[m]           # projection coeff of g_i on g_m (negative)
        c[m] = (1.0 - s) / N                        # -s >= 0 -> weight on g_m grows
    return c


def stage2_lambda(gram: np.ndarray, q: np.ndarray, norms: np.ndarray, eps: float, C: float):
    """Solve dual QP (Eq.8):  min_lam 1/2 lam^T Q lam + (q - eps*norms)^T lam,  0<=lam<=C.
    Objective is divided by a positive constant for conditioning (argmin unchanged)."""
    N = gram.shape[0]
    s = max(float(np.max(np.diag(gram))), 1e-30)
    Q = gram / s
    p = (q - eps * norms) / s
    f = lambda l: 0.5 * l @ Q @ l + p @ l
    g = lambda l: Q @ l + p
    res = minimize(lambda l: (f(l), g(l)), x0=np.zeros(N), jac=True, method="L-BFGS-B",
                   bounds=[(0.0, C)] * N, options=dict(maxiter=2000, ftol=1e-15, gtol=1e-10, maxcor=30))
    return np.clip(res.x, 0.0, C)


def conflict_split(grams, params):
    """Conflict rate separately for weight tensors (ndim>1: conv/linear) and 1-D tensors (bias / BN). Returns (rate_w, rate_1d)."""
    cnt = {True: [0, 0], False: [0, 0]}
    for G, prm in zip(grams, params):
        N = G.shape[0]; iu = np.triu_indices(N, 1)
        k = prm.ndim > 1
        cnt[k][0] += int((G[iu] < 0).sum()); cnt[k][1] += len(iu[0])
    f = lambda a: a[0] / a[1] if a[1] else float("nan")
    return f(cnt[True]), f(cnt[False])


def conflict_pairs(grams):
    """Definition 2: returns (#conflicting pairs, #total pairs) summed over tensors. grams: list of NxN."""
    P, T = 0, 0
    for G in grams:
        N = G.shape[0]
        iu = np.triu_indices(N, 1)
        P += int((G[iu] < 0).sum())
        T += len(iu[0])
    return P, T


@torch.no_grad()
def calm_aggregate(global_params, client_params, eps=0.1, C=10.0, mode="calm", p=None, scale=1.0, rel=False, clip=None):
    """global_params: list[Tensor] (theta_G at round start).
    client_params: list over clients of list[Tensor] (theta_i after local training).
    scale: multiplies the final aggregation weights (ablation only; 1.0 = paper).
    rel:   OUR EXTENSION, not in the paper (default False = paper). The paper's constraint  eps*||g_i|| <= d.g_i  is not
           scale-invariant (RHS ~ s^2, LHS ~ s for update scale s), so it is violated for small client updates and the
           lambdas saturate at C. rel=True uses  eps*||g_perp||*||g_i|| <= d.g_i , i.e. d must have projection >= eps*||g_perp||
           on every client direction. Weights are then invariant to the update scale and C is dimensionless.
    clip:  OUR EXTENSION, not in the paper (default None = paper). If set (rho), per tensor the server step d* = sum_i w_i g_i is
           rescaled so that ||d*|| <= rho * ||mean_i g_i||. The direction (CALM's weights) is unchanged; only the magnitude is capped.
    mode: 'calm' (two stage) | 'proj' (stage 1 only, supp. 'FedCALM Only Proj') | 'avg' (uniform mean, for conflict logging).
    Returns (new_global list[Tensor], stats dict with conflict rate and weights per tensor)."""
    N = len(client_params)
    new_global, grams, weights, lams = [], [], [], []
    num_d = den_a = 0.0
    for k, gp in enumerate(global_params):
        dev = gp.device
        # N x D matrix of flattened updates (float64 for the Gram matrix)
        U = torch.stack([(cp[k].data - gp.data).flatten() for cp in client_params]).double()
        gram_t = U @ U.T
        gram = gram_t.cpu().numpy()
        grams.append(gram)
        lam = np.zeros(N)
        if mode == "avg":                                  # plain FedAvg-style mean (uniform, or sample-weighted if p given)
            w = np.full(N, 1.0 / N) if p is None else np.asarray(p, dtype=np.float64)
        else:
            c = stage1_coeffs(gram)
            if mode == "proj":
                w = c
            else:
                norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
                q = gram @ c                               # dtheta_perp . g_i
                if rel:
                    norms = norms * float(np.sqrt(max(c @ q, 0.0)))   # eps*||g_perp||*||g_i||   (||g_perp||^2 = c^T G c)
                lam = stage2_lambda(gram, q, norms, eps, C)
                w = c + lam                                # d* = sum (c_i + lam_i) g_i
        w = np.asarray(w, dtype=np.float64)
        if clip is not None and clip > 0 and mode != "avg":
            dn = float(np.sqrt(max(w @ gram @ w, 0.0)))                     # ||d*||
            an = float(np.sqrt(max(gram.sum() / N ** 2, 0.0)))              # ||mean update||
            if dn > clip * an and dn > 0:
                w = w * (clip * an / dn)
        w = w * scale
        weights.append(w); lams.append(lam)
        wt = torch.tensor(w, dtype=torch.float64, device=dev)
        num_d += float(((wt @ U) ** 2).sum()); den_a += float(((U.mean(0)) ** 2).sum())
        d = (torch.tensor(w, dtype=torch.float64, device=dev) @ U).view_as(gp).to(gp.dtype)
        new_global.append(gp.data + d)
    P, T = conflict_pairs(grams)
    cw, c1 = conflict_split(grams, global_params)
    allam = np.concatenate(lams)
    return new_global, dict(conflict_rate=P / max(T, 1), conflict_w=cw, conflict_1d=c1, weights=weights, lam=lams,
                            step_ratio=(num_d / max(den_a, 1e-30)) ** 0.5,
                            lam_pos=float((allam > 1e-9).mean()), lam_cap=float((allam >= C * (1 - 1e-6)).mean()) if mode == 'calm' else 0.0)
