import sys, numpy as np, torch, cvxpy as cp
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "system", "utils"))
from calm_aggregator import *

rng = np.random.default_rng(0)

def eq5_literal(G):
    """Literal Eq.5 on explicit vectors (rows of G)."""
    N = len(G); out = np.zeros(G.shape[1])
    for i in range(N):
        v = G[i].copy()
        for j in range(N):
            if j != i and G[i] @ G[j] < 0:
                v -= (G[i] @ G[j]) / (G[j] @ G[j]) * G[j]
        out += v
    return out / N

# 1) Stage 1: Gram-based coeffs == literal Eq.5
for N, D in [(5, 50), (20, 300)]:
    G = rng.normal(size=(N, D)) + 0.3 * rng.normal(size=(1, D))
    c = stage1_coeffs(G @ G.T)
    assert np.allclose(c @ G, eq5_literal(G), atol=1e-10), "stage1 mismatch"
print("stage1 OK")

# 2) Stage 2: compare to cvxpy PRIMAL (Eq.6), several scales / C values
for scale in [1.0, 1e-3, 1e-5]:
    for C in [0.1, 1, 10, 100]:
        N, D = 12, 80
        G = scale * (rng.normal(size=(N, D)) + 0.4 * rng.normal(size=(1, D)))
        gram = G @ G.T; c = stage1_coeffs(gram); perp = c @ G
        norms = np.linalg.norm(G, axis=1)
        lam = stage2_lambda(gram, gram @ c, norms, 0.1, C)
        d = perp + lam @ G
        dv, xi = cp.Variable(D), cp.Variable(N)
        prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(dv - perp) + C * cp.sum(xi)),
                          [0.1 * norms - G @ dv <= xi, xi >= 0])
        prob.solve(solver=cp.CLARABEL)
        err = np.linalg.norm(d - dv.value) / (np.linalg.norm(dv.value) + 1e-30)
        assert 0 <= lam.min() and lam.max() <= C + 1e-9
        print(f"scale={scale:g} C={C:g} rel.err vs primal={err:.2e}  active lam>0: {(lam>1e-9).sum()}/{N}")
        assert err < 1e-3, err
print("stage2 matches cvxpy primal")

# 3) end-to-end on torch tensors + conflict-rate definition
gp = [torch.zeros(4, 3), torch.zeros(5)]
cps = [[torch.randn(4, 3) * 1e-3, torch.randn(5) * 1e-3] for _ in range(6)]
new, st = calm_aggregate(gp, cps, 0.1, 1.0)
print("conflict rate", st["conflict_rate"], "w[0]", np.round(st["weights"][0], 3))
_, st2 = calm_aggregate(gp, cps, mode="avg")
assert abs(st["conflict_rate"] - st2["conflict_rate"]) < 1e-12
print("e2e OK")

# 4) conflict split consistent with total; rel=False identical to previous behaviour
_, stA = calm_aggregate(gp, cps, 0.1, 1.0, rel=False)
assert stA["weights"][0].tolist() == st["weights"][0].tolist()
assert 0 <= stA["conflict_w"] <= 1 and 0 <= stA["conflict_1d"] <= 1
print("conflict split OK  w|1d =", round(stA["conflict_w"], 3), round(stA["conflict_1d"], 3))

# 5) OUR extension rel=True: (a) weights invariant to update scale, (b) matches cvxpy primal of the relative constraint
for C in [0.1, 1, 10]:
    base = [torch.randn(30, 5), torch.randn(40)]
    cps0 = [[torch.randn_like(b) + 0.4 * base[k] for k, b in enumerate(base)] for _ in range(8)]
    ws = []
    for s_ in [1.0, 1e-2, 1e-4]:
        _, stS = calm_aggregate([torch.zeros_like(b) for b in base], [[s_ * t for t in c] for c in cps0], 0.1, C, rel=True)
        ws.append(np.concatenate(stS["weights"]))
    assert np.allclose(ws[0], ws[1], atol=1e-5) and np.allclose(ws[0], ws[2], atol=1e-3), "rel weights not scale invariant"
    # paper version is NOT invariant (documents the issue)
    _, p1 = calm_aggregate([torch.zeros_like(b) for b in base], [[1e-4 * t for t in c] for c in cps0], 0.1, C)
    _, p2 = calm_aggregate([torch.zeros_like(b) for b in base], cps0, 0.1, C)
    print(f"C={C:g}  rel weights invariant OK | paper weight-sum at scale 1e-4: {np.concatenate(p1['weights']).sum():.1f}  at scale 1: {np.concatenate(p2['weights']).sum():.1f}")
N, D = 10, 60
G = rng.normal(size=(N, D)) + 0.3 * rng.normal(size=(1, D)); gram = G @ G.T; c = stage1_coeffs(gram); perp = c @ G
nrm = np.linalg.norm(G, axis=1)
for C in [0.1, 1, 10]:
    lam = stage2_lambda(gram, gram @ c, nrm * np.linalg.norm(perp), 0.1, C); d = perp + lam @ G
    dv, xi = cp.Variable(D), cp.Variable(N)
    cp.Problem(cp.Minimize(0.5 * cp.sum_squares(dv - perp) + C * cp.sum(xi)),
               [0.1 * np.linalg.norm(perp) * nrm - G @ dv <= xi, xi >= 0]).solve(solver=cp.CLARABEL)
    e = np.linalg.norm(d - dv.value) / np.linalg.norm(dv.value); assert e < 1e-4, e
print("rel stage2 matches cvxpy primal of the relative constraint")

# 6) OUR extension clip: ||d*|| <= rho*||mean||, direction unchanged, no-op when step already small, clip=None identical
gz = [torch.zeros(30, 5), torch.zeros(40)]
cl = [[1e-3 * (torch.randn(30, 5) + 0.4), 1e-3 * (torch.randn(40) + 0.4)] for _ in range(8)]
_, s0 = calm_aggregate(gz, cl, 0.1, 10.0)
for rho in [1.0, 2.0]:
    n1, s1 = calm_aggregate(gz, cl, 0.1, 10.0, clip=rho)
    for k in range(2):
        U = torch.stack([c[k].flatten() for c in cl]).double().numpy()
        d0, d1 = s0["weights"][k] @ U, s1["weights"][k] @ U
        mean = U.mean(0)
        assert np.linalg.norm(d1) <= rho * np.linalg.norm(mean) * (1 + 1e-9), "clip violated"
        assert abs(d0 @ d1 / (np.linalg.norm(d0) * np.linalg.norm(d1)) - 1) < 1e-9, "direction changed"
    print(f"clip rho={rho:g} OK  step/avg {s0['step_ratio']:.1f}x -> {s1['step_ratio']:.2f}x")
_, s2 = calm_aggregate(gz, cl, 0.1, 10.0, clip=1e6)
assert all(np.allclose(a, b) for a, b in zip(s0["weights"], s2["weights"])), "clip should be a no-op when step already below cap"
print("ALL TESTS OK")
