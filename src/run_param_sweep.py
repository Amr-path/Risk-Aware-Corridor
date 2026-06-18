#!/usr/bin/env python3
"""
run_param_sweep.py  --  Phase 3 (E6/E7): justify every default empirically
and audit the risk-adjusted heuristic's admissibility.

Sweeps (one factor at a time, around the paper defaults):
  ILS   : alpha0 (initial corridor fraction), K_max (max attempts)
  AILS  : alpha (density exponent), omega (window half-size)
  RILS  : beta  (risk exponent)
For each setting: mean node-reduction vs A* and mean path-cost ratio.

rho-bar admissibility audit (backs Proposition "Conditional Heuristic
Admissibility"): for rho_bar in {0,0.15,0.3,0.5}, run A* with the risk-adjusted
heuristic h_lambda = h*(1+lambda*rho_bar) and measure the fraction of queries
whose returned cost EXCEEDS the true optimum (computed with the admissible
h0). rho_bar=0 must give 0 violations; larger rho_bar trades guidance for
occasional inadmissibility -- the audit quantifies exactly where.

Outputs: results/param_sweep.csv, results/rho_audit.csv, results/defaults.json
Usage: python3 experiments/run_param_sweep.py [--quick|--heavy]
"""
import argparse, csv, heapq, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_all_experiments as R
from _progress import Progress
try:
    from run_supplementary_experiments import rils_astar
    _HAVE_RILS = True
except Exception:
    _HAVE_RILS = False

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def astar_hlambda(gm, start, goal, lam, rho_bar):
    """A* using the risk-adjusted heuristic h_lambda = h*(1+lam*rho_bar)."""
    sr, sc = start; gr, gc = goal
    kappa = 1.0 + lam * rho_bar
    open_list = []; g_score = {(sr, sc): 0.0}; came = {}; closed = set()
    heapq.heappush(open_list, (R.heuristic_octile(sr, sc, gr, gc) * kappa, 0, sr, sc))
    cnt = 1
    while open_list:
        f, _, r, c = heapq.heappop(open_list)
        if (r, c) in closed:
            continue
        closed.add((r, c))
        if (r, c) == (gr, gc):
            path = []; cur = (gr, gc)
            while cur in came:
                path.append(cur); cur = came[cur]
            path.append(start); path.reverse(); return path
        for nr, nc, base in gm.neighbors_8(r, c):
            if (nr, nc) in closed:
                continue
            rc = lam * gm.risk[nr, nc] if (gm.risk is not None and lam > 0) else 0.0
            ng = g_score[(r, c)] + base + rc
            if ng < g_score.get((nr, nc), float("inf")):
                g_score[(nr, nc)] = ng; came[(nr, nc)] = (r, c)
                h = R.heuristic_octile(nr, nc, gr, gc) * kappa
                heapq.heappush(open_list, (ng + h, cnt, nr, nc)); cnt += 1
    return None


def make_map(N, density, risk_type, seed):
    obs = R.generate_random_grid(N, density, seed=seed)
    risk = R.generate_risk_layer(N, risk_type, seed=seed)
    return R.GridMap(width=N, height=N, obstacles=obs, risk=risk)


def reduction_and_quality(run_fn, gm, s, g, lam):
    pa, na, _ = R.astar(gm, s, g, lam)
    res = run_fn(gm, s, g, lam)
    path, nodes = res[0], res[1]
    if path is None or pa is None:
        return None
    red = 1.0 - nodes / max(na, 1)
    ca = R.compute_path_cost(pa, gm, lam)
    cc = R.compute_path_cost(path, gm, lam)
    ratio = cc / ca if ca > 0 else 1.0
    return red, ratio


def sweep(N, maps, lam=1.0, seed=3):
    rng = np.random.RandomState(seed)
    rows = []
    configs = []
    for a0 in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]:
        configs.append(("ILS.alpha0", a0,
                        lambda gm, s, g, l, a0=a0: R.ils_astar(gm, s, g, l, initial_width_frac=a0)))
    for km in [1, 3, 5, 10]:
        configs.append(("ILS.Kmax", km,
                        lambda gm, s, g, l, km=km: R.ils_astar(gm, s, g, l, max_attempts=km)))
    for al in [0.5, 1.0, 2.0]:
        configs.append(("AILS.alpha", al,
                        lambda gm, s, g, l, al=al: R.ails_astar(gm, s, g, l, alpha=al)))
    for om in [1, 3, 5, 7, 9]:
        configs.append(("AILS.omega", om,
                        lambda gm, s, g, l, om=om: R.ails_astar(gm, s, g, l, omega=om)))
    if _HAVE_RILS:
        for be in [0.5, 1.0, 2.0]:
            configs.append(("RILS.beta", be,
                            lambda gm, s, g, l, be=be: rils_astar(gm, s, g, l, beta=be)))

    pb = Progress(len(configs) * maps, desc="param ")
    for name, val, fn in configs:
        reds, ratios = [], []
        for m in range(maps):
            sd = int(rng.randint(0, 1 << 30))
            gm = make_map(N, 0.20, "gradient", sd)
            s, g = (0, 0), (N - 1, N - 1)
            out = reduction_and_quality(fn, gm, s, g, lam)
            pb.tick()
            if out:
                reds.append(out[0]); ratios.append(out[1])
        rows.append(dict(param=name, value=val,
                         node_reduction_pct=round(100 * float(np.mean(reds)), 2),
                         cost_ratio=round(float(np.mean(ratios)), 5),
                         n=len(reds)))
    pb.close()
    with open(os.path.join(RESULTS, "param_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows


def rho_audit(N, maps, lam=1.0, seed=5):
    rng = np.random.RandomState(seed)
    rows = []
    rho_bars = [0.0, 0.15, 0.3, 0.5]
    risk_types = ["gradient", "hotspot", "uniform"]
    pb = Progress(len(rho_bars) * len(risk_types) * maps, desc="rho   ")
    for rt in risk_types:
        for rb in rho_bars:
            viol = 0; tot = 0; max_excess = 0.0
            for m in range(maps):
                sd = int(rng.randint(0, 1 << 30))
                gm = make_map(N, 0.20, rt, sd)
                s, g = (0, 0), (N - 1, N - 1)
                popt, _, _ = R.astar(gm, s, g, lam)   # admissible h0 -> true optimum
                ph = astar_hlambda(gm, s, g, lam, rb)
                pb.tick()
                if popt is None or ph is None:
                    continue
                copt = R.compute_path_cost(popt, gm, lam)
                ch = R.compute_path_cost(ph, gm, lam)
                tot += 1
                if ch > copt * (1 + 1e-9):
                    viol += 1
                    max_excess = max(max_excess, ch / copt - 1.0)
            rows.append(dict(risk_type=rt, rho_bar=rb, n=tot,
                             violation_pct=round(100 * viol / max(tot, 1), 2),
                             max_excess_pct=round(100 * max_excess, 3)))
    pb.close()
    with open(os.path.join(RESULTS, "rho_audit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows


def pick_defaults(sweep_rows):
    """Choose the alpha0 at the knee: smallest cost-ratio inflation per unit speed.
    Reports the data-justified default rather than asserting 0.05."""
    ils = [r for r in sweep_rows if r["param"] == "ILS.alpha0"]
    # knee: maximise node_reduction while cost_ratio-1 <= 0.005
    good = [r for r in ils if r["cost_ratio"] - 1 <= 0.005]
    best = max(good, key=lambda r: r["node_reduction_pct"]) if good else max(ils, key=lambda r: r["node_reduction_pct"])
    return {"ILS.alpha0_recommended": best["value"],
            "criterion": "max node reduction with mean cost ratio <= 1.005",
            "node_reduction_pct": best["node_reduction_pct"],
            "cost_ratio": best["cost_ratio"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--heavy", action="store_true")
    args = ap.parse_args()
    if args.quick:
        N, maps_s, maps_r = 80, 3, 3
    elif args.heavy:
        N, maps_s, maps_r = 200, 50, 50
    else:
        N, maps_s, maps_r = 128, 10, 10
    print(f"RILS available: {_HAVE_RILS}")
    sweep_rows = sweep(N, maps_s)
    audit_rows = rho_audit(N, maps_r)
    defaults = pick_defaults(sweep_rows)
    with open(os.path.join(RESULTS, "defaults.json"), "w") as f:
        json.dump(defaults, f, indent=2)
    print("\n[defaults.json]"); print(json.dumps(defaults, indent=2))
    print("\n[rho admissibility audit] (rho_bar=0 must be 0% violations)")
    for r in audit_rows:
        print(f"  {r['risk_type']:9s} rho_bar={r['rho_bar']:.2f}  "
              f"violations={r['violation_pct']:5.1f}%  max_excess={r['max_excess_pct']:.2f}%")


if __name__ == "__main__":
    main()
