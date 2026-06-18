#!/usr/bin/env python3
"""
run_random_endpoint.py  --  referee fix B5: does the Experiment-1 picture
survive RANDOM start-goal endpoints (vs the fixed corner-to-corner geometry)?

Re-runs the Experiment-1 protocol (200x200 risk grids, 3 densities, 3 risk
types, lambda in {0.5,1.0}) but with random start-goal pairs at minimum
separation 40% of the diagonal, for A*, ILS, AILS, and (if available) RILS.
Reports speedup vs A*, node-reduction, and cost ratio so the paper can state
whether the 7.90x / ILS-leads-AILS conclusions are endpoint-dependent.

Usage:  python3 experiments/run_random_endpoint.py [--quick|--heavy]
Outputs: results/random_endpoint.csv
"""
import argparse, csv, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_all_experiments as R
from _progress import Progress
try:
    from run_supplementary_experiments import rils_astar
    HAVE_RILS = True
except Exception:
    HAVE_RILS = False

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def random_pair(N, rng, min_frac=0.40):
    diag = math.hypot(N, N)
    for _ in range(2000):
        s = (int(rng.randint(0, N)), int(rng.randint(0, N)))
        g = (int(rng.randint(0, N)), int(rng.randint(0, N)))
        if math.hypot(s[0]-g[0], s[1]-g[1]) >= min_frac*diag:
            return s, g
    return (0, 0), (N-1, N-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--heavy", action="store_true")
    args = ap.parse_args()
    N = 200
    maps = 5 if args.quick else (100 if args.heavy else 30)
    lams = [1.0] if args.quick else [0.5, 1.0]
    dens = [0.10, 0.20, 0.30]
    risks = ["gradient", "hotspot", "uniform"]
    rng = np.random.RandomState(2027)
    rows = []
    pb = Progress(len(dens)*len(risks)*len(lams)*maps, desc="rand-ep ")
    for d in dens:
        for rt in risks:
            for lam in lams:
                acc = {k: {"red": [], "ratio": [], "spd": []} for k in ["ILS", "AILS", "RILS"]}
                for m in range(maps):
                    sd = int(rng.randint(0, 1 << 30))
                    obs = R.generate_random_grid(N, d, seed=sd)
                    risk = R.generate_risk_layer(N, rt, seed=sd)
                    gm = R.GridMap(width=N, height=N, obstacles=obs, risk=risk)
                    s, g = random_pair(N, rng)
                    pa, na, ta = R.astar(gm, s, g, lam)
                    if pa is None:
                        pb.tick(); continue
                    ca = R.compute_path_cost(pa, gm, lam)
                    runs = [("ILS", R.ils_astar(gm, s, g, lam)),
                            ("AILS", R.ails_astar(gm, s, g, lam))]
                    if HAVE_RILS:
                        runs.append(("RILS", rils_astar(gm, s, g, lam)))
                    for tag, res in runs:
                        path, nodes, tt = res[0], res[1], res[2]
                        if path is None:
                            continue
                        acc[tag]["red"].append(1 - nodes/max(na, 1))
                        acc[tag]["ratio"].append(R.compute_path_cost(path, gm, lam)/ca if ca > 0 else 1.0)
                        acc[tag]["spd"].append(ta/tt if tt > 0 else float("nan"))
                    pb.tick()
                for tag in ["ILS", "AILS", "RILS"]:
                    if not acc[tag]["red"]:
                        continue
                    rows.append(dict(density=d, risk=rt, lam=lam, method=tag,
                                     speedup=round(float(np.nanmean(acc[tag]["spd"])), 3),
                                     node_red_pct=round(100*float(np.mean(acc[tag]["red"])), 2),
                                     cost_ratio=round(float(np.mean(acc[tag]["ratio"])), 5),
                                     n=len(acc[tag]["red"])))
    pb.close()
    out = os.path.join(RESULTS, "random_endpoint.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # summary: per method, mean node reduction at lambda=1.0 and best speedup
    import collections
    print(f"\nRandom-endpoint re-run (N={N}, {maps} maps/cell, vs fixed-corner Experiment 1):")
    print(f"{'method':6s} {'mean node_red%':>14s} {'mean speedup':>13s} {'mean cost_ratio':>16s}")
    for tag in ["ILS", "AILS", "RILS"]:
        sub = [r for r in rows if r["method"] == tag and r["lam"] == 1.0]
        if not sub:
            continue
        nr = np.mean([r["node_red_pct"] for r in sub])
        sp = np.nanmean([r["speedup"] for r in sub])
        cr = np.mean([r["cost_ratio"] for r in sub])
        print(f"{tag:6s} {nr:13.1f}% {sp:12.2f}x {cr:16.4f}")
    print(f"\nCSV: {out}\nInterpretation: compare ILS vs AILS mean speedup to test whether the "
          "fixed-corner ranking (ILS>AILS) flips under random endpoints.")


if __name__ == "__main__":
    main()
