"""
run_experiment_12.py
====================
Experiment 12: Real Port Validation on Penang Port (NBCT).

Loads experiments/data/penang_port.npz produced by download_penang_port.py
and runs A*, JPS, ILS, AILS, RILS on 30 random start-goal pairs at three
risk weights (lambda in {0.0, 0.5, 1.0}).

Output: results/exp12_penang_port.csv

USAGE (on your laptop, after download_penang_port.py has run):
    cd "/Users/amralshahed/Downloads/PHD-Thesis-Apr/Risk_Aware_Corridor_Pathfinding"
    python experiments/run_experiment_12.py

Expected runtime: 5-15 minutes on a typical laptop.
"""
from __future__ import annotations
import os
import sys
import time
import csv
import numpy as np

# Make sibling experiment modules importable
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Reuse the canonical algorithm implementations from the existing suite
from run_supplementary_experiments import (
    GridMap, astar, ils_astar, ails_astar, rils_astar,
    generate_random_endpoints,
    compute_path_cost, compute_exposure, path_length_euclidean,
)
from run_all_experiments import jps_astar


NPZ_PATH = os.path.join(HERE, "data", "penang_port.npz")
OUT_CSV  = os.path.join(os.path.dirname(HERE), "results", "exp12_penang_port.csv")

LAMBDAS = [0.0, 0.5, 1.0]
N_TRIALS = 30                  # 30 random start-goal pairs
MIN_DIST_FRAC = 0.4            # endpoints must be at least 40% of grid apart
SEED = 20260518                # paper repro seed


def expand_jps_path(sparse_path):
    """JPS returns only jump points (not every cell along the way).
    Expand to a dense 8-connected cell sequence so that downstream
    cost/exposure metrics measure the actual path, not the jump-point
    skeleton. Each segment between consecutive jump points (a, b) is
    octile-optimal: take diagonal steps until aligned with b on a row
    or column, then take cardinal steps the rest of the way.
    """
    if not sparse_path or len(sparse_path) < 2:
        return sparse_path
    dense = [sparse_path[0]]
    for i in range(len(sparse_path) - 1):
        r, c = sparse_path[i]
        r2, c2 = sparse_path[i + 1]
        while (r, c) != (r2, c2):
            dr = 0 if r == r2 else (1 if r2 > r else -1)
            dc = 0 if c == c2 else (1 if c2 > c else -1)
            r, c = r + dr, c + dc
            dense.append((r, c))
    return dense


def safe_metrics(path, grid_map, lam, expand=False):
    """Path-quality metrics that don't blow up on None paths.
    `expand=True` first densifies a sparse path (used for JPS output).
    """
    if path is None:
        return None, None, None, None
    if expand:
        path = expand_jps_path(path)
    cost = compute_path_cost(path, grid_map, lam)
    expo = compute_exposure(path, grid_map)
    length = path_length_euclidean(path)
    return cost, expo, length, len(path)


def main():
    if not os.path.exists(NPZ_PATH):
        print(f"ERROR: {NPZ_PATH} not found.", file=sys.stderr)
        print("Run experiments/download_penang_port.py first.", file=sys.stderr)
        sys.exit(2)

    print(f"Loading port grid: {NPZ_PATH}")
    data = np.load(NPZ_PATH, allow_pickle=True)
    obstacles = data["obstacles"]
    risk = data["risk"]
    H, W = obstacles.shape
    diag = np.sqrt(H**2 + W**2)
    print(f"  {H}x{W}, obstacle density {100*obstacles.mean():.1f}%, "
          f"mean risk {risk.mean():.3f}")

    gm = GridMap(W, H, obstacles, risk)

    # Generate 30 random endpoint pairs (shared across all lambdas
    # so per-pair comparisons are meaningful)
    rng = np.random.RandomState(SEED)
    endpoints = []
    print(f"Generating {N_TRIALS} random endpoint pairs...")
    for i in range(N_TRIALS):
        start, goal = generate_random_endpoints(W, obstacles, rng,
                                                 min_dist_frac=MIN_DIST_FRAC)
        endpoints.append((start, goal))
    print(f"  done ({len(endpoints)} pairs)")

    rows = []
    print(f"Running algorithms for lambda in {LAMBDAS}...")
    for lam in LAMBDAS:
        per_alg = {a: dict(time=[], nodes=[], cost=[], expo=[], length=[],
                           failures=0)
                   for a in ["astar", "jps", "ils", "ails", "rils"]}

        for i, (start, goal) in enumerate(endpoints, 1):
            # ---- A* (baseline) ----
            pa, na, ta = astar(gm, start, goal, lam=lam)
            per_alg["astar"]["time"].append(ta)
            per_alg["astar"]["nodes"].append(na)
            ca, ea, la_, _ = safe_metrics(pa, gm, lam)
            if pa is None:
                per_alg["astar"]["failures"] += 1
            per_alg["astar"]["cost"].append(ca)
            per_alg["astar"]["expo"].append(ea)
            per_alg["astar"]["length"].append(la_)

            # ---- JPS (only meaningful at lambda=0) ----
            if lam == 0.0:
                pj, nj, tj = jps_astar(gm, start, goal)
                per_alg["jps"]["time"].append(tj)
                per_alg["jps"]["nodes"].append(nj)
                # JPS returns only jump points; expand before scoring cost/exposure
                cj, ej, lj, _ = safe_metrics(pj, gm, lam, expand=True)
                if pj is None:
                    per_alg["jps"]["failures"] += 1
                per_alg["jps"]["cost"].append(cj)
                per_alg["jps"]["expo"].append(ej)
                per_alg["jps"]["length"].append(lj)

            # ---- ILS ----
            res = ils_astar(gm, start, goal, lam=lam, initial_width_frac=0.05)
            pi, ni, ti = res[0], res[1], res[2]
            per_alg["ils"]["time"].append(ti)
            per_alg["ils"]["nodes"].append(ni)
            ci, ei, li, _ = safe_metrics(pi, gm, lam)
            if pi is None:
                per_alg["ils"]["failures"] += 1
            per_alg["ils"]["cost"].append(ci)
            per_alg["ils"]["expo"].append(ei)
            per_alg["ils"]["length"].append(li)

            # ---- AILS ----
            res = ails_astar(gm, start, goal, lam=lam,
                              r_min=2, r_max=max(3, int(0.10*min(H, W))),
                              alpha=1.0, omega=3)
            paI, naI, taI = res[0], res[1], res[2]
            per_alg["ails"]["time"].append(taI)
            per_alg["ails"]["nodes"].append(naI)
            cI, eI, lI, _ = safe_metrics(paI, gm, lam)
            if paI is None:
                per_alg["ails"]["failures"] += 1
            per_alg["ails"]["cost"].append(cI)
            per_alg["ails"]["expo"].append(eI)
            per_alg["ails"]["length"].append(lI)

            # ---- RILS ----
            res = rils_astar(gm, start, goal, lam=lam,
                              r_base_frac=0.05, r_max_frac=0.15,
                              beta=1.0, omega=5)
            pr, nr_, tr = res[0], res[1], res[2]
            per_alg["rils"]["time"].append(tr)
            per_alg["rils"]["nodes"].append(nr_)
            cr, er, lr_, _ = safe_metrics(pr, gm, lam)
            if pr is None:
                per_alg["rils"]["failures"] += 1
            per_alg["rils"]["cost"].append(cr)
            per_alg["rils"]["expo"].append(er)
            per_alg["rils"]["length"].append(lr_)

            if i % 5 == 0:
                print(f"  lambda={lam}: {i}/{N_TRIALS} pairs done")

        # Aggregate
        def nanmean(xs):
            xs = [x for x in xs if x is not None]
            return float(np.mean(xs)) if xs else float("nan")
        def nanstd(xs):
            xs = [x for x in xs if x is not None]
            return float(np.std(xs, ddof=1)) if len(xs) > 1 else float("nan")

        row = dict(lambd=lam, n_valid=N_TRIALS)
        astar_time = nanmean(per_alg["astar"]["time"])
        astar_nodes = nanmean(per_alg["astar"]["nodes"])
        astar_cost = nanmean(per_alg["astar"]["cost"])
        astar_expo = nanmean(per_alg["astar"]["expo"])

        row.update({
            "astar_time_ms": astar_time,
            "astar_nodes": astar_nodes,
            "astar_cost": astar_cost,
            "astar_failures": per_alg["astar"]["failures"],
        })

        for alg in ["jps", "ils", "ails", "rils"]:
            if alg == "jps" and lam != 0.0:
                # Skip JPS reporting at lambda > 0 (not applicable)
                for k in ("time_ms", "nodes", "speedup", "node_red_pct",
                          "cost_ratio", "exposure_ratio", "failures"):
                    row[f"{alg}_{k}"] = float("nan")
                continue
            tm = nanmean(per_alg[alg]["time"])
            ts = nanstd(per_alg[alg]["time"])
            nm = nanmean(per_alg[alg]["nodes"])
            cm = nanmean(per_alg[alg]["cost"])
            em = nanmean(per_alg[alg]["expo"])
            speedup = astar_time / tm if tm > 0 else float("nan")
            node_red = 100 * (1 - nm / astar_nodes) if astar_nodes else float("nan")
            cost_ratio = cm / astar_cost if astar_cost else float("nan")
            expo_ratio = em / astar_expo if astar_expo else float("nan")
            row[f"{alg}_time_ms"] = tm
            row[f"{alg}_time_std"] = ts
            row[f"{alg}_nodes"] = nm
            row[f"{alg}_speedup"] = speedup
            row[f"{alg}_node_red_pct"] = node_red
            row[f"{alg}_cost_ratio"] = cost_ratio
            row[f"{alg}_exposure_ratio"] = expo_ratio
            row[f"{alg}_failures"] = per_alg[alg]["failures"]

        rows.append(row)

        print(f"  lambda={lam} summary:")
        print(f"    A*    : {astar_time:7.1f}ms  {astar_nodes:7.0f} nodes")
        for alg in ["jps", "ils", "ails", "rils"]:
            if alg == "jps" and lam != 0.0:
                continue
            print(f"    {alg.upper():5}: {row[f'{alg}_time_ms']:7.1f}ms "
                  f"speedup={row[f'{alg}_speedup']:5.2f}x  "
                  f"red={row[f'{alg}_node_red_pct']:5.1f}%  "
                  f"cost_ratio={row[f'{alg}_cost_ratio']:.4f}")

    # Write CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print()
    print(f"Wrote {OUT_CSV}")
    print()
    print("Now paste the numbers into Table 12 in main.tex,")
    print("or send me the CSV and I'll generate the LaTeX table for you.")


if __name__ == "__main__":
    main()
