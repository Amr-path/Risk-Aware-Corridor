#!/usr/bin/env python3
"""
run_scaling.py  --  Phase 3 (E2): empirical validation of the new theory.

Two checks, both tied to theorems added to the manuscript:

  PART A  Corridor-size bound (Theorem "Corridor-size bound"):
          |C(r)| <= (4r+1)(D_inf + 2r + 1)   for every line and width r.
          Pure geometry; asserts the bound is never violated and reports
          how tight it is.

  PART B  Node-expansion reduction (Corollary "Node-expansion reduction"):
          restricted search expands O(alpha N^2) nodes vs Theta(N^2),
          i.e. a Theta(1/alpha) reduction that is stable as N grows.
          Sweeps grid size N and corridor fraction alpha, measuring
          nodes(A*) / nodes(ILS) and |V|/|C|.

Outputs CSV to results/scaling_partA.csv and results/scaling_partB.csv and
prints a PASS/FAIL verdict for the analytic bound.

Usage:
  python3 experiments/run_scaling.py                 # default sizes
  python3 experiments/run_scaling.py --quick         # tiny, for a smoke test
  python3 experiments/run_scaling.py --heavy         # 200..2000, long run
"""
import argparse, csv, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_all_experiments as R
from _progress import Progress

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def corridor_cells_and_bound(H, W, start, goal, r):
    """Return (|C(r)|, analytic_bound) for half-width r (Chebyshev)."""
    cw = 2 * r + 1                       # full width whose //2 == r
    mask = R.build_corridor_mask(H, W, start, goal, cw)
    count = int(mask.sum())
    Dinf = max(abs(start[0] - goal[0]), abs(start[1] - goal[1]))
    bound = (4 * r + 1) * (Dinf + 2 * r + 1)
    return count, bound


def part_a(sizes, seed=7, pairs_per_size=8):
    rng = np.random.RandomState(seed)
    rows = []
    violations = 0
    worst_ratio = 0.0
    pb = Progress(len(sizes) * pairs_per_size, desc="scaling A ")
    for N in sizes:
        for _ in range(pairs_per_size):
            pb.tick()
            s = (int(rng.randint(0, N)), int(rng.randint(0, N)))
            g = (int(rng.randint(0, N)), int(rng.randint(0, N)))
            if s == g:
                continue
            for r in [1, 2, 3, int(0.01 * N), int(0.05 * N), int(0.10 * N)]:
                if r < 1:
                    continue
                count, bound = corridor_cells_and_bound(N, N, s, g, r)
                ok = count <= bound
                if not ok:
                    violations += 1
                worst_ratio = max(worst_ratio, count / bound)
                rows.append(dict(N=N, sr=s[0], sc=s[1], gr=g[0], gc=g[1],
                                 r=r, cells=count, bound=bound,
                                 ratio=round(count / bound, 4), ok=int(ok)))
    pb.close()
    with open(os.path.join(RESULTS, "scaling_partA.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[PART A] corridor-size bound checked on {len(rows)} (line,r) cases.")
    print(f"         violations: {violations}  (must be 0)")
    print(f"         tightest case: empirical/bound = {worst_ratio:.3f} (<=1 confirms the bound)")
    return violations == 0


def part_b(sizes, alphas, maps_per=5, seed=11):
    rng = np.random.RandomState(seed)
    rows = []
    pb = Progress(len(sizes) * len(alphas) * maps_per, desc="scaling B ")
    for N in sizes:
        for ai, alpha in enumerate(alphas):
            ratios_nodes = []
            ratios_space = []
            for m in range(maps_per):
                pb.tick()
                sd = int(rng.randint(0, 1 << 30))
                obs = R.generate_random_grid(N, 0.20, seed=sd)
                risk = R.generate_risk_layer(N, "gradient", seed=sd)
                gm = R.GridMap(width=N, height=N, obstacles=obs, risk=risk)
                s, g = (0, 0), (N - 1, N - 1)
                _, na, _ = R.astar(gm, s, g, lam=1.0)
                _, ni, _, _ = R.ils_astar(gm, s, g, lam=1.0,
                                          initial_width_frac=alpha, max_attempts=10)
                # realized initial corridor cell count
                diag = int(math.sqrt(N * N + N * N))
                r0 = max(3, int(alpha * diag)) // 2
                cells, _ = corridor_cells_and_bound(N, N, s, g, r0)
                ratios_nodes.append(na / max(ni, 1))
                ratios_space.append((N * N) / max(cells, 1))
            row = dict(N=N, alpha=alpha,
                       node_reduction_factor=round(float(np.mean(ratios_nodes)), 3),
                       space_reduction_VC=round(float(np.mean(ratios_space)), 3),
                       predicted_1_over_4alpha=round(1.0 / (4 * alpha), 3))
            rows.append(row)
            pb.stream.write("\r" + " " * 100 + "\r")  # clear bar before the row line
            print(f"[PART B] N={N:5d} alpha={alpha:.2f}  "
                  f"node_reduction={row['node_reduction_factor']:6.2f}x  "
                  f"|V|/|C|={row['space_reduction_VC']:6.2f}  "
                  f"~1/(4a)={row['predicted_1_over_4alpha']:.2f}")
    pb.close()
    with open(os.path.join(RESULTS, "scaling_partB.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--heavy", action="store_true")
    args = ap.parse_args()
    if args.quick:
        sizes_a = [64, 128]; sizes_b = [64, 128]; alphas = [0.05, 0.10]; maps = 2
    elif args.heavy:
        sizes_a = [200, 300, 500, 750, 1000, 1500, 2000]
        sizes_b = [200, 300, 500, 750, 1000, 1500, 2000]
        alphas = [0.02, 0.05, 0.10, 0.20]; maps = 10
    else:
        sizes_a = [128, 256, 512]; sizes_b = [128, 256, 512]
        alphas = [0.05, 0.10]; maps = 4
    ok = part_a(sizes_a)
    part_b(sizes_b, alphas, maps_per=maps)
    print("\nVERDICT:", "corridor-size bound HOLDS on all tested cases."
          if ok else "BOUND VIOLATED -- investigate.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
