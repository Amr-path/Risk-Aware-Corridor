#!/usr/bin/env python3
"""
run_cpp_bench.py  --  Phase 3 (E3/E4): compiled wall-clock crossover.

Drives the C++ reference (cpp_reference/pathfind) to measure wall-clock and
node expansions at COMPILED speed, settling the questions the paper could only
conjecture in the Python regime:
  - the AILS grid-size crossover (where AILS overtakes A* in wall-clock);
  - D* Lite vs A*-rerun / ILS on clustered re-planning;
  - ILS vs JPS wall-clock on uniform-cost grids.

Node counts here equal the Python ones (see verify_parity.py); the new
information is the wall-clock at production speed.

Usage:
  cd cpp_reference && make && cd ..
  python3 experiments/run_cpp_bench.py --sizes 200 500 1000 --maps 10
  python3 experiments/run_cpp_bench.py --replan        # D* Lite re-planning mode
"""
import argparse, csv, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_all_experiments as R
from _progress import Progress

BIN = os.environ.get("PATHFIND_BIN", os.path.join(HERE, "..", "cpp_reference", "pathfind"))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def instance(gm, s, g, lam, algo, replan=None):
    H, W = gm.height, gm.width
    L = [f"GRID {H} {W}"]
    for r in range(H):
        L.append("".join('@' if gm.obstacles[r, c] else '.' for c in range(W)))
    if gm.risk is not None:
        L.append("RISK 1")
        for r in range(H):
            L.append(" ".join("%.17g" % gm.risk[r, c] for c in range(W)))
    else:
        L.append("RISK 0")
    L.append(f"QUERY {s[0]} {s[1]} {g[0]} {g[1]} {lam:.17g}")
    L.append(f"ALGO {algo}")
    if replan:
        L.append(f"REPLAN {len(replan)}")
        L += [f"{r} {c}" for (r, c) in replan]
    return "\n".join(L) + "\n"


def run(text):
    p = subprocess.run([BIN], input=text, capture_output=True, text=True)
    a = p.stdout.strip().split(",")
    return dict(nodes=int(a[1]), cost=float(a[2]), ms=float(a[3]), solved=int(a[5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=[200, 300, 500, 750, 1000])
    ap.add_argument("--maps", type=int, default=10)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--replan", action="store_true", help="D* Lite clustered re-planning mode")
    ap.add_argument("--reps", type=int, default=5, help="timing repeats (median)")
    args = ap.parse_args()
    if not os.path.exists(BIN):
        sys.exit("Build first:  cd cpp_reference && make")

    algos = ["astar", "ils", "ails"] + (["dstar"] if args.replan else ["jps"])
    rows = []
    pb = Progress(len(args.sizes) * args.maps * len(algos), desc="cppbench ")
    rng = np.random.RandomState(7)
    for N in args.sizes:
        agg = {a: dict(ms=[], nodes=[]) for a in algos}
        for m in range(args.maps):
            sd = int(rng.randint(0, 1 << 30))
            obs = R.generate_random_grid(N, 0.20, seed=sd)
            risk = R.generate_risk_layer(N, "gradient", seed=sd)
            gm = R.GridMap(width=N, height=N, obstacles=obs, risk=risk)
            s, g = (0, 0), (N - 1, N - 1)
            replan = None
            if args.replan:
                # 5 clustered blocks near mid-path
                replan = [(N // 2 + dr, N // 2 + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
            for a in algos:
                lam = 0.0 if a == "jps" else args.lam
                best = None
                for _ in range(args.reps):
                    r = run(instance(gm, s, g, lam, a, replan if a == "dstar" else None))
                    best = r["ms"] if best is None else min(best, r["ms"])
                agg[a]["ms"].append(best); agg[a]["nodes"].append(r["nodes"])
                pb.tick()
        base = np.median(agg["astar"]["ms"])
        for a in algos:
            mms = float(np.median(agg[a]["ms"]))
            rows.append(dict(N=N, algo=a, median_ms=round(mms, 4),
                             speedup_vs_astar=round(base / mms, 3) if mms > 0 else float("nan"),
                             mean_nodes=int(np.mean(agg[a]["nodes"]))))
    pb.close()
    tag = "replan" if args.replan else "static"
    out = os.path.join(RESULTS, f"cpp_bench_{tag}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nCompiled wall-clock ({tag}, lambda={args.lam}, median of {args.reps} reps):")
    print(f"{'N':>6s} {'algo':6s} {'median_ms':>10s} {'x vs A*':>8s} {'nodes':>9s}")
    for r in rows:
        print(f"{r['N']:6d} {r['algo']:6s} {r['median_ms']:10.3f} {r['speedup_vs_astar']:8.2f} {r['mean_nodes']:9d}")
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
