#!/usr/bin/env python3
"""
run_exp13_autopilot.py  --  Experiment 13: Autopilot Waypoint-Mission Feasibility.

Generates the data the paper's Experiment 13 reports. It reuses the SAME planners
and grid/risk generators as every other experiment (run_all_experiments.py), so the
numbers are produced by the identical code path, not a re-implementation.

PIPELINE (matches the paper's Setup paragraph):
  1. Plan a grid path with ILS (corridor-restricted A*) and with A* (optimal baseline).
  2. Convert the grid path to waypoints, 1 m per cell.
  3. Compress by collinear-waypoint removal (keep only vertices where heading changes).
  4. A 2 m proximity threshold + return-to-launch terminator are part of the emitted
     mission; they do not affect the geometric metrics below.

METRICS (per the paper's "Parameter sweep" paragraph), each defined explicitly:
  - plan_time_ms     : ILS planning wall-clock (ms), median over maps.
  - node_red_pct     : 100*(1 - ils_nodes / astar_nodes).
  - opt_ratio        : Euclidean length of the ILS path / Euclidean length of the A*
                       (optimal) path. 1.0000 means the corridor contained the optimum.
  - waypoints_after  : number of waypoints remaining after collinear compression.
  - compression_pct  : 100*(1 - waypoints_after / cells_in_path).
  - mean_heading_deg : mean absolute turn angle (deg) between consecutive segments of
                       the FULL (dense) cell path -- collinear segments contribute 0,
                       so this is the "smoothness" figure (~12 deg in the paper).
  - pipeline_speedup : astar_time / ils_time  (<1 means the pipeline is slower than A*,
                       expected on these sub-100x100 grids).

SWEEP: grid size in {30..100}, obstacle density in {10%..30%}, initial corridor width
       in {3%..15%}, 100 independent maps per configuration (paper's setting).
       Start/goal are corner-to-corner (0,0)->(N-1,N-1), as in Experiment 1.
       Planning is geometric (lambda = 0): mission geometry, not risk-weighted cost.

Usage:
  python3 experiments/run_exp13_autopilot.py            # full sweep (paper setting)
  python3 experiments/run_exp13_autopilot.py --quick    # fast smoke test
Output: results/exp13_autopilot.csv  (+ a printed summary that mirrors the paper's claims)
"""
import argparse, csv, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_all_experiments as R
from _progress import Progress

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def step_dir(a, b):
    """Unit integer direction from cell a to cell b (sign of each delta)."""
    return (int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1])))


def compress_collinear(path):
    """Keep endpoints + vertices where the heading changes (ArduPilot-style)."""
    if len(path) <= 2:
        return list(path)
    out = [path[0]]
    for i in range(1, len(path) - 1):
        if step_dir(path[i - 1], path[i]) != step_dir(path[i], path[i + 1]):
            out.append(path[i])
    out.append(path[-1])
    return out


def mean_heading_change_deg(path):
    """Mean absolute turn angle (deg) over consecutive segments of the dense path."""
    if len(path) < 3:
        return 0.0
    angs = []
    for i in range(1, len(path) - 1):
        v1 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        v2 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        crs = v1[0] * v2[1] - v1[1] * v2[0]
        angs.append(abs(math.degrees(math.atan2(crs, dot))))
    return float(np.mean(angs)) if angs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke test (fewer maps/configs)")
    args = ap.parse_args()

    sizes = [50] if args.quick else [30, 40, 50, 60, 70, 80, 90, 100]
    densities = [0.10, 0.20, 0.30] if args.quick else [0.10, 0.15, 0.20, 0.25, 0.30]
    widths = [0.05] if args.quick else [0.03, 0.05, 0.07, 0.10, 0.15]
    n_maps = 10 if args.quick else 100
    lam = 0.0  # geometric mission planning

    rng = np.random.RandomState(1313)
    rows = []
    total = len(sizes) * len(densities) * len(widths) * n_maps
    pb = Progress(total, desc="exp13   ")

    for N in sizes:
        for d in densities:
            for wf in widths:
                acc = {k: [] for k in
                       ("time", "nodered", "opt", "wp_after", "compress", "heading", "spd")}
                for m in range(n_maps):
                    sd = int(rng.randint(0, 1 << 30))
                    obs = R.generate_random_grid(N, d, seed=sd)
                    gm = R.GridMap(width=N, height=N, obstacles=obs, risk=None)
                    s, g = (0, 0), (N - 1, N - 1)
                    pa, na, ta = R.astar(gm, s, g, lam)
                    if pa is None:
                        pb.tick(); continue
                    res = R.ils_astar(gm, s, g, lam, initial_width_frac=wf)
                    pi, ni, ti = res[0], res[1], res[2]
                    if pi is None:
                        pb.tick(); continue
                    la = R.path_length_euclidean(pa)
                    li = R.path_length_euclidean(pi)
                    comp = compress_collinear(pi)
                    acc["time"].append(ti * 1000.0)
                    acc["nodered"].append(100.0 * (1 - ni / max(na, 1)))
                    acc["opt"].append(li / la if la > 0 else 1.0)
                    acc["wp_after"].append(len(comp))
                    acc["compress"].append(100.0 * (1 - len(comp) / max(len(pi), 1)))
                    acc["heading"].append(mean_heading_change_deg(pi))
                    acc["spd"].append(ta / ti if ti > 0 else float("nan"))
                    pb.tick()
                if not acc["opt"]:
                    continue
                rows.append(dict(
                    size=N, density=d, corridor_width=wf, n_valid=len(acc["opt"]),
                    plan_time_ms=round(float(np.median(acc["time"])), 4),
                    node_red_pct=round(float(np.mean(acc["nodered"])), 3),
                    opt_ratio=round(float(np.mean(acc["opt"])), 6),
                    waypoints_after=round(float(np.mean(acc["wp_after"])), 2),
                    compression_pct=round(float(np.mean(acc["compress"])), 2),
                    mean_heading_deg=round(float(np.mean(acc["heading"])), 2),
                    pipeline_speedup=round(float(np.nanmean(acc["spd"])), 4)))
    pb.close()

    out = os.path.join(RESULTS, "exp13_autopilot.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- summary that mirrors the paper's Experiment-13 claims ----
    def agg(sub, key):
        v = [r[key] for r in sub]
        return float(np.mean(v)), float(np.min(v)), float(np.max(v))

    print(f"\nExperiment 13 -- autopilot waypoint-mission feasibility "
          f"(lambda=0, {n_maps} maps/config, {len(rows)} configs)\n")
    print("Path optimality ratio by density (paper: 1.0000 up to 25%, ~1.0015 at 30%):")
    for d in densities:
        sub = [r for r in rows if r["density"] == d]
        if sub:
            mo = np.mean([r["opt_ratio"] for r in sub])
            print(f"  density {int(d*100):3d}%   mean opt_ratio = {mo:.4f}")
    cm, cmn, cmx = agg(rows, "compression_pct")
    hm, hmn, hmx = agg(rows, "mean_heading_deg")
    sm, smn, smx = agg(rows, "pipeline_speedup")
    print(f"\nWaypoint compression  (paper 70-71%): mean {cm:.1f}%  [{cmn:.1f}, {cmx:.1f}]")
    print(f"Mean heading change   (paper ~12 deg): mean {hm:.1f} deg  [{hmn:.1f}, {hmx:.1f}]")
    print(f"Pipeline speedup      (paper 0.24-0.44x): mean {sm:.2f}x  [{smn:.2f}, {smx:.2f}]")
    print("\nIndependence from corridor width (opt/compress/heading should be ~flat):")
    for wf in widths:
        sub = [r for r in rows if r["corridor_width"] == wf]
        if sub:
            print(f"  width {int(wf*100):2d}%   opt={np.mean([r['opt_ratio'] for r in sub]):.4f}"
                  f"  compress={np.mean([r['compression_pct'] for r in sub]):.1f}%"
                  f"  heading={np.mean([r['mean_heading_deg'] for r in sub]):.1f}deg")
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
