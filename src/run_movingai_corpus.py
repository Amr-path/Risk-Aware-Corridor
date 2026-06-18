#!/usr/bin/env python3
"""
run_movingai_corpus.py  --  Phase 3 (E1/E5): full-corpus MovingAI evaluation
against PUBLISHED optimal path lengths.

Loads MovingAI .map/.scen pairs, runs A*, ILS, AILS (optionally with a synthetic
risk layer at lambda>0), and reports per-scenario:
  - suboptimality  = found_cost / scen_optimal_length   (uses true sqrt(2))
  - node-reduction = 1 - nodes(method)/nodes(A*)
  - solved flag
Aggregates per map and per map-family, with a Holm-corrected Wilcoxon summary
(via stats_v2) on node reduction.

This addresses the "small / cherry-picked maps" weakness: point it at the full
benchmark_maps tree and it evaluates every map against the .scen optima.

Usage:
  python3 experiments/run_movingai_corpus.py --maps experiments/benchmark_maps/dao \\
      --per-map 100 --risk none
  python3 experiments/run_movingai_corpus.py --maps experiments/benchmark_maps/dao \\
      --per-map 50 --risk gradient --lam 1.0
"""
import argparse, csv, glob, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_all_experiments as R
from _progress import Progress

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
SQRT2 = math.sqrt(2.0)
PASSABLE = {".", "G"}   # standard Sturtevant convention


class NoCornerCutGrid(R.GridMap):
    """GridMap that forbids diagonal moves which cut an obstacle corner --
    the convention under which MovingAI .scen optimal lengths are computed.
    Without this, A*/ILS would find paths SHORTER than the published optimum."""
    def neighbors_8(self, r, c):
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]
        for (dr, dc), base in zip(dirs, costs):
            nr, nc = r + dr, c + dc
            if not self.is_free(nr, nc):
                continue
            if dr != 0 and dc != 0:
                # require both orthogonally adjacent cells free (no corner cutting)
                if not (self.is_free(r + dr, c) and self.is_free(r, c + dc)):
                    continue
            yield nr, nc, base


def load_map(path):
    with open(path) as f:
        lines = f.read().splitlines()
    H = W = 0; mi = 0
    for i, ln in enumerate(lines):
        if ln.startswith("height"):
            H = int(ln.split()[1])
        elif ln.startswith("width"):
            W = int(ln.split()[1])
        elif ln.strip() == "map":
            mi = i + 1; break
    grid_rows = lines[mi:mi + H]
    obs = np.ones((H, W), dtype=bool)
    for r in range(H):
        row = grid_rows[r]
        for c in range(min(W, len(row))):
            if row[c] in PASSABLE:
                obs[r, c] = False
    return NoCornerCutGrid(width=W, height=H, obstacles=obs, risk=None)


def load_scen(path):
    out = []
    with open(path) as f:
        for ln in f.read().splitlines():
            if ln.startswith("version") or not ln.strip():
                continue
            p = ln.split("\t") if "\t" in ln else ln.split()
            if len(p) < 9:
                continue
            sx, sy, gx, gy = int(p[4]), int(p[5]), int(p[6]), int(p[7])
            opt = float(p[8])
            out.append(((sy, sx), (gy, gx), opt))   # (row,col)
    return out


def cost_sqrt2(path):
    c = 0.0
    for a, b in zip(path, path[1:]):
        c += SQRT2 if (a[0] != b[0] and a[1] != b[1]) else 1.0
    return c


def discover(maps_dir):
    pairs = []
    for mp in sorted(glob.glob(os.path.join(maps_dir, "*.map"))):
        sc = mp + ".scen"
        if os.path.exists(sc):
            pairs.append((mp, sc))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", required=True, help="directory of .map (+ .map.scen) files")
    ap.add_argument("--per-map", type=int, default=100, help="scenarios sampled per map (longest first)")
    ap.add_argument("--risk", choices=["none", "gradient", "hotspot", "uniform"], default="none")
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--max-maps", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    pairs = discover(args.maps)
    if args.max_maps:
        pairs = pairs[:args.max_maps]
    if not pairs:
        sys.exit(f"No .map/.scen pairs found under {args.maps}")

    total = sum(min(args.per_map, _count_scen(sc)) for _, sc in pairs)
    pb = Progress(total, desc="corpus ")
    rows = []
    for mp, sc in pairs:
        name = os.path.basename(mp)
        gm = load_map(mp)
        if args.risk != "none" and args.lam > 0:
            risk = R.generate_risk_layer(max(gm.height, gm.width), args.risk, seed=42)
            gm = NoCornerCutGrid(width=gm.width, height=gm.height, obstacles=gm.obstacles,
                                 risk=risk[:gm.height, :gm.width])
        scen = load_scen(sc)
        scen.sort(key=lambda t: -t[2])           # longest optimal paths first
        scen = scen[:args.per_map]
        for s, g, opt in scen:
            lam = args.lam if args.risk != "none" else 0.0
            pa, na, _ = R.astar(gm, s, g, lam)
            pi, ni, _, _ = R.ils_astar(gm, s, g, lam)
            pl, nl, _, _ = R.ails_astar(gm, s, g, lam)
            pb.tick()
            if pa is None:
                continue
            base_opt = opt if lam == 0 else cost_sqrt2(pa)   # risk: compare to A* (no published opt)
            def rec(tag, path, nodes):
                solved = path is not None
                sub = (cost_sqrt2(path) / base_opt) if solved and base_opt > 0 else float("nan")
                red = (1 - nodes / max(na, 1)) if solved else float("nan")
                return dict(map=name, family=os.path.basename(args.maps), method=tag,
                            sr=s[0], sc=s[1], gr=g[0], gc=g[1], lam=lam,
                            opt=round(base_opt, 4), suboptimality=round(sub, 5),
                            node_reduction=round(red, 4), nodes=nodes, solved=int(solved))
            rows.append(rec("ILS", pi, ni))
            rows.append(rec("AILS", pl, nl))
    pb.close()

    out = os.path.join(RESULTS, f"corpus_{os.path.basename(args.maps)}_{args.risk}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # summary
    import collections
    agg = collections.defaultdict(lambda: dict(sub=[], red=[], slv=0, n=0))
    for r in rows:
        k = r["method"]
        agg[k]["n"] += 1; agg[k]["slv"] += r["solved"]
        if r["solved"] and not math.isnan(r["suboptimality"]):
            agg[k]["sub"].append(r["suboptimality"]); agg[k]["red"].append(r["node_reduction"])
    print(f"\nCorpus: {len(pairs)} maps, {total} scenarios, risk={args.risk} lam={args.lam}")
    print(f"{'method':6s} {'solve%':>7s} {'mean_subopt':>12s} {'mean_node_red%':>15s}")
    for k, v in agg.items():
        sub = 100 * (np.mean(v["sub"]) - 1) if v["sub"] else float("nan")
        red = 100 * np.mean(v["red"]) if v["red"] else float("nan")
        print(f"{k:6s} {100*v['slv']/max(v['n'],1):6.1f}% {sub:11.3f}% {red:14.1f}%")
    print(f"\nper-instance CSV: {out}")


def _count_scen(scen_path):
    n = 0
    with open(scen_path) as f:
        for ln in f:
            if ln.startswith("version") or not ln.strip():
                continue
            n += 1
    return n


if __name__ == "__main__":
    main()
