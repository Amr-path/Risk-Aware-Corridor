#!/usr/bin/env python3
# =============================================================================
# Adaptivity ablation (referee M4): does AILS's *adaptive shape* help, or is it
# just a wider average corridor?
#
# For each instance we build TWO corridors with the SAME painting machinery,
# differing only in the half-width:
#   (a) AILS    : per-cell half-width radius(p) = r_min + (r_max-r_min)*dens(p)^alpha   (variable)
#   (b) FIXED   : constant half-width tuned so the corridor has the SAME total AREA
#                 (cell budget) as AILS -- a "de-adapted" AILS of equal area
# Both are searched by the identical restricted A*; A* (full grid) is the baseline.
# If AILS beats the equal-mean-width FIXED corridor on node reduction and/or
# feasibility, the gain is the ADAPTIVITY, not the average width.
#
# Run from the project root on the Mac (numpy/scipy already present):
#   PYTHONHASHSEED=0 python3 experiments/run_ablation_adaptivity.py
#   PYTHONHASHSEED=0 python3 experiments/run_ablation_adaptivity.py --sizes 500 --maps 15
#
# Output: results/ablation_adaptivity.csv  + a printed headline.
# Runtime: tens of minutes (A* on 500^2 is the slow part), same order as Exp 9.
# =============================================================================
import os, sys, csv, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_supplementary_experiments as E   # the exact Exp-9 stack
import heapq


def ails_mask_and_meanwidth(gm, start, goal, r_min=2, r_max=None, alpha=1.0, omega=3):
    """Exact AILS base (attempt-0) corridor: returns (mask, mean_halfwidth, area)."""
    h, w = gm.height, gm.width
    if r_max is None:
        r_max = max(r_min + 1, int(0.1 * min(h, w)))      # AILS default
    integral = E.compute_integral_image(gm.obstacles)
    line = E.bresenham_line(start[0], start[1], goal[0], goal[1])
    mask = np.zeros((h, w), dtype=bool)
    radii = []
    for lr, lc in line:
        density = E.query_density(integral, lr, lc, omega, h, w)
        radius = int(r_min + (r_max - r_min) * (density ** alpha))   # = AILS half-width
        radius = max(r_min, min(radius, r_max))
        radii.append(radius)
        mask[max(0, lr-radius):min(h, lr+radius+1),
             max(0, lc-radius):min(w, lc+radius+1)] = True
    return mask, float(np.mean(radii)), int(mask.sum())


def fixed_mask_matched_area(gm, start, goal, target_area):
    """Constant-half-width corridor whose AREA matches target_area as closely as
    possible (binary search; corridor area is monotone in width). Returns (mask, area, hw)."""
    h, w = gm.height, gm.width
    lo, hi = 1, max(h, w)
    best_m = best_a = best_hw = None
    while lo <= hi:
        mid = (lo + hi) // 2
        m = E.build_corridor_mask(h, w, start, goal, 2 * mid)
        a = int(m.sum())
        if best_a is None or abs(a - target_area) < abs(best_a - target_area):
            best_m, best_a, best_hw = m, a, mid
        if a < target_area:
            lo = mid + 1
        else:
            hi = mid - 1
    return best_m, best_a, best_hw


def search_mask(gm, start, goal, lam, mask):
    """Restricted A* (identical inner loop to ils_astar). Returns (path_or_None, nodes)."""
    sr, sc = start; gr, gc = goal
    open_list = []; g = {(sr, sc): 0.0}; came = {}; closed = set(); nodes = 0
    heapq.heappush(open_list, (E.heuristic_octile(sr, sc, gr, gc), 0, sr, sc)); cnt = 1
    while open_list:
        f, _, r, c = heapq.heappop(open_list)
        if (r, c) in closed: continue
        closed.add((r, c))
        if (r, c) == (gr, gc):
            path = []; cur = (gr, gc)
            while cur in came: path.append(cur); cur = came[cur]
            path.append(start); path.reverse()
            return path, nodes
        nodes += 1
        for nr, nc, base_cost in gm.neighbors_8(r, c):
            if not mask[nr, nc] or (nr, nc) in closed: continue
            risk = lam * gm.risk[nr, nc] if (gm.risk is not None and lam > 0) else 0.0
            ng = g[(r, c)] + base_cost + risk
            if ng < g.get((nr, nc), float('inf')):
                g[(nr, nc)] = ng; came[(nr, nc)] = (r, c)
                heapq.heappush(open_list, (ng + E.heuristic_octile(nr, nc, gr, gc), cnt, nr, nc)); cnt += 1
    return None, nodes


def random_endpoints(gm, diag, rng, min_frac=0.40, tries=2000):
    h, w = gm.height, gm.width; need = min_frac * diag
    free = lambda r, c: gm.obstacles[r, c] == 0
    for _ in range(tries):
        sr, sc, gr, gc = rng.integers(0, h), rng.integers(0, w), rng.integers(0, h), rng.integers(0, w)
        if free(sr, sc) and free(gr, gc) and np.hypot(sr-gr, sc-gc) >= need:
            return (int(sr), int(sc)), (int(gr), int(gc))
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[300, 500])
    ap.add_argument("--risks", nargs="+", default=["gradient", "hotspot", "uniform"])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--maps", type=int, default=8)
    args = ap.parse_args()

    rows = []
    for size in args.sizes:
        diag = int(np.sqrt(2) * size)
        for risk in args.risks:
            for m in range(args.maps):
                seed = E.make_instance_seed(risk, 0.20, args.lam, m, offset=900000)
                obs = E.generate_random_grid(size, 0.20, seed=seed)
                rk = E.generate_risk_layer(size, risk, seed=seed + 10000)
                gm = E.GridMap(size, size, obs, rk)
                rng = np.random.default_rng(seed + 777)
                s, g = random_endpoints(gm, diag, rng)
                if s is None: continue

                pa, na, _ = E.astar(gm, s, g, args.lam)
                if pa is None: continue
                ca = E.compute_path_cost(pa, gm, args.lam)

                m_ails, rbar, area_ails = ails_mask_and_meanwidth(gm, s, g)
                m_fix, area_fix, hw_fix = fixed_mask_matched_area(gm, s, g, area_ails)
                p_ails, n_ails = search_mask(gm, s, g, args.lam, m_ails)
                p_fix,  n_fix  = search_mask(gm, s, g, args.lam, m_fix)

                rows.append(dict(
                    size=size, risk=risk, lam=args.lam, m=m, rbar=round(rbar, 2), hw_fix=hw_fix,
                    astar_nodes=na,
                    area_ails=area_ails, area_fix=area_fix,
                    ails_ok=int(p_ails is not None), fix_ok=int(p_fix is not None),
                    ails_nodes=n_ails, fix_nodes=n_fix,
                    ails_red=round((1 - n_ails/na)*100, 2) if p_ails is not None else None,
                    fix_red=round((1 - n_fix/na)*100, 2) if p_fix is not None else None,
                    ails_opt=round(E.compute_path_cost(p_ails, gm, args.lam)/ca, 4) if p_ails is not None else None,
                    fix_opt=round(E.compute_path_cost(p_fix, gm, args.lam)/ca, 4) if p_fix is not None else None,
                ))
                print(f"  {size}^2 {risk} m{m}: r̄={rbar:.1f} area AILS/FIX={area_ails}/{area_fix} "
                      f"red AILS/FIX={rows[-1]['ails_red']}/{rows[-1]['fix_red']} "
                      f"ok={rows[-1]['ails_ok']}/{rows[-1]['fix_ok']}")

    out = os.path.join(HERE, "..", "results", "ablation_adaptivity.csv")
    if rows:
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print(f"\nsaved {out}\n")

    # ---- headline (the matched-mean-width ablation result) ----
    both = [r for r in rows if r["ails_ok"] and r["fix_ok"]]
    if both:
        a = np.mean([r["ails_red"] for r in both]); fx = np.mean([r["fix_red"] for r in both])
        win = np.mean([r["ails_red"] > r["fix_red"] for r in both]) * 100
        aa = np.mean([r["area_ails"] for r in both]); af = np.mean([r["area_fix"] for r in both])
        print("=" * 64)
        print(f"ABLATION (matched corridor AREA, base corridors, n={len(both)} both-feasible):")
        print(f"  mean corridor AREA   AILS {aa:,.0f}  vs  FIXED {af:,.0f}  "
              f"({100*(aa-af)/af:+.1f}% gap -> areas matched)")
        print(f"  mean NODE REDUCTION  AILS {a:.1f}%   vs  FIXED {fx:.1f}%   "
              f"(adaptivity gain {a-fx:+.1f} pts; AILS wins {win:.0f}% of cases)")
        print(f"  mean PATH OPT ratio  AILS {np.mean([r['ails_opt'] for r in both]):.4f}  "
              f"vs FIXED {np.mean([r['fix_opt'] for r in both]):.4f}")
    ao = sum(r["ails_ok"] and not r["fix_ok"] for r in rows)
    fo = sum(r["fix_ok"] and not r["ails_ok"] for r in rows)
    print(f"  FEASIBILITY at equal area: AILS-only-feasible {ao}, FIXED-only-feasible {fo} "
          f"(of {len(rows)} instances)")
    print("=" * 64)
    print("Interpretation: AILS reduction > FIXED reduction at equal corridor area  =>  "
          "the gain comes from the density-adaptive SHAPE, not a larger corridor budget.")


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("[note] run with  PYTHONHASHSEED=0  for deterministic instances")
    main()
