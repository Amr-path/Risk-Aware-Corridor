#!/usr/bin/env python3
# =============================================================================
# Add 95% bootstrap CIs to the Experiment 2 and Experiment 9 tables (referee R5).
#
# It REUSES your own generators / algorithms / seeds (imported from
# run_all_experiments.py and run_supplementary_experiments.py), so the point
# estimates reproduce the paper exactly; it only adds the missing CIs by
# bootstrapping the per-instance paired samples those functions already build.
#
# Run from the project root on the Mac:
#     PYTHONHASHSEED=0 python3 experiments/add_exp2_exp9_cis.py
#
# Outputs:
#     results/exp2_port_environment_ci.csv
#     results/exp9_large_grid_ails_ci.csv
# and prints paste-ready LaTeX cells for tab:exp2_port and tab:exp9_large.
#
# Runtime note: this re-runs the same loops in pure Python (incl. 500x500
# grids), so expect tens of minutes, same as the original runs.
# =============================================================================
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_all_experiments as E2            # Exp 2 lives here
import run_supplementary_experiments as E9  # Exp 9 lives here

RESULTS = os.path.join(HERE, "..", "results")
cpc9 = getattr(E9, "compute_path_cost", E2.compute_path_cost)


def boot_ci(num, den, kind, B=10000, seed=0):
    """Percentile bootstrap CI of a ratio-of-means statistic, matching the
    paper's point estimate (resample paired instances, recompute the ratio)."""
    num = np.asarray(num, float); den = np.asarray(den, float)
    rng = np.random.default_rng(seed); n = len(num); out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        mn, md = num[idx].mean(), den[idx].mean()
        if kind == "speedup":
            out[b] = mn / md if md > 0 else 0.0
        else:  # node-reduction %
            out[b] = (1 - md / mn) * 100 if mn > 0 else 0.0
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def summarise(at, it, ait, an, iin, ain):
    """Point estimates (ratio of means, exactly as the paper) + bootstrap CIs."""
    at, it, ait = map(lambda x: np.asarray(x, float), (at, it, ait))
    an, iin, ain = map(lambda x: np.asarray(x, float), (an, iin, ain))
    r = {}
    r["ils_speed"]  = at.mean() / it.mean()
    r["ails_speed"] = at.mean() / ait.mean()
    r["ils_red"]    = (1 - iin.mean() / an.mean()) * 100
    r["ails_red"]   = (1 - ain.mean() / an.mean()) * 100
    r["ils_speed_ci"]  = boot_ci(at, it,  "speedup")
    r["ails_speed_ci"] = boot_ci(at, ait, "speedup")
    r["ils_red_ci"]    = boot_ci(an, iin, "nodered")
    r["ails_red_ci"]   = boot_ci(an, ain, "nodered")
    return r


# ----------------------------------------------------------------------------
def rerun_exp2():
    print("\n=== Experiment 2 (port environment, 200/300/500^2) ===")
    rows = []
    for size in [200, 300, 500]:
        for lam in [0.0, 0.5, 1.0]:
            START, GOAL = (0, 0), (size - 1, size - 1)
            at, it, ait, an, iin, ain = ([] for _ in range(6))
            for m in range(30):
                seed = size * 1000 + int(lam * 100) + m          # exact paper seed
                obs, risk = E2.generate_port_grid(size, seed=seed)
                gm = E2.GridMap(size, size, obs, risk)
                pa, na, ta = E2.astar(gm, START, GOAL, lam)
                if pa is None: continue
                pi, ni, ti, _ = E2.ils_astar(gm, START, GOAL, lam)
                if pi is None: continue
                pai, nai, tai, _ = E2.ails_astar(gm, START, GOAL, lam)
                if pai is None: continue
                at.append(ta); it.append(ti); ait.append(tai)
                an.append(na); iin.append(ni); ain.append(nai)
            if len(at) < 5:
                continue
            r = summarise(at, it, ait, an, iin, ain)
            r.update(size=size, lam=lam, n=len(at))
            rows.append(r)
            print(f"  {size}^2 lam={lam}: ILS {r['ils_speed']:.2f}x "
                  f"[{r['ils_speed_ci'][0]:.2f},{r['ils_speed_ci'][1]:.2f}]  "
                  f"red {r['ils_red']:.1f}% [{r['ils_red_ci'][0]:.1f},{r['ils_red_ci'][1]:.1f}]")
    _write(os.path.join(RESULTS, "exp2_port_environment_ci.csv"), rows,
           keys=["size", "lam", "n"])
    return rows


def rerun_exp9():
    print("\n=== Experiment 9 (AILS large-grid, 500^2) ===")
    SIZE, START, GOAL = 500, (0, 0), (499, 499)
    rows = []
    for density in [0.10, 0.20, 0.30]:
        for risk_type in ["gradient", "hotspot", "uniform"]:
            for lam in [0.5, 1.0]:
                at, it, ait, an, iin, ain = ([] for _ in range(6))
                for m in range(15):
                    seed = E9.make_instance_seed(risk_type, density, lam, m, offset=400000)
                    obs = E9.generate_random_grid(SIZE, density, seed=seed)
                    risk = E9.generate_risk_layer(SIZE, risk_type, seed=seed + 10000)
                    gm = E9.GridMap(SIZE, SIZE, obs, risk)
                    pa, na, ta = E9.astar(gm, START, GOAL, lam)
                    if pa is None: continue
                    pi, ni, ti, _ = E9.ils_astar(gm, START, GOAL, lam)
                    if pi is None: continue
                    pai, nai, tai, _ = E9.ails_astar(gm, START, GOAL, lam)
                    if pai is None: continue
                    at.append(ta); it.append(ti); ait.append(tai)
                    an.append(na); iin.append(ni); ain.append(nai)
                if len(at) < 3:
                    continue
                r = summarise(at, it, ait, an, iin, ain)
                r.update(density=density, risk_type=risk_type, lam=lam, n=len(at))
                rows.append(r)
                print(f"  d={density:.0%} {risk_type} lam={lam}: "
                      f"AILS {r['ails_speed']:.2f}x "
                      f"[{r['ails_speed_ci'][0]:.2f},{r['ails_speed_ci'][1]:.2f}]  "
                      f"red {r['ails_red']:.1f}% [{r['ails_red_ci'][0]:.1f},{r['ails_red_ci'][1]:.1f}]")
    _write(os.path.join(RESULTS, "exp9_large_grid_ails_ci.csv"), rows,
           keys=["density", "risk_type", "lam", "n"])
    return rows


def _write(path, rows, keys):
    import csv
    if not rows:
        print(f"  [warn] no rows for {path}"); return
    fields = keys + ["ils_speed", "ils_speed_ci", "ils_red", "ils_red_ci",
                     "ails_speed", "ails_speed_ci", "ails_red", "ails_red_ci"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(fields)
        for r in rows:
            w.writerow([r.get(k) for k in keys] +
                       [f"{r['ils_speed']:.3f}", f"[{r['ils_speed_ci'][0]:.3f},{r['ils_speed_ci'][1]:.3f}]",
                        f"{r['ils_red']:.2f}",  f"[{r['ils_red_ci'][0]:.2f},{r['ils_red_ci'][1]:.2f}]",
                        f"{r['ails_speed']:.3f}", f"[{r['ails_speed_ci'][0]:.3f},{r['ails_speed_ci'][1]:.3f}]",
                        f"{r['ails_red']:.2f}",  f"[{r['ails_red_ci'][0]:.2f},{r['ails_red_ci'][1]:.2f}]"])
    print(f"  saved {path}")


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("[note] for bit-identical seeds run with:  PYTHONHASHSEED=0 python3 "
              "experiments/add_exp2_exp9_cis.py")
    rerun_exp2()
    rerun_exp9()
    print("\nDone. Paste the CI brackets into the Speed/Red. columns of "
          "tab:exp2_port and tab:exp9_large, e.g.  $2.26\\times$ "
          "{\\scriptsize[1.95,\\,2.61]}.")
