#!/usr/bin/env python3
"""
stats_v2.py  --  Phase 3 (E9): corrected statistical protocol.

Fixes the three defects of the original stats_recompute.py for a Q1 venue:
  1. reports matched-pairs RANK-BISERIAL correlation (the paper's stated
     effect size) in addition to Cohen's d_z;
  2. uses a VALID pooled percentile bootstrap CI on the per-instance relative
     improvement (the old code averaged per-cell CI endpoints, which is not a CI);
  3. applies Holm correction across a PRE-REGISTERED family of comparisons,
     and uses Wilcoxon signed-rank with zero_method='pratt' (keeps tied pairs,
     common with integer node counts).

Library use:
    from stats_v2 import paired_report
    rep = paired_report(baseline, contender, lower_is_better=True)

CLI use (point at any per-instance CSV with a baseline and a contender column):
    python3 experiments/stats_v2.py --csv results/exp1_risk_annotated.csv \\
        --baseline nodes_astar --contender nodes_ils --group density risk_type

Runs a self-test on synthetic data when called with --selftest.
"""
import argparse, sys
import numpy as np

try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _ranks(a):
    """Average ranks of 1..n for array a (1-based, ties averaged)."""
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(1, len(a) + 1)
    # average ties
    a_sorted = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
        i = j + 1
    return r


def rank_biserial(baseline, contender):
    """Matched-pairs rank-biserial r = (W+ - W-)/(W+ + W-) on nonzero diffs.
    Positive r => contender < baseline more often (improvement if lower-is-better)."""
    d = np.asarray(baseline, float) - np.asarray(contender, float)
    nz = d[d != 0]
    if len(nz) == 0:
        return 0.0
    ar = _ranks(np.abs(nz))
    w_pos = ar[nz > 0].sum()   # baseline > contender (contender better, lower)
    w_neg = ar[nz < 0].sum()
    tot = w_pos + w_neg
    return float((w_pos - w_neg) / tot) if tot > 0 else 0.0


def cohens_dz(baseline, contender):
    d = np.asarray(baseline, float) - np.asarray(contender, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def bootstrap_ci(rel, B=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the MEAN of per-instance values `rel`,
    resampling instances with replacement (a valid CI, unlike averaging endpoints)."""
    rng = np.random.default_rng(seed)
    rel = np.asarray(rel, float)
    n = len(rel)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(B, n))
    means = rel[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def paired_report(baseline, contender, lower_is_better=True, label="", B=10000):
    """Full paired report for one comparison cell."""
    b = np.asarray(baseline, float)
    c = np.asarray(contender, float)
    assert len(b) == len(c) and len(b) > 0
    # relative improvement: positive = contender better
    denom = np.where(b == 0, np.nan, b)
    rel = (b - c) / denom if lower_is_better else (c - b) / np.where(c == 0, np.nan, c)
    rel = rel[~np.isnan(rel)]
    if _HAVE_SCIPY and np.any(b != c):
        try:
            stat, p = wilcoxon(b, c, zero_method="pratt", alternative="two-sided")
        except ValueError:
            stat, p = np.nan, 1.0
    else:
        stat, p = np.nan, 1.0
    lo, hi = bootstrap_ci(rel, B=B)
    return dict(label=label, n=len(b),
                mean_rel_impr=float(np.mean(rel)),
                ci_lo=lo, ci_hi=hi,
                rank_biserial=rank_biserial(b, c),
                cohens_dz=cohens_dz(b, c),
                wilcoxon_p=float(p))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, preserving order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def family_report(cells, lower_is_better=True):
    """cells: list of (label, baseline_array, contender_array). Holm across the family."""
    reps = [paired_report(b, c, lower_is_better, label=lab) for lab, b, c in cells]
    adj = holm([r["wilcoxon_p"] for r in reps])
    for r, a in zip(reps, adj):
        r["holm_p"] = float(a)
        r["significant"] = bool(a < 0.05)
    return reps


def _print_table(reps):
    print(f"{'cell':28s} {'n':>5s} {'impr%':>8s} {'95% CI':>18s} "
          f"{'r_rb':>6s} {'d_z':>7s} {'holm_p':>9s} sig")
    for r in reps:
        ci = f"[{100*r['ci_lo']:+.1f},{100*r['ci_hi']:+.1f}]"
        print(f"{r['label'][:28]:28s} {r['n']:5d} {100*r['mean_rel_impr']:7.2f} "
              f"{ci:>18s} {r['rank_biserial']:6.3f} {r['cohens_dz']:7.2f} "
              f"{r['holm_p']:9.2e} {'*' if r['significant'] else ''}")


def selftest():
    rng = np.random.default_rng(1)
    cells = []
    for k, shift in enumerate([0.0, 0.2, 0.5]):
        base = rng.gamma(3, 200, size=120)
        cont = base * (1 - shift) + rng.normal(0, 5, size=120)
        cells.append((f"cell{k}_shift{shift}", base, cont))
    reps = family_report(cells, lower_is_better=True)
    _print_table(reps)
    assert reps[0]["holm_p"] >= 0  # sanity
    assert reps[2]["significant"]   # the 50% improvement must be significant
    print("\nselftest OK (rank-biserial, valid bootstrap CI, Holm family).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--baseline")
    ap.add_argument("--contender")
    ap.add_argument("--group", nargs="*", default=[])
    ap.add_argument("--higher-is-better", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest or not args.csv:
        selftest(); return
    import pandas as pd
    df = pd.read_csv(args.csv)
    lib = not args.higher_is_better
    if args.group:
        cells = []
        for key, sub in df.groupby(args.group):
            lab = "|".join(map(str, key if isinstance(key, tuple) else (key,)))
            cells.append((lab, sub[args.baseline].values, sub[args.contender].values))
    else:
        cells = [("all", df[args.baseline].values, df[args.contender].values)]
    reps = family_report(cells, lower_is_better=lib)
    _print_table(reps)


if __name__ == "__main__":
    main()
