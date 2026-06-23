#!/usr/bin/env python3
"""Regenerate figures/pareto_frontier.pdf from results/exp7_corridor_sensitivity.csv.
Each point is the grand mean over all density-risk-lambda configurations at one
corridor fraction alpha; x = mean suboptimality (%), y = mean speedup; bars are
95% CIs. Inset zooms the low-suboptimality cluster (alpha >= 0.07); the default
alpha = 0.05 is circled at the knee.
"""
import csv, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "results", "exp7_corridor_sensitivity.csv")
OUT = os.path.join(HERE, "..", "figures", "pareto_frontier.pdf")

rows = list(csv.DictReader(open(CSV)))
alphas = sorted({float(r["alpha"]) for r in rows})

A, S, Scilo, Scihi, Q, Qcilo, Qcihi = [], [], [], [], [], [], []
for a in alphas:
    rr = [r for r in rows if float(r["alpha"]) == a]
    sp = np.array([float(r["speedup_mean"]) for r in rr])
    opt = np.array([(float(r["opt_ratio_mean"]) - 1.0) * 100.0 for r in rr])
    n = len(sp)
    sp_m, sp_ci = sp.mean(), 1.96 * sp.std(ddof=1) / math.sqrt(n)
    q_m, q_ci = opt.mean(), 1.96 * opt.std(ddof=1) / math.sqrt(n)
    A.append(a); S.append(sp_m); Scilo.append(sp_ci); Scihi.append(sp_ci)
    Q.append(q_m); Qcilo.append(q_ci); Qcihi.append(q_ci)

A = np.array(A); S = np.array(S); Q = np.array(Q)
Sci = np.array(Scilo); Qci = np.array(Qcilo)

# report knee
i05 = list(A).index(0.05)
print(f"alpha=0.05 grand mean: speedup={S[i05]:.2f}x  suboptimality={Q[i05]:.2f}%")
for a, s, q in zip(A, S, Q):
    print(f"  alpha={a:.2f}  speedup={s:.2f}x  subopt={q:.3f}%")

plt.rcParams.update({"font.size": 12, "font.family": "serif", "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(7.2, 4.6))

ax.errorbar(Q, S, xerr=Qci, yerr=Sci, fmt="-o", color="#1f4e79", ecolor="#9bb8d3",
            elinewidth=1, capsize=3, markersize=6, lw=1.6, zorder=3, label="Pareto frontier")
for a, q, s in zip(A, Q, S):
    ax.annotate(rf"$\alpha={a:g}$", (q, s), textcoords="offset points",
                xytext=(8, 5), fontsize=9, color="#333")
# circle the default alpha=0.05
ax.scatter([Q[i05]], [S[i05]], s=320, facecolors="none", edgecolors="#c00000",
           linewidths=2.0, zorder=4)
ax.annotate("default", (Q[i05], S[i05]), textcoords="offset points",
            xytext=(10, -16), fontsize=10, color="#c00000", weight="bold")

ax.set_xlabel("Mean path suboptimality (\\%)")
ax.set_ylabel(r"Mean speedup ($\times$)")
ax.grid(True, ls=":", alpha=0.5)
ax.set_title("Speedup--quality Pareto frontier (Experiment 7)")

# inset: zoom alpha >= 0.07 (low suboptimality cluster)
mask = A >= 0.07
axin = inset_axes(ax, width="46%", height="46%", loc="upper right", borderpad=1.4)
axin.errorbar(Q[mask], S[mask], xerr=Qci[mask], yerr=Sci[mask], fmt="-o",
              color="#1f4e79", ecolor="#9bb8d3", elinewidth=1, capsize=2.5,
              markersize=5, lw=1.4)
for a, q, s in zip(A[mask], Q[mask], S[mask]):
    axin.annotate(rf"$\alpha={a:g}$", (q, s), textcoords="offset points",
                  xytext=(5, 4), fontsize=8, color="#333")
axin.grid(True, ls=":", alpha=0.5)
axin.set_title("zoom: $\\alpha \\geq 0.07$", fontsize=9)
axin.tick_params(labelsize=8)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
