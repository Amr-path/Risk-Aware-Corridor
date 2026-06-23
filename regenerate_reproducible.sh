#!/usr/bin/env bash
# =============================================================================
# Reproducible regeneration of the hash-seed-affected experiments.
#
# WHY THIS EXISTS
#   The previous instance seed used Python's hash() on the risk-type string,
#   which is randomized per process unless PYTHONHASHSEED is fixed. That made
#   Experiment 1 and supplementary Experiments 1b/6/7/8/9 non-reproducible.
#   The driver code is now patched (fixed RISK_TYPE_SEED map + structured,
#   collision-free make_instance_seed). This script re-runs exactly those
#   experiments under a pinned hash seed so the released CSVs, the paper's
#   headline numbers, AND the reported Wilcoxon/Holm statistics all come from
#   ONE reproducible run.
#
# IMPORTANT
#   * Speedup is a wall-clock ratio and is therefore HARDWARE-DEPENDENT. Run
#     this on the SAME reference machine used for the paper (the Intel Core i7,
#     per Section "Experimental Design") so the regenerated speedups are
#     comparable to the published ones. Node-count reductions are
#     hardware-independent and will reproduce anywhere.
#   * Expected runtime: roughly 30-60 min total (Exp 9 on 500x500 is the long
#     pole). Run it in a terminal that stays open, or under `nohup`/`tmux`.
#   * This OVERWRITES result CSVs in ../results and ./results. Back them up
#     first if you want to diff old vs new (recommended).
#
# USAGE
#   cd experiments
#   bash regenerate_reproducible.sh            # full reproducible run
#   bash regenerate_reproducible.sh --backup   # tar the current CSVs first
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# Pin the hash seed so any residual hash()-dependent ordering is deterministic.
export PYTHONHASHSEED=0

if [[ "${1:-}" == "--backup" ]]; then
  ts=$(date +%Y%m%d_%H%M%S)
  echo "[backup] saving current CSVs to results_backup_${ts}.tar.gz"
  tar -czf "results_backup_${ts}.tar.gz" ../results ./results 2>/dev/null || true
fi

echo "============================================================"
echo " Reproducible regeneration  (PYTHONHASHSEED=$PYTHONHASHSEED)"
echo " Start: $(date)"
echo "============================================================"

# Experiment 1 (main driver) -- the headline risk-annotated grid sweep.
echo ">>> [1/6] Experiment 1  (run_all_experiments.py 1)"
python3 -u run_all_experiments.py 1

# Supplementary experiments that used the hash() seed.
for exp in 1b 6 7 8 9; do
  echo ">>> Experiment ${exp}  (run_supplementary_experiments.py ${exp})"
  python3 -u run_supplementary_experiments.py "${exp}"
done

echo "============================================================"
echo " Regeneration complete: $(date)"
echo " Overwritten CSVs:"
echo "   results/exp1_risk_annotated.csv"
echo "   results/exp1b_random_startgoal.csv"
echo "   results/exp6_weighted_astar.csv"
echo "   results/exp7_corridor_sensitivity.csv"
echo "   results/exp8_rils_evaluation.csv"
echo "   results/exp9_large_grid_ails.csv"
echo
echo " NEXT: update the paper's numbers from these CSVs, then recompile."
echo "   python3 check_latex.py        # cross-checks several tables vs CSV"
echo "============================================================"
