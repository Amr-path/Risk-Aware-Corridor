#!/usr/bin/env bash
# =============================================================================
# FULL reproducible re-run of EVERY experiment on the M1 Pro MacBook.
#
# WHY
#   The instance seed is now patched (deterministic make_instance_seed); we
#   re-run the entire suite on ONE machine (this M1 Pro) so every number in the
#   paper -- node counts AND wall-clock speedups -- is internally consistent and
#   reproducible from a single hardware target. After this completes, the paper
#   text is updated to state "MacBook Pro (M1 Pro), 16 GB" throughout.
#
# WHAT IT RUNS  (each writes CSV/TXT into results/ or experiments/results/)
#   build  C++ reference (arm64)                  -> cpp_reference/pathfind
#   core   exp 1,2,3,4,5                          -> results/exp{1..5}*.csv
#   supp   exp 1b,4b,5b,6,7,8,9 (+sanity)         -> results/exp{1b,4b,5b,6,7,8,9}*.csv
#   ext    exp 10b,10c,11                         -> results/exp10b,exp10c,exp11*.csv
#   bench  exp 10 (game/random MovingAI)          -> results/exp10_benchmark.csv
#   dao    full Dragon Age corpus (none+gradient) -> experiments/results/corpus_dao_*.csv
#   penang exp 12 (real port)                     -> results/exp12_penang_port.csv
#   auto   exp 13 autopilot waypoint sweep        -> experiments/results/exp13_autopilot.csv
#   rand   random-endpoint replication            -> experiments/results/random_endpoint.csv
#   sweep  param sweep + rho audit (--heavy)      -> experiments/results/param_sweep.csv, rho_audit.csv
#   parity Python<->C++ node-count parity         -> experiments/results/parity_report.txt
#   cppbn  compiled wall-clock (static + replan)  -> experiments/results/cpp_bench_{static,replan}.csv
#   scale  Exp-14 scaling 200..2000 (--heavy)     -> experiments/results/scaling_part{A,B}.csv  [LONGEST]
#
# RUNTIME: several hours total. The long poles are `sweep`, `scale`, and `dao`.
#   Run it overnight. It continues past any single step's failure and prints a
#   PASS/FAIL summary at the end, so nothing is lost if one step errors.
#
# USAGE (from the project root, in Terminal):
#   bash run_all_m1.sh
#   # or detached so you can close Terminal:
#   nohup bash run_all_m1.sh > run_all_m1.out 2>&1 &
#   tail -f run_all_m1.out
#
# Re-running is safe: each step overwrites its own outputs. A timestamped backup
# of the current results is taken first.
# =============================================================================
set -u
cd "$(dirname "$0")"                      # project root
export PYTHONHASHSEED=0                   # pin for the patched hash-affected exps

PY=python3
LOGDIR="run_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/SUMMARY.txt"
: > "$SUMMARY"

echo "============================================================"
echo " FULL M1 RE-RUN   start: $(date)"
echo " PYTHONHASHSEED=$PYTHONHASHSEED   logs: $LOGDIR/"
echo "============================================================"

# --- dependency check -------------------------------------------------------
$PY - <<'PYCHECK' || { echo "Install deps first:  $PY -m pip install --user numpy scipy"; exit 1; }
import numpy, scipy
print("numpy", numpy.__version__, "scipy", scipy.__version__)
PYCHECK

# --- one-time backup of current results ------------------------------------
ts=$(date +%Y%m%d_%H%M%S)
tar -czf "results_backup_${ts}.tar.gz" results experiments/results 2>/dev/null \
  && echo "[backup] results_backup_${ts}.tar.gz written"

# --- overall progress tracking ----------------------------------------------
SCRIPT_START=$SECONDS
STEP=0
TOTAL=16          # 1 build + 15 experiment steps

hms () { local s=$1; printf '%dh%02dm%02ds' $((s/3600)) $(((s%3600)/60)) $((s%60)); }

progress_bar () { # $1=done $2=total -- prints an overall step bar + elapsed time
  local done=$1 total=$2 width=28 i bar=""
  local filled=$(( done * width / total ))
  for ((i=0; i<filled; i++)); do bar+="#"; done
  for ((i=filled; i<width; i++)); do bar+="."; done
  printf '\n========[ %s ] %d%%  step %d/%d   elapsed %s ========\n' \
    "$bar" $(( done * 100 / total )) "$done" "$total" "$(hms $((SECONDS - SCRIPT_START)))"
}

# --- helper: run a step, stream its LIVE progress, time it, never abort ------
run_step () {
  local name="$1"; shift
  STEP=$((STEP + 1))
  local log="$LOGDIR/${name}.log"
  progress_bar "$STEP" "$TOTAL"
  echo ">>> [$STEP/$TOTAL] $name  start $(date '+%H:%M:%S')  ::  $*"
  local t0=$SECONDS
  # tee = you see each script's own percentage+ETA bar live AND it is logged
  "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]} dt=$((SECONDS - t0))
  if [ "$rc" -eq 0 ]; then
    printf 'PASS  %-10s  %s  %s\n' "$name" "$(hms $dt)" "$*" | tee -a "$SUMMARY"
  else
    printf 'FAIL(%d) %-10s %s  %s   [see %s]\n' "$rc" "$name" "$(hms $dt)" "$*" "$log" | tee -a "$SUMMARY"
  fi
}

# --- 0. build the C++ reference (non-fatal: a prebuilt arm64 binary exists) --
STEP=$((STEP + 1))
progress_bar "$STEP" "$TOTAL"
echo ">>> [$STEP/$TOTAL] build  C++ reference"
( cd cpp_reference && make clean && make ) 2>&1 | tee "$LOGDIR/build.log"
if [ "${PIPESTATUS[0]}" -eq 0 ]; then
  echo "PASS  build       C++ pathfind" | tee -a "$SUMMARY"
else
  echo "WARN  build       make failed (need Xcode CLT?); using existing cpp_reference/pathfind" | tee -a "$SUMMARY"
fi

# --- cheap/medium experiments first (fail fast) -----------------------------
run_step core   $PY experiments/run_all_experiments.py
run_step supp   $PY experiments/run_supplementary_experiments.py
run_step ext    $PY experiments/run_extended_benchmarks.py
run_step bench  $PY experiments/run_benchmark_experiment.py
run_step penang $PY experiments/run_experiment_12.py
run_step auto   $PY experiments/run_exp13_autopilot.py
run_step rand   $PY experiments/run_random_endpoint.py

# --- DAO corpus: uniform-cost (vs published optima) + gradient (vs A*) ------
run_step dao_none  $PY experiments/run_movingai_corpus.py --maps experiments/benchmark_maps/dao --per-map 100 --risk none
run_step dao_grad  $PY experiments/run_movingai_corpus.py --maps experiments/benchmark_maps/dao --per-map 50 --risk gradient --lam 1.0

# --- C++ parity + compiled wall-clock (needs the binary) --------------------
run_step parity     $PY experiments/verify_parity.py --n 1000 --size 256
run_step cppbn_stat $PY experiments/run_cpp_bench.py --sizes 200 500 1000 2000 --maps 10
run_step cppbn_repl $PY experiments/run_cpp_bench.py --sizes 200 500 1000 --maps 10 --replan

# --- long poles last --------------------------------------------------------
run_step sweep  $PY experiments/run_param_sweep.py --heavy
run_step scale  $PY experiments/run_scaling.py --heavy

# --- optional: regenerate the benchmark figure (best-effort; needs matplotlib) ---
run_step figure $PY experiments/run_figure_only.py

echo ""
echo "============================================================"
echo " FULL M1 RE-RUN   done: $(date)"
echo "------------------------------------------------------------"
cat "$SUMMARY"
echo "============================================================"
echo " Next: tell the assistant it's done; it will read the refreshed"
echo " CSVs and update every number + the M1 hardware note in main.tex."
