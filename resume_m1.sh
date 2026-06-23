#!/usr/bin/env bash
# =============================================================================
# RESUME the full M1 re-run, skipping every step that already finished.
#
# Use this ONLY if the original `run_all_m1.sh` process was stopped (full
# shutdown / restart / battery death). If you merely closed the lid (sleep),
# you do NOT need this -- the original run resumes on its own when you reopen.
#
# How it knows what is done: it reads the PASS records that run_all_m1.sh (and
# any earlier resume) already wrote into run_logs_*/SUMMARY.txt and
# resume_logs_*/SUMMARY.txt, and runs only the steps that have no PASS yet.
# Safe to run repeatedly.
#
# USAGE (from project root):
#   nohup bash resume_m1.sh > resume_m1.out 2>&1 &
#   tail -f resume_m1.out
# =============================================================================
set -u
cd "$(dirname "$0")"
export PYTHONHASHSEED=0
PY=python3

# steps already completed in any prior run/resume (field 2 of each PASS line)
DONE_STEPS="$(cat run_logs_*/SUMMARY.txt resume_logs_*/SUMMARY.txt 2>/dev/null \
              | awk '/^PASS/{print $2}' | sort -u | tr '\n' ' ')"

LOGDIR="resume_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/SUMMARY.txt"; : > "$SUMMARY"
SCRIPT_START=$SECONDS
STEP=0
TOTAL=16

echo "============================================================"
echo " RESUME M1 RE-RUN   start: $(date)"
echo " already completed: ${DONE_STEPS:-<none found>}"
echo " logs: $LOGDIR/"
echo "============================================================"

$PY - <<'PYCHECK' || { echo "Install deps first:  $PY -m pip install --user numpy scipy"; exit 1; }
import numpy, scipy
print("numpy", numpy.__version__, "scipy", scipy.__version__)
PYCHECK

hms () { local s=$1; printf '%dh%02dm%02ds' $((s/3600)) $(((s%3600)/60)) $((s%60)); }
progress_bar () {
  local done=$1 total=$2 width=28 i bar=""
  local filled=$(( done * width / total ))
  for ((i=0; i<filled; i++)); do bar+="#"; done
  for ((i=filled; i<width; i++)); do bar+="."; done
  printf '\n========[ %s ] %d%%  step %d/%d   elapsed %s ========\n' \
    "$bar" $(( done * 100 / total )) "$done" "$total" "$(hms $((SECONDS - SCRIPT_START)))"
}
should_run () { case " $DONE_STEPS " in *" $1 "*) return 1;; *) return 0;; esac; }

do_run () {
  local name="$1"; shift
  local log="$LOGDIR/${name}.log" t0=$SECONDS
  echo ">>> [$STEP/$TOTAL] $name  start $(date '+%H:%M:%S')  ::  $*"
  "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]} dt=$((SECONDS - t0))
  if [ "$rc" -eq 0 ]; then
    printf 'PASS  %-10s  %s  %s\n' "$name" "$(hms $dt)" "$*" | tee -a "$SUMMARY"
  else
    printf 'FAIL(%d) %-10s %s  %s   [see %s]\n' "$rc" "$name" "$(hms $dt)" "$*" "$log" | tee -a "$SUMMARY"
  fi
}

# run a step only if it has not already PASSed
maybe () {
  local name="$1"; shift
  STEP=$((STEP + 1))
  progress_bar "$STEP" "$TOTAL"
  if should_run "$name"; then
    do_run "$name" "$@"
  else
    echo ">>> [$STEP/$TOTAL] $name  SKIP (completed in a previous run)"
    printf 'SKIP  %-10s  (prior run)\n' "$name" | tee -a "$SUMMARY"
  fi
}

# ---- same 16 steps, same order as run_all_m1.sh ----------------------------
maybe build      bash -c 'cd cpp_reference && make clean && make'
maybe core       $PY experiments/run_all_experiments.py
maybe supp       $PY experiments/run_supplementary_experiments.py
maybe ext        $PY experiments/run_extended_benchmarks.py
maybe bench      $PY experiments/run_benchmark_experiment.py
maybe penang     $PY experiments/run_experiment_12.py
maybe auto       $PY experiments/run_exp13_autopilot.py
maybe rand       $PY experiments/run_random_endpoint.py
maybe dao_none   $PY experiments/run_movingai_corpus.py --maps experiments/benchmark_maps/dao --per-map 100 --risk none
maybe dao_grad   $PY experiments/run_movingai_corpus.py --maps experiments/benchmark_maps/dao --per-map 50 --risk gradient --lam 1.0
maybe parity     $PY experiments/verify_parity.py --n 1000 --size 256
maybe cppbn_stat $PY experiments/run_cpp_bench.py --sizes 200 500 1000 2000 --maps 10
maybe cppbn_repl $PY experiments/run_cpp_bench.py --sizes 200 500 1000 --maps 10 --replan
maybe sweep      $PY experiments/run_param_sweep.py --heavy
maybe scale      $PY experiments/run_scaling.py --heavy
maybe figure     $PY experiments/run_figure_only.py

echo ""
echo "============================================================"
echo " RESUME done: $(date)"
cat "$SUMMARY"
echo "============================================================"
echo " Tell the assistant 'done' to update the paper from the new CSVs."
