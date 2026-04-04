# Risk-Aware Corridor-Constrained Pathfinding for UAV Navigation

This repository contains the source code, experimental scripts, and results for the paper:

> **Risk-Aware Corridor-Constrained Pathfinding for UAV Navigation in Biosecurity-Sensitive Environments**
> Amr Elshahed, Majid Khan Bin Majahar Ali, Ahmad Sufril Azlan Mohamed, Farah Aini Binti Abdullah
> *Submitted to Expert Systems with Applications (Elsevier)*

## Overview

We investigate corridor-constrained pathfinding on risk-annotated grids, where traversal cost combines Euclidean distance with spatial risk: `cost = dist + lambda * risk(cell)`. Three corridor variants are evaluated:

- **ILS** (Incremental Line Search): fixed-width corridor along the Bresenham line
- **AILS** (Adaptive ILS): density-adaptive corridor using integral images
- **RILS** (Risk-Responsive ILS): risk-adaptive corridor width (new contribution)

Thirteen coordinated experiments (6,000+ planned paths) evaluate the approach across risk-annotated grids, procedurally generated port environments, Jump Point Search comparison, D\*Lite and Weighted A\* baselines, corridor width sensitivity analysis, Moving AI Lab benchmarks, maze stress tests, multi-heuristic comparison, and progressive obstacle discovery missions.

## Repository Structure

```
src/                          # Core algorithm implementations
  run_all_experiments.py      # Experiments 1-9 (core + supplementary)
  run_benchmark_experiment.py # Experiment 10 (Moving AI Lab benchmarks)
  run_extended_benchmarks.py  # Experiments 10b, 10c, 11 (risk benchmarks, mazes, heuristics)
  run_figure_only.py          # Generate Figure 2 (corridor comparison)
  run_maze_only.py            # Experiment 10c runner
  run_risk_bench_only.py      # Experiment 10b runner
  run_heuristic_only.py       # Experiment 11 runner
  run_supplementary_experiments.py  # Experiments 6-9

results/                      # Experimental results (CSV)
  exp1_risk_annotated.csv     # Exp 1: Risk-annotated grids (5,400 paths)
  exp1b_random_startgoal.csv  # Exp 1b: Random start-goal variant
  exp2_port_environment.csv   # Exp 2: Port environment validation
  exp3_jps_comparison.csv     # Exp 3: JPS comparison
  exp4_replanning.csv         # Exp 4: Dynamic re-planning
  exp4b_dstar_replanning.csv  # Exp 4b: D*Lite re-planning comparison
  exp5_progressive.csv        # Exp 5: Progressive obstacle discovery
  exp5b_dstar_progressive.csv # Exp 5b: D*Lite progressive comparison
  exp6_weighted_astar.csv     # Exp 6: Weighted A* comparison
  exp7_corridor_sensitivity.csv # Exp 7: Corridor width sensitivity
  exp8_rils_evaluation.csv    # Exp 8: RILS evaluation
  exp9_large_grid_ails.csv    # Exp 9: Large-grid AILS validation
  exp10_benchmark.csv         # Exp 10: Moving AI Lab benchmarks
  exp10b_risk_benchmarks.csv  # Exp 10b: Risk-annotated benchmarks
  exp10c_maze_benchmarks.csv  # Exp 10c: Maze benchmarks
  exp11_multi_heuristic.csv   # Exp 11: Multi-heuristic comparison

figures/                      # Generated figures
  corridor_comparison_real.pdf  # Figure 2: 4-panel corridor comparison
  corridor_comparison_real.png  # Figure 2 (PNG version)

paper/                        # LaTeX manuscript
  main.tex                    # Full manuscript source
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib (for figure generation)

```bash
pip install numpy matplotlib
```

## Running Experiments

### Full experiment suite (Experiments 1-9)
```bash
cd src
python run_all_experiments.py
```

### Moving AI Lab benchmarks (Experiment 10)
Download benchmark maps from [movingai.com/benchmarks](https://www.movingai.com/benchmarks/) and place them in `src/benchmark_maps/`, then:
```bash
cd src
python run_benchmark_experiment.py
```

### Extended experiments (Experiments 10b, 10c, 11)
```bash
cd src
python run_extended_benchmarks.py
# Or run individually:
python run_risk_bench_only.py   # Exp 10b
python run_maze_only.py         # Exp 10c
python run_heuristic_only.py    # Exp 11
python run_figure_only.py       # Figure 2
```

## Key Results

| Experiment | Key Finding |
|---|---|
| Exp 1 | ILS achieves up to 7.90x speedup on risk-annotated grids |
| Exp 3 | ILS surpasses JPS at obstacle densities >= 20% |
| Exp 6 | ILS maintains <0.3% suboptimality vs wA*'s 2-5% |
| Exp 8 | RILS achieves best path quality at cost of lower speedup |
| Exp 10b | Same maps: 0.99x at lambda=0, 2.5-3.2x at lambda=1.0 |
| Exp 10c | Mazes are worst case: 0.53x with 35-65% solve rates |
| Exp 11 | Corridor speedups robust across 4 heuristic functions |

## Algorithms

All algorithms are implemented in Python for fair comparison under identical language overhead. The core implementations include:

- **A\*** with octile distance heuristic (baseline)
- **ILS A\*** with configurable corridor width and incremental expansion
- **AILS A\*** with integral-image-based density-adaptive corridors
- **RILS A\*** with risk-integral-image-based risk-responsive corridors
- **Jump Point Search** (canonical implementation)
- **D\* Lite** (incremental re-planning baseline)
- **Weighted A\*** (bounded-suboptimal baseline)

## Risk Distributions

Three risk distributions are tested:
- **Gradient**: risk increases linearly from top-left to bottom-right
- **Hotspot**: Gaussian-centred high-risk zones
- **Uniform**: constant risk across all cells

## Citation

```bibtex
@article{elshahed2026risk,
  title={Risk-Aware Corridor-Constrained Pathfinding for {UAV} Navigation in Biosecurity-Sensitive Environments},
  author={Elshahed, Amr and Ali, Majid Khan Bin Majahar and Mohamed, Ahmad Sufril Azlan and Abdullah, Farah Aini Binti},
  journal={Expert Systems with Applications},
  year={2026},
  note={Under Review}
}
```

## License

This project is released for academic and research purposes. Please cite the paper if you use this code.

## Contact

Amr Elshahed - amr.elshahed@student.usm.my
School of Computer Sciences, Universiti Sains Malaysia
