# Update notes (Experiment 12 added)

This commit adds the real-world port validation (Experiment 12) requested by
peer review v2 §5.1: a real-port topology rasterised from OpenStreetMap data,
to complement the procedurally generated grids in Experiments 1–11.

## New files

```
src/download_penang_port.py         # OSM downloader + rasteriser (Penang Port NBCT)
src/run_experiment_12.py            # Five-algorithm runner on the real port grid
results/exp12_penang_port.csv       # 30 random start-goal pairs at lambda in {0.0, 0.5, 1.0}
figures/penang_port.png             # Figure 4 in the manuscript
UPDATE_NOTES.md                     # this file
```

Updated files: `README.md`, `requirements.txt`, `.gitignore`.

## Headline result

| λ     | AILS speedup | RILS cost ratio | JPS (λ=0) |
|-------|--------------|------------------|-----------|
| 0.0   | 1.58×        | 1.003            | 0.55× speed, 96.7% node red, cost 1.000 |
| 0.5   | 2.90×        | 1.000            | n/a |
| 1.0   | 3.44×        | 1.000            | n/a |

The risk-weighted activation pattern established on synthetic grids in
Experiments 1, 10b, etc. transfers to the real port topology: at λ=0 the
corridor methods give little benefit, while at λ≥0.5 AILS recovers a clear
speedup advantage and RILS retains exact path optimality.

## Reproduction

```bash
pip install -r requirements.txt
cd src
python download_penang_port.py   # 1–3 min (Overpass API)
python run_experiment_12.py      # 5–15 min on a laptop
```

Bounding box: 5.3950°–5.4150°N, 100.3500°–100.3800°E (North Butterworth
Container Terminal area). Projection EPSG:3375 (RSO Borneo). The 500×500
grid is regenerated each time `download_penang_port.py` runs; results are
deterministic given a fixed random seed (`SEED = 20260518`).

Map data © OpenStreetMap contributors, licensed under the Open Database
License (ODbL).
