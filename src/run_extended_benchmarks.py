#!/usr/bin/env python3
"""
Extended Benchmark Experiments:
  - Exp 10b: Risk-annotated benchmarks (λ > 0) on Moving AI Lab maps
  - Exp 10c: Maze benchmark evaluation
  - Exp 11:  Multi-heuristic comparison (octile, Euclidean, Manhattan)
  - Figure generation: Real data corridor visualization

Requirements:
  pip install numpy matplotlib requests

Usage:
  python3 run_extended_benchmarks.py

Author: Amr Elshahed
"""

import numpy as np
import heapq
import time
import csv
import os
import sys
import math
from typing import Optional, Tuple, List

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available. Figure generation will be skipped.")

SEED = 42
np.random.seed(SEED)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
FIGURES_DIR = os.path.join(BASE_DIR, '..', 'figures')
MAPS_DIR = os.path.join(BASE_DIR, 'benchmark_maps')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MAPS_DIR, exist_ok=True)

# ==============================================================
# GRID / PATHFINDING (shared with run_benchmark_experiment.py)
# ==============================================================
class GridMap:
    def __init__(self, width, height, obstacles, risk=None):
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.risk = risk

    def is_free(self, r, c):
        return 0 <= r < self.height and 0 <= c < self.width and not self.obstacles[r, c]

    def neighbors_8(self, r, c):
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]
        for (dr, dc), base_cost in zip(dirs, costs):
            nr, nc = r+dr, c+dc
            if self.is_free(nr, nc):
                yield nr, nc, base_cost


def octile_h(r1, c1, r2, c2):
    dr = abs(r1 - r2); dc = abs(c1 - c2)
    return max(dr, dc) + (1.414 - 1.0) * min(dr, dc)

def euclidean_h(r1, c1, r2, c2):
    return math.sqrt((r1-r2)**2 + (c1-c2)**2)

def manhattan_h(r1, c1, r2, c2):
    return abs(r1 - r2) + abs(c1 - c2)

def risk_adjusted_h(r1, c1, r2, c2, lam, rho_bar):
    """h_lambda = h_octile * (1 + lambda * rho_bar)"""
    h = octile_h(r1, c1, r2, c2)
    return h * (1 + lam * rho_bar)


def parse_map_file(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    height = width = 0; map_start = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('height'): height = int(line.split()[1])
        elif line.startswith('width'): width = int(line.split()[1])
        elif line == 'map': map_start = i + 1; break
    obstacles = np.zeros((height, width), dtype=bool)
    for r in range(height):
        row_str = lines[map_start + r].strip()
        for c in range(min(width, len(row_str))):
            if row_str[c] in ('@', 'O', 'T', 'W'):
                obstacles[r, c] = True
    return obstacles, height, width


def parse_scen_file(filepath):
    scenarios = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('version') or not line: continue
            parts = line.split('\t')
            if len(parts) < 9: parts = line.split()
            if len(parts) < 9: continue
            try:
                sx, sy = int(parts[4]), int(parts[5])
                gx, gy = int(parts[6]), int(parts[7])
                opt_len = float(parts[8])
                scenarios.append({'start': (sy, sx), 'goal': (gy, gx),
                                  'optimal_length': opt_len, 'bucket': int(parts[0])})
            except (ValueError, IndexError):
                continue
    return scenarios


def bresenham_line(r0, c0, r1, c1):
    points = []
    dr = abs(r1 - r0); dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1; sc = 1 if c0 < c1 else -1
    err = dr - dc; r, c = r0, c0
    while True:
        points.append((r, c))
        if r == r1 and c == c1: break
        e2 = 2 * err
        if e2 > -dc: err -= dc; r += sr
        if e2 < dr: err += dr; c += sc
    return points


def build_corridor_mask(H, W, line_points, radius):
    mask = np.zeros((H, W), dtype=bool)
    r_int = int(np.ceil(radius))
    for (lr, lc) in line_points:
        mask[max(0,lr-r_int):min(H,lr+r_int+1), max(0,lc-r_int):min(W,lc+r_int+1)] = True
    return mask


# Generic A* with selectable heuristic
def astar_generic(grid, start, goal, lam=0.0, heuristic_fn=None, rho_bar=0.0, max_nodes=200000):
    """A* with selectable heuristic function."""
    if heuristic_fn is None:
        heuristic_fn = octile_h
    sr, sc = start; gr, gc = goal
    if not grid.is_free(sr, sc) or not grid.is_free(gr, gc):
        return None, 0, float('inf')

    open_list = [(heuristic_fn(sr, sc, gr, gc), 0, sr, sc)]
    g_cost = {(sr, sc): 0.0}; closed = set(); parent = {}; expanded = 0
    while open_list and expanded < max_nodes:
        f, _, r, c = heapq.heappop(open_list)
        if (r, c) in closed: continue
        closed.add((r, c)); expanded += 1
        if (r, c) == (gr, gc):
            path = [(r, c)]
            while (r, c) != (sr, sc): r, c = parent[(r, c)]; path.append((r, c))
            path.reverse()
            return path, expanded, g_cost[(gr, gc)]
        for nr, nc, base_cost in grid.neighbors_8(r, c):
            if (nr, nc) in closed: continue
            risk_cost = lam * grid.risk[nr, nc] if grid.risk is not None and lam > 0 else 0
            new_g = g_cost[(r, c)] + base_cost + risk_cost
            if new_g < g_cost.get((nr, nc), float('inf')):
                g_cost[(nr, nc)] = new_g; parent[(nr, nc)] = (r, c)
                h = heuristic_fn(nr, nc, gr, gc)
                heapq.heappush(open_list, (new_g + h, expanded, nr, nc))
    return None, expanded, float('inf')


def astar_corridor_generic(grid, start, goal, corridor_mask, lam=0.0, heuristic_fn=None, max_nodes=200000):
    if heuristic_fn is None: heuristic_fn = octile_h
    sr, sc = start; gr, gc = goal
    if not grid.is_free(sr, sc) or not grid.is_free(gr, gc):
        return None, 0, float('inf')
    open_list = [(heuristic_fn(sr, sc, gr, gc), 0, sr, sc)]
    g_cost = {(sr, sc): 0.0}; closed = set(); parent = {}; expanded = 0
    while open_list and expanded < max_nodes:
        f, _, r, c = heapq.heappop(open_list)
        if (r, c) in closed: continue
        closed.add((r, c)); expanded += 1
        if (r, c) == (gr, gc):
            path = [(r, c)]
            while (r, c) != (sr, sc): r, c = parent[(r, c)]; path.append((r, c))
            path.reverse()
            return path, expanded, g_cost[(gr, gc)]
        for nr, nc, base_cost in grid.neighbors_8(r, c):
            if (nr, nc) in closed or not corridor_mask[nr, nc]: continue
            risk_cost = lam * grid.risk[nr, nc] if grid.risk is not None and lam > 0 else 0
            new_g = g_cost[(r, c)] + base_cost + risk_cost
            if new_g < g_cost.get((nr, nc), float('inf')):
                g_cost[(nr, nc)] = new_g; parent[(nr, nc)] = (r, c)
                h = heuristic_fn(nr, nc, gr, gc)
                heapq.heappush(open_list, (new_g + h, expanded, nr, nc))
    return None, expanded, float('inf')


def ils_search(grid, start, goal, alpha=0.05, max_attempts=10, lam=0.0, heuristic_fn=None):
    if heuristic_fn is None: heuristic_fn = octile_h
    sr, sc = start; gr, gc = goal
    H, W = grid.height, grid.width
    diag = np.sqrt(H**2 + W**2)
    radius = alpha * diag; delta_r = 0.02 * diag
    line_points = bresenham_line(sr, sc, gr, gc)
    total_expanded = 0
    for attempt in range(max_attempts):
        mask = build_corridor_mask(H, W, line_points, radius)
        mask[sr, sc] = True; mask[gr, gc] = True
        path, expanded, cost = astar_corridor_generic(grid, start, goal, mask, lam, heuristic_fn)
        total_expanded += expanded
        if path is not None:
            return path, total_expanded, cost, attempt + 1
        radius += delta_r
    return None, total_expanded, float('inf'), max_attempts


def build_integral_image(grid_2d):
    return grid_2d.astype(float).cumsum(axis=0).cumsum(axis=1)

def query_integral(integral, r, c, w, H, W):
    r0 = max(0, r - w); c0 = max(0, c - w)
    r1 = min(H - 1, r + w); c1 = min(W - 1, c + w)
    total = integral[r1, c1]
    if r0 > 0: total -= integral[r0-1, c1]
    if c0 > 0: total -= integral[r1, c0-1]
    if r0 > 0 and c0 > 0: total += integral[r0-1, c0-1]
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    return total / area if area > 0 else 0

def ails_search(grid, start, goal, r_min_frac=0.02, r_max_frac=0.10,
                alpha_exp=1.0, window=3, max_attempts=10, lam=0.0, heuristic_fn=None):
    if heuristic_fn is None: heuristic_fn = octile_h
    sr, sc = start; gr, gc = goal
    H, W = grid.height, grid.width
    diag = np.sqrt(H**2 + W**2)
    r_min = r_min_frac * diag; r_max = r_max_frac * diag; delta_r = 0.02 * diag
    line_points = bresenham_line(sr, sc, gr, gc)
    integral = build_integral_image(grid.obstacles)
    total_expanded = 0
    for attempt in range(max_attempts):
        mask = np.zeros((H, W), dtype=bool)
        for (lr, lc) in line_points:
            density = query_integral(integral, lr, lc, window, H, W)
            radius = r_min + (r_max - r_min) * (density ** alpha_exp)
            r_int = int(np.ceil(radius))
            mask[max(0,lr-r_int):min(H,lr+r_int+1), max(0,lc-r_int):min(W,lc+r_int+1)] = True
        mask[sr, sc] = True; mask[gr, gc] = True
        path, expanded, cost = astar_corridor_generic(grid, start, goal, mask, lam, heuristic_fn)
        total_expanded += expanded
        if path is not None:
            return path, total_expanded, cost, attempt + 1
        r_min += delta_r; r_max += delta_r
    return None, total_expanded, float('inf'), max_attempts


def rils_search(grid, start, goal, r_base_frac=0.05, r_max_frac=0.15,
                beta=1.0, window=5, max_attempts=10, lam=0.0):
    sr, sc = start; gr, gc = goal
    H, W = grid.height, grid.width
    diag = np.sqrt(H**2 + W**2)
    r_base = r_base_frac * diag; r_max = r_max_frac * diag; delta_r = 0.02 * diag
    line_points = bresenham_line(sr, sc, gr, gc)
    if grid.risk is not None:
        integral_risk = build_integral_image(grid.risk)
    else:
        return ils_search(grid, start, goal, alpha=r_base_frac, max_attempts=max_attempts, lam=lam)
    total_expanded = 0
    for attempt in range(max_attempts):
        mask = np.zeros((H, W), dtype=bool)
        for (lr, lc) in line_points:
            mean_risk = query_integral(integral_risk, lr, lc, window, H, W)
            radius = r_base + (r_max - r_base) * (mean_risk ** beta)
            r_int = int(np.ceil(radius))
            mask[max(0,lr-r_int):min(H,lr+r_int+1), max(0,lc-r_int):min(W,lc+r_int+1)] = True
        mask[sr, sc] = True; mask[gr, gc] = True
        path, expanded, cost = astar_corridor_generic(grid, start, goal, mask, lam)
        total_expanded += expanded
        if path is not None:
            return path, total_expanded, cost, attempt + 1
        r_base += delta_r; r_max += delta_r
    return None, total_expanded, float('inf'), max_attempts


# ==============================================================
# RISK LAYER GENERATORS
# ==============================================================
def generate_gradient_risk(H, W):
    """Gradient risk: linearly increasing from top-left to bottom-right."""
    r = np.arange(H).reshape(-1, 1); c = np.arange(W).reshape(1, -1)
    risk = (r / H + c / W) / 2.0
    return risk.astype(np.float32)

def generate_hotspot_risk(H, W, n_hotspots=5, seed=42):
    """Hotspot risk: Gaussian peaks at random locations."""
    rng = np.random.RandomState(seed)
    risk = np.zeros((H, W), dtype=np.float32)
    for _ in range(n_hotspots):
        cr, cc = rng.randint(0, H), rng.randint(0, W)
        sigma = rng.uniform(H * 0.05, H * 0.15)
        r, c = np.ogrid[:H, :W]
        risk += np.exp(-((r - cr)**2 + (c - cc)**2) / (2 * sigma**2))
    risk = np.clip(risk / risk.max(), 0, 1) if risk.max() > 0 else risk
    return risk

def generate_uniform_risk(H, W, value=0.3):
    return np.full((H, W), value, dtype=np.float32)


# ==============================================================
# HELPER: find benchmark files
# ==============================================================
def find_file_in_dir(directory, filename):
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


# ==============================================================
# EXPERIMENT 10b: RISK-ANNOTATED BENCHMARKS
# ==============================================================
def run_risk_annotated_benchmarks():
    """
    Run ILS, AILS, and RILS on Moving AI Lab maps with synthetic risk layers at λ > 0.
    Tests on random maps (which are available) with gradient and hotspot risk overlays.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 10b: Risk-Annotated Benchmark Evaluation")
    print("=" * 60)

    # Use random maps only (game maps too slow in Python on 512x512)
    entries = []
    for cat_site, zip_prefix, map_file, scen_file, name, category in [
        ('random', 'random', 'random512-20-0.map', 'random512-20-0.map.scen', 'random512-20-0', 'random'),
        ('random', 'random', 'random512-30-0.map', 'random512-30-0.map.scen', 'random512-30-0', 'random'),
    ]:
        cat_dir = os.path.join(MAPS_DIR, cat_site)
        mp = find_file_in_dir(cat_dir, map_file)
        sp = find_file_in_dir(cat_dir, scen_file)
        if mp and sp:
            entries.append((mp, sp, name, category))

    if not entries:
        print("  No benchmark files found. Run run_benchmark_experiment.py first.")
        return

    lambdas = [0.5, 1.0]
    risk_types = ['gradient', 'hotspot']
    scenarios_per_map = 10
    results = []

    for map_path, scen_path, name, category in entries:
        obstacles, H, W = parse_map_file(map_path)
        scenarios = parse_scen_file(scen_path)
        if not scenarios: continue

        # Select longer-path scenarios
        scenarios.sort(key=lambda s: s['optimal_length'], reverse=True)
        selected = scenarios[:scenarios_per_map]

        for risk_type in risk_types:
            if risk_type == 'gradient':
                risk_layer = generate_gradient_risk(H, W)
            else:
                risk_layer = generate_hotspot_risk(H, W)

            for lam in lambdas:
                grid = GridMap(W, H, obstacles, risk=risk_layer)
                print(f"\n--- {name} | {risk_type} | λ={lam} ---")

                a_times = []; a_nodes = []; a_costs = []
                i_times = []; i_nodes = []; i_costs = []
                ai_times = []; ai_nodes = []; ai_costs = []
                ri_times = []; ri_nodes = []; ri_costs = []
                n_solved = {'astar': 0, 'ils': 0, 'ails': 0, 'rils': 0}

                for scen in selected:
                    s, g = scen['start'], scen['goal']
                    if not grid.is_free(s[0], s[1]) or not grid.is_free(g[0], g[1]):
                        continue

                    # A*
                    t0 = time.perf_counter()
                    path_a, nodes_a, cost_a = astar_generic(grid, s, g, lam)
                    t_a = (time.perf_counter() - t0) * 1000
                    if path_a is None: continue
                    n_solved['astar'] += 1
                    a_times.append(t_a); a_nodes.append(nodes_a); a_costs.append(cost_a)

                    # ILS
                    t0 = time.perf_counter()
                    path_i, nodes_i, cost_i, _ = ils_search(grid, s, g, alpha=0.05, lam=lam)
                    t_i = (time.perf_counter() - t0) * 1000
                    i_times.append(t_i); i_nodes.append(nodes_i)
                    i_costs.append(cost_i if path_i else float('inf'))
                    if path_i: n_solved['ils'] += 1

                    # AILS
                    t0 = time.perf_counter()
                    path_ai, nodes_ai, cost_ai, _ = ails_search(grid, s, g, lam=lam)
                    t_ai = (time.perf_counter() - t0) * 1000
                    ai_times.append(t_ai); ai_nodes.append(nodes_ai)
                    ai_costs.append(cost_ai if path_ai else float('inf'))
                    if path_ai: n_solved['ails'] += 1

                    # RILS
                    t0 = time.perf_counter()
                    path_ri, nodes_ri, cost_ri, _ = rils_search(grid, s, g, lam=lam)
                    t_ri = (time.perf_counter() - t0) * 1000
                    ri_times.append(t_ri); ri_nodes.append(nodes_ri)
                    ri_costs.append(cost_ri if path_ri else float('inf'))
                    if path_ri: n_solved['rils'] += 1

                n_valid = len(a_times)
                if n_valid == 0: continue

                # Only count commonly solved
                valid = [i_costs[j] < float('inf') and ai_costs[j] < float('inf')
                         and ri_costs[j] < float('inf') for j in range(n_valid)]
                n_common = sum(valid)
                if n_common == 0: continue

                # Filter
                at = [a_times[j] for j in range(n_valid) if valid[j]]
                ac = [a_costs[j] for j in range(n_valid) if valid[j]]
                an = [a_nodes[j] for j in range(n_valid) if valid[j]]
                it = [i_times[j] for j in range(n_valid) if valid[j]]
                ic = [i_costs[j] for j in range(n_valid) if valid[j]]
                inn = [i_nodes[j] for j in range(n_valid) if valid[j]]
                ait_ = [ai_times[j] for j in range(n_valid) if valid[j]]
                aic = [ai_costs[j] for j in range(n_valid) if valid[j]]
                ain = [ai_nodes[j] for j in range(n_valid) if valid[j]]
                rit = [ri_times[j] for j in range(n_valid) if valid[j]]
                ric = [ri_costs[j] for j in range(n_valid) if valid[j]]
                rin = [ri_nodes[j] for j in range(n_valid) if valid[j]]

                mean_at = np.mean(at); mean_an = np.mean(an); mean_ac = np.mean(ac)

                def calc_metrics(times, nodes, costs, label):
                    mt = np.mean(times); mn = np.mean(nodes); mc = np.mean(costs)
                    spd = mean_at / mt if mt > 0 else 0
                    nred = (1 - mn / mean_an) * 100 if mean_an > 0 else 0
                    opt_ratios = [costs[j] / ac[j] for j in range(len(costs))]
                    opt = np.mean(opt_ratios)
                    opt_std = np.std(opt_ratios)
                    return spd, nred, opt, opt_std

                ils_spd, ils_nred, ils_opt, ils_ostd = calc_metrics(it, inn, ic, 'ILS')
                ails_spd, ails_nred, ails_opt, ails_ostd = calc_metrics(ait_, ain, aic, 'AILS')
                rils_spd, rils_nred, rils_opt, rils_ostd = calc_metrics(rit, rin, ric, 'RILS')

                print(f"  {n_common} commonly solved | A* mean: {mean_at:.1f}ms, {mean_an:.0f} nodes")
                print(f"  ILS:  {ils_spd:.2f}x, {ils_nred:.1f}% red, opt={ils_opt:.4f}")
                print(f"  AILS: {ails_spd:.2f}x, {ails_nred:.1f}% red, opt={ails_opt:.4f}")
                print(f"  RILS: {rils_spd:.2f}x, {rils_nred:.1f}% red, opt={rils_opt:.4f}")

                results.append({
                    'map_name': name, 'category': category, 'risk_type': risk_type,
                    'lambda': lam, 'n_common': n_common,
                    'astar_time': mean_at, 'astar_nodes': mean_an,
                    'ils_speedup': ils_spd, 'ils_node_red': ils_nred,
                    'ils_opt': ils_opt, 'ils_opt_std': ils_ostd,
                    'ils_solve_rate': n_solved['ils'] / n_valid * 100,
                    'ails_speedup': ails_spd, 'ails_node_red': ails_nred,
                    'ails_opt': ails_opt, 'ails_opt_std': ails_ostd,
                    'ails_solve_rate': n_solved['ails'] / n_valid * 100,
                    'rils_speedup': rils_spd, 'rils_node_red': rils_nred,
                    'rils_opt': rils_opt, 'rils_opt_std': rils_ostd,
                    'rils_solve_rate': n_solved['rils'] / n_valid * 100,
                })

    # Save CSV
    if results:
        csv_path = os.path.join(RESULTS_DIR, 'exp10b_risk_benchmarks.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader(); w.writerows(results)
        print(f"\nResults saved to: {csv_path}")
    return results


# ==============================================================
# EXPERIMENT 10c: MAZE BENCHMARKS (generate synthetic if needed)
# ==============================================================
def generate_maze(size=256, corridor_width=1, seed=42):
    """Generate a perfect maze using randomized DFS with proper corridor widths."""
    rng = np.random.RandomState(seed)
    cell_size = corridor_width + 1  # wall(1) + corridor(corridor_width)
    n_cells_h = (size - 1) // cell_size
    n_cells_w = (size - 1) // cell_size

    grid = np.ones((size, size), dtype=bool)  # all walls

    def cell_center(cr, cc):
        return (1 + cr * cell_size, 1 + cc * cell_size)

    # Carve each cell
    for r in range(n_cells_h):
        for c in range(n_cells_w):
            cy, cx = cell_center(r, c)
            for dy in range(corridor_width):
                for dx in range(corridor_width):
                    ry, rx = cy + dy, cx + dx
                    if 0 <= ry < size and 0 <= rx < size:
                        grid[ry, rx] = False

    # DFS to connect cells by carving walls between them
    visited = set()
    stack = [(0, 0)]
    visited.add((0, 0))

    while stack:
        cr, cc = stack[-1]
        neighbors = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < n_cells_h and 0 <= nc < n_cells_w and (nr, nc) not in visited:
                neighbors.append((nr, nc, dr, dc))

        if neighbors:
            nr, nc, dr, dc = neighbors[rng.randint(len(neighbors))]
            visited.add((nr, nc))

            # Carve the wall between cells
            cy1, cx1 = cell_center(cr, cc)
            cy2, cx2 = cell_center(nr, nc)

            # Wall is between the two cell centers
            if dr != 0:  # vertical passage
                wy = min(cy1, cy2) + corridor_width if dr > 0 else max(cy1, cy2) - 1
                for dx in range(corridor_width):
                    if 0 <= wy < size and 0 <= cx1 + dx < size:
                        grid[wy, cx1 + dx] = False
            else:  # horizontal passage
                wx = min(cx1, cx2) + corridor_width if dc > 0 else max(cx1, cx2) - 1
                for dy in range(corridor_width):
                    if 0 <= cy1 + dy < size and 0 <= wx < size:
                        grid[cy1 + dy, wx] = False

            stack.append((nr, nc))
        else:
            stack.pop()

    return grid


def run_maze_benchmarks():
    """Run benchmarks on maze maps (downloaded or generated)."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 10c: Maze Benchmark Evaluation")
    print("=" * 60)

    maze_configs = [
        ('maze256-1-0', 1),
        ('maze256-2-0', 2),
        ('maze256-4-0', 4),
        ('maze256-8-0', 8),
    ]

    results = []
    n_scenarios = 20

    for maze_name, corridor_width in maze_configs:
        # Try to find downloaded maze
        cat_dir = os.path.join(MAPS_DIR, 'mazes')
        map_path = find_file_in_dir(cat_dir, f'{maze_name}.map')
        scen_path = find_file_in_dir(cat_dir, f'{maze_name}.map.scen')

        use_downloaded = False
        if map_path and scen_path:
            # Verify map file is not an HTML error page
            with open(map_path) as f:
                first_line = f.readline().strip()
            if first_line.startswith('<!DOCTYPE') or first_line.startswith('<html'):
                print(f"\n--- {maze_name} (downloaded map is corrupt HTML, using generated) ---")
                use_downloaded = False
            else:
                print(f"\n--- {maze_name} (downloaded) ---")
                obstacles, H, W = parse_map_file(map_path)
                scenarios = parse_scen_file(scen_path)
                scenarios.sort(key=lambda s: s['optimal_length'], reverse=True)
                selected = scenarios[:n_scenarios]
                use_downloaded = True

        if not use_downloaded:
            print(f"\n--- {maze_name} (generated, corridor_width={corridor_width}) ---")
            obstacles = generate_maze(256, corridor_width, seed=SEED + corridor_width)
            H, W = 256, 256
            # Generate random scenarios
            rng = np.random.RandomState(SEED + corridor_width)
            free_cells = list(zip(*np.where(~obstacles)))
            selected = []
            attempts = 0
            while len(selected) < n_scenarios and attempts < n_scenarios * 20:
                attempts += 1
                si = rng.randint(len(free_cells))
                gi = rng.randint(len(free_cells))
                s = free_cells[si]; g = free_cells[gi]
                dist = abs(s[0]-g[0]) + abs(s[1]-g[1])
                if dist > 80:  # long paths for 256 grid
                    selected.append({'start': s, 'goal': g, 'optimal_length': 0})
            print(f"  Generated {len(selected)} scenarios")

        density = np.mean(obstacles)
        grid = GridMap(W, H, obstacles, risk=None)
        print(f"  Grid: {W}x{H}, density: {density:.1%}")

        a_times = []; a_nodes = []; a_costs = []
        i_times = []; i_nodes = []; i_costs = []
        ai_times = []; ai_nodes = []; ai_costs = []
        n_solved = {'astar': 0, 'ils': 0, 'ails': 0}

        for scen in selected:
            s, g = scen['start'], scen['goal']
            if not grid.is_free(s[0], s[1]) or not grid.is_free(g[0], g[1]):
                continue

            t0 = time.perf_counter()
            path_a, nodes_a, cost_a = astar_generic(grid, s, g)
            t_a = (time.perf_counter() - t0) * 1000
            if path_a is None: continue
            n_solved['astar'] += 1
            a_times.append(t_a); a_nodes.append(nodes_a); a_costs.append(cost_a)

            t0 = time.perf_counter()
            path_i, nodes_i, cost_i, _ = ils_search(grid, s, g, alpha=0.05)
            t_i = (time.perf_counter() - t0) * 1000
            i_times.append(t_i); i_nodes.append(nodes_i)
            i_costs.append(cost_i if path_i else float('inf'))
            if path_i: n_solved['ils'] += 1

            t0 = time.perf_counter()
            path_ai, nodes_ai, cost_ai, _ = ails_search(grid, s, g)
            t_ai = (time.perf_counter() - t0) * 1000
            ai_times.append(t_ai); ai_nodes.append(nodes_ai)
            ai_costs.append(cost_ai if path_ai else float('inf'))
            if path_ai: n_solved['ails'] += 1

        n_valid = len(a_times)
        if n_valid == 0: continue

        valid = [i_costs[j] < float('inf') and ai_costs[j] < float('inf') for j in range(n_valid)]
        n_common = sum(valid)
        if n_common < 3:
            print(f"  Only {n_common} commonly solved, reporting partial results")
            # Report ILS and AILS solve rates
            results.append({
                'map_name': maze_name, 'corridor_width': corridor_width,
                'density': f"{density:.1%}", 'n_scenarios': n_valid, 'n_common': n_common,
                'ils_solve_rate': n_solved['ils'] / n_valid * 100,
                'ails_solve_rate': n_solved['ails'] / n_valid * 100,
                'ils_speedup': 0, 'ils_node_red': 0, 'ils_opt': 0,
                'ails_speedup': 0, 'ails_node_red': 0, 'ails_opt': 0,
            })
            continue

        at = [a_times[j] for j in range(n_valid) if valid[j]]
        ac = [a_costs[j] for j in range(n_valid) if valid[j]]
        an = [a_nodes[j] for j in range(n_valid) if valid[j]]
        it_ = [i_times[j] for j in range(n_valid) if valid[j]]
        ic = [i_costs[j] for j in range(n_valid) if valid[j]]
        inn_ = [i_nodes[j] for j in range(n_valid) if valid[j]]
        ait_ = [ai_times[j] for j in range(n_valid) if valid[j]]
        aic = [ai_costs[j] for j in range(n_valid) if valid[j]]
        ain_ = [ai_nodes[j] for j in range(n_valid) if valid[j]]

        mean_at = np.mean(at); mean_an = np.mean(an)
        ils_spd = mean_at / np.mean(it_) if np.mean(it_) > 0 else 0
        ils_nred = (1 - np.mean(inn_) / mean_an) * 100
        ils_opt = np.mean([ic[j] / ac[j] for j in range(len(ac))])
        ails_spd = mean_at / np.mean(ait_) if np.mean(ait_) > 0 else 0
        ails_nred = (1 - np.mean(ain_) / mean_an) * 100
        ails_opt = np.mean([aic[j] / ac[j] for j in range(len(ac))])

        print(f"  {n_common} commonly solved:")
        print(f"  ILS:  {ils_spd:.2f}x, {ils_nred:.1f}% red, opt={ils_opt:.4f}, solve={n_solved['ils']/n_valid*100:.0f}%")
        print(f"  AILS: {ails_spd:.2f}x, {ails_nred:.1f}% red, opt={ails_opt:.4f}, solve={n_solved['ails']/n_valid*100:.0f}%")

        results.append({
            'map_name': maze_name, 'corridor_width': corridor_width,
            'density': f"{density:.1%}", 'n_scenarios': n_valid, 'n_common': n_common,
            'ils_speedup': ils_spd, 'ils_node_red': ils_nred, 'ils_opt': ils_opt,
            'ils_solve_rate': n_solved['ils'] / n_valid * 100,
            'ails_speedup': ails_spd, 'ails_node_red': ails_nred, 'ails_opt': ails_opt,
            'ails_solve_rate': n_solved['ails'] / n_valid * 100,
        })

    if results:
        csv_path = os.path.join(RESULTS_DIR, 'exp10c_maze_benchmarks.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader(); w.writerows(results)
        print(f"\nResults saved to: {csv_path}")
    return results


# ==============================================================
# EXPERIMENT 11: MULTI-HEURISTIC COMPARISON
# ==============================================================
def run_multi_heuristic():
    """Compare octile, Euclidean, Manhattan heuristics on risk-annotated grids."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 11: Multi-Heuristic Comparison")
    print("=" * 60)

    grid_size = 200
    densities = [0.1, 0.2, 0.3]
    risk_types = ['gradient', 'hotspot', 'uniform']
    lambdas = [1.0]
    n_maps = 20
    results = []

    heuristic_configs = [
        ('octile', octile_h),
        ('euclidean', euclidean_h),
        ('manhattan', manhattan_h),
        ('risk_adj_03', lambda r1,c1,r2,c2: risk_adjusted_h(r1,c1,r2,c2, 1.0, 0.3)),
    ]

    for density in densities:
        for risk_type in risk_types:
            for lam in lambdas:
                print(f"\n--- density={density}, risk={risk_type}, λ={lam} ---")

                heur_results = {name: {'times': [], 'nodes': [], 'costs': []}
                                for name, _ in heuristic_configs}
                ils_heur_results = {name: {'times': [], 'nodes': [], 'costs': [], 'solved': 0}
                                    for name, _ in heuristic_configs}

                for map_i in range(n_maps):
                    rng = np.random.RandomState(SEED + map_i)
                    obstacles = rng.random((grid_size, grid_size)) < density

                    if risk_type == 'gradient':
                        risk_layer = generate_gradient_risk(grid_size, grid_size)
                    elif risk_type == 'hotspot':
                        risk_layer = generate_hotspot_risk(grid_size, grid_size, seed=SEED + map_i)
                    else:
                        risk_layer = generate_uniform_risk(grid_size, grid_size)

                    grid = GridMap(grid_size, grid_size, obstacles, risk=risk_layer)

                    # Random start-goal
                    free = list(zip(*np.where(~obstacles)))
                    if len(free) < 2: continue
                    idx = rng.choice(len(free), 2, replace=False)
                    s, g = free[idx[0]], free[idx[1]]
                    if abs(s[0]-g[0]) + abs(s[1]-g[1]) < 100: continue

                    for hname, hfn in heuristic_configs:
                        # A* with this heuristic
                        t0 = time.perf_counter()
                        path, nodes, cost = astar_generic(grid, s, g, lam, hfn)
                        t = (time.perf_counter() - t0) * 1000
                        if path:
                            heur_results[hname]['times'].append(t)
                            heur_results[hname]['nodes'].append(nodes)
                            heur_results[hname]['costs'].append(cost)

                        # ILS with this heuristic
                        t0 = time.perf_counter()
                        path_i, nodes_i, cost_i, _ = ils_search(grid, s, g, alpha=0.05, lam=lam, heuristic_fn=hfn)
                        t_i = (time.perf_counter() - t0) * 1000
                        if path_i:
                            ils_heur_results[hname]['times'].append(t_i)
                            ils_heur_results[hname]['nodes'].append(nodes_i)
                            ils_heur_results[hname]['costs'].append(cost_i)
                            ils_heur_results[hname]['solved'] += 1

                for hname, _ in heuristic_configs:
                    h_data = heur_results[hname]
                    i_data = ils_heur_results[hname]
                    if not h_data['times'] or not i_data['times']: continue

                    mean_at = np.mean(h_data['times'])
                    mean_an = np.mean(h_data['nodes'])
                    mean_ac = np.mean(h_data['costs'])
                    mean_it = np.mean(i_data['times'])
                    mean_in = np.mean(i_data['nodes'])
                    mean_ic = np.mean(i_data['costs'])

                    spd = mean_at / mean_it if mean_it > 0 else 0
                    nred = (1 - mean_in / mean_an) * 100
                    opt = mean_ic / mean_ac if mean_ac > 0 else 1.0

                    results.append({
                        'density': density, 'risk_type': risk_type, 'lambda': lam,
                        'heuristic': hname,
                        'astar_time': mean_at, 'astar_nodes': mean_an,
                        'ils_speedup': spd, 'ils_node_red': nred, 'ils_opt': opt,
                        'n_maps': len(h_data['times']),
                        'ils_solved': i_data['solved'],
                    })

                    print(f"  {hname}: A* {mean_at:.1f}ms/{mean_an:.0f}n -> ILS {spd:.2f}x, {nred:.1f}% red, opt={opt:.4f}")

    if results:
        csv_path = os.path.join(RESULTS_DIR, 'exp11_multi_heuristic.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader(); w.writerows(results)
        print(f"\nResults saved to: {csv_path}")
    return results


# ==============================================================
# FIGURE 2 REPLACEMENT: Real-data corridor visualization
# ==============================================================
def generate_figure2_replacement():
    """Generate a real-data visualization showing ILS, AILS, and RILS corridors."""
    if not HAS_MPL:
        print("matplotlib not available, skipping figure generation")
        return

    print("\n" + "=" * 60)
    print("Generating Figure 2 replacement (real data visualization)")
    print("=" * 60)

    # Create a 200x200 grid with hotspot risk
    size = 200
    rng = np.random.RandomState(SEED)
    obstacles = rng.random((size, size)) < 0.20

    risk = generate_hotspot_risk(size, size, n_hotspots=4, seed=42)
    grid = GridMap(size, size, obstacles, risk=risk)

    start = (10, 10)
    goal = (185, 185)

    # Ensure start/goal are free
    obstacles[start[0], start[1]] = False
    obstacles[goal[0], goal[1]] = False

    lam = 1.0

    # Get paths and corridors
    # A* path
    path_astar, _, _ = astar_generic(grid, start, goal, lam)

    # ILS path and corridor
    sr, sc = start; gr, gc = goal
    H, W = size, size
    diag = np.sqrt(H**2 + W**2)
    line_points = bresenham_line(sr, sc, gr, gc)
    ils_mask = build_corridor_mask(H, W, line_points, 0.05 * diag)
    path_ils, _, _, _ = ils_search(grid, start, goal, alpha=0.05, lam=lam)

    # AILS corridor
    integral_obs = build_integral_image(obstacles)
    ails_mask = np.zeros((H, W), dtype=bool)
    r_min = 0.02 * diag; r_max = 0.10 * diag
    for (lr, lc) in line_points:
        density = query_integral(integral_obs, lr, lc, 3, H, W)
        radius = r_min + (r_max - r_min) * (density ** 1.0)
        r_int = int(np.ceil(radius))
        ails_mask[max(0,lr-r_int):min(H,lr+r_int+1), max(0,lc-r_int):min(W,lc+r_int+1)] = True

    # RILS corridor
    integral_risk = build_integral_image(risk)
    rils_mask = np.zeros((H, W), dtype=bool)
    r_base = 0.05 * diag; r_max_r = 0.15 * diag
    for (lr, lc) in line_points:
        mean_risk = query_integral(integral_risk, lr, lc, 5, H, W)
        radius = r_base + (r_max_r - r_base) * (mean_risk ** 1.0)
        r_int = int(np.ceil(radius))
        rils_mask[max(0,lr-r_int):min(H,lr+r_int+1), max(0,lc-r_int):min(W,lc+r_int+1)] = True
    path_rils, _, _, _ = rils_search(grid, start, goal, lam=lam)

    # Create figure: 2x2 panel
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # Panel A: Grid with risk overlay
    ax = axes[0, 0]
    bg = np.ones((size, size, 3))
    # Risk as red channel
    for r in range(size):
        for c in range(size):
            if obstacles[r, c]:
                bg[r, c] = [0.2, 0.2, 0.2]
            else:
                rv = risk[r, c]
                bg[r, c] = [1.0, 1.0 - 0.6*rv, 1.0 - 0.8*rv]
    ax.imshow(bg, origin='upper', interpolation='nearest')
    if path_astar:
        pr = [p[0] for p in path_astar]; pc = [p[1] for p in path_astar]
        ax.plot(pc, pr, 'b-', linewidth=1.5, alpha=0.8, label='A* path')
    ax.plot(start[1], start[0], 'go', markersize=10, zorder=5)
    ax.plot(goal[1], goal[0], 'r*', markersize=12, zorder=5)
    ax.set_title('(a) Risk-annotated grid with A* path', fontsize=12, fontweight='bold')
    ax.set_xlim(0, size); ax.set_ylim(size, 0)
    ax.legend(loc='upper right', fontsize=9)

    # Panel B: ILS corridor
    ax = axes[0, 1]
    bg2 = bg.copy()
    corridor_overlay = np.zeros((size, size, 4))
    corridor_overlay[ils_mask, :] = [0.0, 0.4, 1.0, 0.15]
    ax.imshow(bg2, origin='upper', interpolation='nearest')
    ax.imshow(corridor_overlay, origin='upper', interpolation='nearest')
    # Bresenham line
    lr = [p[0] for p in line_points]; lc = [p[1] for p in line_points]
    ax.plot(lc, lr, 'b--', linewidth=0.8, alpha=0.5, label='Bresenham line')
    if path_ils:
        pr = [p[0] for p in path_ils]; pc = [p[1] for p in path_ils]
        ax.plot(pc, pr, 'b-', linewidth=1.5, alpha=0.9, label='ILS path')
    ax.plot(start[1], start[0], 'go', markersize=10, zorder=5)
    ax.plot(goal[1], goal[0], 'r*', markersize=12, zorder=5)
    ax.set_title('(b) ILS: Uniform-width corridor', fontsize=12, fontweight='bold')
    ax.set_xlim(0, size); ax.set_ylim(size, 0)
    ax.legend(loc='upper right', fontsize=9)

    # Panel C: AILS corridor
    ax = axes[1, 0]
    bg3 = bg.copy()
    corridor_overlay2 = np.zeros((size, size, 4))
    corridor_overlay2[ails_mask, :] = [0.0, 0.8, 0.0, 0.15]
    ax.imshow(bg3, origin='upper', interpolation='nearest')
    ax.imshow(corridor_overlay2, origin='upper', interpolation='nearest')
    ax.plot(lc, lr, 'g--', linewidth=0.8, alpha=0.5, label='Bresenham line')
    path_ails_v, _, _, _ = ails_search(grid, start, goal, lam=lam)
    if path_ails_v:
        pr = [p[0] for p in path_ails_v]; pc = [p[1] for p in path_ails_v]
        ax.plot(pc, pr, 'g-', linewidth=1.5, alpha=0.9, label='AILS path')
    ax.plot(start[1], start[0], 'go', markersize=10, zorder=5)
    ax.plot(goal[1], goal[0], 'r*', markersize=12, zorder=5)
    ax.set_title('(c) AILS: Density-adaptive corridor', fontsize=12, fontweight='bold')
    ax.set_xlim(0, size); ax.set_ylim(size, 0)
    ax.legend(loc='upper right', fontsize=9)

    # Panel D: RILS corridor
    ax = axes[1, 1]
    bg4 = bg.copy()
    corridor_overlay3 = np.zeros((size, size, 4))
    corridor_overlay3[rils_mask, :] = [0.8, 0.0, 0.0, 0.15]
    ax.imshow(bg4, origin='upper', interpolation='nearest')
    ax.imshow(corridor_overlay3, origin='upper', interpolation='nearest')
    ax.plot(lc, lr, 'r--', linewidth=0.8, alpha=0.5, label='Bresenham line')
    if path_rils:
        pr = [p[0] for p in path_rils]; pc = [p[1] for p in path_rils]
        ax.plot(pc, pr, 'r-', linewidth=1.5, alpha=0.9, label='RILS path')
    ax.plot(start[1], start[0], 'go', markersize=10, zorder=5)
    ax.plot(goal[1], goal[0], 'r*', markersize=12, zorder=5)
    ax.set_title('(d) RILS: Risk-responsive corridor', fontsize=12, fontweight='bold')
    ax.set_xlim(0, size); ax.set_ylim(size, 0)
    ax.legend(loc='upper right', fontsize=9)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'corridor_comparison_real.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")

    # Also save as PDF for LaTeX
    fig_pdf = os.path.join(FIGURES_DIR, 'corridor_comparison_real.pdf')
    fig.savefig(fig_pdf, dpi=300, bbox_inches='tight')
    print(f"PDF saved to: {fig_pdf}")

    return fig_path


# ==============================================================
# MAIN
# ==============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("EXTENDED BENCHMARK EXPERIMENTS")
    print("=" * 60)

    # 1. Generate Figure 2 replacement
    generate_figure2_replacement()

    # 2. Maze benchmarks
    maze_results = run_maze_benchmarks()

    # 3. Risk-annotated benchmarks
    risk_bench_results = run_risk_annotated_benchmarks()

    # 4. Multi-heuristic comparison
    heur_results = run_multi_heuristic()

    print("\n" + "=" * 60)
    print("ALL EXTENDED EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")
