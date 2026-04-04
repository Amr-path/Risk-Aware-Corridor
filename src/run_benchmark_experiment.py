#!/usr/bin/env python3
"""
Experiment 10: Moving AI Lab Benchmark Evaluation

Evaluates ILS and AILS on standard pathfinding benchmarks from
Sturtevant (2012) to establish external validity and enable direct
comparison with published results.

Benchmark maps:
  - Game maps: dao (Dragon Age: Origins), bg (Baldur's Gate)
  - Mazes: maze512-1-0 through maze512-1-7
  - Random: random512-10-0 through random512-40-0

The script downloads benchmark .map files and .scen (scenario) files
from the Moving AI Lab website, then runs A*, ILS, and AILS on
a sample of scenarios per map.

Usage:
  python3 run_benchmark_experiment.py

Requirements:
  pip install numpy scipy requests

Author: Amr Elshahed
"""

import numpy as np
import heapq
import time
import csv
import os
import sys
from typing import Optional, Tuple, List, Set
from collections import defaultdict

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package required. Install with: pip install requests")
    sys.exit(1)

SEED = 42
np.random.seed(SEED)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
MAPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_maps')
os.makedirs(MAPS_DIR, exist_ok=True)

# ==============================================================
# BENCHMARK MAP CONFIGURATIONS
# ==============================================================
# We use a curated subset covering different map types:
#   - Game maps (structured, indoor environments)
#   - Mazes (narrow corridors, challenging for corridor methods)
#   - Random maps (similar to our procedural grids, for validation)
#
# Moving AI Lab distributes benchmarks as ZIP archives per category.
# We download the ZIPs and extract individual files.
#
# ZIP download URLs (from https://movingai.com/benchmarks/grids.html):
#   Maps: https://movingai.com/benchmarks/{category}/{category}-map.zip
#   Scen: https://movingai.com/benchmarks/{category}/{category}-scen.zip

BASE_URL = 'https://www.movingai.com/benchmarks'

# (category_on_site, zip_prefix, map_filename, scen_filename, display_name, our_category)
BENCHMARK_ENTRIES = [
    # DAO (Dragon Age: Origins) game maps
    ('dao', 'dao', 'lak303d.map', 'lak303d.map.scen', 'lak303d', 'game'),
    ('dao', 'dao', 'den501d.map', 'den501d.map.scen', 'den501d', 'game'),
    ('dao', 'dao', 'brc202d.map', 'brc202d.map.scen', 'brc202d', 'game'),
    ('dao', 'dao', 'ost003d.map', 'ost003d.map.scen', 'ost003d', 'game'),
    # Mazes (512x512 with varying corridor widths)
    ('mazes', 'maze', 'maze512-1-0.map', 'maze512-1-0.map.scen', 'maze512-1-0', 'maze'),
    ('mazes', 'maze', 'maze512-2-0.map', 'maze512-2-0.map.scen', 'maze512-2-0', 'maze'),
    ('mazes', 'maze', 'maze512-4-0.map', 'maze512-4-0.map.scen', 'maze512-4-0', 'maze'),
    ('mazes', 'maze', 'maze512-8-0.map', 'maze512-8-0.map.scen', 'maze512-8-0', 'maze'),
    # Random maps (512x512, 10-40% obstacle density)
    ('random', 'random', 'random512-10-0.map', 'random512-10-0.map.scen', 'random512-10-0', 'random'),
    ('random', 'random', 'random512-20-0.map', 'random512-20-0.map.scen', 'random512-20-0', 'random'),
    ('random', 'random', 'random512-30-0.map', 'random512-30-0.map.scen', 'random512-30-0', 'random'),
    ('random', 'random', 'random512-40-0.map', 'random512-40-0.map.scen', 'random512-40-0', 'random'),
]

# ==============================================================
# MAP PARSER (Moving AI Lab .map format)
# ==============================================================
def parse_map_file(filepath):
    """Parse a .map file in Moving AI Lab format.

    Format:
        type octile
        height H
        width W
        map
        <H lines of W chars each>

    Characters: . = passable, @ = obstacle, T = tree (obstacle),
                G = passable (grass), S = passable (swamp)
    """
    with open(filepath) as f:
        lines = f.readlines()

    height = width = 0
    map_start = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('height'):
            height = int(line.split()[1])
        elif line.startswith('width'):
            width = int(line.split()[1])
        elif line == 'map':
            map_start = i + 1
            break

    obstacles = np.zeros((height, width), dtype=bool)
    for r in range(height):
        row_str = lines[map_start + r].strip()
        for c in range(min(width, len(row_str))):
            ch = row_str[c]
            if ch in ('@', 'O', 'T', 'W'):  # obstacles
                obstacles[r, c] = True
            # '.', 'G', 'S' are passable

    return obstacles, height, width


def parse_scen_file(filepath):
    """Parse a .scen file in Moving AI Lab format.

    Format (tab-separated):
        version 1
        bucket  map  width  height  startx  starty  goalx  goaly  optimal_length
    """
    scenarios = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('version') or not line:
                continue
            parts = line.split('\t')
            if len(parts) < 9:
                parts = line.split()
            if len(parts) < 9:
                continue
            try:
                bucket = int(parts[0])
                sx, sy = int(parts[4]), int(parts[5])
                gx, gy = int(parts[6]), int(parts[7])
                opt_len = float(parts[8])
                scenarios.append({
                    'bucket': bucket,
                    'start': (sy, sx),  # (row, col) format
                    'goal': (gy, gx),
                    'optimal_length': opt_len,
                })
            except (ValueError, IndexError):
                continue
    return scenarios


# ==============================================================
# PATHFINDING ALGORITHMS (reuse from supplementary experiments)
# ==============================================================
class GridMap:
    """Binary occupancy grid with optional risk layer."""
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
    dr = abs(r1 - r2)
    dc = abs(c1 - c2)
    return max(dr, dc) + (1.414 - 1.0) * min(dr, dc)


def astar(grid, start, goal, lam=0.0):
    """Standard A* on grid with optional risk weighting."""
    sr, sc = start
    gr, gc = goal
    if not grid.is_free(sr, sc) or not grid.is_free(gr, gc):
        return None, 0, float('inf')

    open_list = [(octile_h(sr, sc, gr, gc), 0, sr, sc)]
    g_cost = {(sr, sc): 0.0}
    closed = set()
    parent = {}
    expanded = 0

    while open_list:
        f, _, r, c = heapq.heappop(open_list)
        if (r, c) in closed:
            continue
        closed.add((r, c))
        expanded += 1

        if (r, c) == (gr, gc):
            # Reconstruct path
            path = [(r, c)]
            while (r, c) != (sr, sc):
                r, c = parent[(r, c)]
                path.append((r, c))
            path.reverse()
            return path, expanded, g_cost[(gr, gc)]

        for nr, nc, base_cost in grid.neighbors_8(r, c):
            if (nr, nc) in closed:
                continue
            risk_cost = lam * grid.risk[nr, nc] if grid.risk is not None and lam > 0 else 0
            new_g = g_cost[(r, c)] + base_cost + risk_cost
            if new_g < g_cost.get((nr, nc), float('inf')):
                g_cost[(nr, nc)] = new_g
                parent[(nr, nc)] = (r, c)
                h = octile_h(nr, nc, gr, gc)
                heapq.heappush(open_list, (new_g + h, expanded, nr, nc))

    return None, expanded, float('inf')


def bresenham_line(r0, c0, r1, c1):
    """Bresenham's line algorithm returning list of (r, c)."""
    points = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return points


def build_corridor_mask(grid, line_points, radius):
    """Build boolean mask of corridor around Bresenham line."""
    H, W = grid.height, grid.width
    mask = np.zeros((H, W), dtype=bool)
    r_int = int(np.ceil(radius))
    for (lr, lc) in line_points:
        r_min = max(0, lr - r_int)
        r_max = min(H - 1, lr + r_int)
        c_min = max(0, lc - r_int)
        c_max = min(W - 1, lc + r_int)
        mask[r_min:r_max+1, c_min:c_max+1] = True
    return mask


def astar_corridor(grid, start, goal, corridor_mask, lam=0.0):
    """A* restricted to corridor mask."""
    sr, sc = start
    gr, gc = goal
    if not grid.is_free(sr, sc) or not grid.is_free(gr, gc):
        return None, 0, float('inf')

    open_list = [(octile_h(sr, sc, gr, gc), 0, sr, sc)]
    g_cost = {(sr, sc): 0.0}
    closed = set()
    parent = {}
    expanded = 0

    while open_list:
        f, _, r, c = heapq.heappop(open_list)
        if (r, c) in closed:
            continue
        closed.add((r, c))
        expanded += 1

        if (r, c) == (gr, gc):
            path = [(r, c)]
            while (r, c) != (sr, sc):
                r, c = parent[(r, c)]
                path.append((r, c))
            path.reverse()
            return path, expanded, g_cost[(gr, gc)]

        for nr, nc, base_cost in grid.neighbors_8(r, c):
            if (nr, nc) in closed or not corridor_mask[nr, nc]:
                continue
            risk_cost = lam * grid.risk[nr, nc] if grid.risk is not None and lam > 0 else 0
            new_g = g_cost[(r, c)] + base_cost + risk_cost
            if new_g < g_cost.get((nr, nc), float('inf')):
                g_cost[(nr, nc)] = new_g
                parent[(nr, nc)] = (r, c)
                h = octile_h(nr, nc, gr, gc)
                heapq.heappush(open_list, (new_g + h, expanded, nr, nc))

    return None, expanded, float('inf')


def ils_search(grid, start, goal, alpha=0.05, max_attempts=10, lam=0.0):
    """ILS: Incremental Line Search with uniform corridor."""
    sr, sc = start
    gr, gc = goal
    diag = np.sqrt(grid.height**2 + grid.width**2)
    radius = alpha * diag
    delta_r = 0.02 * diag

    line_points = bresenham_line(sr, sc, gr, gc)

    total_expanded = 0
    for attempt in range(max_attempts):
        mask = build_corridor_mask(grid, line_points, radius)
        # Ensure start and goal are in corridor
        mask[sr, sc] = True
        mask[gr, gc] = True

        path, expanded, cost = astar_corridor(grid, start, goal, mask, lam)
        total_expanded += expanded
        if path is not None:
            return path, total_expanded, cost, attempt + 1
        radius += delta_r

    return None, total_expanded, float('inf'), max_attempts


def build_integral_image(grid_2d):
    """Build integral image (summed area table) for O(1) region queries."""
    return grid_2d.astype(float).cumsum(axis=0).cumsum(axis=1)


def query_integral(integral, r, c, w, H, W):
    """Query sum in window of half-width w centred at (r,c)."""
    r0 = max(0, r - w)
    c0 = max(0, c - w)
    r1 = min(H - 1, r + w)
    c1 = min(W - 1, c + w)
    total = integral[r1, c1]
    if r0 > 0:
        total -= integral[r0-1, c1]
    if c0 > 0:
        total -= integral[r1, c0-1]
    if r0 > 0 and c0 > 0:
        total += integral[r0-1, c0-1]
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    return total / area if area > 0 else 0


def ails_search(grid, start, goal, r_min_frac=0.02, r_max_frac=0.10,
                alpha_exp=1.0, window=3, max_attempts=10, lam=0.0):
    """AILS: Adaptive ILS with density-based variable corridor."""
    sr, sc = start
    gr, gc = goal
    H, W = grid.height, grid.width
    diag = np.sqrt(H**2 + W**2)
    r_min = r_min_frac * diag
    r_max = r_max_frac * diag
    delta_r = 0.02 * diag

    line_points = bresenham_line(sr, sc, gr, gc)
    integral = build_integral_image(grid.obstacles)

    total_expanded = 0
    for attempt in range(max_attempts):
        mask = np.zeros((H, W), dtype=bool)
        for (lr, lc) in line_points:
            density = query_integral(integral, lr, lc, window, H, W)
            radius = r_min + (r_max - r_min) * (density ** alpha_exp)
            r_int = int(np.ceil(radius))
            rr0 = max(0, lr - r_int)
            rr1 = min(H - 1, lr + r_int)
            cc0 = max(0, lc - r_int)
            cc1 = min(W - 1, lc + r_int)
            mask[rr0:rr1+1, cc0:cc1+1] = True
        mask[sr, sc] = True
        mask[gr, gc] = True

        path, expanded, cost = astar_corridor(grid, start, goal, mask, lam)
        total_expanded += expanded
        if path is not None:
            return path, total_expanded, cost, attempt + 1
        r_min += delta_r
        r_max += delta_r

    return None, total_expanded, float('inf'), max_attempts


# ==============================================================
# DOWNLOAD BENCHMARK FILES
# ==============================================================
import zipfile
import io

def download_file(url, local_path):
    """Download a file if it doesn't already exist."""
    if os.path.exists(local_path):
        return True
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(resp.content)
            return True
        return False
    except Exception:
        return False


def download_and_extract_zip(url, extract_dir):
    """Download a ZIP archive and extract it."""
    print(f"  Downloading ZIP: {url} ...")
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                zf.extractall(extract_dir)
            print(f"    Extracted {len(zf.namelist())} files")
            return True
        else:
            print(f"    HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"    Failed: {e}")
        return False


def find_file_in_dir(directory, filename):
    """Recursively find a file in a directory tree."""
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


def download_benchmarks():
    """Download all benchmark map and scenario files.

    Strategy:
    1. Try downloading ZIP archives per category (the standard distribution method)
    2. Try multiple individual file URL patterns as fallback
    3. Report what's available
    """
    print("=" * 60)
    print("Downloading Moving AI Lab benchmark files...")
    print("=" * 60)

    # Determine which categories we need
    categories_needed = set()
    for (cat_site, zip_prefix, mf, sf, name, cat) in BENCHMARK_ENTRIES:
        categories_needed.add((cat_site, zip_prefix))

    # Step 1: Try downloading ZIP archives
    for cat_site, zip_prefix in categories_needed:
        cat_dir = os.path.join(MAPS_DIR, cat_site)
        os.makedirs(cat_dir, exist_ok=True)

        # Try multiple URL patterns for the ZIP files
        map_zip_urls = [
            f"{BASE_URL}/{cat_site}/{zip_prefix}-map.zip",
            f"{BASE_URL}/{cat_site}/{cat_site}-map.zip",
            f"{BASE_URL}/{cat_site}/maps.zip",
        ]
        scen_zip_urls = [
            f"{BASE_URL}/{cat_site}/{zip_prefix}-scen.zip",
            f"{BASE_URL}/{cat_site}/{cat_site}-scen.zip",
            f"{BASE_URL}/{cat_site}/scenarios.zip",
            f"{BASE_URL}/{cat_site}/scen.zip",
        ]

        print(f"\n--- Category: {cat_site} ---")

        # Try map ZIPs
        map_ok = False
        for url in map_zip_urls:
            if download_and_extract_zip(url, cat_dir):
                map_ok = True
                break

        if not map_ok:
            print(f"  ZIP download failed for {cat_site} maps, trying individual files...")

        # Try scenario ZIPs
        scen_ok = False
        for url in scen_zip_urls:
            if download_and_extract_zip(url, cat_dir):
                scen_ok = True
                break

        if not scen_ok:
            print(f"  ZIP download failed for {cat_site} scenarios, trying individual files...")

    # Step 2: For any files still missing, try individual URL patterns
    for (cat_site, zip_prefix, map_file, scen_file, name, cat) in BENCHMARK_ENTRIES:
        cat_dir = os.path.join(MAPS_DIR, cat_site)

        # Check if map file exists (might be in subdirectory after zip extraction)
        map_path = find_file_in_dir(cat_dir, map_file)
        if map_path is None:
            # Try individual download with multiple URL patterns
            individual_urls = [
                f"{BASE_URL}/{cat_site}/{map_file}",
                f"{BASE_URL}/{zip_prefix}/{map_file}",
                f"{BASE_URL}/{cat_site}/maps/{map_file}",
            ]
            for url in individual_urls:
                local = os.path.join(cat_dir, map_file)
                if download_file(url, local):
                    print(f"  Downloaded individual: {map_file}")
                    break

        # Check if scen file exists
        scen_path = find_file_in_dir(cat_dir, scen_file)
        if scen_path is None:
            individual_urls = [
                f"{BASE_URL}/{cat_site}/{scen_file}",
                f"{BASE_URL}/{zip_prefix}/{scen_file}",
                f"{BASE_URL}/{cat_site}/scen/{scen_file}",
                f"{BASE_URL}/{cat_site}/scenarios/{scen_file}",
            ]
            for url in individual_urls:
                local = os.path.join(cat_dir, scen_file)
                if download_file(url, local):
                    print(f"  Downloaded individual: {scen_file}")
                    break

    # Step 3: Count successes
    success = 0
    missing = []
    for (cat_site, zip_prefix, map_file, scen_file, name, cat) in BENCHMARK_ENTRIES:
        cat_dir = os.path.join(MAPS_DIR, cat_site)
        mp = find_file_in_dir(cat_dir, map_file)
        sp = find_file_in_dir(cat_dir, scen_file)
        if mp and sp:
            success += 1
        else:
            missing_items = []
            if not mp: missing_items.append(f"{map_file}")
            if not sp: missing_items.append(f"{scen_file}")
            missing.append(f"  {name}: missing {', '.join(missing_items)}")

    print(f"\nFound {success}/{len(BENCHMARK_ENTRIES)} complete benchmark sets.")
    if missing:
        print("Missing files:")
        for m in missing:
            print(m)
        print(f"\nIf automatic download failed, please download manually:")
        print(f"  1. Visit https://movingai.com/benchmarks/grids.html")
        print(f"  2. Download map and scenario ZIPs for: dao, mazes, random")
        print(f"  3. Extract to: {MAPS_DIR}/<category>/")
        print(f"     e.g., {MAPS_DIR}/dao/lak303d.map")
        print(f"           {MAPS_DIR}/dao/lak303d.map.scen")

    return success


# ==============================================================
# EXPERIMENT 10: BENCHMARK EVALUATION
# ==============================================================
def run_benchmark_experiment(scenarios_per_map=100):
    """Run ILS and AILS evaluation on Moving AI Lab benchmarks.

    For each benchmark map:
    - Sample up to `scenarios_per_map` scenarios from the .scen file
      (selecting from higher buckets = longer paths for more meaningful eval)
    - Run A*, ILS (alpha=0.05), and AILS
    - Record: nodes expanded, wall-clock time, path cost, path optimality
    - Compare against known optimal path length from .scen file
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 10: Moving AI Lab Benchmark Evaluation")
    print("=" * 60)

    results = []

    for (cat_site, zip_prefix, map_file, scen_file, name, category) in BENCHMARK_ENTRIES:
        cat_dir = os.path.join(MAPS_DIR, cat_site)
        map_local = find_file_in_dir(cat_dir, map_file)
        scen_local = find_file_in_dir(cat_dir, scen_file)

        if not map_local or not scen_local:
            print(f"\nSkipping {name}: files not found")
            continue

        print(f"\n--- {name} ({category}) ---")

        # Parse map and scenarios
        obstacles, H, W = parse_map_file(map_local)
        scenarios = parse_scen_file(scen_local)

        if not scenarios:
            print(f"  No valid scenarios found")
            continue

        grid = GridMap(W, H, obstacles, risk=None)

        # Compute obstacle density
        density = np.mean(obstacles)
        print(f"  Grid: {W}x{H}, density: {density:.1%}, scenarios: {len(scenarios)}")

        # Select scenarios: prefer longer paths (higher buckets)
        # Sort by optimal length descending and take top N
        scenarios.sort(key=lambda s: s['optimal_length'], reverse=True)
        selected = scenarios[:scenarios_per_map]

        # Also include some medium-length paths
        if len(scenarios) > scenarios_per_map * 2:
            mid = len(scenarios) // 2
            mid_scenarios = scenarios[mid:mid + scenarios_per_map // 2]
            selected = selected[:scenarios_per_map // 2] + mid_scenarios

        print(f"  Selected {len(selected)} scenarios (path lengths: "
              f"{selected[-1]['optimal_length']:.1f} to {selected[0]['optimal_length']:.1f})")

        # Run experiments
        astar_times = []; astar_nodes = []; astar_costs = []
        ils_times = []; ils_nodes = []; ils_costs = []; ils_attempts_list = []
        ails_times = []; ails_nodes = []; ails_costs = []; ails_attempts_list = []
        optimal_lengths = []
        n_solved_astar = 0; n_solved_ils = 0; n_solved_ails = 0

        for i, scen in enumerate(selected):
            start = scen['start']
            goal = scen['goal']
            opt_len = scen['optimal_length']

            # Check start/goal validity
            if not grid.is_free(start[0], start[1]) or not grid.is_free(goal[0], goal[1]):
                continue
            if opt_len <= 0:
                continue

            # A*
            t0 = time.perf_counter()
            path_a, nodes_a, cost_a = astar(grid, start, goal)
            t_a = (time.perf_counter() - t0) * 1000

            if path_a is None:
                continue

            n_solved_astar += 1
            astar_times.append(t_a)
            astar_nodes.append(nodes_a)
            astar_costs.append(cost_a)
            optimal_lengths.append(opt_len)

            # ILS
            t0 = time.perf_counter()
            path_i, nodes_i, cost_i, attempts_i = ils_search(grid, start, goal, alpha=0.05)
            t_i = (time.perf_counter() - t0) * 1000

            if path_i is not None:
                n_solved_ils += 1
                ils_times.append(t_i)
                ils_nodes.append(nodes_i)
                ils_costs.append(cost_i)
                ils_attempts_list.append(attempts_i)
            else:
                ils_times.append(t_i)
                ils_nodes.append(nodes_i)
                ils_costs.append(float('inf'))
                ils_attempts_list.append(10)

            # AILS
            t0 = time.perf_counter()
            path_ai, nodes_ai, cost_ai, attempts_ai = ails_search(
                grid, start, goal, r_min_frac=0.02, r_max_frac=0.10
            )
            t_ai = (time.perf_counter() - t0) * 1000

            if path_ai is not None:
                n_solved_ails += 1
                ails_times.append(t_ai)
                ails_nodes.append(nodes_ai)
                ails_costs.append(cost_ai)
                ails_attempts_list.append(attempts_ai)
            else:
                ails_times.append(t_ai)
                ails_nodes.append(nodes_ai)
                ails_costs.append(float('inf'))
                ails_attempts_list.append(10)

            if (i + 1) % 25 == 0:
                print(f"  Processed {i+1}/{len(selected)} scenarios...")

        n_valid = len(astar_times)
        if n_valid == 0:
            print(f"  No valid scenarios completed")
            continue

        # Compute metrics (only for scenarios where all algorithms found a path)
        valid_mask = [ils_costs[j] < float('inf') and ails_costs[j] < float('inf')
                      for j in range(n_valid)]
        n_all_solved = sum(valid_mask)

        if n_all_solved == 0:
            print(f"  No scenarios solved by all algorithms")
            continue

        # Filter to commonly solved
        at = [astar_times[j] for j in range(n_valid) if valid_mask[j]]
        an = [astar_nodes[j] for j in range(n_valid) if valid_mask[j]]
        ac = [astar_costs[j] for j in range(n_valid) if valid_mask[j]]
        it = [ils_times[j] for j in range(n_valid) if valid_mask[j]]
        inn = [ils_nodes[j] for j in range(n_valid) if valid_mask[j]]
        ic = [ils_costs[j] for j in range(n_valid) if valid_mask[j]]
        ait = [ails_times[j] for j in range(n_valid) if valid_mask[j]]
        ain = [ails_nodes[j] for j in range(n_valid) if valid_mask[j]]
        aic = [ails_costs[j] for j in range(n_valid) if valid_mask[j]]

        # Compute means
        mean_at = np.mean(at); mean_an = np.mean(an); mean_ac = np.mean(ac)
        mean_it = np.mean(it); mean_in = np.mean(inn); mean_ic = np.mean(ic)
        mean_ait = np.mean(ait); mean_ain = np.mean(ain); mean_aic = np.mean(aic)

        # Speedup and node reduction
        ils_speedup = mean_at / mean_it if mean_it > 0 else 0
        ils_node_red = (1 - mean_in / mean_an) * 100 if mean_an > 0 else 0
        ils_opt = mean_ic / mean_ac if mean_ac > 0 else 1.0

        ails_speedup = mean_at / mean_ait if mean_ait > 0 else 0
        ails_node_red = (1 - mean_ain / mean_an) * 100 if mean_an > 0 else 0
        ails_opt = mean_aic / mean_ac if mean_ac > 0 else 1.0

        # Stds for CIs
        ils_opt_std = np.std([ic[j] / ac[j] for j in range(n_all_solved)])
        ails_opt_std = np.std([aic[j] / ac[j] for j in range(n_all_solved)])

        # ILS solve rate
        ils_solve_rate = n_solved_ils / n_valid * 100
        ails_solve_rate = n_solved_ails / n_valid * 100

        print(f"  Results ({n_all_solved} commonly solved):")
        print(f"    A*:   {mean_at:.1f} ms, {mean_an:.0f} nodes")
        print(f"    ILS:  {ils_speedup:.2f}x speedup, {ils_node_red:.1f}% red, opt={ils_opt:.4f}, solve={ils_solve_rate:.0f}%")
        print(f"    AILS: {ails_speedup:.2f}x speedup, {ails_node_red:.1f}% red, opt={ails_opt:.4f}, solve={ails_solve_rate:.0f}%")

        # Store result
        results.append({
            'map_name': name,
            'category': category,
            'grid_size': f"{W}x{H}",
            'density': f"{density:.1%}",
            'n_scenarios': n_valid,
            'n_all_solved': n_all_solved,
            'astar_time_mean': mean_at,
            'astar_nodes_mean': mean_an,
            'ils_speedup': ils_speedup,
            'ils_node_red': ils_node_red,
            'ils_opt_mean': ils_opt,
            'ils_opt_std': ils_opt_std,
            'ils_solve_rate': ils_solve_rate,
            'ails_speedup': ails_speedup,
            'ails_node_red': ails_node_red,
            'ails_opt_mean': ails_opt,
            'ails_opt_std': ails_opt_std,
            'ails_solve_rate': ails_solve_rate,
        })

    # Write CSV
    if results:
        csv_path = os.path.join(RESULTS_DIR, 'exp10_benchmark.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {csv_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE (for LaTeX)")
    print("=" * 60)
    print(f"{'Map':<20} {'Cat':<8} {'Size':<10} {'N':>4} "
          f"{'ILS Spd':>8} {'ILS Red%':>9} {'ILS Opt':>8} "
          f"{'AILS Spd':>9} {'AILS Red%':>10} {'AILS Opt':>9}")
    print("-" * 110)

    for r in results:
        print(f"{r['map_name']:<20} {r['category']:<8} {r['grid_size']:<10} {r['n_all_solved']:>4} "
              f"{r['ils_speedup']:>7.2f}x {r['ils_node_red']:>8.1f}% {r['ils_opt_mean']:>8.4f} "
              f"{r['ails_speedup']:>8.2f}x {r['ails_node_red']:>9.1f}% {r['ails_opt_mean']:>9.4f}")

    # Category averages
    print("\nCategory averages:")
    for cat in ['game', 'maze', 'random']:
        cat_results = [r for r in results if r['category'] == cat]
        if cat_results:
            avg_ils_spd = np.mean([r['ils_speedup'] for r in cat_results])
            avg_ils_red = np.mean([r['ils_node_red'] for r in cat_results])
            avg_ils_opt = np.mean([r['ils_opt_mean'] for r in cat_results])
            avg_ails_spd = np.mean([r['ails_speedup'] for r in cat_results])
            avg_ails_red = np.mean([r['ails_node_red'] for r in cat_results])
            avg_ails_opt = np.mean([r['ails_opt_mean'] for r in cat_results])
            print(f"  {cat:<10}: ILS {avg_ils_spd:.2f}x/{avg_ils_red:.1f}%/{avg_ils_opt:.4f}  "
                  f"AILS {avg_ails_spd:.2f}x/{avg_ails_red:.1f}%/{avg_ails_opt:.4f}")

    return results


# ==============================================================
# MAIN
# ==============================================================
if __name__ == '__main__':
    n_downloaded = download_benchmarks()
    if n_downloaded == 0:
        print("\nERROR: Could not download any complete benchmark sets.")
        print("Please download manually from https://movingai.com/benchmarks/grids.html")
        print(f"and extract to: {MAPS_DIR}/<category>/")
        print()
        print("Expected structure:")
        print(f"  {MAPS_DIR}/dao/lak303d.map + lak303d.map.scen")
        print(f"  {MAPS_DIR}/mazes/maze512-1-0.map + maze512-1-0.map.scen")
        print(f"  {MAPS_DIR}/random/random512-10-0.map + random512-10-0.map.scen")
        sys.exit(1)

    results = run_benchmark_experiment(scenarios_per_map=100)

    if results:
        print(f"\n{'='*60}")
        print("EXPERIMENT 10 COMPLETE")
        print(f"Results saved to: {os.path.join(RESULTS_DIR, 'exp10_benchmark.csv')}")
        print(f"{'='*60}")
    else:
        print("\nNo results generated. Check that benchmark files were downloaded correctly.")
