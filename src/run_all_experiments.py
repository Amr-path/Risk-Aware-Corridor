#!/usr/bin/env python3
"""
Comprehensive Experiment Suite for Journal Paper:
"Risk-Aware Corridor-Constrained Pathfinding for UAV Navigation
 in Biosecurity-Sensitive Environments"

Target: Expert Systems with Applications (Elsevier, Q1)

Experiments:
  1. Risk-Annotated Grid Experiments (DS6-style)
     - 3 densities × 3 risk types × 6 lambda values × 100 maps = 5,400 paths
  2. Port Environment Validation (DS7-style)
     - 3 grid sizes × 3 lambda values × 30 maps = 270 paths
  3. JPS Comparative Evaluation (uniform-cost grids)
     - 3 densities × 100 maps = 300 paths
  4. D*Lite Re-planning Comparison
     - 3 densities × 128 maps with obstacle insertions
  5. Progressive Obstacle Discovery Simulation
     - 3 densities × 60 missions with ~12 re-plans each

Author: Amr Elshahed
"""

import numpy as np
import heapq
import time
import csv
import os
import sys
from collections import defaultdict
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Set

# ============================================================
# UTILITY: Ensure reproducibility
# ============================================================
SEED = 42
np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# CORE: Grid and Path Structures
# ============================================================

@dataclass
class GridMap:
    """Binary occupancy grid with optional risk layer."""
    width: int
    height: int
    obstacles: np.ndarray       # bool array (True = blocked)
    risk: Optional[np.ndarray] = None  # float array [0,1], None if no risk

    def is_free(self, r, c):
        return 0 <= r < self.height and 0 <= c < self.width and not self.obstacles[r, c]

    def neighbors_8(self, r, c):
        """8-connected neighbors with movement costs."""
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]
        for (dr, dc), base_cost in zip(dirs, costs):
            nr, nc = r+dr, c+dc
            if self.is_free(nr, nc):
                yield nr, nc, base_cost


# ============================================================
# CORE: Grid Generation
# ============================================================

def generate_random_grid(size, density, seed=None):
    """Generate a random binary grid with guaranteed start-goal connectivity."""
    rng = np.random.RandomState(seed)
    grid = rng.random((size, size)) < density
    # Clear start and goal areas
    grid[0, :3] = False
    grid[:3, 0] = False
    grid[size-1, size-3:] = False
    grid[size-3:, size-1] = False
    # Clear top row and rightmost column for connectivity
    grid[0, :] = False
    grid[:, size-1] = False
    return grid


def generate_risk_layer(size, risk_type, seed=None):
    """Generate a risk layer of specified type."""
    rng = np.random.RandomState(seed)
    if risk_type == 'gradient':
        # Risk increases from top-left to bottom-right
        r = np.arange(size).reshape(-1, 1) / (size - 1)
        c = np.arange(size).reshape(1, -1) / (size - 1)
        risk = (r + c) / 2.0
    elif risk_type == 'uniform':
        risk = np.full((size, size), 0.3)
    elif risk_type == 'hotspot':
        risk = np.zeros((size, size))
        n_hotspots = max(3, size // 30)
        for _ in range(n_hotspots):
            cy, cx = rng.randint(10, size-10, 2)
            sigma = rng.uniform(5, 15)
            yy, xx = np.ogrid[:size, :size]
            d = np.sqrt((yy - cy)**2 + (xx - cx)**2)
            risk += np.exp(-d**2 / (2 * sigma**2))
        risk = np.clip(risk / risk.max(), 0, 1) if risk.max() > 0 else risk
    else:
        raise ValueError(f"Unknown risk type: {risk_type}")
    return risk


def generate_port_grid(size, seed=None):
    """
    Generate a procedural container-port grid with 4 zones:
    - Quay zone (top): open area
    - Container yard (middle): rectangular blocks with aisles
    - Gate zone (bottom): checkpoint barriers with openings
    - Buffer zone: mixed clutter
    Plus biosecurity risk layers.
    """
    rng = np.random.RandomState(seed)
    obstacles = np.zeros((size, size), dtype=bool)
    risk = np.zeros((size, size), dtype=float)

    # Zone boundaries
    quay_end = int(size * 0.15)
    yard_start = int(size * 0.20)
    yard_end = int(size * 0.70)
    gate_start = int(size * 0.75)
    gate_end = int(size * 0.85)

    # Quay zone: sparse obstacles (cranes, equipment)
    n_quay_obs = max(3, size // 25)
    for _ in range(n_quay_obs):
        r = rng.randint(2, quay_end - 2)
        c = rng.randint(2, size - 2)
        w = rng.randint(2, 6)
        h = rng.randint(1, 3)
        obstacles[max(0, r):min(size, r+h), max(0, c):min(size, c+w)] = True

    # Container yard: rectangular blocks with aisles
    block_h = rng.randint(6, 10)
    block_w = rng.randint(15, 25)
    aisle_h = rng.randint(3, 5)
    aisle_w = rng.randint(4, 6)

    row = yard_start
    while row + block_h < yard_end:
        col = aisle_w
        while col + block_w < size - aisle_w:
            # Place a container block
            obstacles[row:row+block_h, col:col+block_w] = True
            # Add risk around blocks (biosecurity inspection zones)
            r_start = max(0, row - 2)
            r_end = min(size, row + block_h + 2)
            c_start = max(0, col - 2)
            c_end = min(size, col + block_w + 2)
            risk[r_start:r_end, c_start:c_end] = np.maximum(
                risk[r_start:r_end, c_start:c_end],
                rng.uniform(0.3, 0.7)
            )
            col += block_w + aisle_w
        row += block_h + aisle_h

    # Gate zone: barriers with controlled openings
    obstacles[gate_start:gate_start+2, :] = True  # wall
    n_gates = max(3, size // 40)
    gate_positions = sorted(rng.choice(range(5, size-5), size=n_gates, replace=False))
    for gp in gate_positions:
        gate_width = rng.randint(3, 6)
        obstacles[gate_start:gate_start+2, gp:gp+gate_width] = False
        # High risk at gate checkpoints
        risk[gate_start-2:gate_start+4, gp:gp+gate_width] = rng.uniform(0.6, 0.9)

    # Buffer zone: random clutter
    buffer_obs = rng.random((size - gate_end, size)) < 0.08
    obstacles[gate_end:, :] = buffer_obs

    # Add hotspot risk zones (biosecurity hazards)
    n_hotspots = max(2, size // 60)
    for _ in range(n_hotspots):
        cy = rng.randint(yard_start, yard_end)
        cx = rng.randint(10, size - 10)
        sigma = rng.uniform(8, 20)
        yy, xx = np.ogrid[:size, :size]
        d = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        risk += 0.5 * np.exp(-d**2 / (2 * sigma**2))

    risk = np.clip(risk, 0, 1)

    # Ensure start/goal areas are clear
    obstacles[0, :5] = False
    obstacles[:5, 0] = False
    obstacles[size-1, size-5:] = False
    obstacles[size-5:, size-1] = False

    return obstacles, risk


# ============================================================
# CORE: Pathfinding Algorithms
# ============================================================

def heuristic_octile(r1, c1, r2, c2):
    """Octile distance heuristic for 8-connected grids."""
    dr = abs(r1 - r2)
    dc = abs(c1 - c2)
    return max(dr, dc) + (1.414 - 1) * min(dr, dc)


def astar(grid_map, start, goal, lam=0.0):
    """
    A* on 8-connected grid with optional risk weighting.
    cost(cell) = base_move_cost + lambda * risk(cell)
    Returns: (path, nodes_expanded, time_ms)
    """
    sr, sc = start
    gr, gc = goal
    t0 = time.perf_counter()

    open_list = []
    g_score = {}
    came_from = {}
    closed = set()
    nodes_expanded = 0

    g_score[(sr, sc)] = 0.0
    h0 = heuristic_octile(sr, sc, gr, gc)
    heapq.heappush(open_list, (h0, 0, sr, sc))
    counter = 1

    while open_list:
        f, _, r, c = heapq.heappop(open_list)

        if (r, c) in closed:
            continue
        closed.add((r, c))

        if (r, c) == (gr, gc):
            # Reconstruct path
            path = []
            cur = (gr, gc)
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            path.reverse()
            t1 = time.perf_counter()
            return path, nodes_expanded, (t1 - t0) * 1000

        nodes_expanded += 1

        for nr, nc, base_cost in grid_map.neighbors_8(r, c):
            if (nr, nc) in closed:
                continue
            risk_cost = lam * grid_map.risk[nr, nc] if (grid_map.risk is not None and lam > 0) else 0.0
            move_cost = base_cost + risk_cost
            ng = g_score[(r, c)] + move_cost
            if ng < g_score.get((nr, nc), float('inf')):
                g_score[(nr, nc)] = ng
                came_from[(nr, nc)] = (r, c)
                h = heuristic_octile(nr, nc, gr, gc)
                heapq.heappush(open_list, (ng + h, counter, nr, nc))
                counter += 1

    t1 = time.perf_counter()
    return None, nodes_expanded, (t1 - t0) * 1000


def bresenham_line(r0, c0, r1, c1):
    """Bresenham's line algorithm returning list of (r,c) cells."""
    cells = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


def build_corridor_mask(height, width, start, goal, corridor_width):
    """Build a boolean corridor mask around the Bresenham line."""
    mask = np.zeros((height, width), dtype=bool)
    line = bresenham_line(start[0], start[1], goal[0], goal[1])
    hw = corridor_width // 2
    for r, c in line:
        r_lo = max(0, r - hw)
        r_hi = min(height, r + hw + 1)
        c_lo = max(0, c - hw)
        c_hi = min(width, c + hw + 1)
        mask[r_lo:r_hi, c_lo:c_hi] = True
    return mask


def ils_astar(grid_map, start, goal, lam=0.0, initial_width_frac=0.05, max_attempts=10):
    """
    ILS A*: Corridor-constrained A* with incremental expansion.
    Returns: (path, nodes_expanded, time_ms, n_expansions)
    """
    sr, sc = start
    gr, gc = goal
    h, w = grid_map.height, grid_map.width
    diag = int(np.sqrt(h**2 + w**2))
    base_width = max(3, int(initial_width_frac * diag))

    t0 = time.perf_counter()
    total_nodes = 0

    for attempt in range(max_attempts):
        corridor_width = base_width + attempt * max(2, base_width // 2)
        mask = build_corridor_mask(h, w, start, goal, corridor_width)

        open_list = []
        g_score = {}
        came_from = {}
        closed = set()
        nodes_expanded = 0

        g_score[(sr, sc)] = 0.0
        h0 = heuristic_octile(sr, sc, gr, gc)
        heapq.heappush(open_list, (h0, 0, sr, sc))
        counter = 1

        found = False
        while open_list:
            f, _, r, c = heapq.heappop(open_list)

            if (r, c) in closed:
                continue
            closed.add((r, c))

            if (r, c) == (gr, gc):
                path = []
                cur = (gr, gc)
                while cur in came_from:
                    path.append(cur)
                    cur = came_from[cur]
                path.append(start)
                path.reverse()
                total_nodes += nodes_expanded
                t1 = time.perf_counter()
                return path, total_nodes, (t1 - t0) * 1000, attempt + 1

            nodes_expanded += 1

            for nr, nc, base_cost in grid_map.neighbors_8(r, c):
                if not mask[nr, nc] or (nr, nc) in closed:
                    continue
                risk_cost = lam * grid_map.risk[nr, nc] if (grid_map.risk is not None and lam > 0) else 0.0
                move_cost = base_cost + risk_cost
                ng = g_score[(r, c)] + move_cost
                if ng < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = ng
                    came_from[(nr, nc)] = (r, c)
                    hv = heuristic_octile(nr, nc, gr, gc)
                    heapq.heappush(open_list, (ng + hv, counter, nr, nc))
                    counter += 1

        total_nodes += nodes_expanded

    t1 = time.perf_counter()
    return None, total_nodes, (t1 - t0) * 1000, max_attempts


def compute_integral_image(grid):
    """Compute integral image (summed area table) for obstacle density queries."""
    return np.cumsum(np.cumsum(grid.astype(np.float64), axis=0), axis=1)


def query_density(integral, r, c, half_w, height, width):
    """O(1) density query using integral image."""
    r0 = max(0, r - half_w)
    c0 = max(0, c - half_w)
    r1 = min(height - 1, r + half_w)
    c1 = min(width - 1, c + half_w)

    total = integral[r1, c1]
    if r0 > 0:
        total -= integral[r0 - 1, c1]
    if c0 > 0:
        total -= integral[r1, c0 - 1]
    if r0 > 0 and c0 > 0:
        total += integral[r0 - 1, c0 - 1]

    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    return total / area if area > 0 else 0.0


def ails_astar(grid_map, start, goal, lam=0.0, r_min=2, r_max=None,
               alpha=1.0, omega=3, max_attempts=10):
    """
    AILS A*: Adaptive corridor-constrained A* with density-based width.
    Uses integral images for O(1) density queries.
    Returns: (path, nodes_expanded, time_ms, n_expansions)
    """
    sr, sc = start
    gr, gc = goal
    h, w = grid_map.height, grid_map.width
    if r_max is None:
        r_max = max(r_min + 1, int(0.1 * min(h, w)))

    t0 = time.perf_counter()

    # Build integral image for density estimation
    integral = compute_integral_image(grid_map.obstacles)
    line = bresenham_line(sr, sc, gr, gc)

    total_nodes = 0

    for attempt in range(max_attempts):
        expansion_bonus = attempt * max(1, r_min)

        # Build adaptive corridor mask
        mask = np.zeros((h, w), dtype=bool)
        for lr, lc in line:
            density = query_density(integral, lr, lc, omega, h, w)
            radius = int(r_min + (r_max - r_min) * (density ** alpha)) + expansion_bonus
            radius = max(r_min, min(radius, r_max + expansion_bonus))
            r_lo = max(0, lr - radius)
            r_hi = min(h, lr + radius + 1)
            c_lo = max(0, lc - radius)
            c_hi = min(w, lc + radius + 1)
            mask[r_lo:r_hi, c_lo:c_hi] = True

        open_list = []
        g_score = {}
        came_from = {}
        closed = set()
        nodes_expanded = 0

        g_score[(sr, sc)] = 0.0
        h0 = heuristic_octile(sr, sc, gr, gc)
        heapq.heappush(open_list, (h0, 0, sr, sc))
        counter = 1

        while open_list:
            f, _, r, c = heapq.heappop(open_list)

            if (r, c) in closed:
                continue
            closed.add((r, c))

            if (r, c) == (gr, gc):
                path = []
                cur = (gr, gc)
                while cur in came_from:
                    path.append(cur)
                    cur = came_from[cur]
                path.append(start)
                path.reverse()
                total_nodes += nodes_expanded
                t1 = time.perf_counter()
                return path, total_nodes, (t1 - t0) * 1000, attempt + 1

            nodes_expanded += 1

            for nr, nc, base_cost in grid_map.neighbors_8(r, c):
                if not mask[nr, nc] or (nr, nc) in closed:
                    continue
                risk_cost = lam * grid_map.risk[nr, nc] if (grid_map.risk is not None and lam > 0) else 0.0
                move_cost = base_cost + risk_cost
                ng = g_score[(r, c)] + move_cost
                if ng < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = ng
                    came_from[(nr, nc)] = (r, c)
                    hv = heuristic_octile(nr, nc, gr, gc)
                    heapq.heappush(open_list, (ng + hv, counter, nr, nc))
                    counter += 1

        total_nodes += nodes_expanded

    t1 = time.perf_counter()
    return None, total_nodes, (t1 - t0) * 1000, max_attempts


def jps_astar(grid_map, start, goal):
    """
    Jump Point Search on uniform-cost 8-connected grid.
    Returns: (path, nodes_expanded, time_ms)
    """
    sr, sc = start
    gr, gc = goal
    h, w = grid_map.height, grid_map.width
    t0 = time.perf_counter()

    def forced_neighbors(r, c, dr, dc):
        """Check for forced neighbors in the given direction."""
        forced = []
        if dr != 0 and dc != 0:
            # Diagonal movement
            if not grid_map.is_free(r - dr, c) and grid_map.is_free(r - dr, c + dc):
                forced.append((-dr, dc))
            if not grid_map.is_free(r, c - dc) and grid_map.is_free(r + dr, c - dc):
                forced.append((dr, -dc))
        elif dr != 0:
            # Vertical movement
            if not grid_map.is_free(r, c - 1) and grid_map.is_free(r + dr, c - 1):
                forced.append((dr, -1))
            if not grid_map.is_free(r, c + 1) and grid_map.is_free(r + dr, c + 1):
                forced.append((dr, 1))
        else:
            # Horizontal movement
            if not grid_map.is_free(r - 1, c) and grid_map.is_free(r - 1, c + dc):
                forced.append((-1, dc))
            if not grid_map.is_free(r + 1, c) and grid_map.is_free(r + 1, c + dc):
                forced.append((1, dc))
        return forced

    def jump(r, c, dr, dc, goal):
        nr, nc = r + dr, c + dc
        if not grid_map.is_free(nr, nc):
            return None
        if (nr, nc) == goal:
            return (nr, nc)

        # Check forced neighbors
        if forced_neighbors(nr, nc, dr, dc):
            return (nr, nc)

        # Diagonal: must check horizontal and vertical components
        if dr != 0 and dc != 0:
            if jump(nr, nc, dr, 0, goal) is not None:
                return (nr, nc)
            if jump(nr, nc, 0, dc, goal) is not None:
                return (nr, nc)

        # Continue jumping
        return jump(nr, nc, dr, dc, goal)

    sys.setrecursionlimit(max(10000, h * w // 10))

    open_list = []
    g_score = {}
    came_from = {}
    nodes_expanded = 0

    g_score[(sr, sc)] = 0.0
    h0 = heuristic_octile(sr, sc, gr, gc)
    heapq.heappush(open_list, (h0, 0, sr, sc))
    counter = 1

    while open_list:
        f, _, r, c = heapq.heappop(open_list)

        if (r, c) == (gr, gc):
            path = []
            cur = (gr, gc)
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            path.reverse()
            t1 = time.perf_counter()
            return path, nodes_expanded, (t1 - t0) * 1000

        if f > g_score.get((r, c), float('inf')) + heuristic_octile(r, c, gr, gc) + 1e-9:
            continue

        nodes_expanded += 1

        # Generate successors
        if (r, c) == start:
            # From start, try all 8 directions
            directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        else:
            pr, pc = came_from.get((r, c), (r, c))
            dr = 0 if r == pr else (1 if r > pr else -1)
            dc = 0 if c == pc else (1 if c > pc else -1)
            # Natural neighbors + forced neighbors
            directions = []
            if dr != 0 and dc != 0:
                # Diagonal: natural = diagonal, both cardinals
                directions = [(dr, dc), (dr, 0), (0, dc)]
                directions.extend(forced_neighbors(r, c, dr, dc))
            elif dr != 0:
                directions = [(dr, 0)]
                directions.extend(forced_neighbors(r, c, dr, dc))
            elif dc != 0:
                directions = [(0, dc)]
                directions.extend(forced_neighbors(r, c, dr, dc))
            else:
                directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

        for ddr, ddc in directions:
            jp = jump(r, c, ddr, ddc, goal)
            if jp is not None:
                jr, jc = jp
                dist = np.sqrt((jr - r)**2 + (jc - c)**2)
                ng = g_score[(r, c)] + dist
                if ng < g_score.get((jr, jc), float('inf')):
                    g_score[(jr, jc)] = ng
                    came_from[(jr, jc)] = (r, c)
                    hv = heuristic_octile(jr, jc, gr, gc)
                    heapq.heappush(open_list, (ng + hv, counter, jr, jc))
                    counter += 1

    t1 = time.perf_counter()
    return None, nodes_expanded, (t1 - t0) * 1000


def compute_path_cost(path, grid_map, lam=0.0):
    """Compute total cost of a path."""
    if path is None or len(path) < 2:
        return float('inf')
    cost = 0.0
    for i in range(1, len(path)):
        r1, c1 = path[i-1]
        r2, c2 = path[i]
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        base = 1.414 if (dr + dc == 2) else 1.0
        risk_cost = lam * grid_map.risk[r2, c2] if (grid_map.risk is not None and lam > 0) else 0.0
        cost += base + risk_cost
    return cost


def compute_exposure(path, grid_map):
    """Compute cumulative risk exposure along a path."""
    if path is None or grid_map.risk is None:
        return 0.0
    return sum(grid_map.risk[r, c] for r, c in path)


def path_length_euclidean(path):
    """Compute Euclidean path length."""
    if path is None or len(path) < 2:
        return float('inf')
    length = 0.0
    for i in range(1, len(path)):
        r1, c1 = path[i-1]
        r2, c2 = path[i]
        length += np.sqrt((r2-r1)**2 + (c2-c1)**2)
    return length


# ============================================================
# EXPERIMENT 1: Risk-Annotated Grid Experiments (DS6)
# ============================================================

def run_experiment_1():
    """
    DS6-style: Risk-annotated grid experiments.
    3 densities × 3 risk types × 6 lambda values × 100 maps = 5,400 paths
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Risk-Annotated Grid Experiments (DS6)")
    print("="*70)

    SIZE = 200
    DENSITIES = [0.10, 0.20, 0.30]
    RISK_TYPES = ['gradient', 'hotspot', 'uniform']
    LAMBDAS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    N_MAPS = 100
    START = (0, 0)
    GOAL = (SIZE-1, SIZE-1)

    results = []

    for density in DENSITIES:
        for risk_type in RISK_TYPES:
            for lam in LAMBDAS:
                print(f"\n  Density={density:.0%}, Risk={risk_type}, lambda={lam}")
                astar_times = []
                ils_times = []
                astar_nodes_list = []
                ils_nodes_list = []
                path_ratios = []
                exposure_ratios = []
                ils_expansions_list = []
                success_count = 0

                for m in range(N_MAPS):
                    seed = int(density * 1000 + hash(risk_type) % 1000 + lam * 100 + m)
                    obs = generate_random_grid(SIZE, density, seed=seed)
                    risk = generate_risk_layer(SIZE, risk_type, seed=seed + 10000)
                    gm = GridMap(SIZE, SIZE, obs, risk)

                    # Standard A*
                    path_a, nodes_a, time_a = astar(gm, START, GOAL, lam)
                    if path_a is None:
                        continue

                    # ILS A*
                    path_i, nodes_i, time_i, n_exp = ils_astar(gm, START, GOAL, lam)
                    if path_i is None:
                        continue

                    success_count += 1
                    astar_times.append(time_a)
                    ils_times.append(time_i)
                    astar_nodes_list.append(nodes_a)
                    ils_nodes_list.append(nodes_i)
                    ils_expansions_list.append(n_exp)

                    cost_a = compute_path_cost(path_a, gm, lam)
                    cost_i = compute_path_cost(path_i, gm, lam)
                    path_ratios.append(cost_i / cost_a if cost_a > 0 else 1.0)

                    exp_a = compute_exposure(path_a, gm)
                    exp_i = compute_exposure(path_i, gm)
                    exposure_ratios.append(exp_i / exp_a if exp_a > 0 else 1.0)

                if success_count < 5:
                    print(f"    Skipped (too few valid maps: {success_count})")
                    continue

                at = np.array(astar_times)
                it = np.array(ils_times)
                an = np.array(astar_nodes_list)
                iin = np.array(ils_nodes_list)

                speedup = np.mean(at) / np.mean(it) if np.mean(it) > 0 else 0
                node_red = (1 - np.mean(iin) / np.mean(an)) * 100 if np.mean(an) > 0 else 0
                mean_opt = np.mean(path_ratios)
                mean_exp = np.mean(exposure_ratios)
                mean_expansions = np.mean(ils_expansions_list)

                # Statistical test
                if len(at) > 1:
                    t_stat, p_val = stats.ttest_rel(an, iin)
                    d_val = np.mean(an - iin) / np.std(an - iin) if np.std(an - iin) > 0 else 0
                else:
                    t_stat, p_val, d_val = 0, 1, 0

                results.append({
                    'density': density,
                    'risk_type': risk_type,
                    'lambda': lam,
                    'n_valid': success_count,
                    'astar_time_ms': np.mean(at),
                    'ils_time_ms': np.mean(it),
                    'speedup': speedup,
                    'astar_nodes': np.mean(an),
                    'ils_nodes': np.mean(iin),
                    'node_reduction_pct': node_red,
                    'path_opt_ratio': mean_opt,
                    'exposure_ratio': mean_exp,
                    'mean_expansions': mean_expansions,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'cohens_d': d_val
                })

                print(f"    N={success_count}, Speedup={speedup:.2f}x, "
                      f"NodeRed={node_red:.1f}%, OptRatio={mean_opt:.4f}, "
                      f"ExpRatio={mean_exp:.4f}")

    # Save results
    outfile = os.path.join(RESULTS_DIR, 'exp1_risk_annotated.csv')
    with open(outfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Results saved to {outfile}")
    return results


# ============================================================
# EXPERIMENT 2: Port Environment Validation (DS7)
# ============================================================

def run_experiment_2():
    """
    DS7-style: Port environment validation.
    3 grid sizes × 3 lambda values × 30 maps = 270 paths
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Port Environment Validation (DS7)")
    print("="*70)

    SIZES = [200, 300, 500]
    LAMBDAS = [0.0, 0.5, 1.0]
    N_MAPS = 30

    results = []

    for size in SIZES:
        for lam in LAMBDAS:
            print(f"\n  Size={size}, lambda={lam}")
            astar_times = []
            ils_times = []
            ails_times = []
            astar_nodes_list = []
            ils_nodes_list = []
            ails_nodes_list = []
            ils_opt_list = []
            ails_opt_list = []
            success_count = 0

            START = (0, 0)
            GOAL = (size-1, size-1)

            for m in range(N_MAPS):
                seed = size * 1000 + int(lam * 100) + m
                obs, risk = generate_port_grid(size, seed=seed)
                gm = GridMap(size, size, obs, risk)

                # Standard A*
                path_a, nodes_a, time_a = astar(gm, START, GOAL, lam)
                if path_a is None:
                    continue

                # ILS A*
                path_i, nodes_i, time_i, _ = ils_astar(gm, START, GOAL, lam)
                if path_i is None:
                    continue

                # AILS A*
                path_ai, nodes_ai, time_ai, _ = ails_astar(gm, START, GOAL, lam)
                if path_ai is None:
                    continue

                success_count += 1
                astar_times.append(time_a)
                ils_times.append(time_i)
                ails_times.append(time_ai)
                astar_nodes_list.append(nodes_a)
                ils_nodes_list.append(nodes_i)
                ails_nodes_list.append(nodes_ai)

                cost_a = compute_path_cost(path_a, gm, lam)
                cost_i = compute_path_cost(path_i, gm, lam)
                cost_ai = compute_path_cost(path_ai, gm, lam)
                ils_opt_list.append(cost_i / cost_a if cost_a > 0 else 1.0)
                ails_opt_list.append(cost_ai / cost_a if cost_a > 0 else 1.0)

            if success_count < 5:
                print(f"    Skipped (too few valid maps: {success_count})")
                continue

            at = np.array(astar_times)
            it = np.array(ils_times)
            ait = np.array(ails_times)
            an = np.array(astar_nodes_list)
            iin = np.array(ils_nodes_list)
            ain = np.array(ails_nodes_list)

            results.append({
                'size': size,
                'lambda': lam,
                'n_valid': success_count,
                'astar_time_ms': np.mean(at),
                'ils_time_ms': np.mean(it),
                'ails_time_ms': np.mean(ait),
                'ils_speedup': np.mean(at) / np.mean(it) if np.mean(it) > 0 else 0,
                'ails_speedup': np.mean(at) / np.mean(ait) if np.mean(ait) > 0 else 0,
                'astar_nodes': np.mean(an),
                'ils_nodes': np.mean(iin),
                'ails_nodes': np.mean(ain),
                'ils_node_red_pct': (1 - np.mean(iin)/np.mean(an)) * 100,
                'ails_node_red_pct': (1 - np.mean(ain)/np.mean(an)) * 100,
                'ils_opt_ratio': np.mean(ils_opt_list),
                'ails_opt_ratio': np.mean(ails_opt_list),
            })

            print(f"    N={success_count}, "
                  f"ILS: {results[-1]['ils_speedup']:.2f}x / {results[-1]['ils_node_red_pct']:.1f}%, "
                  f"AILS: {results[-1]['ails_speedup']:.2f}x / {results[-1]['ails_node_red_pct']:.1f}%")

    outfile = os.path.join(RESULTS_DIR, 'exp2_port_environment.csv')
    with open(outfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Results saved to {outfile}")
    return results


# ============================================================
# EXPERIMENT 3: JPS Comparative Evaluation
# ============================================================

def run_experiment_3():
    """
    JPS comparison on uniform-cost grids.
    3 densities × 100 maps = 300 paths
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: JPS Comparative Evaluation (Uniform-Cost)")
    print("="*70)

    SIZE = 200
    DENSITIES = [0.10, 0.20, 0.30]
    N_MAPS = 100
    START = (0, 0)
    GOAL = (SIZE-1, SIZE-1)

    results = []

    for density in DENSITIES:
        print(f"\n  Density={density:.0%}")
        astar_times = []
        ils_times = []
        jps_times = []
        astar_nodes_list = []
        ils_nodes_list = []
        jps_nodes_list = []
        ils_opt_list = []
        success_count = 0

        for m in range(N_MAPS):
            seed = int(density * 10000 + m + 50000)
            obs = generate_random_grid(SIZE, density, seed=seed)
            gm = GridMap(SIZE, SIZE, obs, None)

            # Standard A*
            path_a, nodes_a, time_a = astar(gm, START, GOAL)
            if path_a is None:
                continue

            # ILS A*
            path_i, nodes_i, time_i, _ = ils_astar(gm, START, GOAL, lam=0.0,
                                                     initial_width_frac=0.03)
            if path_i is None:
                continue

            # JPS
            try:
                path_j, nodes_j, time_j = jps_astar(gm, START, GOAL)
                if path_j is None:
                    continue
            except RecursionError:
                continue

            success_count += 1
            astar_times.append(time_a)
            ils_times.append(time_i)
            jps_times.append(time_j)
            astar_nodes_list.append(nodes_a)
            ils_nodes_list.append(nodes_i)
            jps_nodes_list.append(nodes_j)

            len_a = path_length_euclidean(path_a)
            len_i = path_length_euclidean(path_i)
            ils_opt_list.append(len_i / len_a if len_a > 0 else 1.0)

        if success_count < 5:
            print(f"    Skipped (too few valid maps: {success_count})")
            continue

        at = np.array(astar_times)
        it = np.array(ils_times)
        jt = np.array(jps_times)
        an = np.array(astar_nodes_list)
        iin = np.array(ils_nodes_list)
        jn = np.array(jps_nodes_list)

        # Statistical tests
        t_ils, p_ils = stats.ttest_rel(an, iin) if len(an) > 1 else (0, 1)
        t_jps, p_jps = stats.ttest_rel(an, jn) if len(an) > 1 else (0, 1)

        d_ils = np.mean(an - iin) / np.std(an - iin) if np.std(an - iin) > 0 else 0
        d_jps = np.mean(an - jn) / np.std(an - jn) if np.std(an - jn) > 0 else 0

        results.append({
            'density': density,
            'n_valid': success_count,
            'astar_time_ms': np.mean(at),
            'ils_time_ms': np.mean(it),
            'jps_time_ms': np.mean(jt),
            'astar_nodes': np.mean(an),
            'ils_nodes': np.mean(iin),
            'jps_nodes': np.mean(jn),
            'ils_node_red_pct': (1 - np.mean(iin)/np.mean(an)) * 100,
            'jps_node_red_pct': (1 - np.mean(jn)/np.mean(an)) * 100,
            'ils_opt_ratio': np.mean(ils_opt_list),
            't_ils': t_ils,
            'p_ils': p_ils,
            'd_ils': d_ils,
            't_jps': t_jps,
            'p_jps': p_jps,
            'd_jps': d_jps,
        })

        print(f"    N={success_count}")
        print(f"    A*:  {np.mean(at):.2f}ms, {np.mean(an):.0f} nodes")
        print(f"    ILS: {np.mean(it):.2f}ms, {np.mean(iin):.0f} nodes ({results[-1]['ils_node_red_pct']:.1f}% red)")
        print(f"    JPS: {np.mean(jt):.2f}ms, {np.mean(jn):.0f} nodes ({results[-1]['jps_node_red_pct']:.1f}% red)")

    outfile = os.path.join(RESULTS_DIR, 'exp3_jps_comparison.csv')
    with open(outfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Results saved to {outfile}")
    return results


# ============================================================
# EXPERIMENT 4: D*Lite Re-planning Comparison
# ============================================================

def run_experiment_4():
    """
    Dynamic re-planning comparison: A* re-run vs AILS vs D*Lite-approximation.
    3 densities × 128 maps with 5 obstacle clusters inserted mid-path.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Dynamic Re-planning Comparison")
    print("="*70)

    SIZE = 200
    DENSITIES = [0.10, 0.20, 0.30]
    N_MAPS = 128
    START = (0, 0)
    GOAL = (SIZE-1, SIZE-1)
    N_CLUSTERS = 5
    CLUSTER_RADIUS = 3

    results = []

    for density in DENSITIES:
        print(f"\n  Density={density:.0%}")
        astar_replan_times = []
        ails_replan_times = []
        astar_replan_nodes = []
        ails_replan_nodes = []
        success_count = 0

        for m in range(N_MAPS):
            seed = int(density * 10000 + m + 80000)
            obs = generate_random_grid(SIZE, density, seed=seed)
            gm = GridMap(SIZE, SIZE, obs.copy(), None)

            # Initial path
            path_init, _, _ = astar(gm, START, GOAL)
            if path_init is None or len(path_init) < 20:
                continue

            # Insert obstacle clusters at ~40% along path
            insert_idx = int(len(path_init) * 0.4)
            rng = np.random.RandomState(seed + 99999)
            new_obs = obs.copy()

            for cl in range(N_CLUSTERS):
                if insert_idx + cl * 5 >= len(path_init):
                    break
                cr, cc = path_init[min(insert_idx + cl * 5, len(path_init)-1)]
                for dr in range(-CLUSTER_RADIUS, CLUSTER_RADIUS + 1):
                    for dc in range(-CLUSTER_RADIUS, CLUSTER_RADIUS + 1):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < SIZE and 0 <= nc < SIZE:
                            new_obs[nr, nc] = True

            # Ensure start/goal are still free
            new_obs[0, :3] = False
            new_obs[:3, 0] = False
            new_obs[SIZE-1, SIZE-3:] = False
            new_obs[SIZE-3:, SIZE-1] = False

            gm_new = GridMap(SIZE, SIZE, new_obs, None)

            # Re-plan start: current position (40% along path)
            replan_start = path_init[max(0, insert_idx - 5)]

            # A* full re-run
            path_ar, nodes_ar, time_ar = astar(gm_new, replan_start, GOAL)
            if path_ar is None:
                continue

            # AILS re-plan
            path_ai, nodes_ai, time_ai, _ = ails_astar(gm_new, replan_start, GOAL)
            if path_ai is None:
                continue

            success_count += 1
            astar_replan_times.append(time_ar)
            ails_replan_times.append(time_ai)
            astar_replan_nodes.append(nodes_ar)
            ails_replan_nodes.append(nodes_ai)

        if success_count < 5:
            print(f"    Skipped (too few valid cases: {success_count})")
            continue

        at = np.array(astar_replan_times)
        ait = np.array(ails_replan_times)
        an = np.array(astar_replan_nodes)
        ain = np.array(ails_replan_nodes)

        t_stat, p_val = stats.ttest_rel(an, ain) if len(an) > 1 else (0, 1)
        d_val = np.mean(an - ain) / np.std(an - ain) if np.std(an - ain) > 0 else 0

        results.append({
            'density': density,
            'n_valid': success_count,
            'astar_replan_time_ms': np.mean(at),
            'ails_replan_time_ms': np.mean(ait),
            'astar_replan_nodes': np.mean(an),
            'ails_replan_nodes': np.mean(ain),
            'node_reduction_pct': (1 - np.mean(ain)/np.mean(an)) * 100,
            'speedup': np.mean(at) / np.mean(ait) if np.mean(ait) > 0 else 0,
            't_statistic': t_stat,
            'p_value': p_val,
            'cohens_d': d_val,
        })

        print(f"    N={success_count}")
        print(f"    A* rerun:  {np.mean(at):.2f}ms, {np.mean(an):.0f} nodes")
        print(f"    AILS:      {np.mean(ait):.2f}ms, {np.mean(ain):.0f} nodes "
              f"({results[-1]['node_reduction_pct']:.1f}% red)")

    outfile = os.path.join(RESULTS_DIR, 'exp4_replanning.csv')
    with open(outfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Results saved to {outfile}")
    return results


# ============================================================
# EXPERIMENT 5: Progressive Obstacle Discovery
# ============================================================

def run_experiment_5():
    """
    Progressive obstacle discovery simulation.
    Agent discovers obstacles every 20 steps, re-plans from current position.
    3 densities × 60 missions
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: Progressive Obstacle Discovery Simulation")
    print("="*70)

    SIZE = 200
    DENSITIES = [0.10, 0.20, 0.30]
    N_MISSIONS = 60
    DISCOVERY_INTERVAL = 20
    SENSOR_RANGE = 15
    N_NEW_OBS = 3
    OBS_SIZE = 3
    MAX_REPLANS = 20

    results = []

    for density in DENSITIES:
        print(f"\n  Density={density:.0%}")

        astar_total_nodes_list = []
        ils_total_nodes_list = []
        astar_total_time_list = []
        ils_total_time_list = []
        astar_replans_list = []
        ils_replans_list = []
        astar_success = 0
        ils_success = 0
        path_ratio_list = []

        for m in range(N_MISSIONS):
            seed = int(density * 10000 + m + 200000)
            base_obs = generate_random_grid(SIZE, density, seed=seed)
            rng = np.random.RandomState(seed + 500000)

            START = (0, 0)
            GOAL = (SIZE-1, SIZE-1)

            for planner_name in ['astar', 'ils']:
                current_obs = base_obs.copy()
                gm = GridMap(SIZE, SIZE, current_obs, None)
                pos = START
                total_nodes = 0
                total_time = 0.0
                n_replans = 0
                total_path = [pos]
                mission_success = False

                for replan in range(MAX_REPLANS):
                    # Plan path
                    if planner_name == 'astar':
                        path, nodes, t_ms = astar(gm, pos, GOAL)
                    else:
                        path, nodes, t_ms, _ = ils_astar(gm, pos, GOAL)

                    total_nodes += nodes
                    total_time += t_ms
                    n_replans += 1

                    if path is None:
                        break

                    # Move along path for DISCOVERY_INTERVAL steps
                    steps_taken = 0
                    path_blocked = False
                    for i in range(1, len(path)):
                        r, c = path[i]
                        if current_obs[r, c]:
                            path_blocked = True
                            break
                        pos = (r, c)
                        total_path.append(pos)
                        steps_taken += 1
                        if pos == GOAL:
                            mission_success = True
                            break
                        if steps_taken >= DISCOVERY_INTERVAL:
                            break

                    if mission_success:
                        break

                    # Discover new obstacles ahead
                    for _ in range(N_NEW_OBS):
                        # Place obstacles near current position + forward
                        cr = pos[0] + rng.randint(0, SENSOR_RANGE)
                        cc = pos[1] + rng.randint(-SENSOR_RANGE//2, SENSOR_RANGE)
                        for dr in range(-OBS_SIZE//2, OBS_SIZE//2 + 1):
                            for dc in range(-OBS_SIZE//2, OBS_SIZE//2 + 1):
                                nr, nc = cr + dr, cc + dc
                                if 0 <= nr < SIZE and 0 <= nc < SIZE:
                                    current_obs[nr, nc] = True
                    # Keep start/goal clear
                    current_obs[0, :3] = False
                    current_obs[:3, 0] = False
                    current_obs[SIZE-1, SIZE-3:] = False
                    current_obs[SIZE-3:, SIZE-1] = False
                    gm = GridMap(SIZE, SIZE, current_obs, None)

                if planner_name == 'astar':
                    if mission_success:
                        astar_success += 1
                    astar_total_nodes_list.append(total_nodes)
                    astar_total_time_list.append(total_time)
                    astar_replans_list.append(n_replans)
                    astar_path_len = path_length_euclidean(total_path)
                else:
                    if mission_success:
                        ils_success += 1
                    ils_total_nodes_list.append(total_nodes)
                    ils_total_time_list.append(total_time)
                    ils_replans_list.append(n_replans)
                    ils_path_len = path_length_euclidean(total_path)

            if astar_path_len > 0 and astar_path_len < float('inf') and ils_path_len < float('inf'):
                path_ratio_list.append(ils_path_len / astar_path_len)

        an = np.array(astar_total_nodes_list)
        iin = np.array(ils_total_nodes_list)
        at = np.array(astar_total_time_list)
        it = np.array(ils_total_time_list)

        t_stat, p_val = stats.ttest_rel(an, iin) if len(an) > 1 else (0, 1)
        d_val = np.mean(an - iin) / np.std(an - iin) if np.std(an - iin) > 0 else 0

        results.append({
            'density': density,
            'n_missions': N_MISSIONS,
            'astar_success_rate': astar_success / N_MISSIONS,
            'ils_success_rate': ils_success / N_MISSIONS,
            'astar_mean_replans': np.mean(astar_replans_list),
            'ils_mean_replans': np.mean(ils_replans_list),
            'astar_mean_total_nodes': np.mean(an),
            'ils_mean_total_nodes': np.mean(iin),
            'node_reduction_pct': (1 - np.mean(iin)/np.mean(an)) * 100 if np.mean(an) > 0 else 0,
            'astar_mean_total_time_ms': np.mean(at),
            'ils_mean_total_time_ms': np.mean(it),
            'time_ratio': np.mean(it) / np.mean(at) if np.mean(at) > 0 else 0,
            'mean_path_ratio': np.mean(path_ratio_list) if path_ratio_list else 0,
            'max_path_ratio': np.max(path_ratio_list) if path_ratio_list else 0,
            't_statistic': t_stat,
            'p_value': p_val,
            'cohens_d': d_val,
        })

        print(f"    A* success: {astar_success}/{N_MISSIONS} ({astar_success/N_MISSIONS:.1%})")
        print(f"    ILS success: {ils_success}/{N_MISSIONS} ({ils_success/N_MISSIONS:.1%})")
        print(f"    A* nodes: {np.mean(an):.0f}, ILS nodes: {np.mean(iin):.0f} "
              f"({results[-1]['node_reduction_pct']:.1f}% red)")
        print(f"    Mean replans: A*={np.mean(astar_replans_list):.1f}, ILS={np.mean(ils_replans_list):.1f}")
        if path_ratio_list:
            print(f"    Path ratio: mean={np.mean(path_ratio_list):.4f}, max={np.max(path_ratio_list):.4f}")

    outfile = os.path.join(RESULTS_DIR, 'exp5_progressive.csv')
    with open(outfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Results saved to {outfile}")
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("COMPREHENSIVE EXPERIMENT SUITE")
    print("Risk-Aware Corridor-Constrained Pathfinding")
    print("for UAV Navigation in Biosecurity-Sensitive Environments")
    print("=" * 70)
    print(f"Results directory: {RESULTS_DIR}")

    all_results = {}

    # Run selected experiments based on command-line args
    experiments = {
        '1': ('Risk-Annotated Grids (DS6)', run_experiment_1),
        '2': ('Port Environment (DS7)', run_experiment_2),
        '3': ('JPS Comparison', run_experiment_3),
        '4': ('Re-planning Comparison', run_experiment_4),
        '5': ('Progressive Discovery', run_experiment_5),
    }

    if len(sys.argv) > 1:
        exp_ids = sys.argv[1:]
    else:
        exp_ids = ['1', '2', '3', '4', '5']

    for eid in exp_ids:
        if eid in experiments:
            name, func = experiments[eid]
            print(f"\n{'='*70}")
            print(f"Starting: {name}")
            print(f"{'='*70}")
            t0 = time.time()
            all_results[eid] = func()
            elapsed = time.time() - t0
            print(f"\n  Completed in {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    for eid in exp_ids:
        if eid in experiments:
            name, _ = experiments[eid]
            print(f"  Experiment {eid}: {name} - DONE")
    print(f"\nResults saved to: {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
