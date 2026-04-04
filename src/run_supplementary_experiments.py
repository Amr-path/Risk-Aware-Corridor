#!/usr/bin/env python3
"""
Supplementary Experiments for Revised Journal Paper:
"Risk-Aware Corridor-Constrained Pathfinding for UAV Navigation
 in Biosecurity-Sensitive Environments"

Addressing Reviewer Concerns:
  1b. Random start-goal replication of Exp 1 + wA* baseline
  4b. D*Lite re-planning comparison
  5b. D*Lite progressive discovery comparison
  6.  Weighted A* comparison across risk configurations
  7.  Corridor width sensitivity sweep
  8.  Risk-Responsive ILS (RILS) evaluation (new contribution)
  9.  AILS large-grid validation (500x500)

Usage:
  python3 run_supplementary_experiments.py          # run all
  python3 run_supplementary_experiments.py 1b 6 7   # run selected

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
from typing import Optional, Tuple, List, Set

# ============================================================
SEED = 42
np.random.seed(SEED)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# GRID MAP
# ============================================================
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

# ============================================================
# GRID GENERATION
# ============================================================
def generate_random_grid(size, density, seed=None):
    rng = np.random.RandomState(seed)
    grid = rng.random((size, size)) < density
    grid[0, :3] = False; grid[:3, 0] = False
    grid[size-1, size-3:] = False; grid[size-3:, size-1] = False
    grid[0, :] = False; grid[:, size-1] = False
    return grid

def generate_risk_layer(size, risk_type, seed=None):
    rng = np.random.RandomState(seed)
    if risk_type == 'gradient':
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
    rng = np.random.RandomState(seed)
    obstacles = np.zeros((size, size), dtype=bool)
    risk = np.zeros((size, size), dtype=float)
    quay_end = int(size * 0.15)
    yard_start = int(size * 0.20); yard_end = int(size * 0.70)
    gate_start = int(size * 0.75); gate_end = int(size * 0.85)
    n_quay = max(3, size // 25)
    for _ in range(n_quay):
        r = rng.randint(2, quay_end - 2); c = rng.randint(2, size - 2)
        w = rng.randint(2, 6); h = rng.randint(1, 3)
        obstacles[max(0,r):min(size,r+h), max(0,c):min(size,c+w)] = True
    block_h = rng.randint(6, 10); block_w = rng.randint(15, 25)
    aisle_h = rng.randint(3, 5); aisle_w = rng.randint(4, 6)
    row = yard_start
    while row + block_h < yard_end:
        col = aisle_w
        while col + block_w < size - aisle_w:
            obstacles[row:row+block_h, col:col+block_w] = True
            rs, re = max(0, row-2), min(size, row+block_h+2)
            cs, ce = max(0, col-2), min(size, col+block_w+2)
            risk[rs:re, cs:ce] = np.maximum(risk[rs:re, cs:ce], rng.uniform(0.3, 0.7))
            col += block_w + aisle_w
        row += block_h + aisle_h
    obstacles[gate_start:gate_start+2, :] = True
    n_gates = max(3, size // 40)
    gate_pos = sorted(rng.choice(range(5, size-5), size=n_gates, replace=False))
    for gp in gate_pos:
        gw = rng.randint(3, 6)
        obstacles[gate_start:gate_start+2, gp:gp+gw] = False
        risk[gate_start-2:gate_start+4, gp:gp+gw] = rng.uniform(0.6, 0.9)
    buffer_obs = rng.random((size - gate_end, size)) < 0.08
    obstacles[gate_end:, :] = buffer_obs
    n_hs = max(2, size // 60)
    for _ in range(n_hs):
        cy = rng.randint(yard_start, yard_end); cx = rng.randint(10, size-10)
        sigma = rng.uniform(8, 20)
        yy, xx = np.ogrid[:size, :size]
        d = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        risk += 0.5 * np.exp(-d**2 / (2 * sigma**2))
    risk = np.clip(risk, 0, 1)
    obstacles[0, :5] = False; obstacles[:5, 0] = False
    obstacles[size-1, size-5:] = False; obstacles[size-5:, size-1] = False
    return obstacles, risk

def generate_random_endpoints(size, obstacles, rng, min_dist_frac=0.3):
    """Generate random start-goal pair with minimum separation."""
    min_dist = int(size * min_dist_frac)
    for _ in range(1000):
        sr, sc = rng.randint(0, size, 2)
        gr, gc = rng.randint(0, size, 2)
        if obstacles[sr, sc] or obstacles[gr, gc]:
            continue
        dist = abs(sr - gr) + abs(sc - gc)
        if dist >= min_dist:
            return (int(sr), int(sc)), (int(gr), int(gc))
    # Fallback: diagonal corners
    return (0, 0), (size-1, size-1)

# ============================================================
# CORE ALGORITHMS
# ============================================================

def heuristic_octile(r1, c1, r2, c2):
    dr = abs(r1 - r2); dc = abs(c1 - c2)
    return max(dr, dc) + (1.414 - 1) * min(dr, dc)

def astar(grid_map, start, goal, lam=0.0):
    sr, sc = start; gr, gc = goal
    t0 = time.perf_counter()
    open_list = []; g_score = {}; came_from = {}; closed = set()
    nodes_expanded = 0
    g_score[(sr, sc)] = 0.0
    h0 = heuristic_octile(sr, sc, gr, gc)
    heapq.heappush(open_list, (h0, 0, sr, sc)); counter = 1

    while open_list:
        f, _, r, c = heapq.heappop(open_list)
        if (r, c) in closed: continue
        closed.add((r, c))
        if (r, c) == (gr, gc):
            path = []; cur = (gr, gc)
            while cur in came_from: path.append(cur); cur = came_from[cur]
            path.append(start); path.reverse()
            return path, nodes_expanded, (time.perf_counter() - t0) * 1000
        nodes_expanded += 1
        for nr, nc, base_cost in grid_map.neighbors_8(r, c):
            if (nr, nc) in closed: continue
            risk_cost = lam * grid_map.risk[nr, nc] if (grid_map.risk is not None and lam > 0) else 0.0
            ng = g_score[(r, c)] + base_cost + risk_cost
            if ng < g_score.get((nr, nc), float('inf')):
                g_score[(nr, nc)] = ng; came_from[(nr, nc)] = (r, c)
                h = heuristic_octile(nr, nc, gr, gc)
                heapq.heappush(open_list, (ng + h, counter, nr, nc)); counter += 1
    return None, nodes_expanded, (time.perf_counter() - t0) * 1000


def weighted_astar(grid_map, start, goal, lam=0.0, w=1.5):
    """Weighted A*: f(n) = g(n) + w * h(n). Returns w-bounded suboptimal path."""
    sr, sc = start; gr, gc = goal
    t0 = time.perf_counter()
    open_list = []; g_score = {}; came_from = {}; closed = set()
    nodes_expanded = 0
    g_score[(sr, sc)] = 0.0
    h0 = w * heuristic_octile(sr, sc, gr, gc)
    heapq.heappush(open_list, (h0, 0, sr, sc)); counter = 1

    while open_list:
        f, _, r, c = heapq.heappop(open_list)
        if (r, c) in closed: continue
        closed.add((r, c))
        if (r, c) == (gr, gc):
            path = []; cur = (gr, gc)
            while cur in came_from: path.append(cur); cur = came_from[cur]
            path.append(start); path.reverse()
            return path, nodes_expanded, (time.perf_counter() - t0) * 1000
        nodes_expanded += 1
        for nr, nc, base_cost in grid_map.neighbors_8(r, c):
            if (nr, nc) in closed: continue
            risk_cost = lam * grid_map.risk[nr, nc] if (grid_map.risk is not None and lam > 0) else 0.0
            ng = g_score[(r, c)] + base_cost + risk_cost
            if ng < g_score.get((nr, nc), float('inf')):
                g_score[(nr, nc)] = ng; came_from[(nr, nc)] = (r, c)
                h = w * heuristic_octile(nr, nc, gr, gc)
                heapq.heappush(open_list, (ng + h, counter, nr, nc)); counter += 1
    return None, nodes_expanded, (time.perf_counter() - t0) * 1000


def bresenham_line(r0, c0, r1, c1):
    cells = []; dr = abs(r1 - r0); dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1; sc = 1 if c0 < c1 else -1
    err = dr - dc; r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1: break
        e2 = 2 * err
        if e2 > -dc: err -= dc; r += sr
        if e2 < dr: err += dr; c += sc
    return cells

def build_corridor_mask(height, width, start, goal, corridor_width):
    mask = np.zeros((height, width), dtype=bool)
    line = bresenham_line(start[0], start[1], goal[0], goal[1])
    hw = corridor_width // 2
    for r, c in line:
        r_lo, r_hi = max(0, r-hw), min(height, r+hw+1)
        c_lo, c_hi = max(0, c-hw), min(width, c+hw+1)
        mask[r_lo:r_hi, c_lo:c_hi] = True
    return mask

def ils_astar(grid_map, start, goal, lam=0.0, initial_width_frac=0.05, max_attempts=10):
    sr, sc = start; gr, gc = goal
    h, w = grid_map.height, grid_map.width
    diag = int(np.sqrt(h**2 + w**2))
    base_width = max(3, int(initial_width_frac * diag))
    t0 = time.perf_counter(); total_nodes = 0

    for attempt in range(max_attempts):
        corridor_width = base_width + attempt * max(2, base_width // 2)
        mask = build_corridor_mask(h, w, start, goal, corridor_width)
        open_list = []; g_score = {}; came_from = {}; closed = set()
        nodes_expanded = 0
        g_score[(sr, sc)] = 0.0
        h0 = heuristic_octile(sr, sc, gr, gc)
        heapq.heappush(open_list, (h0, 0, sr, sc)); counter = 1

        while open_list:
            f, _, r, c = heapq.heappop(open_list)
            if (r, c) in closed: continue
            closed.add((r, c))
            if (r, c) == (gr, gc):
                path = []; cur = (gr, gc)
                while cur in came_from: path.append(cur); cur = came_from[cur]
                path.append(start); path.reverse()
                total_nodes += nodes_expanded
                return path, total_nodes, (time.perf_counter() - t0)*1000, attempt+1
            nodes_expanded += 1
            for nr, nc, base_cost in grid_map.neighbors_8(r, c):
                if not mask[nr, nc] or (nr, nc) in closed: continue
                risk_cost = lam * grid_map.risk[nr, nc] if (grid_map.risk is not None and lam > 0) else 0.0
                ng = g_score[(r, c)] + base_cost + risk_cost
                if ng < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = ng; came_from[(nr, nc)] = (r, c)
                    hv = heuristic_octile(nr, nc, gr, gc)
                    heapq.heappush(open_list, (ng + hv, counter, nr, nc)); counter += 1
        total_nodes += nodes_expanded
    return None, total_nodes, (time.perf_counter() - t0)*1000, max_attempts

def compute_integral_image(arr):
    return np.cumsum(np.cumsum(arr.astype(np.float64), axis=0), axis=1)

def query_density(integral, r, c, half_w, height, width):
    r0 = max(0, r - half_w); c0 = max(0, c - half_w)
    r1 = min(height - 1, r + half_w); c1 = min(width - 1, c + half_w)
    total = integral[r1, c1]
    if r0 > 0: total -= integral[r0-1, c1]
    if c0 > 0: total -= integral[r1, c0-1]
    if r0 > 0 and c0 > 0: total += integral[r0-1, c0-1]
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    return total / area if area > 0 else 0.0

def ails_astar(grid_map, start, goal, lam=0.0, r_min=2, r_max=None,
               alpha=1.0, omega=3, max_attempts=10):
    sr, sc = start; gr, gc = goal
    h, w = grid_map.height, grid_map.width
    if r_max is None: r_max = max(r_min + 1, int(0.1 * min(h, w)))
    t0 = time.perf_counter()
    integral = compute_integral_image(grid_map.obstacles)
    line = bresenham_line(sr, sc, gr, gc)
    total_nodes = 0

    for attempt in range(max_attempts):
        expansion_bonus = attempt * max(1, r_min)
        mask = np.zeros((h, w), dtype=bool)
        for lr, lc in line:
            density = query_density(integral, lr, lc, omega, h, w)
            radius = int(r_min + (r_max - r_min) * (density ** alpha)) + expansion_bonus
            radius = max(r_min, min(radius, r_max + expansion_bonus))
            mask[max(0,lr-radius):min(h,lr+radius+1), max(0,lc-radius):min(w,lc+radius+1)] = True

        open_list = []; g_score = {}; came_from = {}; closed = set()
        nodes_expanded = 0
        g_score[(sr, sc)] = 0.0; h0 = heuristic_octile(sr, sc, gr, gc)
        heapq.heappush(open_list, (h0, 0, sr, sc)); counter = 1
        while open_list:
            f, _, r, c = heapq.heappop(open_list)
            if (r, c) in closed: continue
            closed.add((r, c))
            if (r, c) == (gr, gc):
                path = []; cur = (gr, gc)
                while cur in came_from: path.append(cur); cur = came_from[cur]
                path.append(start); path.reverse()
                total_nodes += nodes_expanded
                return path, total_nodes, (time.perf_counter()-t0)*1000, attempt+1
            nodes_expanded += 1
            for nr, nc, base_cost in grid_map.neighbors_8(r, c):
                if not mask[nr, nc] or (nr, nc) in closed: continue
                risk_cost = lam * grid_map.risk[nr, nc] if (grid_map.risk is not None and lam > 0) else 0.0
                ng = g_score[(r, c)] + base_cost + risk_cost
                if ng < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = ng; came_from[(nr, nc)] = (r, c)
                    hv = heuristic_octile(nr, nc, gr, gc)
                    heapq.heappush(open_list, (ng + hv, counter, nr, nc)); counter += 1
        total_nodes += nodes_expanded
    return None, total_nodes, (time.perf_counter()-t0)*1000, max_attempts


# ============================================================
# NEW ALGORITHM: Risk-Responsive ILS (RILS)
# ============================================================
def rils_astar(grid_map, start, goal, lam=0.0, r_base_frac=0.05,
               r_max_frac=0.15, beta=1.0, omega=5, max_attempts=10):
    """
    Risk-Responsive ILS: corridor width adapts to LOCAL RISK level.
    - High risk areas -> wider corridor (allows risk-avoidance detours)
    - Low risk areas  -> narrow corridor (maximum search-space pruning)

    Distinct from AILS (adapts to obstacle DENSITY).
    RILS adapts to the RISK DISTRIBUTION.

    Parameters:
        r_base_frac: base corridor width as fraction of grid diagonal
        r_max_frac:  max corridor width as fraction of grid diagonal
        beta:        risk-sensitivity exponent (higher = more responsive)
        omega:       risk query window half-size
    """
    sr, sc = start; gr, gc = goal
    h, w = grid_map.height, grid_map.width
    diag = int(np.sqrt(h**2 + w**2))
    r_base = max(3, int(r_base_frac * diag))
    r_max = max(r_base + 1, int(r_max_frac * diag))

    # If no risk layer, fall back to standard ILS
    if grid_map.risk is None or lam <= 0:
        return ils_astar(grid_map, start, goal, lam, r_base_frac, max_attempts)

    t0 = time.perf_counter()
    risk_integral = compute_integral_image(grid_map.risk)
    line = bresenham_line(sr, sc, gr, gc)
    total_nodes = 0

    for attempt in range(max_attempts):
        expansion_bonus = attempt * max(1, r_base // 2)

        # Build risk-responsive corridor mask
        mask = np.zeros((h, w), dtype=bool)
        for lr, lc in line:
            mean_risk = query_density(risk_integral, lr, lc, omega, h, w)
            # Scale: wider where risk is high (need room for detours)
            # Narrower where risk is low (safe to prune aggressively)
            radius = int(r_base + (r_max - r_base) * (mean_risk ** beta)) + expansion_bonus
            radius = max(r_base, min(radius, r_max + expansion_bonus))
            mask[max(0,lr-radius):min(h,lr+radius+1),
                 max(0,lc-radius):min(w,lc+radius+1)] = True

        open_list = []; g_score = {}; came_from = {}; closed = set()
        nodes_expanded = 0
        g_score[(sr, sc)] = 0.0
        h0 = heuristic_octile(sr, sc, gr, gc)
        heapq.heappush(open_list, (h0, 0, sr, sc)); counter = 1

        while open_list:
            f, _, r, c = heapq.heappop(open_list)
            if (r, c) in closed: continue
            closed.add((r, c))
            if (r, c) == (gr, gc):
                path = []; cur = (gr, gc)
                while cur in came_from: path.append(cur); cur = came_from[cur]
                path.append(start); path.reverse()
                total_nodes += nodes_expanded
                return path, total_nodes, (time.perf_counter()-t0)*1000, attempt+1
            nodes_expanded += 1
            for nr, nc, base_cost in grid_map.neighbors_8(r, c):
                if not mask[nr, nc] or (nr, nc) in closed: continue
                risk_cost = lam * grid_map.risk[nr, nc]
                ng = g_score[(r, c)] + base_cost + risk_cost
                if ng < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = ng; came_from[(nr, nc)] = (r, c)
                    hv = heuristic_octile(nr, nc, gr, gc)
                    heapq.heappush(open_list, (ng + hv, counter, nr, nc)); counter += 1
        total_nodes += nodes_expanded

    return None, total_nodes, (time.perf_counter()-t0)*1000, max_attempts


# ============================================================
# NEW ALGORITHM: D* Lite
# ============================================================
class DStarLite:
    """
    D* Lite implementation following Koenig & Likhachev (2002).
    Searches backward from goal to start. Supports incremental
    re-planning after map changes.
    """
    INF = float('inf')

    def __init__(self, grid_map, start, goal, lam=0.0):
        self.grid = grid_map
        self.s_start = start
        self.s_goal = goal
        self.s_last = start
        self.lam = lam
        self.km = 0
        self.g = {}
        self.rhs = {}
        self.heap = []
        self.in_queue = {}   # node -> (k1, k2) if active in queue
        self._counter = 0
        self.nodes_expanded = 0

    def _edge_cost(self, u, v):
        """Cost of traversing from u to v on the current grid."""
        r2, c2 = v
        if not self.grid.is_free(r2, c2):
            return self.INF
        r1, c1 = u
        dr, dc = abs(r2 - r1), abs(c2 - c1)
        base = 1.414 if (dr + dc == 2) else 1.0
        risk_cost = 0.0
        if self.grid.risk is not None and self.lam > 0:
            risk_cost = self.lam * self.grid.risk[r2, c2]
        return base + risk_cost

    def _h(self, s):
        """Heuristic: distance from s_start to s (used in key calculation)."""
        return heuristic_octile(self.s_start[0], self.s_start[1], s[0], s[1])

    def _get_g(self, s):
        return self.g.get(s, self.INF)

    def _get_rhs(self, s):
        return self.rhs.get(s, self.INF)

    def _calculate_key(self, s):
        min_val = min(self._get_g(s), self._get_rhs(s))
        return (min_val + self._h(s) + self.km, min_val)

    def _queue_insert(self, s, key):
        self._counter += 1
        heapq.heappush(self.heap, (key[0], key[1], self._counter, s[0], s[1]))
        self.in_queue[s] = key

    def _queue_remove(self, s):
        if s in self.in_queue:
            del self.in_queue[s]

    def _queue_top_key(self):
        while self.heap:
            k1, k2, _, r, c = self.heap[0]
            s = (r, c)
            if s in self.in_queue and self.in_queue[s] == (k1, k2):
                return (k1, k2)
            heapq.heappop(self.heap)
        return (self.INF, self.INF)

    def _queue_pop(self):
        while self.heap:
            k1, k2, _, r, c = self.heap[0]
            s = (r, c)
            heapq.heappop(self.heap)
            if s in self.in_queue and self.in_queue[s] == (k1, k2):
                del self.in_queue[s]
                return s, (k1, k2)
        return None, (self.INF, self.INF)

    def _successors(self, s):
        """Get traversable neighbors of s on current grid."""
        result = []
        r, c = s
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid.height and 0 <= nc < self.grid.width:
                result.append((nr, nc))
        return result

    def _update_vertex(self, u):
        if u != self.s_goal:
            min_rhs = self.INF
            for s_prime in self._successors(u):
                cost = self._edge_cost(u, s_prime)
                if cost < self.INF:
                    val = cost + self._get_g(s_prime)
                    if val < min_rhs:
                        min_rhs = val
            self.rhs[u] = min_rhs
        self._queue_remove(u)
        if self._get_g(u) != self._get_rhs(u):
            self._queue_insert(u, self._calculate_key(u))

    def initialize(self):
        """Initialize D* Lite: set goal rhs=0, insert into queue."""
        self.rhs[self.s_goal] = 0
        self._queue_insert(self.s_goal, self._calculate_key(self.s_goal))

    def compute_shortest_path(self):
        """Main D* Lite planning loop. Returns nodes expanded in this call."""
        expanded = 0
        max_iters = self.grid.height * self.grid.width * 2  # safety limit
        iters = 0

        while iters < max_iters:
            iters += 1
            top_key = self._queue_top_key()
            start_key = self._calculate_key(self.s_start)

            if top_key >= start_key and self._get_rhs(self.s_start) == self._get_g(self.s_start):
                break

            u, k_old = self._queue_pop()
            if u is None:
                break

            k_new = self._calculate_key(u)

            if k_old < k_new:
                self._queue_insert(u, k_new)
            elif self._get_g(u) > self._get_rhs(u):
                self.g[u] = self.rhs[u]
                expanded += 1
                for s in self._successors(u):
                    self._update_vertex(s)
            else:
                self.g[u] = self.INF
                self._update_vertex(u)
                expanded += 1
                for s in self._successors(u):
                    self._update_vertex(s)

        self.nodes_expanded += expanded
        return expanded

    def get_path(self):
        """Extract path from s_start to s_goal using g-values."""
        if self._get_g(self.s_start) >= self.INF:
            return None
        path = [self.s_start]
        current = self.s_start
        visited = set()
        max_len = self.grid.height * self.grid.width

        while current != self.s_goal and len(path) < max_len:
            if current in visited:
                return None
            visited.add(current)
            best_next = None
            best_cost = self.INF
            for s_prime in self._successors(current):
                cost = self._edge_cost(current, s_prime)
                if cost >= self.INF:
                    continue
                val = cost + self._get_g(s_prime)
                if val < best_cost:
                    best_cost = val
                    best_next = s_prime
            if best_next is None:
                return None
            path.append(best_next)
            current = best_next
        return path if current == self.s_goal else None

    def update_map(self, old_obstacles, new_obstacles):
        """
        Notify D* Lite of obstacle changes. Call before compute_shortest_path().
        Updates km and affected vertices.
        """
        self.km += heuristic_octile(self.s_last[0], self.s_last[1],
                                    self.s_start[0], self.s_start[1])
        self.s_last = self.s_start

        # Find changed cells
        changed = np.argwhere(old_obstacles != new_obstacles)

        affected = set()
        for idx in range(len(changed)):
            r, c = int(changed[idx, 0]), int(changed[idx, 1])
            affected.add((r, c))
            for s in self._successors((r, c)):
                affected.add(s)

        for u in affected:
            self._update_vertex(u)

    def move_start(self, new_start):
        """Move the start position (agent moved)."""
        self.s_start = new_start


# ============================================================
# UTILITIES
# ============================================================
def compute_path_cost(path, grid_map, lam=0.0):
    if path is None or len(path) < 2: return float('inf')
    cost = 0.0
    for i in range(1, len(path)):
        r1, c1 = path[i-1]; r2, c2 = path[i]
        dr, dc = abs(r2-r1), abs(c2-c1)
        base = 1.414 if (dr+dc == 2) else 1.0
        risk_cost = lam * grid_map.risk[r2, c2] if (grid_map.risk is not None and lam > 0) else 0.0
        cost += base + risk_cost
    return cost

def compute_exposure(path, grid_map):
    if path is None or grid_map.risk is None: return 0.0
    return sum(grid_map.risk[r, c] for r, c in path)

def path_length_euclidean(path):
    if path is None or len(path) < 2: return float('inf')
    return sum(np.sqrt((path[i][0]-path[i-1][0])**2 + (path[i][1]-path[i-1][1])**2)
               for i in range(1, len(path)))

def ci_95(data):
    """Return (mean, std, ci_low, ci_high) for 95% confidence interval."""
    n = len(data)
    if n < 2: return np.mean(data), 0.0, np.mean(data), np.mean(data)
    m = np.mean(data); s = np.std(data, ddof=1)
    se = s / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, n - 1)
    return m, s, m - t_crit * se, m + t_crit * se


# ============================================================
# EXP 1b: Random Start-Goal Risk-Annotated Grids + wA*
# ============================================================
def run_experiment_1b():
    """
    Replicate Experiment 1 with random start-goal pairs and wA* baseline.
    3 densities x 3 risk types x 3 lambdas x 50 maps = 1,350 paths
    Algorithms: A*, wA*(2.0), ILS
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1b: Random Start-Goal + wA* Baseline")
    print("="*70)

    SIZE = 200
    DENSITIES = [0.10, 0.20, 0.30]
    RISK_TYPES = ['gradient', 'hotspot', 'uniform']
    LAMBDAS = [0.5, 1.0, 2.0]
    N_MAPS = 50
    W_WEIGHT = 2.0  # wA* weight

    results = []
    for density in DENSITIES:
        for risk_type in RISK_TYPES:
            for lam in LAMBDAS:
                print(f"  d={density:.0%}, risk={risk_type}, lam={lam}")
                astar_times, ils_times, wa_times = [], [], []
                astar_nodes_l, ils_nodes_l, wa_nodes_l = [], [], []
                ils_opt_l, wa_opt_l = [], []
                success = 0

                for m in range(N_MAPS):
                    seed = int(density*1000 + hash(risk_type)%1000 + lam*100 + m + 900000)
                    obs = generate_random_grid(SIZE, density, seed=seed)
                    risk = generate_risk_layer(SIZE, risk_type, seed=seed+10000)
                    gm = GridMap(SIZE, SIZE, obs, risk)
                    rng = np.random.RandomState(seed + 777)
                    start, goal = generate_random_endpoints(SIZE, obs, rng)

                    pa, na, ta = astar(gm, start, goal, lam)
                    if pa is None: continue
                    pi, ni, ti, _ = ils_astar(gm, start, goal, lam)
                    if pi is None: continue
                    pw, nw, tw = weighted_astar(gm, start, goal, lam, w=W_WEIGHT)
                    if pw is None: continue

                    success += 1
                    astar_times.append(ta); ils_times.append(ti); wa_times.append(tw)
                    astar_nodes_l.append(na); ils_nodes_l.append(ni); wa_nodes_l.append(nw)

                    ca = compute_path_cost(pa, gm, lam)
                    ci = compute_path_cost(pi, gm, lam)
                    cw = compute_path_cost(pw, gm, lam)
                    ils_opt_l.append(ci/ca if ca > 0 else 1.0)
                    wa_opt_l.append(cw/ca if ca > 0 else 1.0)

                if success < 5: continue
                at = np.array(astar_times); it = np.array(ils_times); wt = np.array(wa_times)
                an = np.array(astar_nodes_l); inn = np.array(ils_nodes_l); wn = np.array(wa_nodes_l)

                at_m, at_s, at_cl, at_ch = ci_95(at)
                it_m, it_s, it_cl, it_ch = ci_95(it)
                wt_m, wt_s, wt_cl, wt_ch = ci_95(wt)
                an_m, an_s, _, _ = ci_95(an)
                inn_m, inn_s, _, _ = ci_95(inn)
                wn_m, wn_s, _, _ = ci_95(wn)

                results.append({
                    'density': density, 'risk_type': risk_type, 'lambda': lam,
                    'n_valid': success,
                    'astar_time_mean': at_m, 'astar_time_std': at_s,
                    'ils_time_mean': it_m, 'ils_time_std': it_s,
                    'wa_time_mean': wt_m, 'wa_time_std': wt_s,
                    'ils_speedup': at_m/it_m if it_m > 0 else 0,
                    'wa_speedup': at_m/wt_m if wt_m > 0 else 0,
                    'astar_nodes_mean': an_m, 'astar_nodes_std': an_s,
                    'ils_nodes_mean': inn_m, 'ils_nodes_std': inn_s,
                    'wa_nodes_mean': wn_m, 'wa_nodes_std': wn_s,
                    'ils_node_red_pct': (1 - inn_m/an_m)*100 if an_m > 0 else 0,
                    'wa_node_red_pct': (1 - wn_m/an_m)*100 if an_m > 0 else 0,
                    'ils_opt_mean': np.mean(ils_opt_l), 'ils_opt_std': np.std(ils_opt_l),
                    'wa_opt_mean': np.mean(wa_opt_l), 'wa_opt_std': np.std(wa_opt_l),
                })
                print(f"    N={success}, ILS={results[-1]['ils_speedup']:.2f}x, "
                      f"wA*={results[-1]['wa_speedup']:.2f}x")

    outfile = os.path.join(RESULTS_DIR, 'exp1b_random_startgoal.csv')
    if results:
        with open(outfile, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader(); writer.writerows(results)
    print(f"  Saved to {outfile}")
    return results


# ============================================================
# EXP 4b: D*Lite Re-planning Comparison
# ============================================================
def run_experiment_4b():
    """
    Re-planning with D*Lite baseline.
    Algorithms: A* re-run, D*Lite incremental, ILS re-plan
    3 densities x 100 maps = 300 cases
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4b: D*Lite Re-planning Comparison")
    print("="*70)

    SIZE = 200
    DENSITIES = [0.10, 0.20, 0.30]
    N_MAPS = 100
    N_CLUSTERS = 5; CLUSTER_RADIUS = 3

    results = []
    for density in DENSITIES:
        print(f"  Density={density:.0%}")
        astar_t, dstar_t, ils_t = [], [], []
        astar_n, dstar_n, ils_n = [], [], []
        astar_opt, dstar_opt, ils_opt = [], [], []
        success = 0

        for m in range(N_MAPS):
            seed = int(density*10000 + m + 180000)
            obs = generate_random_grid(SIZE, density, seed=seed)
            gm = GridMap(SIZE, SIZE, obs.copy(), None)
            START = (0, 0); GOAL = (SIZE-1, SIZE-1)

            # Initial path (all planners start from same state)
            path_init, _, _ = astar(gm, START, GOAL)
            if path_init is None or len(path_init) < 20: continue

            # Initialize D*Lite on original grid
            dsl = DStarLite(gm, START, GOAL)
            dsl.initialize()
            dsl.compute_shortest_path()
            dsl_init_path = dsl.get_path()
            if dsl_init_path is None: continue

            # Insert obstacle clusters at ~40% along path
            insert_idx = int(len(path_init) * 0.4)
            rng = np.random.RandomState(seed + 99999)
            new_obs = obs.copy()
            for cl in range(N_CLUSTERS):
                idx = min(insert_idx + cl*5, len(path_init)-1)
                cr, cc = path_init[idx]
                for dr in range(-CLUSTER_RADIUS, CLUSTER_RADIUS+1):
                    for dc in range(-CLUSTER_RADIUS, CLUSTER_RADIUS+1):
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < SIZE and 0 <= nc < SIZE:
                            new_obs[nr, nc] = True
            new_obs[0, :3] = False; new_obs[:3, 0] = False
            new_obs[SIZE-1, SIZE-3:] = False; new_obs[SIZE-3:, SIZE-1] = False

            replan_start = path_init[max(0, insert_idx - 5)]
            gm_new = GridMap(SIZE, SIZE, new_obs, None)

            # A* full re-run
            t0 = time.perf_counter()
            pa, na, _ = astar(gm_new, replan_start, GOAL)
            ta = (time.perf_counter() - t0) * 1000
            if pa is None: continue

            # D*Lite incremental re-plan
            dsl.grid = gm_new
            dsl.move_start(replan_start)
            dsl.nodes_expanded = 0
            t0 = time.perf_counter()
            dsl.update_map(obs, new_obs)
            nd = dsl.compute_shortest_path()
            td = (time.perf_counter() - t0) * 1000
            pd = dsl.get_path()
            if pd is None: continue

            # ILS re-plan
            t0 = time.perf_counter()
            pi, ni, _, _ = ils_astar(gm_new, replan_start, GOAL)
            ti = (time.perf_counter() - t0) * 1000
            if pi is None: continue

            success += 1
            astar_t.append(ta); dstar_t.append(td); ils_t.append(ti)
            astar_n.append(na); dstar_n.append(dsl.nodes_expanded); ils_n.append(ni)

            cost_a = compute_path_cost(pa, gm_new, 0.0)
            cost_d = compute_path_cost(pd, gm_new, 0.0)
            cost_i = compute_path_cost(pi, gm_new, 0.0)
            astar_opt.append(1.0)
            dstar_opt.append(cost_d/cost_a if cost_a > 0 else 1.0)
            ils_opt.append(cost_i/cost_a if cost_a > 0 else 1.0)

        if success < 5: continue
        at = np.array(astar_t); dt = np.array(dstar_t); it = np.array(ils_t)
        an = np.array(astar_n); dn = np.array(dstar_n); inn = np.array(ils_n)

        at_m, at_s, _, _ = ci_95(at)
        dt_m, dt_s, _, _ = ci_95(dt)
        it_m, it_s, _, _ = ci_95(it)
        an_m, an_s, _, _ = ci_95(an)
        dn_m, dn_s, _, _ = ci_95(dn)
        inn_m, inn_s, _, _ = ci_95(inn)

        results.append({
            'density': density, 'n_valid': success,
            'astar_time_mean': at_m, 'astar_time_std': at_s,
            'dstar_time_mean': dt_m, 'dstar_time_std': dt_s,
            'ils_time_mean': it_m, 'ils_time_std': it_s,
            'dstar_speedup': at_m/dt_m if dt_m > 0 else 0,
            'ils_speedup': at_m/it_m if it_m > 0 else 0,
            'astar_nodes_mean': an_m, 'astar_nodes_std': an_s,
            'dstar_nodes_mean': dn_m, 'dstar_nodes_std': dn_s,
            'ils_nodes_mean': inn_m, 'ils_nodes_std': inn_s,
            'dstar_node_red_pct': (1 - dn_m/an_m)*100 if an_m > 0 else 0,
            'ils_node_red_pct': (1 - inn_m/an_m)*100 if an_m > 0 else 0,
            'dstar_opt_mean': np.mean(dstar_opt), 'dstar_opt_std': np.std(dstar_opt),
            'ils_opt_mean': np.mean(ils_opt), 'ils_opt_std': np.std(ils_opt),
        })
        print(f"    N={success}, D*Lite={results[-1]['dstar_speedup']:.2f}x, "
              f"ILS={results[-1]['ils_speedup']:.2f}x")

    outfile = os.path.join(RESULTS_DIR, 'exp4b_dstar_replanning.csv')
    if results:
        with open(outfile, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader(); writer.writerows(results)
    print(f"  Saved to {outfile}")
    return results


# ============================================================
# EXP 5b: D*Lite Progressive Discovery
# ============================================================
def run_experiment_5b():
    """
    Progressive obstacle discovery with D*Lite baseline.
    Algorithms: A*, D*Lite, ILS
    3 densities x 40 missions = 120 missions
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5b: D*Lite Progressive Discovery")
    print("="*70)

    SIZE = 200; DENSITIES = [0.10, 0.20, 0.30]
    N_MISSIONS = 40; DISC_INTERVAL = 20; SENSOR = 15
    N_NEW_OBS = 3; OBS_SIZE = 3; MAX_REPLANS = 20
    START = (0, 0); GOAL = (SIZE-1, SIZE-1)

    results = []
    for density in DENSITIES:
        print(f"  Density={density:.0%}")
        planner_data = {p: {'nodes': [], 'time': [], 'replans': [], 'success': 0,
                           'path_lens': []}
                       for p in ['astar', 'dstar', 'ils']}

        for m in range(N_MISSIONS):
            seed = int(density*10000 + m + 300000)
            base_obs = generate_random_grid(SIZE, density, seed=seed)
            rng = np.random.RandomState(seed + 500000)

            for planner_name in ['astar', 'dstar', 'ils']:
                current_obs = base_obs.copy()
                gm = GridMap(SIZE, SIZE, current_obs, None)
                pos = START; total_nodes = 0; total_time = 0.0
                n_replans = 0; total_path = [pos]; mission_success = False

                # Initialize D*Lite if needed
                if planner_name == 'dstar':
                    dsl = DStarLite(gm, START, GOAL)
                    dsl.initialize()
                    t0 = time.perf_counter()
                    dsl.compute_shortest_path()
                    total_time += (time.perf_counter() - t0) * 1000
                    total_nodes += dsl.nodes_expanded

                for replan in range(MAX_REPLANS):
                    # Plan
                    if planner_name == 'astar':
                        path, nodes, t_ms = astar(gm, pos, GOAL)
                        total_nodes += nodes; total_time += t_ms; n_replans += 1
                    elif planner_name == 'dstar':
                        if replan > 0:  # Already have initial search
                            dsl.nodes_expanded = 0
                            t0 = time.perf_counter()
                            dsl.compute_shortest_path()
                            t_ms = (time.perf_counter() - t0) * 1000
                            total_nodes += dsl.nodes_expanded; total_time += t_ms
                        path = dsl.get_path()
                        n_replans += 1
                    else:
                        path, nodes, t_ms, _ = ils_astar(gm, pos, GOAL)
                        total_nodes += nodes; total_time += t_ms; n_replans += 1

                    if path is None: break

                    # Move along path
                    steps = 0
                    for i in range(1, len(path)):
                        r, c = path[i]
                        if current_obs[r, c]: break
                        pos = (r, c); total_path.append(pos); steps += 1
                        if pos == GOAL: mission_success = True; break
                        if steps >= DISC_INTERVAL: break

                    if mission_success: break

                    # Discover new obstacles
                    old_obs = current_obs.copy()
                    for _ in range(N_NEW_OBS):
                        cr = pos[0] + rng.randint(0, SENSOR)
                        cc = pos[1] + rng.randint(-SENSOR//2, SENSOR)
                        for dr in range(-OBS_SIZE//2, OBS_SIZE//2+1):
                            for dc in range(-OBS_SIZE//2, OBS_SIZE//2+1):
                                nr, nc = cr+dr, cc+dc
                                if 0 <= nr < SIZE and 0 <= nc < SIZE:
                                    current_obs[nr, nc] = True
                    current_obs[0,:3]=False; current_obs[:3,0]=False
                    current_obs[SIZE-1,SIZE-3:]=False; current_obs[SIZE-3:,SIZE-1]=False
                    gm = GridMap(SIZE, SIZE, current_obs, None)

                    if planner_name == 'dstar':
                        dsl.grid = gm
                        dsl.move_start(pos)
                        t0 = time.perf_counter()
                        dsl.update_map(old_obs, current_obs)
                        total_time += (time.perf_counter() - t0) * 1000

                pd = planner_data[planner_name]
                if mission_success: pd['success'] += 1
                pd['nodes'].append(total_nodes)
                pd['time'].append(total_time)
                pd['replans'].append(n_replans)
                pd['path_lens'].append(path_length_euclidean(total_path))

        row = {'density': density, 'n_missions': N_MISSIONS}
        for p in ['astar', 'dstar', 'ils']:
            pd = planner_data[p]
            n_m, n_s, _, _ = ci_95(pd['nodes'])
            t_m, t_s, _, _ = ci_95(pd['time'])
            row[f'{p}_success_rate'] = pd['success'] / N_MISSIONS
            row[f'{p}_nodes_mean'] = n_m; row[f'{p}_nodes_std'] = n_s
            row[f'{p}_time_mean'] = t_m; row[f'{p}_time_std'] = t_s
            row[f'{p}_replans_mean'] = np.mean(pd['replans'])
        # Node reduction relative to A*
        a_nodes = np.mean(planner_data['astar']['nodes'])
        for p in ['dstar', 'ils']:
            p_nodes = np.mean(planner_data[p]['nodes'])
            row[f'{p}_node_red_pct'] = (1 - p_nodes/a_nodes)*100 if a_nodes > 0 else 0
        # Path ratios
        a_lens = planner_data['astar']['path_lens']
        for p in ['dstar', 'ils']:
            p_lens = planner_data[p]['path_lens']
            ratios = [pl/al if al > 0 and al < float('inf') and pl < float('inf')
                     else float('nan') for al, pl in zip(a_lens, p_lens)]
            valid = [r for r in ratios if not np.isnan(r)]
            row[f'{p}_path_ratio_mean'] = np.mean(valid) if valid else 0

        results.append(row)
        for p in ['astar', 'dstar', 'ils']:
            pd = planner_data[p]
            print(f"    {p}: success={pd['success']}/{N_MISSIONS}, "
                  f"nodes={np.mean(pd['nodes']):.0f}")

    outfile = os.path.join(RESULTS_DIR, 'exp5b_dstar_progressive.csv')
    if results:
        with open(outfile, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader(); writer.writerows(results)
    print(f"  Saved to {outfile}")
    return results


# ============================================================
# EXP 6: Weighted A* Comparison
# ============================================================
def run_experiment_6():
    """
    Comprehensive wA* comparison.
    Algorithms: A*, wA*(1.5), wA*(2.0), wA*(3.0), ILS
    3 densities x 3 risk types x 3 lambdas x 50 maps = 1,350 paths
    """
    print("\n" + "="*70)
    print("EXPERIMENT 6: Weighted A* Comparison")
    print("="*70)

    SIZE = 200; DENSITIES = [0.10, 0.20, 0.30]
    RISK_TYPES = ['gradient', 'hotspot', 'uniform']
    LAMBDAS = [0.5, 1.0, 2.0]; N_MAPS = 50
    W_VALUES = [1.5, 2.0, 3.0]
    START = (0, 0); GOAL = (SIZE-1, SIZE-1)

    results = []
    for density in DENSITIES:
        for risk_type in RISK_TYPES:
            for lam in LAMBDAS:
                print(f"  d={density:.0%}, risk={risk_type}, lam={lam}")
                data = {alg: {'time': [], 'nodes': [], 'opt': []}
                       for alg in ['astar', 'wa1.5', 'wa2.0', 'wa3.0', 'ils']}
                success = 0

                for m in range(N_MAPS):
                    seed = int(density*1000 + hash(risk_type)%1000 + lam*100 + m + 600000)
                    obs = generate_random_grid(SIZE, density, seed=seed)
                    risk = generate_risk_layer(SIZE, risk_type, seed=seed+10000)
                    gm = GridMap(SIZE, SIZE, obs, risk)

                    pa, na, ta = astar(gm, START, GOAL, lam)
                    if pa is None: continue

                    all_ok = True
                    paths = {'astar': (pa, na, ta)}
                    for wv in W_VALUES:
                        pw, nw, tw = weighted_astar(gm, START, GOAL, lam, w=wv)
                        if pw is None: all_ok = False; break
                        paths[f'wa{wv}'] = (pw, nw, tw)
                    if not all_ok: continue

                    pi, ni, ti, _ = ils_astar(gm, START, GOAL, lam)
                    if pi is None: continue
                    paths['ils'] = (pi, ni, ti)

                    success += 1
                    ca = compute_path_cost(pa, gm, lam)
                    for alg, (p, n, t) in paths.items():
                        data[alg]['time'].append(t)
                        data[alg]['nodes'].append(n)
                        c = compute_path_cost(p, gm, lam)
                        data[alg]['opt'].append(c/ca if ca > 0 else 1.0)

                if success < 5: continue
                row = {'density': density, 'risk_type': risk_type, 'lambda': lam,
                       'n_valid': success}
                a_time_m = np.mean(data['astar']['time'])
                a_nodes_m = np.mean(data['astar']['nodes'])
                for alg in ['astar', 'wa1.5', 'wa2.0', 'wa3.0', 'ils']:
                    tm, ts, _, _ = ci_95(data[alg]['time'])
                    nm, ns, _, _ = ci_95(data[alg]['nodes'])
                    om, ostd, _, _ = ci_95(data[alg]['opt'])
                    row[f'{alg}_time_mean'] = tm; row[f'{alg}_time_std'] = ts
                    row[f'{alg}_nodes_mean'] = nm; row[f'{alg}_nodes_std'] = ns
                    row[f'{alg}_opt_mean'] = om; row[f'{alg}_opt_std'] = ostd
                    row[f'{alg}_speedup'] = a_time_m/tm if tm > 0 else 0
                    row[f'{alg}_node_red'] = (1 - nm/a_nodes_m)*100 if a_nodes_m > 0 else 0
                results.append(row)

    outfile = os.path.join(RESULTS_DIR, 'exp6_weighted_astar.csv')
    if results:
        with open(outfile, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader(); writer.writerows(results)
    print(f"  Saved to {outfile}")
    return results


# ============================================================
# EXP 7: Corridor Width Sensitivity Sweep
# ============================================================
def run_experiment_7():
    """
    Corridor width sensitivity analysis.
    Sweep alpha_0 from 1% to 30% of grid diagonal.
    3 densities x 2 risk types x 2 lambdas x 10 alphas x 30 maps = 3,600 paths
    """
    print("\n" + "="*70)
    print("EXPERIMENT 7: Corridor Width Sensitivity Sweep")
    print("="*70)

    SIZE = 200; DENSITIES = [0.10, 0.20, 0.30]
    RISK_TYPES = ['gradient', 'hotspot']
    LAMBDAS = [0.5, 1.0]
    ALPHAS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
    N_MAPS = 30
    START = (0, 0); GOAL = (SIZE-1, SIZE-1)

    results = []
    for density in DENSITIES:
        for risk_type in RISK_TYPES:
            for lam in LAMBDAS:
                # First run A* once for this config (baseline)
                astar_costs = []; astar_times = []; astar_nodes_l = []
                seeds = []
                for m in range(N_MAPS):
                    seed = int(density*1000 + hash(risk_type)%1000 + lam*100 + m + 700000)
                    seeds.append(seed)
                    obs = generate_random_grid(SIZE, density, seed=seed)
                    risk = generate_risk_layer(SIZE, risk_type, seed=seed+10000)
                    gm = GridMap(SIZE, SIZE, obs, risk)
                    pa, na, ta = astar(gm, START, GOAL, lam)
                    if pa is not None:
                        astar_costs.append(compute_path_cost(pa, gm, lam))
                        astar_times.append(ta)
                        astar_nodes_l.append(na)
                    else:
                        astar_costs.append(None)
                        astar_times.append(None)
                        astar_nodes_l.append(None)

                for alpha in ALPHAS:
                    print(f"  d={density:.0%}, risk={risk_type}, lam={lam}, alpha={alpha}")
                    speedups, node_reds, opt_ratios = [], [], []
                    success_count = 0; fail_count = 0

                    for m in range(N_MAPS):
                        if astar_costs[m] is None: continue
                        seed = seeds[m]
                        obs = generate_random_grid(SIZE, density, seed=seed)
                        risk = generate_risk_layer(SIZE, risk_type, seed=seed+10000)
                        gm = GridMap(SIZE, SIZE, obs, risk)

                        pi, ni, ti, _ = ils_astar(gm, START, GOAL, lam,
                                                    initial_width_frac=alpha)
                        if pi is None:
                            fail_count += 1; continue
                        success_count += 1
                        ci = compute_path_cost(pi, gm, lam)
                        speedups.append(astar_times[m] / ti if ti > 0 else 0)
                        node_reds.append((1 - ni/astar_nodes_l[m])*100)
                        opt_ratios.append(ci / astar_costs[m] if astar_costs[m] > 0 else 1.0)

                    if success_count < 5: continue
                    sm, ss, _, _ = ci_95(speedups)
                    nm, ns, _, _ = ci_95(node_reds)
                    om, ostd, _, _ = ci_95(opt_ratios)
                    results.append({
                        'density': density, 'risk_type': risk_type, 'lambda': lam,
                        'alpha': alpha, 'n_valid': success_count,
                        'n_failed': fail_count,
                        'speedup_mean': sm, 'speedup_std': ss,
                        'node_red_mean': nm, 'node_red_std': ns,
                        'opt_ratio_mean': om, 'opt_ratio_std': ostd,
                    })

    outfile = os.path.join(RESULTS_DIR, 'exp7_corridor_sensitivity.csv')
    if results:
        with open(outfile, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader(); writer.writerows(results)
    print(f"  Saved to {outfile}")
    return results


# ============================================================
# EXP 8: Risk-Responsive ILS (RILS) Evaluation
# ============================================================
def run_experiment_8():
    """
    RILS evaluation: compare A*, ILS, RILS, AILS on risk grids.
    3 densities x 3 risk types x 3 lambdas x 50 maps = 1,350 paths
    """
    print("\n" + "="*70)
    print("EXPERIMENT 8: Risk-Responsive ILS (RILS) Evaluation")
    print("="*70)

    SIZE = 200; DENSITIES = [0.10, 0.20, 0.30]
    RISK_TYPES = ['gradient', 'hotspot', 'uniform']
    LAMBDAS = [0.5, 1.0, 2.0]; N_MAPS = 50
    START = (0, 0); GOAL = (SIZE-1, SIZE-1)

    results = []
    for density in DENSITIES:
        for risk_type in RISK_TYPES:
            for lam in LAMBDAS:
                print(f"  d={density:.0%}, risk={risk_type}, lam={lam}")
                data = {alg: {'time': [], 'nodes': [], 'opt': [], 'exposure': []}
                       for alg in ['astar', 'ils', 'rils', 'ails']}
                success = 0

                for m in range(N_MAPS):
                    seed = int(density*1000 + hash(risk_type)%1000 + lam*100 + m + 800000)
                    obs = generate_random_grid(SIZE, density, seed=seed)
                    risk = generate_risk_layer(SIZE, risk_type, seed=seed+10000)
                    gm = GridMap(SIZE, SIZE, obs, risk)

                    pa, na, ta = astar(gm, START, GOAL, lam)
                    if pa is None: continue
                    pi, ni, ti, _ = ils_astar(gm, START, GOAL, lam)
                    if pi is None: continue
                    pr, nr_, tr, _ = rils_astar(gm, START, GOAL, lam)
                    if pr is None: continue
                    pai, nai, tai, _ = ails_astar(gm, START, GOAL, lam)
                    if pai is None: continue

                    success += 1
                    ca = compute_path_cost(pa, gm, lam)
                    ea = compute_exposure(pa, gm)

                    for alg, (p, n, t) in [('astar',(pa,na,ta)), ('ils',(pi,ni,ti)),
                                            ('rils',(pr,nr_,tr)), ('ails',(pai,nai,tai))]:
                        data[alg]['time'].append(t)
                        data[alg]['nodes'].append(n)
                        c = compute_path_cost(p, gm, lam)
                        data[alg]['opt'].append(c/ca if ca > 0 else 1.0)
                        e = compute_exposure(p, gm)
                        data[alg]['exposure'].append(e/ea if ea > 0 else 1.0)

                if success < 5: continue
                row = {'density': density, 'risk_type': risk_type, 'lambda': lam,
                       'n_valid': success}
                a_time_m = np.mean(data['astar']['time'])
                a_nodes_m = np.mean(data['astar']['nodes'])
                for alg in ['astar', 'ils', 'rils', 'ails']:
                    tm, ts, _, _ = ci_95(data[alg]['time'])
                    nm, ns, _, _ = ci_95(data[alg]['nodes'])
                    om, ostd, _, _ = ci_95(data[alg]['opt'])
                    em, es, _, _ = ci_95(data[alg]['exposure'])
                    row[f'{alg}_time_mean'] = tm; row[f'{alg}_time_std'] = ts
                    row[f'{alg}_nodes_mean'] = nm; row[f'{alg}_nodes_std'] = ns
                    row[f'{alg}_opt_mean'] = om; row[f'{alg}_opt_std'] = ostd
                    row[f'{alg}_exposure_mean'] = em; row[f'{alg}_exposure_std'] = es
                    row[f'{alg}_speedup'] = a_time_m/tm if tm > 0 else 0
                    row[f'{alg}_node_red'] = (1-nm/a_nodes_m)*100 if a_nodes_m > 0 else 0
                results.append(row)

    outfile = os.path.join(RESULTS_DIR, 'exp8_rils_evaluation.csv')
    if results:
        with open(outfile, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader(); writer.writerows(results)
    print(f"  Saved to {outfile}")
    return results


# ============================================================
# EXP 9: AILS Large-Grid Validation (500x500)
# ============================================================
def run_experiment_9():
    """
    AILS on larger grids where its overhead is amortised.
    Grid size: 500x500
    3 densities x 3 risk types x 2 lambdas x 15 maps = 270 paths
    Algorithms: A*, ILS, AILS
    """
    print("\n" + "="*70)
    print("EXPERIMENT 9: AILS Large-Grid Validation (500x500)")
    print("="*70)

    SIZE = 500; DENSITIES = [0.10, 0.20, 0.30]
    RISK_TYPES = ['gradient', 'hotspot', 'uniform']
    LAMBDAS = [0.5, 1.0]; N_MAPS = 15
    START = (0, 0); GOAL = (SIZE-1, SIZE-1)

    results = []
    for density in DENSITIES:
        for risk_type in RISK_TYPES:
            for lam in LAMBDAS:
                print(f"  d={density:.0%}, risk={risk_type}, lam={lam} (500x500)")
                data = {alg: {'time': [], 'nodes': [], 'opt': []}
                       for alg in ['astar', 'ils', 'ails']}
                success = 0

                for m in range(N_MAPS):
                    seed = int(density*1000 + hash(risk_type)%1000 + lam*100 + m + 400000)
                    obs = generate_random_grid(SIZE, density, seed=seed)
                    risk = generate_risk_layer(SIZE, risk_type, seed=seed+10000)
                    gm = GridMap(SIZE, SIZE, obs, risk)

                    pa, na, ta = astar(gm, START, GOAL, lam)
                    if pa is None: continue
                    pi, ni, ti, _ = ils_astar(gm, START, GOAL, lam)
                    if pi is None: continue
                    pai, nai, tai, _ = ails_astar(gm, START, GOAL, lam)
                    if pai is None: continue

                    success += 1
                    ca = compute_path_cost(pa, gm, lam)
                    for alg, (p, n, t) in [('astar',(pa,na,ta)),
                                            ('ils',(pi,ni,ti)),
                                            ('ails',(pai,nai,tai))]:
                        data[alg]['time'].append(t)
                        data[alg]['nodes'].append(n)
                        c = compute_path_cost(p, gm, lam)
                        data[alg]['opt'].append(c/ca if ca > 0 else 1.0)

                if success < 3: continue
                row = {'density': density, 'risk_type': risk_type, 'lambda': lam,
                       'n_valid': success, 'grid_size': SIZE}
                a_time_m = np.mean(data['astar']['time'])
                a_nodes_m = np.mean(data['astar']['nodes'])
                for alg in ['astar', 'ils', 'ails']:
                    tm, ts, _, _ = ci_95(data[alg]['time'])
                    nm, ns, _, _ = ci_95(data[alg]['nodes'])
                    om, ostd, _, _ = ci_95(data[alg]['opt'])
                    row[f'{alg}_time_mean'] = tm; row[f'{alg}_time_std'] = ts
                    row[f'{alg}_nodes_mean'] = nm; row[f'{alg}_nodes_std'] = ns
                    row[f'{alg}_opt_mean'] = om; row[f'{alg}_opt_std'] = ostd
                    row[f'{alg}_speedup'] = a_time_m/tm if tm > 0 else 0
                    row[f'{alg}_node_red'] = (1-nm/a_nodes_m)*100 if a_nodes_m > 0 else 0
                results.append(row)
                print(f"    N={success}, ILS={results[-1]['ils_speedup']:.2f}x, "
                      f"AILS={results[-1]['ails_speedup']:.2f}x")

    outfile = os.path.join(RESULTS_DIR, 'exp9_large_grid_ails.csv')
    if results:
        with open(outfile, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader(); writer.writerows(results)
    print(f"  Saved to {outfile}")
    return results


# ============================================================
# SANITY TEST
# ============================================================
def run_sanity_test():
    """Quick sanity test of all algorithms on a small grid."""
    print("\n" + "="*70)
    print("SANITY TEST: Verifying all algorithms")
    print("="*70)

    SIZE = 50; density = 0.15
    obs = generate_random_grid(SIZE, density, seed=12345)
    risk = generate_risk_layer(SIZE, 'gradient', seed=12345)
    gm = GridMap(SIZE, SIZE, obs, risk)
    START = (0, 0); GOAL = (SIZE-1, SIZE-1)
    lam = 1.0

    # A*
    pa, na, ta = astar(gm, START, GOAL, lam)
    assert pa is not None, "A* failed"
    ca = compute_path_cost(pa, gm, lam)
    print(f"  A*:   cost={ca:.2f}, nodes={na}, time={ta:.2f}ms")

    # wA*
    pw, nw, tw = weighted_astar(gm, START, GOAL, lam, w=2.0)
    assert pw is not None, "wA* failed"
    cw = compute_path_cost(pw, gm, lam)
    print(f"  wA*:  cost={cw:.2f}, nodes={nw}, time={tw:.2f}ms, ratio={cw/ca:.4f}")
    assert cw >= ca * 0.999, f"wA* found better path than A*: {cw} < {ca}"

    # ILS
    pi, ni, ti, _ = ils_astar(gm, START, GOAL, lam)
    assert pi is not None, "ILS failed"
    ci_ = compute_path_cost(pi, gm, lam)
    print(f"  ILS:  cost={ci_:.2f}, nodes={ni}, time={ti:.2f}ms, ratio={ci_/ca:.4f}")

    # RILS
    pr, nr_, tr, _ = rils_astar(gm, START, GOAL, lam)
    assert pr is not None, "RILS failed"
    cr = compute_path_cost(pr, gm, lam)
    print(f"  RILS: cost={cr:.2f}, nodes={nr_}, time={tr:.2f}ms, ratio={cr/ca:.4f}")

    # AILS
    pai, nai, tai, _ = ails_astar(gm, START, GOAL, lam)
    assert pai is not None, "AILS failed"
    cai = compute_path_cost(pai, gm, lam)
    print(f"  AILS: cost={cai:.2f}, nodes={nai}, time={tai:.2f}ms, ratio={cai/ca:.4f}")

    # D*Lite initial search
    dsl = DStarLite(gm, START, GOAL, lam)
    dsl.initialize()
    dsl.compute_shortest_path()
    pd = dsl.get_path()
    assert pd is not None, "D*Lite initial search failed"
    cd = compute_path_cost(pd, gm, lam)
    print(f"  D*L:  cost={cd:.2f}, nodes={dsl.nodes_expanded}, ratio={cd/ca:.4f}")
    assert abs(cd - ca) < 0.01, f"D*Lite cost {cd} differs from A* cost {ca}"

    # D*Lite re-planning test
    old_obs = obs.copy()
    new_obs = obs.copy()
    # Block a few cells in middle
    for r in range(24, 27):
        for c in range(24, 27):
            new_obs[r, c] = True
    new_obs[0,:3]=False; new_obs[:3,0]=False
    new_obs[SIZE-1,SIZE-3:]=False; new_obs[SIZE-3:,SIZE-1]=False
    gm_new = GridMap(SIZE, SIZE, new_obs, risk)

    # A* on new grid
    pa2, na2, _ = astar(gm_new, START, GOAL, lam)
    assert pa2 is not None, "A* on modified grid failed"
    ca2 = compute_path_cost(pa2, gm_new, lam)

    # D*Lite incremental
    dsl.grid = gm_new
    dsl.nodes_expanded = 0
    dsl.update_map(old_obs, new_obs)
    dsl.compute_shortest_path()
    pd2 = dsl.get_path()
    assert pd2 is not None, "D*Lite re-plan failed"
    cd2 = compute_path_cost(pd2, gm_new, lam)
    print(f"  D*L replan: cost={cd2:.2f} (A* cost={ca2:.2f}), "
          f"ratio={cd2/ca2:.4f}, incremental nodes={dsl.nodes_expanded}")
    assert cd2 / ca2 < 1.01, f"D*Lite replan too suboptimal: {cd2/ca2:.4f}"

    print("\n  ALL SANITY TESTS PASSED!")


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*70)
    print("SUPPLEMENTARY EXPERIMENTS")
    print("Risk-Aware Corridor-Constrained Pathfinding (Revised)")
    print("="*70)
    print(f"Results directory: {RESULTS_DIR}\n")

    experiments = {
        'sanity': ('Sanity Test', run_sanity_test),
        '1b': ('Random Start-Goal + wA*', run_experiment_1b),
        '4b': ('D*Lite Re-planning', run_experiment_4b),
        '5b': ('D*Lite Progressive', run_experiment_5b),
        '6':  ('Weighted A* Comparison', run_experiment_6),
        '7':  ('Corridor Width Sensitivity', run_experiment_7),
        '8':  ('RILS Evaluation', run_experiment_8),
        '9':  ('AILS Large Grid (500x500)', run_experiment_9),
    }

    if len(sys.argv) > 1:
        exp_ids = sys.argv[1:]
    else:
        exp_ids = ['sanity', '1b', '4b', '5b', '6', '7', '8', '9']

    for eid in exp_ids:
        if eid in experiments:
            name, func = experiments[eid]
            print(f"\n{'='*70}")
            print(f"Starting: {name}")
            print(f"{'='*70}")
            t0 = time.time()
            func()
            print(f"  Completed in {time.time()-t0:.1f}s")

    print("\n" + "="*70)
    print("ALL SUPPLEMENTARY EXPERIMENTS COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}/")
    print("="*70)

if __name__ == '__main__':
    main()
