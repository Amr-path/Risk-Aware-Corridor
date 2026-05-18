"""
download_penang_port.py
=======================
Downloads Penang Port (North Butterworth Container Terminal area) from
OpenStreetMap, rasterises it to a 500x500 occupancy grid, and synthesises
a biosecurity risk layer based on zone semantics.

Output: experiments/data/penang_port.npz with:
    obstacles : 500x500 bool array  (True = obstacle)
    risk      : 500x500 float array (0.0--1.0, biosecurity risk)
    metadata  : dict (bounds, CRS, zone definitions)

Also writes experiments/data/penang_port.png for visual inspection.

USAGE (run on your laptop, NOT in this sandbox):
    pip install osmnx geopandas rasterio shapely pyproj matplotlib
    cd "/Users/amralshahed/Downloads/PHD-Thesis-Apr/Risk_Aware_Corridor_Pathfinding"
    python experiments/download_penang_port.py

Expected runtime: 1-3 minutes (depends on Overpass API load).
"""
from __future__ import annotations
import os
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 1. Geographic bounds for North Butterworth Container Terminal (NBCT)
# ----------------------------------------------------------------------
# Operated by Penang Port Sdn Bhd. This is the main cargo facility and
# the natural choice for a biosecurity-relevant pathfinding study:
# container yard + gate complex + quay + buffer zones.
#
# Approximate bounding box (WGS84 / EPSG:4326):
NORTH = 5.4150   # latitude
SOUTH = 5.3950
EAST  = 100.3800  # longitude
WEST  = 100.3500

GRID_SIZE = 500  # 500x500 grid

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
NPZ_PATH = os.path.join(OUT_DIR, "penang_port.npz")
PNG_PATH = os.path.join(OUT_DIR, "penang_port.png")


def fetch_osm_features():
    """Download OSM features within the bounding box."""
    import osmnx as ox

    print(f"[1/4] Querying OpenStreetMap for Penang Port "
          f"(approx {(NORTH-SOUTH)*111:.1f} km x {(EAST-WEST)*111:.1f} km)...")

    tags = {
        "building": True,
        "landuse": ["industrial", "port", "harbour", "commercial"],
        "man_made": ["pier", "wharf", "breakwater", "storage_tank"],
        "barrier": True,
        "highway": True,
        "waterway": True,
        "natural": ["water", "coastline"],
    }
    # osmnx 2.x signature
    bbox = (WEST, SOUTH, EAST, NORTH)
    try:
        features = ox.features_from_bbox(bbox, tags=tags)
    except TypeError:
        # osmnx 1.x fallback (north,south,east,west)
        features = ox.features_from_bbox(north=NORTH, south=SOUTH,
                                          east=EAST, west=WEST, tags=tags)

    print(f"      Got {len(features)} features.")
    return features


def rasterize_to_grid(features):
    """Rasterise vector features to a 500x500 obstacle grid."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    print(f"[2/4] Rasterising features to {GRID_SIZE}x{GRID_SIZE} grid...")

    # Reproject to a metric CRS for area-correct rasterisation.
    # Malaysia's RSO Borneo (EPSG:3375) is the standard local projection.
    features_m = features.to_crs(epsg=3375)

    # Compute the bounding box of the original bbox in the metric CRS
    bbox_geo = gpd.GeoSeries([box(WEST, SOUTH, EAST, NORTH)], crs="EPSG:4326")
    bbox_m = bbox_geo.to_crs(epsg=3375).iloc[0]
    minx, miny, maxx, maxy = bbox_m.bounds

    # Define the raster transform
    transform = from_bounds(minx, miny, maxx, maxy, GRID_SIZE, GRID_SIZE)

    # Helper: properly check OSM tag values (NaN -> False, "no" -> False)
    NEGATIVE = {None, "", "no", "false", "0", False}

    def has_tag(row, col, expected=None):
        if col not in row.index:
            return False
        val = row[col]
        if pd.isna(val):
            return False
        if val in NEGATIVE:
            return False
        if expected is None:
            return True
        if isinstance(expected, (list, tuple, set)):
            return val in expected
        return val == expected

    # Separate features by semantic class for the risk layer
    obstacles_geom = []   # buildings, barriers, breakwaters
    water_geom = []       # waterways (obstacle for ground UAV survey)
    road_geom = []        # roads (free corridors)
    container_geom = []   # industrial landuse polygons (high-risk inspection zones)

    for idx, row in features_m.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if has_tag(row, "building"):
            obstacles_geom.append(geom)
        elif has_tag(row, "man_made", ["pier", "wharf", "breakwater"]):
            obstacles_geom.append(geom)
        elif has_tag(row, "barrier"):
            obstacles_geom.append(geom)
        elif has_tag(row, "landuse", ["industrial", "port", "harbour", "commercial"]):
            container_geom.append(geom)
        elif has_tag(row, "highway"):
            road_geom.append(geom)
        elif has_tag(row, "waterway"):
            water_geom.append(geom)
        elif has_tag(row, "natural", "water"):
            water_geom.append(geom)

    print(f"      Buildings/barriers: {len(obstacles_geom)}")
    print(f"      Industrial polygons: {len(container_geom)}")
    print(f"      Roads: {len(road_geom)}")
    print(f"      Water: {len(water_geom)}")

    # Rasterise
    def _rasterize(geoms, default=False):
        if not geoms:
            return np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        return rasterize(
            [(g, 1) for g in geoms],
            out_shape=(GRID_SIZE, GRID_SIZE),
            transform=transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)

    obstacles = _rasterize(obstacles_geom)
    container_mask = _rasterize(container_geom)
    road_mask = _rasterize(road_geom)
    water_mask = _rasterize(water_geom)

    # Composite: water is obstacle for ground-level UAV; roads carve out free space
    obstacles |= water_mask
    obstacles &= ~road_mask  # roads override

    # Ensure corners are passable so random endpoints can be placed
    obstacles[:5, :5] = False
    obstacles[-5:, -5:] = False
    obstacles[:5, -5:] = False
    obstacles[-5:, :5] = False

    return obstacles, container_mask, road_mask, transform


def build_risk_layer(obstacles, container_mask, road_mask):
    """Build a zone-aware biosecurity risk layer.

    Risk semantics (UAV biosecurity surveillance context):
      - Container/industrial zones : 0.7--0.9  (inspection hotspots, contamination risk)
      - Gate zones (road-building boundary): 0.5--0.7  (controlled checkpoints)
      - Quay/open areas             : 0.1--0.3  (low risk)
      - Inside obstacles            : 0 (not traversable anyway)

    Spatial smoothing via a Gaussian-like falloff around container blocks
    produces a realistic risk gradient.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    rng = np.random.RandomState(42)
    h, w = obstacles.shape
    risk = np.zeros((h, w), dtype=np.float32)

    # 1. Container zones: high baseline risk with random variation
    risk[container_mask] = rng.uniform(0.7, 0.9, size=container_mask.sum())

    # 2. Falloff around container zones — exponential decay over ~30 cells
    if container_mask.any():
        dist_to_container = distance_transform_edt(~container_mask)
        falloff = np.exp(-dist_to_container / 25.0) * 0.5
        risk = np.maximum(risk, falloff)

    # 3. Gate zones: where roads meet obstacles (ingress checkpoints)
    if road_mask.any() and obstacles.any():
        from scipy.ndimage import binary_dilation
        gate_zone = binary_dilation(obstacles, iterations=3) & road_mask
        if gate_zone.any():
            risk[gate_zone] = np.maximum(risk[gate_zone], rng.uniform(0.5, 0.7, size=gate_zone.sum()))

    # 4. Quay/open background: small ambient risk
    background = (risk == 0) & ~obstacles
    risk[background] = rng.uniform(0.05, 0.20, size=background.sum())

    # 5. Smooth slightly for realism
    risk = gaussian_filter(risk, sigma=1.5)
    risk = np.clip(risk, 0.0, 1.0)

    # Zero out obstacles (not strictly needed but clean)
    risk[obstacles] = 0.0

    return risk.astype(np.float32)


def visualize(obstacles, risk):
    """Save a PNG preview of the port grid."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(obstacles, cmap='gray_r', interpolation='nearest')
    axes[0].set_title(f"Obstacles ({obstacles.sum()} cells = "
                      f"{100*obstacles.mean():.1f}%)")
    axes[0].axis('off')

    im = axes[1].imshow(risk, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title(f"Risk layer (mean {risk.mean():.2f})")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Composite
    composite = np.zeros((*obstacles.shape, 3), dtype=np.float32)
    composite[..., 0] = risk          # red channel = risk
    composite[..., 1] = 0.7 * risk    # green channel
    composite[obstacles] = [0.1, 0.1, 0.1]  # dark gray for obstacles
    axes[2].imshow(composite)
    axes[2].set_title("Composite (red = risk, black = obstacle)")
    axes[2].axis('off')

    plt.suptitle(f"Penang Port (NBCT) — {GRID_SIZE}x{GRID_SIZE} grid",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[4/4] Wrote preview: {PNG_PATH}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    features = fetch_osm_features()
    obstacles, container_mask, road_mask, transform = rasterize_to_grid(features)
    risk = build_risk_layer(obstacles, container_mask, road_mask)

    obs_pct = 100 * obstacles.mean()
    risk_mean = float(risk.mean())
    risk_max = float(risk.max())

    print(f"[3/4] Grid statistics:")
    print(f"      Obstacle density : {obs_pct:.1f}%")
    print(f"      Mean risk        : {risk_mean:.3f}")
    print(f"      Max risk         : {risk_max:.3f}")
    print(f"      Risk >0.5 area   : {100*(risk > 0.5).mean():.1f}%")

    metadata = dict(
        source="OpenStreetMap (Overpass API)",
        place="North Butterworth Container Terminal, Penang, Malaysia",
        bbox=dict(north=NORTH, south=SOUTH, east=EAST, west=WEST),
        crs_geographic="EPSG:4326",
        crs_projected="EPSG:3375 (RSO Borneo)",
        grid_size=GRID_SIZE,
        obstacle_pct=obs_pct,
        mean_risk=risk_mean,
        risk_zones=dict(
            container="0.7-0.9 (inspection hotspots)",
            gate="0.5-0.7 (controlled checkpoints)",
            background="0.05-0.20 (open/quay)",
        ),
    )

    np.savez_compressed(
        NPZ_PATH,
        obstacles=obstacles,
        risk=risk,
        metadata=metadata,
    )
    print(f"      Wrote grid       : {NPZ_PATH}")

    visualize(obstacles, risk)

    print()
    print("Done. Next step:")
    print("    python experiments/run_experiment_12.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        print("If this is an Overpass timeout, retry in a few minutes.", file=sys.stderr)
        print("If osmnx complains about API version, run: pip install -U osmnx", file=sys.stderr)
        raise
