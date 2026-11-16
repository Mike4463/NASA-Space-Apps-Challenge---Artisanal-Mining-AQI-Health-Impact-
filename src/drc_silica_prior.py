from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import box

# Path to local Global Lithological Map (GLiM) vector file (polygons with 'xx' or 'LITHO' class code)
GLIM_PATH = Path("./geology_layers/glim_wgs84.gpkg")

# USGS Mineral Resources Data System (MRDS) WFS endpoint
USGS_MRDS_WFS = "https://mrdata.usgs.gov/wfs/mrds"

# Empirically grounded crystalline-silica potential weights [0.0 - 1.0] by GLiM class
# Reflects typical free alpha-quartz fraction in crushed host rock:
#   pa (Acid Plutonic / Granite-Pegmatite): ~25-40% quartz -> 0.90
#   ss (Siliciclastic Sedimentary / Sandstone-Quartzite): ~50-90% quartz -> 0.95
#   va (Acid Volcanic / Rhyolite): ~20-35% quartz -> 0.80
#   mt (Metamorphic / Schist-Gneiss-Quartzite): ~20-50% quartz -> 0.75
#   sm (Mixed Sedimentary): ~15-30% quartz -> 0.55
#   su (Unconsolidated Sediments / Alluvial): variable -> 0.45
#   sc (Carbonate Sedimentary / Limestone-Dolomite): <5% quartz (unless silicified) -> 0.20
#   pb / vb (Basic Plutonic / Basalt): <2% free quartz -> 0.10
GLIM_SILICA_WEIGHTS = {
    "ss": 0.95,
    "pa": 0.90,
    "va": 0.80,
    "mt": 0.75,
    "sm": 0.55,
    "pi": 0.50,
    "vi": 0.45,
    "su": 0.45,
    "sc": 0.20,
    "ev": 0.15,
    "pb": 0.10,
    "vb": 0.10,
    "wb": 0.00,
    "ig": 0.00,
    "nd": 0.50,
}

# Commodity-specific respirable silica hazard multiplier from USGS MRDS occurrences
COMMODITY_SILICA_BOOST = {
    "gold": 0.30,  # Quartz-vein hard-rock crushing -> extreme silicosis risk
    "tin": 0.25,  # Cassiterite in quartz-muscovite pegmatites/greisens
    "tantalum": 0.25,  # Coltan pegmatite/quartz veins
    "tungsten": 0.25,  # Wolframite quartz veins
    "copper": 0.15,  # Katanga Copperbelt silicified dolomites/shales
    "cobalt": 0.15,  # Co-hosted with Katangan Cu-Co silicified strata
    "diamond": 0.05,  # Primarily alluvial/kimberlite (lower free quartz)
}


def build_study_grid(
    bbox: tuple[float, float, float, float], cell_size_deg: float = 0.05
) -> gpd.GeoDataFrame:
  """Generates a regular WGS84 grid (~5.5 km cells at the equator) covering a DRC corridor."""
  minx, miny, maxx, maxy = bbox
  xs = np.arange(minx, maxx, cell_size_deg)
  ys = np.arange(miny, maxy, cell_size_deg)

  cells = []
  for i, x in enumerate(xs):
    for j, y in enumerate(ys):
      cells.append({
          "cell_id": f"cell_{i}_{j}",
          "geometry": box(x, y, x + cell_size_deg, y + cell_size_deg),
      })
  return gpd.GeoDataFrame(cells, crs="EPSG:4326")


def query_usgs_mrds_occurrences(
    bbox: tuple[float, float, float, float], buffer_deg: float = 0.15
) -> gpd.GeoDataFrame:
  """Queries USGS MRDS WFS for documented mineral deposits within/near the study bounding box."""
  minx, miny, maxx, maxy = bbox
  params = {
      "service": "WFS",
      "version": "1.1.0",
      "request": "GetFeature",
      "typeName": "mrds",
      "outputFormat": "application/json",
      "srsName": "EPSG:4326",
      "bbox": (
          f"{minx - buffer_deg},{miny - buffer_deg},"
          f"{maxx + buffer_deg},{maxy + buffer_deg},EPSG:4326"
      ),
  }
  try:
    resp = requests.get(USGS_MRDS_WFS, params=params, timeout=25)
    resp.raise_for_status()
    gdf = gpd.read_file(resp.text)
    if gdf.crs is None:
      gdf = gdf.set_crs("EPSG:4326")
    return gdf
  except Exception as e:
    print(f"[Warning] USGS MRDS live WFS query unavailable ({e}); proceeding with GLiM lithology only.")
    return gpd.GeoDataFrame(columns=["commod1", "geometry"], crs="EPSG:4326")


def compute_spatial_silica_prior(
    bbox: tuple[float, float, float, float],
    glim_path: Path = GLIM_PATH,
    cell_size_deg: float = 0.05,
    deposit_influence_km: float = 15.0,
) -> gpd.GeoDataFrame:
  """Computes a normalized crystalline-silica weighting factor S_silica in [0, 1] per grid cell

    by combining area-weighted GLiM bedrock quartz potential with kernel-decayed proximity to
    USGS mineral occurrences.
    """
  grid_gdf = build_study_grid(bbox, cell_size_deg=cell_size_deg)
  # Project to African Equal Area (ESRI:102022) for accurate area & distance calculations
  equal_area_crs = "ESRI:102022"
  grid_ea = grid_gdf.to_crs(equal_area_crs)
  grid_ea["cell_area_m2"] = grid_ea.geometry.area

  # --- 1. Area-Weighted GLiM Lithological Quartz Score ---
  if glim_path.exists():
    glim_gdf = gpd.read_file(glim_path, bbox=bbox).to_crs(equal_area_crs)
    # Standardize lithology column name ('xx' in original GLiM release)
    lith_col = "xx" if "xx" in glim_gdf.columns else glim_gdf.columns[0]
    glim_gdf["lith_code"] = glim_gdf[lith_col].astype(str).str.lower().str[:2]
    glim_gdf["base_quartz_w"] = (
        glim_gdf["lith_code"].map(GLIM_SILICA_WEIGHTS).fillna(0.50)
    )

    inter = gpd.overlay(
        grid_ea[["cell_id", "cell_area_m2", "geometry"]],
        glim_gdf[["lith_code", "base_quartz_w", "geometry"]],
        how="intersection",
    )
    inter["weighted_w"] = inter["base_quartz_w"] * (
        inter.geometry.area / inter["cell_area_m2"]
    )
    lith_scores = (
        inter.groupby("cell_id", as_index=False)["weighted_w"]
        .sum()
        .rename(columns={"weighted_w": "glim_silica_score"})
    )
    grid_ea = grid_ea.merge(lith_scores, on="cell_id", how="left")
    grid_ea["glim_silica_score"] = grid_ea["glim_silica_score"].fillna(0.50)
  else:
    print(f"[Info] GLiM file not found at {glim_path}; defaulting bedrock score to 0.50.")
    grid_ea["glim_silica_score"] = 0.50

  # --- 2. USGS MRDS Commodity Proximity Boost ---
  mrds_gdf = query_usgs_mrds_occurrences(bbox)
  if not mrds_gdf.empty:
    mrds_ea = mrds_gdf.to_crs(equal_area_crs)
    commod_col = next(
        (c for c in ["commod1", "COMMOD1", "commodity"] if c in mrds_ea.columns),
        None,
    )

    def score_commodity(val) -> float:
      text = str(val).lower()
      return max(
          (w for k, w in COMMODITY_SILICA_BOOST.items() if k in text),
          default=0.10,
      )

    mrds_ea["commod_boost"] = (
        mrds_ea[commod_col].apply(score_commodity) if commod_col else 0.15
    )

    centroids = grid_ea.geometry.centroid
    radius_m = deposit_influence_km * 1000.0
    boosts = []

    for pt in centroids:
      dists = mrds_ea.geometry.distance(pt)
      nearby = mrds_ea[dists <= radius_m]
      if nearby.empty:
        boosts.append(0.0)
      else:
        # Gaussian distance decay from documented USGS mineral occurrence
        weights = np.exp(-0.5 * (dists[dists <= radius_m] / (radius_m / 2.0)) ** 2)
        boosts.append(float(np.max(nearby["commod_boost"].to_numpy() * weights)))

    grid_ea["usgs_commodity_boost"] = boosts
  else:
    grid_ea["usgs_commodity_boost"] = 0.0

  # --- 3. Combine & Clip to Normalized [0, 1] Prior S_silica ---
  grid_ea["S_silica"] = np.clip(
      grid_ea["glim_silica_score"] + grid_ea["usgs_commodity_boost"], 0.0, 1.0
  ).round(4)

  return grid_ea.to_crs("EPSG:4326")[
      ["cell_id", "glim_silica_score", "usgs_commodity_boost", "S_silica", "geometry"]
  ]


# ==============================================================================
# EXAMPLE EXECUTION: ITURI GOLD BELT VS. KOLWEZI COPPER-COBALT CORRIDOR
# ==============================================================================
if __name__ == "__main__":
  corridors = {
      "DRC_Ituri_Mongbwalu": (29.85, 1.80, 30.25, 2.10),
      "DRC_Kolwezi": (25.25, -10.85, 25.80, -10.55),
  }

  for name, bbox in corridors.items():
    silica_grid = compute_spatial_silica_prior(bbox, cell_size_deg=0.05)
    print(f"\n--- {name} Spatial Crystalline Silica Prior ---")
    print(f"Grid cells: {len(silica_grid)} | Mean S_silica: {silica_grid['S_silica'].mean():.3f} | Max S_silica: {silica_grid['S_silica'].max():.3f}")
    print(silica_grid.drop(columns="geometry").head(5).to_string(index=False))