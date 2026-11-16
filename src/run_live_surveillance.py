from pathlib import Path
import numpy as np
import pandas as pd

from asm_health_surveillance_pipeline import (
    DATA_DIR,
    STUDY_REGIONS,
    TEMPORAL_RANGE,
    build_harmonized_weekly_panel,
    train_and_score_underreporting,
)
from drc_silica_prior import compute_spatial_silica_prior
from fetch_era5_meteorology import download_and_extract_era5
from live_satellite_extractors import (
    extract_mcd19a2_zonal_stats,
    extract_s1_rtc_zonal_stats,
    submit_and_download_s1_rtc,
)

# Cached intermediate CSV paths
SAR_CACHE = DATA_DIR / "live_sar.csv"
AOD_CACHE = DATA_DIR / "live_aod.csv"
ERA5_CACHE = DATA_DIR / "live_era5.csv"

# Expected epidemiological input CSVs
JAMAICA_HEALTH_CSV = Path("./health_data/jamaica_mohw_weekly.csv")
DRC_HEALTH_CSV = Path("./health_data/drc_dhis2_weekly.csv")  # Optional


def load_or_fetch_sar(force_refresh: bool = False) -> pd.DataFrame:
  if SAR_CACHE.exists() and not force_refresh:
    print(f"[Cache Hit] Loading Sentinel-1 RTC series from {SAR_CACHE}")
    return pd.read_csv(SAR_CACHE)

  frames = []
  for region_id, cfg in STUDY_REGIONS.items():
    rtc_dirs = submit_and_download_s1_rtc(
        region_id=region_id,
        bbox=cfg["bbox"],
        temporal=TEMPORAL_RANGE,
        max_scenes=24,  # Adjust based on monthly HyP3 credit budget
        wait_for_completion=True,
    )
    df_reg = extract_s1_rtc_zonal_stats(region_id, cfg["bbox"], rtc_dirs)
    frames.append(df_reg)

  sar_df = pd.concat(frames, ignore_index=True)
  sar_df.to_csv(SAR_CACHE, index=False)
  return sar_df


def load_or_fetch_aod(force_refresh: bool = False) -> pd.DataFrame:
  if AOD_CACHE.exists() and not force_refresh:
    print(f"[Cache Hit] Loading MODIS MCD19A2 AOD series from {AOD_CACHE}")
    return pd.read_csv(AOD_CACHE)

  frames = []
  for region_id, cfg in STUDY_REGIONS.items():
    df_reg = extract_mcd19a2_zonal_stats(
        region_id=region_id,
        bbox=cfg["bbox"],
        temporal=TEMPORAL_RANGE,
        max_granules=365,
    )
    # Placeholder column until Sentinel-5P TROPOMI AER_AI extractor is added
    if not df_reg.empty and "tropomi_ai" not in df_reg.columns:
      df_reg["tropomi_ai"] = df_reg["aod_550"] * 2.2
    frames.append(df_reg)

  aod_df = pd.concat(frames, ignore_index=True)
  aod_df.to_csv(AOD_CACHE, index=False)
  return aod_df


def load_or_fetch_era5(force_refresh: bool = False) -> pd.DataFrame:
  if ERA5_CACHE.exists() and not force_refresh:
    print(f"[Cache Hit] Loading ERA5 meteorology series from {ERA5_CACHE}")
    return pd.read_csv(ERA5_CACHE)
  return download_and_extract_era5(
      STUDY_REGIONS, TEMPORAL_RANGE, out_csv=ERA5_CACHE
  )


def load_epidemiological_records() -> pd.DataFrame:
  """Loads digitized Jamaica MOHW parish weekly surveillance counts (source domain)

    and optional sparse DRC DHIS2 clinic reports (target domain).
    Expected Jamaica columns: region_id, epi_week, resp_rate, completeness
    Expected DRC columns:     region_id, epi_week, reported_drc
    """
  if not JAMAICA_HEALTH_CSV.exists():
    raise FileNotFoundError(
        f"Missing {JAMAICA_HEALTH_CSV}. Place your digitized Jamaica MOHW"
        " weekly CSV there with columns: region_id, epi_week, resp_rate,"
        " completeness"
    )

  jm_health = pd.read_csv(JAMAICA_HEALTH_CSV)
  if "reported_drc" not in jm_health.columns:
    jm_health["reported_drc"] = np.nan

  if DRC_HEALTH_CSV.exists():
    drc_health = pd.read_csv(DRC_HEALTH_CSV)
    drc_health["resp_rate"] = np.nan
    drc_health["completeness"] = 0.0
    return pd.concat([jm_health, drc_health], ignore_index=True)

  print(
      "[Info] No DRC DHIS2 clinic file found; running pure expected-morbidity"
      " inference for DRC."
  )
  return jm_health


def update_drc_silica_priors_from_geology():
  """Updates STUDY_REGIONS silica multipliers dynamically using GLiM + USGS MRDS priors."""
  for region_id, cfg in STUDY_REGIONS.items():
    if cfg["domain"] == 1:  # DRC target corridors
      silica_gdf = compute_spatial_silica_prior(cfg["bbox"], cell_size_deg=0.05)
      mean_s = float(silica_gdf["S_silica"].mean())
      # Map normalized S_silica in [0, 1] to toxicological severity multiplier [1.0, 2.0]
      STUDY_REGIONS[region_id]["silica_factor"] = round(1.0 + mean_s, 3)
      print(
          f"[Geology Prior] {region_id}: mean S_silica={mean_s:.3f} ->"
          f" multiplier={STUDY_REGIONS[region_id]['silica_factor']}"
      )


if __name__ == "__main__":
  # 1. Calibrate DRC crystalline-silica multipliers from GLiM + USGS MRDS geology
  update_drc_silica_priors_from_geology()

  # 2. Fetch or load cached live satellite & meteorological time series
  sar_df = load_or_fetch_sar(force_refresh=False)
  aod_df = load_or_fetch_aod(force_refresh=False)
  era5_df = load_or_fetch_era5(force_refresh=False)

  # 3. Load Jamaica MOHW & DRC health surveillance tables
  health_df = load_epidemiological_records()

  # Azimuthal bearing (radians clockwise from North) from primary mine scar -> population center
  mine_bearings = {
      "JM_St_Ann": 0.60,
      "JM_Manchester": 1.10,
      "JM_Portland_Ctrl": 0.00,
      "DRC_Kolwezi": 0.80,
      "DRC_Ituri_Mongbwalu": 1.40,
  }

  # 4. Harmonize onto W-MON EpiWeeks, screen features via Ledoit-Peche MTS, and train DANN
  panel = build_harmonized_weekly_panel(
      sar_df=sar_df,
      aod_df=aod_df,
      era5_df=era5_df,
      health_df=health_df,
      mine_bearing_rad=mine_bearings,
  )

  alerts_df = train_and_score_underreporting(panel, epochs=45)
  print("\nTop 10 Live DRC Surveillance Alerts:")
  print(alerts_df.head(10).to_string(index=False))