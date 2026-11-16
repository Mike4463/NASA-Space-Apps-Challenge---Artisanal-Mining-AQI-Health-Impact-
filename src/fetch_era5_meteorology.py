from pathlib import Path
import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

ERA5_DIR = Path("./asm_pipeline_workspace/raw_rasters/era5")
ERA5_DIR.mkdir(parents=True, exist_ok=True)


def compute_relative_humidity(
    t2m_k: xr.DataArray, d2m_k: xr.DataArray
) -> xr.DataArray:
  """Computes Relative Humidity (%) from 2m air temperature and 2m dewpoint (Kelvin)

    using the August-Roche-Magnus approximation.
    """
  t_c = t2m_k - 273.15
  td_c = d2m_k - 273.15
  es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
  e = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
  return np.clip((e / es) * 100.0, 0.0, 100.0)


def download_and_extract_era5(
    regions: dict[str, dict],
    temporal: tuple[str, str],
    out_csv: Path = Path("./asm_pipeline_workspace/live_era5.csv"),
) -> pd.DataFrame:
  """Downloads ERA5 single-level reanalysis per region in yearly chunks (to stay well within

    CDS request size limits) and returns a daily DataFrame matching `era5_df`:
    (region_id, date, u10, v10, blh_m, rh_pct).
    """
  client = cdsapi.Client()  # Reads ~/.cdsapirc
  start_dt = pd.to_datetime(temporal[0])
  end_dt = pd.to_datetime(temporal[1])
  years = [str(y) for y in range(start_dt.year, end_dt.year + 1)]

  all_records = []

  for region_id, cfg in regions.items():
    min_lon, min_lat, max_lon, max_lat = cfg["bbox"]
    # Pad bbox by 0.25 deg (1 ERA5 grid cell) so small corridors never slice empty
    area_north_west_south_east = [
        round(max_lat + 0.25, 2),
        round(min_lon - 0.25, 2),
        round(min_lat - 0.25, 2),
        round(max_lon + 0.25, 2),
    ]

    for yr in years:
      nc_file = ERA5_DIR / f"era5_{region_id}_{yr}.nc"
      if not nc_file.exists():
        print(f"[ERA5] Requesting {region_id} ({yr}) from Copernicus CDS...")
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": ["reanalysis"],
                "variable": [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "boundary_layer_height",
                    "2m_temperature",
                    "2m_dewpoint_temperature",
                ],
                "year": [yr],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": area_north_west_south_east,
            },
            str(nc_file),
        )

      with xr.open_dataset(nc_file) as ds:
        # Handle CDS valid_time vs time coordinate naming
        time_dim = "valid_time" if "valid_time" in ds.coords else "time"

        # Spatial mean across the corridor bounding box
        spatial_mean = ds.mean(dim=["latitude", "longitude"], skipna=True)
        rh = compute_relative_humidity(spatial_mean["t2m"], spatial_mean["d2m"])

        df_yr = pd.DataFrame({
            "region_id": region_id,
            "datetime": pd.to_datetime(spatial_mean[time_dim].to_numpy()),
            "u10": spatial_mean["u10"].to_numpy(),
            "v10": spatial_mean["v10"].to_numpy(),
            "blh_m": spatial_mean["blh"].to_numpy(),
            "rh_pct": rh.to_numpy(),
        })

        # Filter to exact requested temporal window and aggregate 6-hourly -> daily
        df_yr = df_yr[
            (df_yr["datetime"] >= start_dt) & (df_yr["datetime"] <= end_dt)
        ].copy()
        df_yr["date"] = df_yr["datetime"].dt.strftime("%Y-%m-%d")

        daily_df = df_yr.groupby(["region_id", "date"], as_index=False).agg(
            u10=("u10", "mean"),
            v10=("v10", "mean"),
            blh_m=("blh_m", "mean"),
            rh_pct=("rh_pct", "mean"),
        )
        all_records.append(daily_df)

  era5_df = pd.concat(all_records, ignore_index=True).sort_values(
      ["region_id", "date"]
  )
  era5_df.to_csv(out_csv, index=False)
  print(f"[ERA5] Saved {len(era5_df)} daily meteorological rows to {out_csv}")
  return era5_df


if __name__ == "__main__":
  from asm_health_surveillance_pipeline import STUDY_REGIONS, TEMPORAL_RANGE

  download_and_extract_era5(STUDY_REGIONS, TEMPORAL_RANGE)