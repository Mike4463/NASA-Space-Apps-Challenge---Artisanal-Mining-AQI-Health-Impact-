from datetime import datetime
from pathlib import Path
import re
import warnings
import zipfile

import asf_search as asf
import earthaccess
import geopandas as gpd
import hyp3_sdk as sdk
import numpy as np
import pandas as pd
import rioxarray as rxr
from shapely.geometry import box
from skimage.feature import graycomatrix, graycoprops

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# 0. DIRECTORY SETUP
# ==============================================================================
WORKSPACE = Path("./asm_pipeline_workspace")
RAW_DIR = WORKSPACE / "raw_rasters"
RTC_DIR = RAW_DIR / "sentinel1_rtc"
AOD_DIR = RAW_DIR / "modis_mcd19a2"

for d in (WORKSPACE, RAW_DIR, RTC_DIR, AOD_DIR):
  d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. SENTINEL-1 RADIOMETRIC TERRAIN CORRECTED (RTC) BACKSCATTER & GLCM TEXTURE
# ==============================================================================
def compute_glcm_variance(
    arr_2d: np.ndarray, levels: int = 32
) -> float:
  """Computes Gray-Level Co-occurrence Matrix (GLCM) contrast/variance on a 2D dB backscatter patch

    to quantify surface roughness from artisanal open-pit mining excavations and tailings piles.
    """
  valid = arr_2d[np.isfinite(arr_2d)]
  if valid.size < 100:
    return np.nan

  # Clip backscatter to typical [-25, 0] dB range and quantize to [0, levels - 1]
  clipped = np.clip(arr_2d, -25.0, 0.0)
  filled = np.where(np.isfinite(clipped), clipped, np.nanmedian(valid))
  quantized = np.floor(((filled + 25.0) / 25.0) * (levels - 1)).astype(
      np.uint8
  )

  # Subsample if raster is very large to keep GLCM fast
  if quantized.shape[0] > 512 or quantized.shape[1] > 512:
    step_r = max(1, quantized.shape[0] // 512)
    step_c = max(1, quantized.shape[1] // 512)
    quantized = quantized[::step_r, ::step_c]

  glcm = graycomatrix(
      quantized,
      distances=[1, 2],
      angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
      levels=levels,
      symmetric=True,
      normed=True,
  )
  return float(np.mean(graycoprops(glcm, "contrast")))


def _extract_scene_timestamp(name: str) -> str | None:
  """Extracts the unique Sentinel-1 acquisition timestamp (e.g., '20240330T110417') from a filename."""
  match = re.search(r"(20\d{6}T\d{6})", name)
  return match.group(1) if match else None


def submit_and_download_s1_rtc(
    region_id: str,
    bbox: tuple[float, float, float, float],
    temporal: tuple[str, str],
    max_scenes: int = 15,
    wait_for_completion: bool = True,
) -> list[Path]:
  """Credit-safe Sentinel-1 RTC manager:

    1. Reuses already-unzipped or downloaded .zip RTC products on local disk.
    2. Recovers existing SUCCEEDED / RUNNING jobs from ASF HyP3 cloud history.
    3. Submits new HyP3 RTC_GAMMA jobs ONLY for granules not yet processed.
    """
  region_rtc_dir = RTC_DIR / region_id
  region_rtc_dir.mkdir(parents=True, exist_ok=True)

  # Unzip any leftover local .zip files that haven't been extracted yet
  for zp in region_rtc_dir.glob("*.zip"):
    out_folder = region_rtc_dir / zp.stem
    if not out_folder.exists():
      with zipfile.ZipFile(zp, "r") as zf:
        zf.extractall(out_folder)

  # Index all locally cached RTC folders by acquisition timestamp
  local_by_ts: dict[str, Path] = {}
  for folder in region_rtc_dir.iterdir():
    if folder.is_dir() and list(folder.rglob("*_VV.tif")):
      ts = _extract_scene_timestamp(folder.name)
      if ts:
        local_by_ts[ts] = folder

  # Query ASF catalog for target granules in the requested window
  min_lon, min_lat, max_lon, max_lat = bbox
  wkt_poly = (
      f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, "
      f"{max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
  )
  search_results = asf.geo_search(
      platform=[asf.PLATFORM.SENTINEL1],
      intersectsWith=wkt_poly,
      processingLevel=[asf.PRODUCT_TYPE.GRD_HD],
      beamMode=[asf.BEAMMODE.IW],
      start=temporal[0],
      end=temporal[1],
      maxResults=max_scenes,
  )

  granule_names = [r.properties["sceneName"] for r in search_results]
  if not granule_names:
    print(f"[{region_id}] No Sentinel-1 scenes found for {temporal}.")
    return list(local_by_ts.values())

  # Check which target granules are still missing from local disk
  missing_granules = []
  resolved_folders: list[Path] = []
  for g_name in granule_names:
    ts = _extract_scene_timestamp(g_name)
    if ts and ts in local_by_ts:
      resolved_folders.append(local_by_ts[ts])
    else:
      missing_granules.append(g_name)

  if not missing_granules:
    print(
        f"[{region_id}] Cache Hit: All {len(resolved_folders)} Sentinel-1 RTC"
        " scenes found locally (0 credits used)."
    )
    return resolved_folders

  # Tier 2: Check existing HyP3 cloud jobs before submitting new ones
  hyp3 = sdk.HyP3()
  job_tag = f"asm_{region_id}"
  existing_jobs = hyp3.find_jobs(name=job_tag)

  cloud_succeeded = sdk.Batch()
  cloud_running = sdk.Batch()
  covered_granules = set()

  for job in existing_jobs:
    if job.status_code in ("SUCCEEDED", "RUNNING", "PENDING"):
      job_granules = job.job_parameters.get("granules", [])
      for jg in job_granules:
        covered_granules.add(jg)
      if job.status_code == "SUCCEEDED" and not job.expired():
        cloud_succeeded += job
      elif job.status_code in ("RUNNING", "PENDING"):
        cloud_running += job

  if len(cloud_succeeded) > 0:
    print(
        f"[{region_id}] Recovering {len(cloud_succeeded)} completed HyP3 jobs"
        " from ASF cloud (0 new credits)..."
    )
    zip_paths = cloud_succeeded.download_files(location=region_rtc_dir)
    for zp in zip_paths:
      out_folder = region_rtc_dir / zp.stem
      if not out_folder.exists():
        with zipfile.ZipFile(zp, "r") as zf:
          zf.extractall(out_folder)
      resolved_folders.append(out_folder)

  # Tier 3: Submit only truly new granules
  to_submit = [g for g in missing_granules if g not in covered_granules]
  new_batch = sdk.Batch()
  for g_name in to_submit:
    new_batch += hyp3.submit_rtc_job(
        granule=g_name,
        name=job_tag,
        dem_matching=True,
        include_dem=False,
        include_inc_map=False,
        radiometry="gamma0",
        scale="decibel",
        resolution=30.0,
    )

  if len(new_batch) > 0:
    print(
        f"[{region_id}] Submitted {len(new_batch)} new HyP3 RTC jobs (Remaining"
        f" credits: {hyp3.check_credits()})."
    )

  active_batch = cloud_running + new_batch
  if len(active_batch) > 0 and wait_for_completion:
    active_batch = hyp3.watch(active_batch)
    zip_paths = active_batch.download_files(location=region_rtc_dir)
    for zp in zip_paths:
      out_folder = region_rtc_dir / zp.stem
      if not out_folder.exists():
        with zipfile.ZipFile(zp, "r") as zf:
          zf.extractall(out_folder)
      resolved_folders.append(out_folder)

  return resolved_folders


def extract_s1_rtc_zonal_stats(
    region_id: str,
    bbox: tuple[float, float, float, float],
    rtc_folders: list[Path],
) -> pd.DataFrame:
  """Clips downloaded HyP3 RTC *_VV.tif and *_VH.tif GeoTIFFs to the study bounding box

    and returns a DataFrame matching `sar_df` (region_id, date, vv_db, vh_db, glcm_var).
    """
  geom_gdf = gpd.GeoDataFrame(geometry=[box(*bbox)], crs="EPSG:4326")
  records = []

  for folder in rtc_folders:
    vv_files = list(folder.rglob("*_VV.tif"))
    vh_files = list(folder.rglob("*_VH.tif"))
    if not vv_files or not vh_files:
      continue

    vv_path, vh_path = vv_files[0], vh_files[0]

    date_match = re.search(r"_(20\d{6})T\d{6}_", vv_path.name)
    if not date_match:
      continue
    acq_date = datetime.strptime(date_match.group(1), "%Y%m%d").strftime(
        "%Y-%m-%d"
    )

    with (
        rxr.open_rasterio(vv_path, masked=True) as da_vv,
        rxr.open_rasterio(vh_path, masked=True) as da_vh,
    ):
      bbox_utm = geom_gdf.to_crs(da_vv.rio.crs).geometry

      vv_clip = da_vv.rio.clip(bbox_utm, drop=True).squeeze()
      vh_clip = da_vh.rio.clip(bbox_utm, drop=True).squeeze()

      vv_vals = vv_clip.to_numpy()
      vh_vals = vh_clip.to_numpy()

      vv_mean = float(np.nanmean(np.where(vv_vals == 0, np.nan, vv_vals)))
      vh_mean = float(np.nanmean(np.where(vh_vals == 0, np.nan, vh_vals)))
      glcm_var = compute_glcm_variance(
          np.where(vv_vals == 0, np.nan, vv_vals)
      )

      records.append({
          "region_id": region_id,
          "date": acq_date,
          "vv_db": round(vv_mean, 4),
          "vh_db": round(vh_mean, 4),
          "glcm_var": round(glcm_var, 4),
      })

  if not records:
    return pd.DataFrame(
        columns=["region_id", "date", "vv_db", "vh_db", "glcm_var"]
    )

  return (
      pd.DataFrame(records)
      .groupby(["region_id", "date"], as_index=False)
      .mean()
  )


# ==============================================================================
# 2. NASA MODIS MAIAC 1KM AOD (MCD19A2) CLOUD-MASKED EXTRACTOR
# ==============================================================================
def extract_mcd19a2_zonal_stats(
    region_id: str,
    bbox: tuple[float, float, float, float],
    temporal: tuple[str, str],
    max_granules: int = 120,
) -> pd.DataFrame:
  """Queries and downloads MODIS MAIAC 1km Aerosol Optical Depth (MCD19A2 v061),

    applies the AOD_QA bitmask to keep only best-quality cloud-free retrievals,
    reprojects from Sinusoidal to WGS84, and computes regional mean 550nm AOD.
    """
  earthaccess.login(strategy="netrc")
  region_aod_dir = AOD_DIR / region_id
  region_aod_dir.mkdir(parents=True, exist_ok=True)

  granules = earthaccess.search_data(
      short_name="MCD19A2",
      bounding_box=bbox,
      temporal=temporal,
      count=max_granules,
  )
  if not granules:
    print(f"[{region_id}] No MCD19A2 granules found for {temporal}.")
    return pd.DataFrame(columns=["region_id", "date", "aod_550"])

  downloaded_files = earthaccess.download(
      granules, local_path=str(region_aod_dir)
  )
  geom_gdf = gpd.GeoDataFrame(geometry=[box(*bbox)], crs="EPSG:4326")
  records = []

  for hdf_path in downloaded_files:
    hdf_path = Path(hdf_path)
    # Extract acquisition year + Julian day-of-year from filename: MCD19A2.AYYYYDDD...
    doy_match = re.search(r"\.A(20\d{2})(\d{3})\.", hdf_path.name)
    if not doy_match:
      continue
    acq_date = datetime.strptime(
        f"{doy_match.group(1)}{doy_match.group(2)}", "%Y%j"
    ).strftime("%Y-%m-%d")

    try:
      opened = rxr.open_rasterio(hdf_path, masked=True)
      if isinstance(opened, list):
        sub_ds = next(
            (
                d
                for d in opened
                if any("Optical_Depth_055" in str(v) for v in d.data_vars)
            ),
            opened[0],
        )
      else:
        sub_ds = opened

      aod_var = next(
          (v for v in sub_ds.data_vars if "Optical_Depth_055" in str(v)), None
      )
      qa_var = next((v for v in sub_ds.data_vars if "AOD_QA" in str(v)), None)
      if aod_var is None:
        continue

      da_aod = sub_ds[aod_var]
      # Scale factor 0.001 if raw integer values exceed physical AOD range
      raw_max = float(np.nanmax(da_aod.to_numpy()))
      if np.isfinite(raw_max) and raw_max > 10.0:
        da_aod = da_aod * 0.001

      # Apply MAIAC MCD19A2 v061 AOD_QA bitmask:
      # Bits 0-2 == 1 (001 = Clear) and Bits 8-11 == 0 (0000 = Best quality AOD)
      if qa_var is not None:
        qa_arr = np.nan_to_num(sub_ds[qa_var].to_numpy(), nan=65535).astype(
            np.uint16
        )
        cloud_clear = (qa_arr & 0b00000111) == 1
        best_qa = ((qa_arr >> 8) & 0b00001111) == 0
        clear_mask = cloud_clear & best_qa
        da_aod = da_aod.where(clear_mask)

      # Collapse multi-orbit dimension (band) to daily pixel median
      if "band" in da_aod.dims:
        da_aod = da_aod.median(dim="band", skipna=True)

      # Clip to study corridor in native MODIS Sinusoidal CRS
      bbox_sinu = geom_gdf.to_crs(da_aod.rio.crs).geometry
      da_clip = da_aod.rio.clip(bbox_sinu, drop=True)
      vals = da_clip.to_numpy()

      # Close open GDAL handles
      if isinstance(opened, list):
        for d in opened:
          d.close()
      else:
        opened.close()

      # Require at least 3% valid cloud-free pixel coverage over the corridor
      valid_frac = np.isfinite(vals).mean()
      if valid_frac < 0.03:
        continue

      mean_aod = float(np.nanmean(vals))
      if np.isfinite(mean_aod) and 0.0 <= mean_aod <= 5.0:
        records.append({
            "region_id": region_id,
            "date": acq_date,
            "aod_550": round(mean_aod, 4),
        })
    except Exception as e:
      print(f"[Warning] Skipping {hdf_path.name}: {e}")
      continue

  if not records:
    return pd.DataFrame(columns=["region_id", "date", "aod_550"])

  return (
      pd.DataFrame(records)
      .groupby(["region_id", "date"], as_index=False)
      .mean()
  )


# ==============================================================================
# 3. QUICK PILOT TEST ON ST. ANN, JAMAICA
# ==============================================================================
if __name__ == "__main__":
  st_ann_bbox = (-77.45, 18.20, -77.05, 18.48)
  window = ("2024-01-01", "2024-03-31")

  # Reuses cached local folders automatically (0 HyP3 credits spent)
  rtc_dirs = submit_and_download_s1_rtc(
      region_id="JM_St_Ann",
      bbox=st_ann_bbox,
      temporal=window,
      max_scenes=6,
      wait_for_completion=True,
  )
  live_sar_df = extract_s1_rtc_zonal_stats(
      "JM_St_Ann", st_ann_bbox, rtc_dirs
  )
  print("Live Sentinel-1 RTC Zonal Series:\n", live_sar_df.to_string(index=False))

  # Download & extract cloud-masked MODIS MCD19A2 AOD zonal time series
  live_aod_df = extract_mcd19a2_zonal_stats(
      region_id="JM_St_Ann",
      bbox=st_ann_bbox,
      temporal=window,
      max_granules=30,
  )
  print(
      "\nLive MODIS MCD19A2 AOD Zonal Series:\n",
      live_aod_df.head(10).to_string(index=False),
  )
