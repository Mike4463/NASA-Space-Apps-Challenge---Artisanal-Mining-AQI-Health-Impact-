from pathlib import Path
import warnings

import asf_search as asf
import earthaccess
import nonlinshrink as nls
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# 0. CONFIGURATION & STUDY CORRIDORS
# ==============================================================================
STUDY_REGIONS = {
    # Source Domain: Jamaica (Bauxite Mining Parishes + Non-Mining Control)
    "JM_St_Ann": {
        "domain": 0,
        "bbox": (-77.45, 18.20, -77.05, 18.48),
        "silica_factor": 1.0,
    },
    "JM_Manchester": {
        "domain": 0,
        "bbox": (-77.65, 17.85, -77.35, 18.22),
        "silica_factor": 1.0,
    },
    "JM_Portland_Ctrl": {
        "domain": 0,
        "bbox": (-76.75, 18.02, -76.25, 18.25),
        "silica_factor": 1.0,
    },
    # Target Domain: DRC Artisanal & Small-Scale Mining Corridors (Kolwezi / Ituri)
    # silica_factor > 1.0 accounts for higher free crystalline silica in DRC quartz/cobalt veins vs. JM bauxite
    "DRC_Kolwezi": {
        "domain": 1,
        "bbox": (25.25, -10.85, 25.80, -10.55),
        "silica_factor": 1.65,
    },
    "DRC_Ituri_Mongbwalu": {
        "domain": 1,
        "bbox": (29.85, 1.80, 30.25, 2.10),
        "silica_factor": 1.80,
    },
}

TEMPORAL_RANGE = ("2023-01-01", "2024-12-31")
DATA_DIR = Path("./asm_pipeline_workspace")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def snap_to_epi_week(date_series: pd.Series) -> pd.Series:
  """Consistently snaps timestamps to Monday-anchored epidemiological week start dates."""
  return pd.to_datetime(date_series).dt.to_period("W-MON").dt.start_time


# ==============================================================================
# 1. NASA EARTHDATA & ASF SENTINEL-1 QUERY ENGINE
# ==============================================================================
class RemoteSensingHarvester:

  def __init__(self, authenticate: bool = True):
    if authenticate:
      earthaccess.login(strategy="netrc")

  def query_nasa_aerosols(
      self,
      bbox: tuple[float, float, float, float],
      temporal: tuple[str, str],
      limit: int = 50,
  ):
    """Queries NASA CMR via earthaccess for MODIS MAIAC 1km AOD (MCD19A2)

        and Sentinel-5P TROPOMI Aerosol Index (S5P_L2__AER_AI_HiR).
        """
    aod_granules = earthaccess.search_data(
        short_name="MCD19A2",
        bounding_box=bbox,
        temporal=temporal,
        count=limit,
    )
    tropomi_granules = earthaccess.search_data(
        short_name="S5P_L2__AER_AI_HiR",
        bounding_box=bbox,
        temporal=temporal,
        count=limit,
    )
    return {"MCD19A2": aod_granules, "TROPOMI_AER": tropomi_granules}

  def query_asf_sentinel1(
      self,
      bbox: tuple[float, float, float, float],
      temporal: tuple[str, str],
      limit: int = 50,
  ):
    """Queries Alaska Satellite Facility (ASF) for Sentinel-1 IW GRD_HD dual-pol (VV+VH) scenes."""
    min_lon, min_lat, max_lon, max_lat = bbox
    wkt_poly = (
        f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, "
        f"{max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
    )
    results = asf.geo_search(
        platform=[asf.PLATFORM.SENTINEL1],
        intersectsWith=wkt_poly,
        processingLevel=[asf.PRODUCT_TYPE.GRD_HD],
        beamMode=[asf.BEAMMODE.IW],
        start=temporal[0],
        end=temporal[1],
        maxResults=limit,
    )
    return results

  def harvest_catalogs(self, temporal: tuple[str, str]) -> pd.DataFrame:
    """Queries metadata across all Jamaica and DRC regions and logs granule availability."""
    summary_rows = []
    for region_id, cfg in STUDY_REGIONS.items():
      nasa_hits = self.query_nasa_aerosols(cfg["bbox"], temporal, limit=25)
      asf_hits = self.query_asf_sentinel1(cfg["bbox"], temporal, limit=25)
      summary_rows.append({
          "region_id": region_id,
          "domain": "Jamaica" if cfg["domain"] == 0 else "DRC",
          "mcd19a2_granules": len(nasa_hits["MCD19A2"]),
          "tropomi_granules": len(nasa_hits["TROPOMI_AER"]),
          "sentinel1_scenes": len(asf_hits),
      })
    df_catalog = pd.DataFrame(summary_rows)
    df_catalog.to_csv(DATA_DIR / "satellite_granule_manifest.csv", index=False)
    return df_catalog


# ==============================================================================
# 2. SPATIO-TEMPORAL HARMONIZATION & PHYSICS FEATURE ENGINEERING
# ==============================================================================
def build_harmonized_weekly_panel(
    sar_df: pd.DataFrame,  # cols: region_id, date, vv_db, vh_db, glcm_var
    aod_df: pd.DataFrame,  # cols: region_id, date, aod_550, tropomi_ai
    era5_df: pd.DataFrame,  # cols: region_id, date, u10, v10, blh_m, rh_pct
    health_df: (
        pd.DataFrame
    ),  # cols: region_id, epi_week, resp_rate, completeness, reported_drc
    mine_bearing_rad: dict[str, float],
) -> pd.DataFrame:
  """Aligns asynchronous SAR, AOD, ERA5 meteorology, and weekly clinic counts onto unified EpiWeeks."""
  sar_df = sar_df.copy()
  aod_df = aod_df.copy()
  era5_df = era5_df.copy()

  for df in (sar_df, aod_df, era5_df):
    df["epi_week"] = snap_to_epi_week(df["date"])

  # Aggregate to weekly cadence per region
  sar_wk = (
      sar_df.groupby(["region_id", "epi_week"], as_index=False)
      .agg(
          vv_db=("vv_db", "mean"),
          vh_db=("vh_db", "mean"),
          sar_texture=("glcm_var", "mean"),
      )
      .sort_values(["region_id", "epi_week"])
  )

  aod_wk = aod_df.groupby(["region_id", "epi_week"], as_index=False).agg(
      aod_median=("aod_550", "median"),
      aod_p90=("aod_550", lambda s: np.nanpercentile(s, 90)),
      tropomi_ai=("tropomi_ai", "mean"),
  )

  era5_wk = era5_df.groupby(["region_id", "epi_week"], as_index=False).agg(
      u10=("u10", "mean"),
      v10=("v10", "mean"),
      blh_m=("blh_m", "mean"),
      rh_pct=("rh_pct", "mean"),
  )

  # Full outer merge on (region_id, epi_week)
  panel = (
      sar_wk.merge(aod_wk, on=["region_id", "epi_week"], how="outer")
      .merge(era5_wk, on=["region_id", "epi_week"], how="outer")
      .sort_values(["region_id", "epi_week"])
      .reset_index(drop=True)
  )

  raw_cols = [
      "vv_db",
      "vh_db",
      "sar_texture",
      "aod_median",
      "aod_p90",
      "tropomi_ai",
      "u10",
      "v10",
      "blh_m",
      "rh_pct",
  ]
  # Forward-fill sparse SAR orbits and interpolate short AOD cloud gaps
  panel[raw_cols] = panel.groupby("region_id")[raw_cols].transform(
      lambda s: s.interpolate(method="linear", limit=3).bfill().ffill()
  )

  # --- Domain-Invariant Physical Interaction Features ---
  panel["vh_vv_ratio"] = panel["vh_db"] - panel["vv_db"]
  panel["sar_disturbance_delta"] = panel["vv_db"] - panel.groupby("region_id")[
      "vv_db"
  ].transform(lambda s: s.rolling(12, min_periods=1).mean())
  # Ground-level PM proxy: column AOD trapped beneath planetary boundary layer height (km)
  panel["surface_pm_proxy"] = panel["aod_p90"] / (
      (panel["blh_m"] / 1000.0) + 0.1
  )
  panel["wind_speed"] = np.sqrt(panel["u10"] ** 2 + panel["v10"] ** 2)

  # Upwind advection alignment (wind blowing from mining scar toward population center)
  bearing = panel["region_id"].map(mine_bearing_rad).fillna(0.0)
  advection = panel["u10"] * np.sin(bearing) + panel["v10"] * np.cos(bearing)
  panel["upwind_dust_flux"] = (
      np.maximum(0.0, advection) * panel["surface_pm_proxy"]
  )

  # Merge health surveillance data snapped to the exact same W-MON period anchor
  health_df = health_df.copy()
  health_df["epi_week"] = snap_to_epi_week(health_df["epi_week"])
  panel = panel.merge(health_df, on=["region_id", "epi_week"], how="left")
  panel["domain"] = panel["region_id"].map(
      lambda r: STUDY_REGIONS[r]["domain"]
  )
  panel["silica_factor"] = panel["region_id"].map(
      lambda r: STUDY_REGIONS[r]["silica_factor"]
  )
  return panel


# ==============================================================================
# 3. LEDOIT-PECHE NONLINEAR SHRINKAGE + MAHALANOBIS-TAGUCHI + RMT-CORAL
# ==============================================================================
def estimate_nonlinear_precision(
    X: np.ndarray, min_samples: int = 50
) -> tuple[np.ndarray, np.ndarray]:
  n, p = X.shape
  if n >= min_samples and n > p:
    try:
      # Add tiny full-rank jitter to prevent exact float collinearity from triggering singular checks
      rng = np.random.default_rng(0)
      X_reg = X + rng.normal(0.0, 1e-6, size=X.shape)
      cov_clean = nls.shrink_cov(X_reg)
    except ValueError:
      cov_clean = LedoitWolf().fit(X).covariance_
  else:
    cov_clean = LedoitWolf().fit(X).covariance_
  precision = np.linalg.pinv(cov_clean, rcond=1e-6)
  return cov_clean, precision


def generate_taguchi_l12_subset(n_features: int) -> np.ndarray:
  """Generates a standard 12-run Plackett-Burman / Taguchi L12 two-level orthogonal array."""
  l12_base = np.array(
      [
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
          [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
          [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
          [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
          [0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1],
          [0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
          [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1],
          [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1],
          [0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
          [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
      ],
      dtype=int,
  )
  if n_features <= 11:
    return l12_base[:, :n_features]
  repeats = int(np.ceil(n_features / 11))
  return np.tile(l12_base, (1, repeats))[:, :n_features]


class LedoitPecheMTS:

  def __init__(self):
    self.mu_normal = None
    self.std_normal = None
    self.cov_normal = None
    self.precision_normal = None
    self.selected_idx = None
    self.gains = None

  def fit_screen(
      self,
      X_normal: np.ndarray,
      X_abnormal: np.ndarray,
      feature_names: list[str],
  ):
    self.mu_normal = np.mean(X_normal, axis=0)
    self.std_normal = np.std(X_normal, axis=0) + 1e-8

    Z_norm = (X_normal - self.mu_normal) / self.std_normal
    Z_abn = (X_abnormal - self.mu_normal) / self.std_normal

    self.cov_normal, self.precision_normal = estimate_nonlinear_precision(
        Z_norm
    )

    oa = generate_taguchi_l12_subset(X_normal.shape[1])
    snrs = np.zeros(oa.shape[0])

    for r in range(oa.shape[0]):
      idx = np.where(oa[r] == 1)[0]
      if len(idx) == 0:
        snrs[r] = -100.0
        continue
      sub_cov = self.cov_normal[np.ix_(idx, idx)]
      sub_prec = np.linalg.pinv(sub_cov, rcond=1e-6)
      diff = Z_abn[:, idx]
      md = np.maximum(np.sum(diff @ sub_prec * diff, axis=1) / len(idx), 1e-8)
      # Larger-the-better Taguchi Signal-to-Noise Ratio
      snrs[r] = -10.0 * np.log10(np.mean(1.0 / (md**2)))

    self.gains = {}
    self.selected_idx = []
    for f, name in enumerate(feature_names):
      snr_in = np.mean(snrs[oa[:, f] == 1])
      snr_out = np.mean(snrs[oa[:, f] == 0])
      gain = float(snr_in - snr_out)
      self.gains[name] = gain
      if gain > 0.0:
        self.selected_idx.append(f)

    # Fallback if all gains <= 0 on synthetic/noisy fold
    if not self.selected_idx:
      self.selected_idx = list(range(len(feature_names)))

    return [feature_names[i] for i in self.selected_idx], self.gains

  def rmt_coral_align(
      self, X_source: np.ndarray, X_target: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    """Standardizes both domains and aligns source (Jamaica) signal covariance to target (DRC)

        using Ledoit-Peche nonlinear covariance estimators.
        """
    Zs = (X_source[:, self.selected_idx] - self.mu_normal[self.selected_idx]) / (
        self.std_normal[self.selected_idx]
    )
    mu_t = np.mean(X_target[:, self.selected_idx], axis=0)
    std_t = np.std(X_target[:, self.selected_idx], axis=0) + 1e-8
    Zt = (X_target[:, self.selected_idx] - mu_t) / std_t

    cov_s, _ = estimate_nonlinear_precision(Zs)
    cov_t, _ = estimate_nonlinear_precision(Zt)

    es, vs = np.linalg.eigh(cov_s)
    et, vt = np.linalg.eigh(cov_t)

    cov_s_inv_sqrt = (
        vs @ np.diag(1.0 / np.sqrt(np.maximum(es, 1e-6))) @ vs.T
    )
    cov_t_sqrt = vt @ np.diag(np.sqrt(np.maximum(et, 1e-6))) @ vt.T

    Zs_aligned = Zs @ cov_s_inv_sqrt @ cov_t_sqrt
    return Zs_aligned, Zt


# ==============================================================================
# 4. SPATIO-TEMPORAL TENSOR DATASET & DOMAIN ADVERSARIAL NEURAL NETWORK (DANN)
# ==============================================================================
class EpiWindowDataset(Dataset):

  def __init__(
      self,
      panel_df: pd.DataFrame,
      X_scaled: np.ndarray,
      seq_len: int = 12,
  ):
    self.samples = []
    panel_df = panel_df.reset_index(drop=True)
    panel_df["_row_idx"] = np.arange(len(panel_df))

    for region_id, grp in panel_df.groupby("region_id"):
      grp = grp.sort_values("epi_week").reset_index(drop=True)
      rows = grp["_row_idx"].to_numpy()
      domain_id = int(grp["domain"].iloc[0])
      silica = float(grp["silica_factor"].iloc[0])

      for t in range(seq_len - 1, len(grp)):
        win_idx = rows[t - seq_len + 1 : t + 1]
        y_val = grp.loc[t, "resp_rate"]
        comp = (
            grp.loc[t, "completeness"]
            if pd.notna(grp.loc[t, "completeness"])
            else 0.0
        )
        rep_drc = (
            grp.loc[t, "reported_drc"]
            if pd.notna(grp.loc[t, "reported_drc"])
            else np.nan
        )

        valid_label = (
            (domain_id == 0) and pd.notna(y_val) and (comp >= 0.5)
        )

        self.samples.append({
            "x": torch.tensor(X_scaled[win_idx], dtype=torch.float32),
            "y": torch.tensor(
                0.0 if pd.isna(y_val) else float(y_val), dtype=torch.float32
            ),
            "mask": torch.tensor(
                1.0 if valid_label else 0.0, dtype=torch.float32
            ),
            "domain": torch.tensor(domain_id, dtype=torch.long),
            "silica_factor": torch.tensor(silica, dtype=torch.float32),
            "region_id": region_id,
            "epi_week": str(grp.loc[t, "epi_week"].date()),
            "reported_drc": (
                float("nan") if pd.isna(rep_drc) else float(rep_drc)
            ),
        })

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    return self.samples[idx]


class GradReverse(Function):

  @staticmethod
  def forward(ctx, x, alpha):
    ctx.alpha = alpha
    return x.view_as(x)

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output.neg() * ctx.alpha, None


class ASMHealthDANN(nn.Module):

  def __init__(self, in_features: int, hidden_dim: int = 48):
    super().__init__()
    self.gru = nn.GRU(
        in_features, hidden_dim, num_layers=2, batch_first=True, dropout=0.15
    )
    self.health_regressor = nn.Sequential(
        nn.Linear(hidden_dim, 32),
        nn.GELU(),
        nn.Linear(32, 1),
        nn.Softplus(),
    )
    self.domain_classifier = nn.Sequential(
        nn.Linear(hidden_dim, 32),
        nn.GELU(),
        nn.Linear(32, 2),
    )

  def forward(self, x: torch.Tensor, alpha: float = 1.0):
    _, h_n = self.gru(x)
    latent = h_n[-1]
    health_pred = self.health_regressor(latent)
    domain_logits = self.domain_classifier(GradReverse.apply(latent, alpha))
    return health_pred, domain_logits


# ==============================================================================
# 5. TRAINING & DRC SURVEILLANCE GAP (UNDERREPORTING) SCORING
# ==============================================================================
def train_and_score_underreporting(panel_df: pd.DataFrame, epochs: int = 35):
  feature_cols = [
      "vv_db",
      "sar_texture",
      "aod_median",
      "aod_p90",
      "tropomi_ai",
      "vh_vv_ratio",
      "sar_disturbance_delta",
      "surface_pm_proxy",
      "wind_speed",
      "upwind_dust_flux",
      "rh_pct",
  ]

  jm_df = panel_df[panel_df["domain"] == 0].copy().reset_index(drop=True)
  drc_df = panel_df[panel_df["domain"] == 1].copy().reset_index(drop=True)

  # Define Jamaica "Normal" (control parish / low exposure) vs "Abnormal" (mining + high respiratory)
  valid_jm = jm_df.dropna(subset=["resp_rate"])
  normal_mask = (valid_jm["region_id"] == "JM_Portland_Ctrl") & (
      valid_jm["resp_rate"] <= valid_jm["resp_rate"].median()
  )
  abnormal_mask = (valid_jm["region_id"].isin(["JM_St_Ann", "JM_Manchester"])) & (
      valid_jm["resp_rate"] > valid_jm["resp_rate"].quantile(0.65)
  )

  X_norm = valid_jm.loc[normal_mask, feature_cols].to_numpy()
  X_abn = valid_jm.loc[abnormal_mask, feature_cols].to_numpy()
  assert (
      len(X_norm) > 0 and len(X_abn) > 0
  ), "Normal or Abnormal reference set is empty—check epi_week alignment."

  mts = LedoitPecheMTS()
  selected_names, gains = mts.fit_screen(
      X_normal=X_norm,
      X_abnormal=X_abn,
      feature_names=feature_cols,
  )
  print(
      f"[MTS + Ledoit-Peche] Selected {len(selected_names)}/{len(feature_cols)}"
      f" features: {selected_names}"
  )

  # Align source (Jamaica) covariance to target (DRC) signal subspace
  Zs_aligned, Zt = mts.rmt_coral_align(
      jm_df[feature_cols].to_numpy(),
      drc_df[feature_cols].to_numpy(),
  )

  jm_ds = EpiWindowDataset(jm_df, Zs_aligned, seq_len=12)
  drc_ds = EpiWindowDataset(drc_df, Zt, seq_len=12)
  combined_loader = DataLoader(
      ConcatDataset([jm_ds, drc_ds]), batch_size=32, shuffle=True
  )

  model = ASMHealthDANN(in_features=len(selected_names), hidden_dim=48)
  optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

  # Train DANN with Ganin et al. progressive GRL schedule
  for epoch in range(epochs):
    model.train()
    p = epoch / float(epochs)
    alpha = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

    for batch in combined_loader:
      optimizer.zero_grad()
      pred_y, pred_dom = model(batch["x"], alpha=alpha)

      y_true = batch["y"].unsqueeze(1)
      mask = batch["mask"].unsqueeze(1)
      loss_health = (
          F.huber_loss(pred_y, y_true, reduction="none", delta=1.5) * mask
      ).sum() / (mask.sum() + 1e-6)
      loss_domain = F.cross_entropy(pred_dom, batch["domain"])

      loss = loss_health + 0.30 * loss_domain
      loss.backward()
      optimizer.step()

  # Estimate residual std on Jamaica validation windows for Z-score standardization
  model.eval()
  jm_residuals = []
  with torch.no_grad():
    for sample in jm_ds:
      if sample["mask"].item() == 1.0:
        pred, _ = model(sample["x"].unsqueeze(0), alpha=0.0)
        jm_residuals.append(abs(pred.item() - sample["y"].item()))
  sigma_pred = max(float(np.std(jm_residuals)), 0.5)

  # Run inference across DRC mining corridors to detect underreporting gaps
  drc_results = []
  with torch.no_grad():
    for sample in drc_ds:
      raw_pred, dom_logits = model(sample["x"].unsqueeze(0), alpha=0.0)
      # Adjust predicted incidence by DRC crystalline silica toxicological factor
      expected_morbidity = raw_pred.item() * sample["silica_factor"].item()
      reported = sample["reported_drc"]
      domain_conf = torch.softmax(dom_logits, dim=1)[0, 1].item()

      gap_score = (
          (expected_morbidity - reported) / sigma_pred
          if not np.isnan(reported)
          else np.nan
      )

      drc_results.append({
          "region_id": sample["region_id"],
          "epi_week": sample["epi_week"],
          "expected_resp_incidence_per_10k": round(expected_morbidity, 2),
          "officially_reported_per_10k": (
              round(reported, 2) if not np.isnan(reported) else None
          ),
          "underreporting_gap_zscore": (
              round(gap_score, 2) if not np.isnan(gap_score) else None
          ),
          "domain_alignment_prob": round(domain_conf, 3),
          "alert_flag": (
              "HIGH_UNDERREPORTING_SUSPECTED"
              if (not np.isnan(gap_score) and gap_score > 2.0)
              else "NORMAL_OR_UNVERIFIED"
          ),
      })

  df_drc_alerts = pd.DataFrame(drc_results).sort_values(
      "underreporting_gap_zscore", ascending=False
  )
  out_path = DATA_DIR / "drc_underreporting_surveillance_alerts.csv"
  df_drc_alerts.to_csv(out_path, index=False)
  print(
      f"[Inference Complete] Saved {len(df_drc_alerts)} DRC weekly risk"
      f" assessments to {out_path}"
  )
  return df_drc_alerts


# ==============================================================================
# 6. DRY-RUN DEMONSTRATION WITH REALISTIC SYNTHETIC DATA
# ==============================================================================
if __name__ == "__main__":
  # Step A: Optionally query live NASA CMR & ASF catalogs (uncomment when credentials are set)
  # harvester = RemoteSensingHarvester(authenticate=True)
  # print(harvester.harvest_catalogs(TEMPORAL_RANGE))

  # Step B: Construct realistic multi-region spatio-temporal streams to verify end-to-end execution
  rng = np.random.default_rng(42)
  dates = pd.date_range("2023-01-02", "2024-12-30", freq="W-MON")

  sar_list, aod_list, era5_list, health_list = [], [], [], []
  for region_id, cfg in STUDY_REGIONS.items():
    is_mining = region_id != "JM_Portland_Ctrl"
    n = len(dates)
    seasonal = np.sin(np.linspace(0, 4 * np.pi, n))

    # SAR backscatter: stronger surface disturbance trend in mining zones
    vv = (
        (-11.5 if is_mining else -14.0)
        + (0.015 * np.arange(n) if is_mining else 0.0)
        + rng.normal(0, 0.4, n)
    )
    vh = vv - 6.5 + rng.normal(0, 0.3, n)
    glcm = (2.8 if is_mining else 1.2) + rng.normal(0, 0.25, n)

    # AOD & TROPOMI: dust spikes in dry seasons + mining plumes
    aod = np.clip(
        (0.38 if is_mining else 0.14) + 0.12 * seasonal + rng.normal(0, 0.06, n),
        0.05,
        1.5,
    )
    tropomi = aod * 2.4 + rng.normal(0, 0.15, n)

    # ERA5 meteorology
    u10 = 2.5 + 1.2 * seasonal + rng.normal(0, 0.6, n)
    v10 = -1.8 + rng.normal(0, 0.6, n)
    blh = 950.0 - 180.0 * seasonal + rng.normal(0, 60.0, n)
    rh = 74.0 + 8.0 * seasonal + rng.normal(0, 3.0, n)

    sar_list.append(
        pd.DataFrame({
            "region_id": region_id,
            "date": dates,
            "vv_db": vv,
            "vh_db": vh,
            "glcm_var": glcm,
        })
    )
    aod_list.append(
        pd.DataFrame({
            "region_id": region_id,
            "date": dates,
            "aod_550": aod,
            "tropomi_ai": tropomi,
        })
    )
    era5_list.append(
        pd.DataFrame({
            "region_id": region_id,
            "date": dates,
            "u10": u10,
            "v10": v10,
            "blh_m": blh,
            "rh_pct": rh,
        })
    )

    # Epidemiological ground truth (known in Jamaica; sparse/underreported in DRC)
    true_resp = (
        4.0
        + 14.0 * (aod / (blh / 1000.0))
        + (3.5 if is_mining else 0.0)
        + rng.normal(0, 0.8, n)
    )
    if cfg["domain"] == 0:
      health_list.append(
          pd.DataFrame({
              "region_id": region_id,
              "epi_week": dates,
              "resp_rate": np.maximum(true_resp, 0.5),
              "completeness": rng.choice([0.0, 1.0], size=n, p=[0.1, 0.9]),
              "reported_drc": np.nan,
          })
      )
    else:
      # Simulate severe DRC clinical underreporting (only ~20% of true morbidity captured in clinics)
      reported_drc = np.maximum(true_resp * 0.20 + rng.normal(0, 0.5, n), 0.1)
      health_list.append(
          pd.DataFrame({
              "region_id": region_id,
              "epi_week": dates,
              "resp_rate": np.nan,
              "completeness": 0.0,
              "reported_drc": reported_drc,
          })
      )

  bearings = {
      "JM_St_Ann": 0.6,
      "JM_Manchester": 1.1,
      "JM_Portland_Ctrl": 0.0,
      "DRC_Kolwezi": 0.8,
      "DRC_Ituri_Mongbwalu": 1.4,
  }

  panel = build_harmonized_weekly_panel(
      pd.concat(sar_list, ignore_index=True),
      pd.concat(aod_list, ignore_index=True),
      pd.concat(era5_list, ignore_index=True),
      pd.concat(health_list, ignore_index=True),
      mine_bearing_rad=bearings,
  )

  alerts_df = train_and_score_underreporting(panel, epochs=30)
  print("\nTop 5 Flagged Underreporting Weeks in DRC:")
  print(alerts_df.head(5).to_string(index=False))
