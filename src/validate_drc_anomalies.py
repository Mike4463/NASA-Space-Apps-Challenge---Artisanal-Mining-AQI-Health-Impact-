import geopandas as gpd
import libpysal
from esda.moran import Moran, Moran_BV_Local
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def run_spatial_autocorrelation_audit(
    grid_gdf: gpd.GeoDataFrame,
    gap_col: str = "underreporting_gap_zscore",
    clinic_dist_col: str = "dist_to_clinic_km",
) -> dict:
  """Computes Global Moran's I on underreporting Z-scores and Bivariate LISA

    against clinic distance to separate access deserts from misdiagnosis hotspots.
    """
  clean_gdf = grid_gdf.dropna(subset=[gap_col]).copy().reset_index(drop=True)

  # Build Queen contiguity spatial weights matrix (row-standardized)
  w = libpysal.weights.Queen.from_dataframe(clean_gdf, use_index=False)
  w.transform = "r"

  y_gap = clean_gdf[gap_col].to_numpy(dtype=float)
  moran_global = Moran(y_gap, w, permutations=999)

  results = {
      "global_morans_I": round(float(moran_global.I), 4),
      "expected_I": round(float(moran_global.EI), 4),
      "p_sim": round(float(moran_global.p_sim), 4),
      "z_sim": round(float(moran_global.z_sim), 3),
  }

  # Bivariate LISA: Underreporting Gap (x) vs. Distance to Clinic (y)
  if clinic_dist_col in clean_gdf.columns:
    x_dist = clean_gdf[clinic_dist_col].to_numpy(dtype=float)
    lisa_bv = Moran_BV_Local(y_gap, x_dist, w, permutations=999)

    # Quadrant codes in esda: 1=HH (Access Desert), 2=LH, 3=LL, 4=HL (Misdiagnosis Hotspot)
    sig_mask = lisa_bv.p_sim < 0.05
    cluster_labels = np.full(len(clean_gdf), "Not_Significant", dtype=object)
    cluster_labels[sig_mask & (lisa_bv.q == 1)] = "HH_Clinic_Access_Desert"
    cluster_labels[sig_mask & (lisa_bv.q == 4)] = "HL_Clinical_Misdiagnosis_Zone"

    clean_gdf["lisa_cluster_type"] = cluster_labels
    results["hh_desert_cells"] = int((cluster_labels == "HH_Clinic_Access_Desert").sum())
    results["hl_misdiagnosis_cells"] = int(
        (cluster_labels == "HL_Clinical_Misdiagnosis_Zone").sum()
    )

  return results, clean_gdf


def run_silica_sensitivity_analysis(
    df_cells: pd.DataFrame,
    base_pred_col: str = "raw_dann_pred",
    silica_col: str = "S_silica",
    chronic_col: str = "C_chronic",
    reported_col: str = "officially_reported_per_10k",
    sigma_pred: float = 1.0,
    top_k_frac: float = 0.15,
) -> pd.DataFrame:
  """Evaluates rank stability (Kendall's tau & Top-K Jaccard overlap) across

    crystalline-silica potency weights beta in [0.0, 0.5, 1.0, 1.5, 2.0].
    """
  betas = [0.0, 0.5, 1.0, 1.5, 2.0]
  k = max(1, int(len(df_cells) * top_k_frac))
  gap_vectors = {}
  top_sets = {}

  for b in betas:
    adj_pred = df_cells[base_pred_col] * (
        1.0 + b * df_cells[silica_col] * df_cells[chronic_col]
    )
    gap_z = (adj_pred - df_cells[reported_col]) / sigma_pred
    gap_vectors[b] = gap_z.to_numpy()
    top_sets[b] = set(df_cells.loc[gap_z.nlargest(k).index, "cell_id"])

  # Compare each beta against the baseline beta=1.0 specification
  ref_b = 1.0
  summary = []
  for b in betas:
    tau, _ = kendalltau(gap_vectors[ref_b], gap_vectors[b])
    jaccard = len(top_sets[ref_b] & top_sets[b]) / len(
        top_sets[ref_b] | top_sets[b]
    )
    summary.append({
        "beta_potency": b,
        "kendall_tau_vs_beta1": round(float(tau), 4),
        "top_15pct_jaccard_overlap": round(float(jaccard), 4),
        "mean_gap_zscore": round(float(np.mean(gap_vectors[b])), 3),
    })

  return pd.DataFrame(summary)


def evaluate_sparse_ngo_lift(
    df_cells: pd.DataFrame,
    gap_col: str = "underreporting_gap_zscore",
    ngo_audit_col: str = "has_verified_ngo_alert",
    top_decile: float = 0.10,
) -> dict:
  """Computes Lift@10% and rank correlation against sparse WHO/MSF/IPIS ground-truth audits."""
  valid = df_cells.dropna(subset=[gap_col, ngo_audit_col]).copy()
  baseline_rate = valid[ngo_audit_col].mean()

  k = max(1, int(len(valid) * top_decile))
  top_cells = valid.nlargest(k, gap_col)
  top_rate = top_cells[ngo_audit_col].mean()

  rho, p_val = spearmanr(valid[gap_col], valid[ngo_audit_col])
  return {
      "n_audited_cells": len(valid),
      "baseline_alert_prevalence": round(float(baseline_rate), 4),
      "top_10pct_precision": round(float(top_rate), 4),
      "lift_at_10pct": round(float(top_rate / (baseline_rate + 1e-8)), 2),
      "spearman_rho": round(float(rho), 4),
      "spearman_p": round(float(p_val), 4),
  }