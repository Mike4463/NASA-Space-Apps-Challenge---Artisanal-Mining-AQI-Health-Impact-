[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24269769.svg)](https://doi.org/10.5281/zenodo.24269769)

# Cross-Continental Epidemiological Surveillance of Artisanal Mining Respiratory Morbidity

A multi-modal Earth observation, domain-adversarial deep learning, and statistical epidemiology framework for detecting underreported respiratory morbidity (silicosis and acute respiratory illness) in artisanal and small-scale mining (ASM) corridors of the Democratic Republic of the Congo (DRC), calibrated against validated clinical surveillance from Jamaican bauxite-mining parishes.

---

## Methodological Architecture

1. **Boundary-Layer & Wind-Advection Exposure Physics**
   - Normalizes cloud-masked MODIS MAIAC 1 km Aerosol Optical Depth (`MCD19A2`) by ERA5 Planetary Boundary Layer Height ($\text{BLH}$) to isolate surface-level particulate concentrations ($\text{Surface PM Proxy} = \text{AOD}_{p90} / (\text{BLH}_{\text{km}} + 0.1)$).
   - Pairs ERA5 10 m wind vectors ($u_{10}, v_{10}$) with Sentinel-1 Radiometric Terrain Corrected (RTC) C-band backscatter ($\sigma^0_{VV}, \sigma^0_{VH}$) and Gray-Level Co-occurrence Matrix (GLCM) surface roughness to separate active open-pit excavation plumes from background humidity.

2. **Random Matrix Theory (RMT) & Taguchi Feature Screening**
   - Applies **Ledoit–Péché analytical nonlinear shrinkage** to denoise sample covariance matrices, selects high-signal predictors via a **Taguchi $L_{12}$ orthogonal array**, and aligns source/target feature covariances via **RMT-CORAL**.

3. **Domain Adversarial Neural Network (DANN)**
   - A 2-layer Gated Recurrent Unit (GRU) encodes 12-week spatio-temporal exposure windows and trains jointly on a masked Huber morbidity regressor (Jamaica source domain) and a Gradient Reversal Layer (GRL) domain discriminator to learn domain-invariant dose-response dynamics.

4. **Lithological Crystalline-Silica Weighting & Epidemiological Triangulation (R)**
   - Weights particulate exposure by host-rock free crystalline silica fraction ($f_{\text{silica}}$: $0.05$ in Jamaican bauxite vs. $0.28\text{–}0.38$ in Katanga copper-cobalt and Ituri quartz-vein gold corridors).
   - Evaluates systematic underreporting relative to the non-mining control (`JM_Portland_Ctrl`) via:
     - **Standardized Incidence Ratios (SIR)** with Exact Poisson confidence intervals (`poisson.test`)
     - **Zero-Inflated Negative Binomial (ZINB)** regression controlling for control-group zero inflation, rural clinic density, diagnostic capacity, and competing tuberculosis burden
     - **Year-Long Difference-in-Differences (DiD)** modeling comparing clinic visit elasticity during acute dust spikes ($\ge P_{80}$)

---

## Repository Structure

- `src/` — Satellite extractors (`Sentinel-1 RTC`, `MODIS MCD19A2`), ERA5 ingestion, silica priors, and the PyTorch `RMT-CORAL + DANN` pipeline.
- `R/` — Epidemiological validation (`epidemiological_underreporting_analysis.R`).
- `health_data/` — Weekly `W-MON` respiratory surveillance tables.
- `asm_pipeline_workspace/` — Processed environmental time series, DANN alert outputs, and publication figures.

---

## Quickstart

### 1. Environment Setup
```bash
conda env create -f environment.yml
conda activate asm_health

## Citation

If you use this pipeline, datasets, or epidemiological framework in your research, please cite it using the Zenodo archive:

> Prichard, M., Jr. (2026). *Cross-Continental Epidemiological Surveillance of Artisanal Mining Respiratory Morbidity* (v1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.24269769

```bibtex
@software{Prichard_ASM_Respiratory_Surveillance_2026,
  author  = {Richard, Michael P., Jr.},
  title   = {Cross-Continental Epidemiological Surveillance of Artisanal Mining Respiratory Morbidity},
  version = {1.0.0},
  year    = {2026},
  publisher = {Zenodo},
  doi     = {10.5281/zenodo.24269769},
  url     = {[https://doi.org/10.5281/zenodo.24269769](https://doi.org/10.5281/zenodo.24269769)}
}