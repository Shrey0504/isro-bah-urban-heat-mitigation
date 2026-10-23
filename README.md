# isro-bah-urban-heat-mitigation

 BHARATIYA ANTARIKSH HACKATHON 2026
 Problem Statement 1: Urban Heat Mitigation (Gujarat)
 Physics-Informed Neural Network for Land Surface Temperature
======================================================================

OVERVIEW
--------
This project predicts Land Surface Temperature (LST) for urban zones
in Ahmedabad, Gujarat, using a Physics-Informed Neural Network (PINN),
and then runs a gradient-based optimizer to recommend vegetation
(NDVI) and built-up density (NDBI) changes that reduce LST in the
hottest zones. It combines satellite/reanalysis data extraction,
model training, and a mitigation-recommendation engine into a single
pipeline.

PIPELINE (main.py)
-------------------
1. Phase 1 - Data Extraction (data_pipeline/extract_gee_data.py)
   Pulls Landsat 8 Collection 2 Level-2 (surface temperature, NDVI,
   NDBI bands) and ERA5-Land hourly reanalysis (2m air temperature,
   10m wind components) from Google Earth Engine for a bounding box
   over Ahmedabad, samples ~2500 points, and writes them to
   ahmedabad_heat_data.csv. If the CSV already exists locally, this
   step is skipped.

2. Phase 2 - PINN Training (models/train_pinn.py)
   Trains a small feed-forward network (Dense 64 -> Dense 32 ->
   sigmoid output, rescaled to a 20-80 degC range) to predict LST
   from [NDVI, NDBI, Air_Temp, Wind_Speed]. Inputs are standardized
   with scikit-learn's StandardScaler before being fed to the network.
   The loss combines:
     - Data loss: standard MSE between predicted and true LST.
     - Physics loss: a simplified surface energy-balance residual
       (Latent Heat + Sensible Heat - Available Energy, normalized),
       computed on the same scaled inputs, weighted by a small
       lambda_physics term so it nudges predictions toward physically
       plausible behavior without dominating the data fit.
   The trained model and the fitted scaler are saved to
   ./saved_models/gujarat_uhi_pinn_model.keras.

3. Phase 3 - Mitigation Optimization (simulation/optimizer.py)
   Identifies the hottest pixel in the dataset (by LST) as the
   high-risk baseline zone, then uses gradient descent (Adam) to
   search, within bounded limits, for the increase in NDVI and/or
   decrease in NDBI that minimizes the model's predicted LST for
   that zone. Outputs baseline LST, optimized LST, estimated
   temperature reduction, and the recommended NDVI/NDBI changes.

HOW TO RUN
-----------
    python main.py

Requirements: earthengine-api, pandas, numpy, tensorflow, scikit-learn.
A Google Earth Engine account/project is only required the first time
(no local CSV present); afterwards the pipeline reuses the cached CSV.

LATEST RESULTS
---------------
Validation MSE:   3.42
Validation RMSE:  1.85 degC
Validation MAE:   1.44 degC
Validation R^2:   0.43

Hottest observed zone (from dataset):
  NDVI 0.14, NDBI 0.04, Air Temp 30.9 degC, Wind Speed 2.17 m/s
  Baseline predicted LST:  47.14 degC
  Optimized predicted LST: 36.35 degC
  Estimated reduction:     10.8 degC
  Recommendation: reduce NDBI (built-up density) by ~0.15;
  no further NDVI increase was needed to reach the optimum in this case.

R^2 of 0.43 means the model explains roughly 43% of the variance in
LST on held-out data - a reasonable first-pass fit for a small MLP on
~2500 sampled points, but there is clear room to improve (see LIMITATIONS).

PROJECT STRUCTURE
------------------
main.py                          - Orchestrates all three phases.
data_pipeline/
  extract_gee_data.py            - Google Earth Engine data extraction.
  __init__.py
models/
  train_pinn.py                  - Model architecture, physics-informed
                                    loss, training loop, evaluation.
  physics_loss.py                - Not used by the pipeline (an earlier,
                                    unused draft of the loss class kept
                                    only for reference - safe to delete).
  __init__.py
simulation/
  optimizer.py                   - Gradient-based NDVI/NDBI mitigation
                                    search over a trained model.
ahmedabad_heat_data.csv          - Cached extracted dataset (generated
                                    on first run, or supplied directly).
saved_models/
  gujarat_uhi_pinn_model.keras   - Trained model, written each run.

KNOWN ISSUES / LIMITATIONS
----------------------------
- The physics-loss constants (250, 300, 400 in the energy-balance
  residual) are illustrative placeholders, not constants calibrated
  against measured Gujarat surface-energy-balance data. Treat the
  physics term as a soft regularizer, not a validated physical model.
- Training data is a single median composite per pixel across the
  full date range (2020-2025), so seasonal variation in LST is
  averaged out rather than modeled explicitly.
- ~2500 sampled points is a small dataset for a neural network;
  R^2 = 0.43 leaves meaningful unexplained variance. More data,
  additional features (e.g. land-use class, elevation, proximity to
  water bodies), or a slightly larger network could improve this.
- The optimizer only searches over NDVI/NDBI for a single fixed
  zone at a time; it does not yet model spatial interactions between
  neighboring zones or propose a city-wide intervention plan.
- physics_loss.py is currently dead code (not imported anywhere) and
  can be safely removed to avoid confusion with the loss class that
  is actually used in train_pinn.py.

DATA SOURCES
-------------
- Landsat 8 Collection 2 Level-2 (LANDSAT/LC08/C02/T1_L2): surface
  temperature (ST_B10), NDVI, NDBI.
- ERA5-Land Hourly (ECMWF/ERA5_LAND/HOURLY): 2m air temperature,
  10m wind components.
- Region: Ahmedabad, Gujarat (bounding box 72.45-72.68 E, 22.95-23.12 N).
