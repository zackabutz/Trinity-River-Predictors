# Trinity River streamflow pipeline

This project contains data from the following stations:
Rosser (`08062500`), Trinidad (`08062700`), and Goodrich (`08066250`). It uses
daily mean discharge (USGS parameter `00060`, statistic `00003`) and NOAA
GHCN-Daily weather. Each row forecasts the next calendar day's mean flow from
information dated on or before the feature date.

## Run

From this folder, with Python 3.10+:

```bash
python retrieve_data.py --start 2005-01-01 --end 2025-12-31
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train_models.py
```

The requirements use TensorFlow's CPU-only build. If this folder's virtual
environment already has the general `tensorflow` package installed, the run
will still complete on CPU but may print CUDA discovery messages. To replace
that package without rebuilding the environment:

```bash
.venv/bin/python -m pip uninstall tensorflow
.venv/bin/python -m pip install -r requirements.txt
```

The retrieval script uses only Python's standard library. Use `--output-dir`
to place data elsewhere, `--refresh` to redownload cached NOAA station files,
and `--help` to see station radius and minimum coverage settings. By default it
checks nearby stations in distance order and selects the nearest with at least
80% complete precipitation and max/min temperature days in the requested
period. It uses NOAA's element inventory to screen candidates, then downloads
compressed station CSV files to verify actual daily coverage.
`data/raw/station_mapping.csv` records the station selected for each
gage. A station is never silently substituted after the model has been built;
inspect the mapping and coverage before treating results as final.

`train_models.py --prepare-only` writes `data/models/model_input.csv` and split
dates without ML packages. The full run fits **separate models at each gage**:
persistence, linear regression, gradient boosting, random forest, SVR, and
LSTM. It writes held-out validation and test predictions plus MAE, RMSE, RAE,
and R² to `data/models/`, and saves fitted models under `data/models/fitted/`.
`predictions.csv` is ordered by target date and gage, with one row per model.
Its `persistence` rows deliberately set `predicted_cfs` to `prior_day_cfs`;
the other rows are fitted-model forecasts for the same target date.
Open `plot_monthly_predictions.ipynb` to chart actual and predicted daily flow
for a chosen gage, one or more models, year, and month. Set `MODELS` to a list
such as `["random_forest", "gradient_boosting", "persistence"]`; each model
appears with its monthly MAE. The notebook reads the saved CSV directly. If the notebook uses this folder's
virtual environment as its kernel, install its plotting packages with
`.venv/bin/python -m pip install -r notebook-requirements.txt`.
The default split uses the earliest 70% of target
dates for training, next 15% for validation, and last 15% for testing. You can
instead provide `--train-end YYYY-MM-DD --val-end YYYY-MM-DD`.

Features include flow lags, rolling flow statistics, current and previous
precipitation, precipitation totals, temperature, seasonality, and site
attributes. Snow and sea-level pressure are used when present. Missing
predictors are imputed from **training rows only**; missing discharge targets
and flow-history gaps are never filled. The LSTM receives consecutive
`--lstm-days` windows of these same features. A target dated `t+1` is built
from data dated through `t`; no random row split is used.

## Data and interpretation notes

- USGS discharge is in cubic feet per second. GHCN precipitation and snow are
  converted to millimeters, temperature to Celsius, and pressure to hPa.
- The model assumes a forecast is issued after day `t` USGS and NOAA daily
  observations become available. GHCN reports can arrive later than the
  observation day, so this is a historical next-day comparison rather than an
  operational real-time flood warning model.
- One nearby weather station does not represent rainfall over the whole
  upstream watershed. Review station distances and storm errors by gage.
- The scripts train fixed example model settings. Hyperparameter search and
  final plots remain separate analysis tasks; compare validation scores before
  interpreting the test results.

Sources: [USGS Water Data API](https://api.waterdata.usgs.gov/docs/ogcapi/),
[USGS migration guide](https://api.waterdata.usgs.gov/docs/ogcapi/migration/),
[GHCN-Daily format and units](https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt),
and [GHCN station CSV layout](https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme-by_station.txt).

## Headwater precipitation comparison

Run `compare_headwater_weather.py` after `retrieve_data.py` to test whether
upstream rain gauges improve next-day flow forecasts:

```bash
.venv/bin/python compare_headwater_weather.py
```

The script downloads four NOAA GHCN-Daily stations, writes their locations and
actual 2005–2025 precipitation coverage to
`data/headwater_comparison/stations.csv`, and compares baseline features with
each fork's rain features separately and all four together. Its default models
are linear regression and gradient boosting; add `--models random_forest` to
run the slower forest comparison. Results are in `metrics.csv` and
`predictions.csv`. A **negative** `mae_change_vs_baseline_cfs` means the added
station helped. Validation and test rows, split dates, training settings, and
target observations are identical across scenarios. Inspect validation scores
when selecting features and use test scores for the final assessment.

The chosen precipitation stations are Antelope (West Fork), Mineral Wells AP
(Clear Fork), Gainesville 5 ENE (Elm Fork), and Sherman (East Fork). Some nearby
stations report precipitation but no temperature; this comparison adds only
rainfall from the headwater stations and retains the original local weather
features. The Elm station is farther from its source than the others because
the closer Bonita 7NW station has substantial gaps in this period. Distances
are straight-line distances to approximate source points, **not** proof that
each station lies within its fork's watershed. Daily GHCN precipitation may
be reported for an observer's 24-hour period rather than the same calendar-day
interval as USGS flow; this remains a retrospective experiment, not an
operational forecast.

Source locations: [West Fork](https://www.tshaonline.org/handbook/entries/west-fork-of-the-trinity-river),
[Clear Fork](https://www.tshaonline.org/handbook/entries/clear-fork-of-the-trinity-river),
[Elm Fork](https://www.tshaonline.org/handbook/entries/elm-fork-of-the-trinity-river),
[East Fork](https://www.tshaonline.org/handbook/entries/east-fork-of-the-trinity-river).
