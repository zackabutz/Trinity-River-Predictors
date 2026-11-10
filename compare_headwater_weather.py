#!/usr/bin/env python3
"""Compare next-day flow models with and without four headwater rain gauges.

Uses identical USGS target rows and chronological splits for every scenario.
Only weather dated on or before the feature date enters a forecast.
"""

from __future__ import annotations

import argparse
import gzip
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from retrieve_data import GHCN, cached, distance_km, ghcn_stations, station_weather
from train_models import FEATURES, assign_splits, build_examples, number, save_csv, scores


# Approximate source locations from the Handbook of Texas. The station choices
# favor long, nearly complete precipitation records over temperature coverage.
FORKS = {
    "west": {"source": (33.45, -98.6333), "station_id": "USC00410271"},  # Antelope
    "clear": {"source": (32.9833, -97.9167), "station_id": "USW00093985"},  # Mineral Wells AP
    "elm": {"source": (33.7, -97.55), "station_id": "USC00413420"},  # Gainesville 5 ENE
    "east": {"source": (33.5333, -96.6833), "station_id": "USC00418274"},  # Sherman
}
def load_rain(data_dir: Path, start: date, end: date, refresh: bool):
    raw_dir = data_dir / "raw" / "noaa"
    stations = {s["station_id"]: s for s in ghcn_stations(cached(
        raw_dir / "ghcnd-stations.txt", f"{GHCN}/ghcnd-stations.txt", refresh))}
    rain = {}
    station_rows = []
    days = (end - start).days + 1
    for fork, config in FORKS.items():
        sid = config["station_id"]
        station = stations[sid]
        raw = gzip.decompress(cached(raw_dir / f"{sid}.csv.gz",
                                     f"{GHCN}/by_station/{sid}.csv.gz", refresh))
        weather = station_weather(raw, start, end)
        values = {date.fromisoformat(row["date"]): row["prcp_mm"]
                  for row in weather if "prcp_mm" in row}
        rain[fork] = values
        lat, lon = config["source"]
        station_rows.append({"fork": fork, "station_id": sid,
                             "station_name": station["station_name"],
                             "latitude": station["latitude"], "longitude": station["longitude"],
                             "source_latitude": lat, "source_longitude": lon,
                             "distance_to_source_km": round(distance_km(lat, lon, station["latitude"], station["longitude"]), 1),
                             "prcp_days": len(values), "period_days": days,
                             "prcp_coverage": round(len(values) / days, 4)})
    return rain, station_rows


def rain_features(rain: dict[date, float], day: date):
    recent = [rain.get(day - timedelta(days=n)) for n in range(7)]
    return [recent[0], recent[1], sum(recent[:3]) if all(x is not None for x in recent[:3]) else None,
            sum(recent) if all(x is not None for x in recent) else None]


def model_for(name: str):
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if name == "linear_regression":
        steps.extend([("scale", StandardScaler()), ("model", LinearRegression())])
    elif name == "gradient_boosting":
        steps.append(("model", GradientBoostingRegressor(random_state=42)))
    elif name == "random_forest":
        steps.append(("model", RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1)))
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline(steps)


def compare(rows, rain, models, output_dir: Path):
    scenarios = {"baseline": (), **{f"plus_{fork}": (fork,) for fork in FORKS},
                 "plus_all": tuple(FORKS)}
    predictions = []
    metrics = []
    for site in sorted({row["site_number"] for row in rows}):
        site_rows = [row for row in rows if row["site_number"] == site]
        dates = [date.fromisoformat(row["feature_date"]) for row in site_rows]
        y = np.asarray([row["target_flow_cfs"] for row in site_rows], dtype=float)
        base = np.asarray([[number(row.get(feature)) if number(row.get(feature)) is not None else np.nan
                            for feature in FEATURES] for row in site_rows], dtype=float)
        extras = {fork: np.asarray([[x if x is not None else np.nan for x in rain_features(values, day)]
                                    for day in dates], dtype=float)
                  for fork, values in rain.items()}
        train_idx = np.asarray([i for i, row in enumerate(site_rows) if row["split"] == "train"])
        if not len(train_idx):
            raise ValueError(f"No training rows at {site}")
        for scenario, forks in scenarios.items():
            matrix = np.column_stack([base] + [extras[fork] for fork in forks])
            keep = np.isfinite(matrix[train_idx]).any(axis=0)
            matrix = matrix[:, keep]
            for name in models:
                fitted = model_for(name).fit(matrix[train_idx], y[train_idx])
                for split in ("validation", "test"):
                    idx = np.asarray([i for i, row in enumerate(site_rows) if row["split"] == split])
                    if not len(idx):
                        raise ValueError(f"No {split} rows at {site}")
                    estimate = np.maximum(0, fitted.predict(matrix[idx]))
                    metrics.append({"site_number": site, "scenario": scenario,
                                    "model": name, "split": split, **scores(y[idx], estimate)})
                    for i, predicted in zip(idx, estimate):
                        predictions.append({"site_number": site, "scenario": scenario,
                                            "model": name, "split": split,
                                            "feature_date": site_rows[i]["feature_date"],
                                            "target_date": site_rows[i]["target_date"],
                                            "actual_cfs": y[i], "predicted_cfs": float(predicted)})
                print(f"{site} {name} {scenario}: fitted", flush=True)
    for scenario in scenarios:
        for name in models:
            for split in ("validation", "test"):
                group = [row for row in predictions if row["scenario"] == scenario
                         and row["model"] == name and row["split"] == split]
                metrics.append({"site_number": "overall", "scenario": scenario,
                                "model": name, "split": split,
                                **scores([r["actual_cfs"] for r in group],
                                         [r["predicted_cfs"] for r in group])})
    lookup = {(r["site_number"], r["model"], r["split"]): r
              for r in metrics if r["scenario"] == "baseline"}
    for row in metrics:
        baseline = lookup[(row["site_number"], row["model"], row["split"])]
        row["mae_change_vs_baseline_cfs"] = row["mae_cfs"] - baseline["mae_cfs"]
        row["rmse_change_vs_baseline_cfs"] = row["rmse_cfs"] - baseline["rmse_cfs"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "metrics.csv", list(metrics[0]), metrics)
    save_csv(output_dir / "predictions.csv", list(predictions[0]), predictions)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--train-end", default="2019-09-22")
    parser.add_argument("--val-end", default="2022-11-11")
    parser.add_argument("--models", nargs="+", choices=("linear_regression", "gradient_boosting", "random_forest"),
                        default=("linear_regression", "gradient_boosting"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if not start <= end:
        parser.error("--start must be on or before --end")
    rows = [row for row in build_examples(args.data_dir / "raw")
            if start <= date.fromisoformat(row["feature_date"]) <= end]
    if not rows:
        parser.error("No base examples; run retrieve_data.py first")
    assign_splits(rows, args.train_end, args.val_end)
    rain, stations = load_rain(args.data_dir, start, end, args.refresh)
    output = args.data_dir / "headwater_comparison"
    save_csv(output / "stations.csv", list(stations[0]), stations)
    metrics = compare(rows, rain, args.models, output)
    (output / "run_config.json").write_text(json.dumps({"start": args.start, "end": args.end,
        "train_end": args.train_end, "val_end": args.val_end, "models": args.models,
        "scenarios": ["baseline"] + [f"plus_{fork}" for fork in FORKS] + ["plus_all"]}, indent=2))
    print(f"Wrote {len(metrics)} score rows to {output}")


if __name__ == "__main__":
    main()
