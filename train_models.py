#!/usr/bin/env python3
"""Join retrieved data, build next-day features, and train the proposal models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

FLOW_LAGS = (1, 2, 7, 14, 30)
FEATURES = (["flow_today_cfs"] + [f"flow_lag_{n}d_cfs" for n in FLOW_LAGS] +
            ["flow_mean_3d_cfs", "flow_mean_7d_cfs", "flow_std_7d_cfs",
             "prcp_mm", "prcp_lag_1d_mm", "prcp_sum_3d_mm", "prcp_sum_7d_mm",
             "tmax_c", "tmin_c", "snow_mm", "snwd_mm", "pressure_hpa", "pressure_change_hpa", "day_sin", "day_cos",
             "drainage_area_sqmi", "elevation_ft"])
META = ["site_number", "feature_date", "target_date", "target_flow_cfs", "split"]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(value) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def complete(values: list[float | None]) -> bool:
    return all(value is not None for value in values)


def build_examples(raw_dir: Path) -> list[dict]:
    sites = {row["site_number"]: row for row in read_csv(raw_dir / "usgs_sites.csv")}
    flow: dict[str, dict[date, float]] = defaultdict(dict)
    weather: dict[str, dict[date, dict]] = defaultdict(dict)
    for row in read_csv(raw_dir / "usgs_daily.csv"):
        value = number(row["flow_cfs"])
        if value is not None and value >= 0:
            flow[row["site_number"]][date.fromisoformat(row["date"])] = value
    for row in read_csv(raw_dir / "noaa_daily.csv"):
        weather[row["site_number"]][date.fromisoformat(row["date"])] = row
    examples = []
    for sid, observations in flow.items():
        site = sites[sid]
        for day in sorted(observations):
            next_day = day + timedelta(days=1)
            if next_day not in observations:
                continue  # Never fill the target or use the next available nonconsecutive day.
            lagged = {f"flow_lag_{n}d_cfs": observations.get(day - timedelta(days=n)) for n in FLOW_LAGS}
            recent3 = [observations.get(day - timedelta(days=n)) for n in range(3)]
            recent7 = [observations.get(day - timedelta(days=n)) for n in range(7)]
            if not complete(list(lagged.values()) + recent7):
                continue  # A gap in flow history would distort the forecast horizon.
            wx = weather[sid]
            today = wx.get(day, {})
            prcp = [number(wx.get(day - timedelta(days=n), {}).get("prcp_mm")) for n in range(7)]
            # A missing precipitation day is missing, never interpreted as zero.
            pressure = number(today.get("pressure_hpa"))
            previous_pressure = number(wx.get(day - timedelta(days=1), {}).get("pressure_hpa"))
            row = {"site_number": sid, "feature_date": day.isoformat(),
                   "target_date": next_day.isoformat(), "target_flow_cfs": observations[next_day],
                   "flow_today_cfs": observations[day], **lagged,
                   "flow_mean_3d_cfs": sum(recent3) / 3,
                   "flow_mean_7d_cfs": sum(recent7) / 7,
                   "flow_std_7d_cfs": (sum((x - sum(recent7) / 7) ** 2 for x in recent7) / 7) ** 0.5,
                   "prcp_mm": prcp[0], "prcp_lag_1d_mm": prcp[1],
                   "prcp_sum_3d_mm": sum(prcp[:3]) if complete(prcp[:3]) else None,
                   "prcp_sum_7d_mm": sum(prcp) if complete(prcp) else None,
                   "tmax_c": number(today.get("tmax_c")), "tmin_c": number(today.get("tmin_c")),
                   "snow_mm": number(today.get("snow_mm")), "snwd_mm": number(today.get("snwd_mm")),
                   "pressure_hpa": pressure,
                   "pressure_change_hpa": pressure - previous_pressure if pressure is not None and previous_pressure is not None else None,
                   "day_sin": math.sin(2 * math.pi * day.timetuple().tm_yday / 365.25),
                   "day_cos": math.cos(2 * math.pi * day.timetuple().tm_yday / 365.25),
                   "drainage_area_sqmi": number(site.get("drainage_area_sqmi")),
                   "elevation_ft": number(site.get("elevation_ft"))}
            examples.append(row)
    return sorted(examples, key=lambda row: (row["target_date"], row["site_number"]))


def assign_splits(rows: list[dict], train_end: str | None, val_end: str | None) -> tuple[str, str]:
    dates = sorted({row["target_date"] for row in rows})
    if len(dates) < 30:
        raise ValueError("Need at least 30 distinct target dates for a chronological split.")
    train_end = train_end or dates[max(0, int(len(dates) * 0.70) - 1)]
    val_end = val_end or dates[max(0, int(len(dates) * 0.85) - 1)]
    if not dates[0] <= train_end < val_end < dates[-1]:
        raise ValueError("Split dates must produce nonempty training, validation, and test periods.")
    for row in rows:
        target = row["target_date"]
        row["split"] = "train" if target <= train_end else "validation" if target <= val_end else "test"
    return train_end, val_end


def save_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def scores(actual, predicted) -> dict:
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = np.sum(np.abs(actual - np.mean(actual)))
    return {"n": len(actual), "mae_cfs": float(mean_absolute_error(actual, predicted)),
            "rmse_cfs": float(np.sqrt(mean_squared_error(actual, predicted))),
            "rae": float(np.sum(np.abs(actual - predicted)) / denominator) if denominator else None,
            "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None}


def train(rows: list[dict], output_dir: Path, lstm_days: int, epochs: int) -> None:
    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: python -m pip install -r requirements.txt") from exc

    # Fit separate models for each gage, as stated in the slide proposal.
    site_ids = sorted({row["site_number"] for row in rows})
    matrix = np.asarray([[number(row.get(col)) if number(row.get(col)) is not None else np.nan
                          for col in FEATURES]
                         for row in rows], dtype=float)
    y = np.asarray([row["target_flow_cfs"] for row in rows], dtype=float)
    all_predictions = []
    features_by_site = {}

    def record(model_name, row_indices, predictions):
        for idx, prediction in zip(row_indices, predictions):
            row = rows[int(idx)]
            all_predictions.append({"model": model_name, "split": row["split"],
                                    "site_number": row["site_number"],
                                    "feature_date": row["feature_date"], "target_date": row["target_date"],
                                    "prior_day_cfs": row["flow_today_cfs"],
                                    "actual_cfs": row["target_flow_cfs"], "predicted_cfs": float(prediction)})

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is needed for LSTM: install requirements.txt, then rerun.") from exc
    # This project runs on a CPU-only machine. Configure TensorFlow before it
    # creates logical devices, even if a GPU-capable wheel was installed.
    tf.config.set_visible_devices([], "GPU")
    tf.keras.utils.set_random_seed(42)
    for sid in site_ids:
        indices = {split: np.asarray([i for i, row in enumerate(rows)
                                      if row["split"] == split and row["site_number"] == sid], dtype=int)
                   for split in ("train", "validation", "test")}
        if any(len(idx) == 0 for idx in indices.values()):
            raise ValueError(f"Site {sid} has no observations in one of the three splits.")
        keep = [i for i in range(matrix.shape[1]) if np.isfinite(matrix[indices["train"], i]).any()]
        site_matrix = matrix[:, keep]
        features_by_site[sid] = [FEATURES[i] for i in keep]

        def pipeline(regressor, scaled=False):
            stages = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
            if scaled:
                stages.append(("scale", StandardScaler()))
            stages.append(("model", regressor))
            return Pipeline(stages)

        models = {
            "linear_regression": pipeline(LinearRegression(), scaled=True),
            "gradient_boosting": pipeline(GradientBoostingRegressor(random_state=42)),
            "random_forest": pipeline(RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1)),
            "svr": pipeline(SVR(C=10, epsilon=0.1), scaled=True),
        }
        for split in ("validation", "test"):
            idx = indices[split]
            record("persistence", idx, np.asarray([rows[int(i)]["flow_today_cfs"] for i in idx]))
        for name, model in models.items():
            model.fit(site_matrix[indices["train"]], y[indices["train"]])
            fitted_dir = output_dir / "fitted" / sid
            fitted_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "features": features_by_site[sid]}, fitted_dir / f"{name}.joblib")
            for split in ("validation", "test"):
                idx = indices[split]
                record(name, idx, np.maximum(0, model.predict(site_matrix[idx])))

        imputer = SimpleImputer(strategy="median", add_indicator=True)
        imputer.fit(site_matrix[indices["train"]])
        scaler = StandardScaler().fit(imputer.transform(site_matrix[indices["train"]]))
        scaled = scaler.transform(imputer.transform(site_matrix))
        y_scale = StandardScaler().fit(y[indices["train"]].reshape(-1, 1))
        sequences = {split: ([], []) for split in indices}
        site_indices = sorted([i for i, row in enumerate(rows) if row["site_number"] == sid],
                              key=lambda i: rows[i]["feature_date"])
        for offset in range(lstm_days - 1, len(site_indices)):
            window = site_indices[offset - lstm_days + 1:offset + 1]
            dates = [date.fromisoformat(rows[i]["feature_date"]) for i in window]
            if any((right - left).days != 1 for left, right in zip(dates, dates[1:])):
                continue
            endpoint = window[-1]
            split = rows[endpoint]["split"]
            sequences[split][0].append(scaled[window])
            sequences[split][1].append(endpoint)
        if any(not values[1] for values in sequences.values()):
            raise ValueError(f"Insufficient consecutive examples for LSTM at {sid}; reduce --lstm-days.")
        x_train, i_train = sequences["train"]
        network = tf.keras.Sequential([tf.keras.layers.Input(shape=(lstm_days, scaled.shape[1])),
                                       tf.keras.layers.LSTM(32), tf.keras.layers.Dense(16, activation="relu"),
                                       tf.keras.layers.Dense(1)])
        network.compile(optimizer="adam", loss="mse")
        x_val, i_val = sequences["validation"]
        network.fit(np.asarray(x_train), y_scale.transform(y[i_train].reshape(-1, 1)),
                    validation_data=(np.asarray(x_val), y_scale.transform(y[i_val].reshape(-1, 1))),
                    epochs=epochs, batch_size=64, verbose=0,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
        network.save(fitted_dir / "lstm.keras")
        joblib.dump({"imputer": imputer, "scaler": scaler, "target_scaler": y_scale,
                     "features": features_by_site[sid], "sequence_days": lstm_days},
                    fitted_dir / "lstm_preprocessing.joblib")
        for split in ("validation", "test"):
            x, idx = sequences[split]
            prediction = y_scale.inverse_transform(network.predict(np.asarray(x), verbose=0)).ravel()
            record("lstm", np.asarray(idx), np.maximum(0, prediction))

    (output_dir / "features.json").write_text(json.dumps(features_by_site, indent=2))
    groups = defaultdict(list)
    for row in all_predictions:
        groups[(row["model"], row["split"], row["site_number"])].append(row)
        groups[(row["model"], row["split"], "overall")].append(row)
    metric_rows = []
    for (model_name, split, sid), group in sorted(groups.items()):
        values = scores([row["actual_cfs"] for row in group], [row["predicted_cfs"] for row in group])
        metric_rows.append({"model": model_name, "split": split, "site_number": sid, **values})
    save_csv(output_dir / "metrics.csv", ["model", "split", "site_number", "n", "mae_cfs", "rmse_cfs", "rae", "r2"], metric_rows)
    all_predictions.sort(key=lambda row: (row["target_date"], row["site_number"], row["model"]))
    save_csv(output_dir / "predictions.csv",
             ["site_number", "feature_date", "target_date", "model", "split",
              "prior_day_cfs", "actual_cfs", "predicted_cfs"], all_predictions)
    print(f"Wrote {len(metric_rows)} metric rows and {len(all_predictions)} predictions to {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--train-end", help="Last target date in training, YYYY-MM-DD (default: 70%% chronological split)")
    parser.add_argument("--val-end", help="Last target date in validation (default: 85%% chronological split)")
    parser.add_argument("--lstm-days", type=int, default=14)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--prepare-only", action="store_true", help="Write model inputs without installing ML libraries")
    args = parser.parse_args()
    try:
        if args.lstm_days < 2 or args.epochs < 1:
            parser.error("--lstm-days must be >=2 and --epochs must be >=1")
        rows = build_examples(args.data_dir / "raw")
        if not rows:
            raise ValueError("No forecast examples found; run retrieve_data.py first.")
        train_end, val_end = assign_splits(rows, args.train_end, args.val_end)
        output_dir = args.data_dir / "models"
        save_csv(output_dir / "model_input.csv", META + FEATURES, rows)
        (output_dir / "split_dates.json").write_text(json.dumps({"train_end": train_end, "validation_end": val_end,
                                                                  "test_end": rows[-1]["target_date"]}, indent=2))
        print(f"Prepared {len(rows)} next-day examples; train <= {train_end}, validation <= {val_end}")
        if not args.prepare_only:
            train(rows, output_dir, args.lstm_days, args.epochs)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
