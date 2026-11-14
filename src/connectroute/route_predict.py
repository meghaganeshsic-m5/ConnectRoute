
"""Illustrative route RSRP projection from local tower data and Salzburg baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree

from connectroute.propagation import (
    fit_log_distance_model,
    predict_log_distance_rsrp,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TOWERS = (
    PROJECT_ROOT / "data" / "processed" / "opencellid_india_lte_2021.parquet"
)

DEFAULT_SALZBURG_FEATURES = (
    PROJECT_ROOT / "data" / "processed" / "salzburg_propagation_features.parquet"
)

DEFAULT_SALZBURG_PREDICTIONS = (
    PROJECT_ROOT / "data" / "processed" / "spatial_baseline_predictions.parquet"
)

MINIMUM_TOWER_SAMPLES = 5


def _utm_crs_for_route(samples: list[dict[str, float]]) -> CRS:

    longitude = float(
        np.mean([sample["longitude"] for sample in samples])
    )

    latitude = float(
        np.mean([sample["latitude"] for sample in samples])
    )

    zone = min(60, max(1, int((longitude + 180) // 6) + 1))

    hemisphere = "north" if latitude >= 0 else "south"

    return CRS.from_dict(
        {
            "proj": "utm",
            "zone": zone,
            "south": hemisphere == "south",
        }
    )


def fit_route_projection_model(
    salzburg_features_path: Path = DEFAULT_SALZBURG_FEATURES,
    salzburg_predictions_path: Path = DEFAULT_SALZBURG_PREDICTIONS,
) -> tuple[dict[str, float | int], dict[str, float]]:

    features = pd.read_parquet(
        salzburg_features_path,
        columns=["tower_1_distance_m", "rsrp_dbm"],
    )

    fitted = fit_log_distance_model(
        features["rsrp_dbm"].to_numpy(dtype=float),
        features["tower_1_distance_m"].to_numpy(dtype=float),
    )

    oof = pd.read_parquet(
        salzburg_predictions_path,
        columns=[
            "rsrp_dbm",
            "predicted_fitted_log_distance_rsrp_dbm",
        ],
    )

    residual = (
        oof["rsrp_dbm"].to_numpy(dtype=float)
        - oof["predicted_fitted_log_distance_rsrp_dbm"].to_numpy(dtype=float)
    )

    residual = residual[np.isfinite(residual)]

    if len(residual) == 0:
        raise ValueError(
            "No finite held-out Salzburg residuals are available."
        )

    calibration = {
        "residual_p05_db": float(np.quantile(residual, 0.05)),
        "residual_p95_db": float(np.quantile(residual, 0.95)),
        "residual_p10_db": float(np.quantile(residual, 0.10)),
        "residual_p90_db": float(np.quantile(residual, 0.90)),
        "oof_residual_rows": int(len(residual)),
        "training_distance_min_m": float(
            features["tower_1_distance_m"].min()
        ),
        "training_distance_max_m": float(
            features["tower_1_distance_m"].max()
        ),
    }

    print("\n========== ROUTE MODEL PARAMETERS ==========")
    print("Intercept:", fitted["intercept_dbm_at_1m"])
    print("Slope:", fitted["slope_db_per_decade"])
    print("Path loss exponent:", fitted["path_loss_exponent"])
    print("Training rows:", fitted["training_rows"])
    print("============================================\n")

    return fitted, calibration


def predict_route_signal(
    samples: list[dict[str, float]],
    towers: pd.DataFrame,
    fitted_model: dict[str, float | int],
    residual_summary: dict[str, float],
    *,
    minimum_tower_samples: int = MINIMUM_TOWER_SAMPLES,
) -> tuple[pd.DataFrame, dict[str, Any]]:

    if not samples:
        raise ValueError("At least one route sample is required.")

    required_towers = {
        "latitude",
        "longitude",
        "samples",
        "radio",
    }

    missing = required_towers.difference(towers.columns)

    if missing:
        raise ValueError(
            "Tower data is missing columns: "
            + ", ".join(sorted(missing))
        )

    if minimum_tower_samples < 0:
        raise ValueError(
            "minimum_tower_samples cannot be negative."
        )

    route = pd.DataFrame(samples)

    coordinate_columns = {
        "latitude",
        "longitude",
    }

    if not coordinate_columns.issubset(route.columns):
        raise ValueError(
            "Route samples require latitude and longitude."
        )

    coordinates = route[
        ["latitude", "longitude"]
    ].to_numpy(dtype=float)

    if not np.isfinite(coordinates).all():
        raise ValueError(
            "Route coordinates must be finite."
        )

    valid_latitude = np.logical_and(
        -90 <= coordinates[:, 0],
        coordinates[:, 0] <= 90,
    )

    valid_longitude = np.logical_and(
        -180 <= coordinates[:, 1],
        coordinates[:, 1] <= 180,
    )

    if not valid_latitude.all() or not valid_longitude.all():
        raise ValueError(
            "Route coordinates must be valid latitude/longitude values."
        )

    tower_radio = towers["radio"].astype("string").str.upper()

    tower_samples = pd.to_numeric(
        towers["samples"],
        errors="coerce",
    )

    tower_latitude = pd.to_numeric(
        towers["latitude"],
        errors="coerce",
    )

    tower_longitude = pd.to_numeric(
        towers["longitude"],
        errors="coerce",
    )

    valid_towers = (
        tower_radio.eq("LTE")
        & tower_samples.ge(minimum_tower_samples)
        & tower_latitude.between(-90, 90)
        & tower_longitude.between(-180, 180)
    )

    valid_towers &= (
        np.isfinite(tower_latitude)
        & np.isfinite(tower_longitude)
    )

    eligible = towers.loc[
        valid_towers
    ].copy().reset_index(drop=True)

    if eligible.empty:
        raise ValueError(
            "No LTE towers meet the minimum sample count in the local tower data."
        )

    tower_latitudes = pd.to_numeric(
        eligible["latitude"],
        errors="raise",
    ).to_numpy(dtype=float)

    tower_longitudes = pd.to_numeric(
        eligible["longitude"],
        errors="raise",
    ).to_numpy(dtype=float)

    within_tower_extent = (
        (coordinates[:, 0] >= tower_latitudes.min())
        & (coordinates[:, 0] <= tower_latitudes.max())
        & (coordinates[:, 1] >= tower_longitudes.min())
        & (coordinates[:, 1] <= tower_longitudes.max())
    )

    if not within_tower_extent.all():
        raise ValueError(
            "The route extends outside the geographic extent of the local LTE "
            "tower data. Signal estimates are unsupported for this route."
        )

    crs = _utm_crs_for_route(samples)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True,
    )

    tower_x, tower_y = transformer.transform(
        tower_longitudes,
        tower_latitudes,
    )

    route_x, route_y = transformer.transform(
        coordinates[:, 1],
        coordinates[:, 0],
    )

    tower_xy = np.column_stack(
        (tower_x, tower_y)
    )

    route_xy = np.column_stack(
        (route_x, route_y)
    )

    distances, positions = cKDTree(
        tower_xy
    ).query(
        route_xy,
        k=1,
    )

    if not np.isfinite(distances).all() or np.any(distances <= 0):
        raise ValueError(
            "Nearest tower distances must be positive finite values."
        )

    predicted = predict_log_distance_rsrp(
        distances,
        float(fitted_model["intercept_dbm_at_1m"]),
        float(fitted_model["slope_db_per_decade"]),
    )

    result = route.copy()

    result["nearest_tower_distance_m"] = distances

    result["nearest_tower_samples"] = pd.to_numeric(
        eligible.iloc[positions]["samples"],
        errors="raise",
    ).to_numpy(dtype=float)

    result["nearest_tower_latitude"] = (
        eligible.iloc[positions]["latitude"].to_numpy(dtype=float)
    )

    result["nearest_tower_longitude"] = (
        eligible.iloc[positions]["longitude"].to_numpy(dtype=float)
    )

    result["predicted_rsrp_dbm"] = predicted

    result["estimated_lower_90_dbm"] = (
        predicted + residual_summary["residual_p05_db"]
    )

    result["estimated_upper_90_dbm"] = (
        predicted + residual_summary["residual_p95_db"]
    )

    result["outside_salzburg_tower_distance_range"] = (
        (distances < residual_summary["training_distance_min_m"])
        | (distances > residual_summary["training_distance_max_m"])
    )

    # Signal classification diagnostics
    strong_points = int(np.sum(predicted >= -80))

    moderate_points = int(
        np.sum((predicted >= -100) & (predicted < -80))
    )

    weak_points = int(np.sum(predicted < -100))

    diagnostics = {
        "minimum_rsrp_dbm": float(np.min(predicted)),
        "maximum_rsrp_dbm": float(np.max(predicted)),
        "mean_rsrp_dbm": float(np.mean(predicted)),
        "median_rsrp_dbm": float(np.median(predicted)),
        "points_below_minus_100": weak_points,
        "points_below_minus_110": int(np.sum(predicted < -110)),
        "strong_points": strong_points,
        "moderate_points": moderate_points,
        "weak_points": weak_points,
        "minimum_tower_distance_m": float(np.min(distances)),
        "maximum_tower_distance_m": float(np.max(distances)),
        "fitted_intercept": float(
            fitted_model["intercept_dbm_at_1m"]
        ),
        "fitted_slope": float(
            fitted_model["slope_db_per_decade"]
        ),
        "path_loss_exponent": float(
            fitted_model["path_loss_exponent"]
        ),
    }

    summary: dict[str, Any] = {
        "eligible_lte_towers": int(len(eligible)),
        "projected_crs": crs.to_string(),
        "minimum_tower_samples": int(minimum_tower_samples),

        "nearest_tower_distance_m": {
            "minimum": float(np.min(distances)),
            "median": float(np.median(distances)),
            "p95": float(np.quantile(distances, 0.95)),
            "maximum": float(np.max(distances)),
        },

        "outside_salzburg_distance_range_fraction": float(
            result["outside_salzburg_tower_distance_range"].mean()
        ),

        "prediction_diagnostics": diagnostics,

        "model_transfer_status": (
            "Illustrative transfer of a Salzburg-fitted distance baseline to "
            "India; not validated against Karnataka measurements."
        ),
    }

    print("\n========== ROUTE PREDICTION DIAGNOSTICS ==========")

    for key, value in diagnostics.items():
        print(f"{key}: {value}")

    print("==================================================\n")

    return result, summary
