
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "salzburg_propagation_features.parquet"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spatial_baseline_predictions.parquet"
)

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "baseline_performance.csv"


def main():

    print("\n========== BASELINE MODEL EVALUATION ==========\n")

    actual = pd.read_parquet(
        FEATURES_PATH,
        columns=["rsrp_dbm"]
    )

    predictions = pd.read_parquet(
        PREDICTIONS_PATH,
        columns=[
            "rsrp_dbm",
            "predicted_fitted_log_distance_rsrp_dbm",
        ]
    )

    df = predictions.copy()

    df = df.dropna(
        subset=[
            "rsrp_dbm",
            "predicted_fitted_log_distance_rsrp_dbm",
        ]
    )

    y_true = df["rsrp_dbm"].to_numpy(dtype=float)

    y_pred = df[
        "predicted_fitted_log_distance_rsrp_dbm"
    ].to_numpy(dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    bias = np.mean(y_pred - y_true)

    results = {
        "Model": "Log Distance Regression",
        "Evaluation_Rows": len(y_true),
        "MAE_dB": mae,
        "RMSE_dB": rmse,
        "R2": r2,
        "Bias_dB": bias,
    }

    print("Model:", results["Model"])
    print("Evaluation rows:", results["Evaluation_Rows"])
    print(f"MAE: {mae:.4f} dB")
    print(f"RMSE: {rmse:.4f} dB")
    print(f"R2: {r2:.4f}")
    print(f"Bias: {bias:.4f} dB")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame([results]).to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSaved results to:", OUTPUT_PATH)

    print("\n===============================================\n")


if __name__ == "__main__":
    main()
