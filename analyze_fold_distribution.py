
import numpy as np
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "experimental_model_predictions.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "fold_distribution_analysis.csv"
)

# Load existing predictions
df = pd.read_parquet(INPUT_FILE)

print("\nDATASET SHAPE:", df.shape)

# Models to compare
models = {
    "Physics Baseline": "physics_baseline_rsrp_dbm",
    "Pure LightGBM": "pure_lightgbm_rsrp_dbm",
    "Hybrid LightGBM": "hybrid_lightgbm_rsrp_dbm",
    "Hybrid XGBoost": "hybrid_xgboost_rsrp_dbm",
}

results = []

# Analyse each geographical fold
for fold, group in df.groupby("fold"):

    actual = group["rsrp_dbm"]

    print("\n" + "=" * 65)
    print(f"FOLD {fold}")
    print("=" * 65)

    print("\nACTUAL SIGNAL DISTRIBUTION")
    print(f"Rows:   {len(group)}")
    print(f"Mean:   {actual.mean():.2f} dBm")
    print(f"Median: {actual.median():.2f} dBm")
    print(f"Std:    {actual.std():.2f} dB")
    print(f"Min:    {actual.min():.2f} dBm")
    print(f"Max:    {actual.max():.2f} dBm")

    print("\nSIGNAL STRENGTH CATEGORIES")
    print(f"Strong (>= -80):       {(actual >= -80).mean() * 100:.2f}%")
    print(f"Moderate (-100 to -80): {((actual < -80) & (actual >= -100)).mean() * 100:.2f}%")
    print(f"Weak (< -100):         {(actual < -100).mean() * 100:.2f}%")

    # Evaluate each model
    for model_name, prediction_column in models.items():

        predicted = group[prediction_column]
        error = predicted - actual

        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error ** 2))
        bias = np.mean(error)

        print(f"\n{model_name}")
        print(f"MAE:  {mae:.2f} dB")
        print(f"RMSE: {rmse:.2f} dB")
        print(f"Bias: {bias:.2f} dB")

        results.append({
            "fold": fold,
            "model": model_name,
            "rows": len(group),
            "actual_mean_dbm": actual.mean(),
            "actual_median_dbm": actual.median(),
            "actual_std_db": actual.std(),
            "actual_min_dbm": actual.min(),
            "actual_max_dbm": actual.max(),
            "strong_percent": (actual >= -80).mean() * 100,
            "moderate_percent": (
                ((actual < -80) & (actual >= -100)).mean() * 100
            ),
            "weak_percent": (actual < -100).mean() * 100,
            "mae_db": mae,
            "rmse_db": rmse,
            "bias_db": bias,
        })

# Save results to a new CSV
results_df = pd.DataFrame(results)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 65)
print("ANALYSIS COMPLETED")
print("=" * 65)
print(f"Results saved to: {OUTPUT_FILE}")
