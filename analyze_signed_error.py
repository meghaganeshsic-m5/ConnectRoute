
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parent

prediction_file = PROJECT / "outputs" / "experimental_model_predictions.parquet"

data = pd.read_parquet(prediction_file)

models = {
    "Physics": "physics_baseline_rsrp_dbm",
    "LightGBM": "pure_lightgbm_rsrp_dbm",
    "Hybrid_LightGBM": "hybrid_lightgbm_rsrp_dbm",
    "XGBoost": "hybrid_xgboost_rsrp_dbm"
}

# Load tower distance from original feature dataset
features = pd.read_parquet(
    PROJECT / "data" / "processed" / "salzburg_propagation_features.parquet",
    columns=["tower_1_distance_m"]
)

distance = features.iloc[
    data["row_index"].to_numpy()
]["tower_1_distance_m"].reset_index(drop=True)

data = data.reset_index(drop=True)
data["tower_distance_m"] = distance

# Create distance groups
data["distance_group"] = pd.cut(
    data["tower_distance_m"],
    bins=[0, 50, 100, 150, 200, 250, 300, 500],
    labels=[
        "0-50m",
        "50-100m",
        "100-150m",
        "150-200m",
        "200-250m",
        "250-300m",
        "300-500m"
    ]
)

results = []

for fold in sorted(data["fold"].unique()):

    group = data[data["fold"] == fold]

    print("\n" + "=" * 75)
    print(f"FOLD {fold}")
    print("=" * 75)

    for distance_group, subset in group.groupby(
        "distance_group",
        observed=True
    ):

        print(f"\nDistance: {distance_group}")
        print("Number of observations:", len(subset))

        for model_name, column in models.items():

            error = subset[column] - subset["rsrp_dbm"]

            bias = error.mean()
            mae = error.abs().mean()

            print(
                f"{model_name:20s} "
                f"Bias: {bias:8.2f} dB | "
                f"MAE: {mae:8.2f} dB"
            )

            results.append({
                "fold": fold,
                "distance_group": str(distance_group),
                "model": model_name,
                "observations": len(subset),
                "bias_db": bias,
                "mae_db": mae
            })

# Save results
output = PROJECT / "outputs" / "signed_error_analysis.csv"

pd.DataFrame(results).to_csv(output, index=False)

print("\nSIGNED ERROR ANALYSIS COMPLETED")
print("Saved to:", output)
