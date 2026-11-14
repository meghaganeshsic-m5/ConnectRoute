import pandas as pd
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parent

prediction_file = PROJECT / "outputs" / "experimental_model_predictions.parquet"
feature_file = PROJECT / "data" / "processed" / "salzburg_propagation_features.parquet"

# Load data
pred = pd.read_parquet(prediction_file)
features = pd.read_parquet(feature_file)

# Align features
geo = features.iloc[pred["row_index"].to_numpy()].reset_index(drop=True)

# Verify alignment
if not np.array_equal(pred["fold"], geo["spatial_fold"]):
    raise ValueError("Fold alignment failed.")

# Combine required columns
data = pd.concat([
    pred.reset_index(drop=True),
    geo[[
        "tower_1_distance_m",
        "tower_1_terrain_roughness_m",
        "tower_1_knife_edge_diffraction_loss_db",
        "tower_1_fresnel_clearance_fraction"
    ]].reset_index(drop=True)
], axis=1)

models = {
    "Physics": "physics_baseline_rsrp_dbm",
    "LightGBM": "pure_lightgbm_rsrp_dbm",
    "Hybrid_LightGBM": "hybrid_lightgbm_rsrp_dbm",
    "XGBoost": "hybrid_xgboost_rsrp_dbm"
}

# Calculate absolute errors
for name, column in models.items():
    data[f"{name}_error"] = (
        data[column] - data["rsrp_dbm"]
    ).abs()

# Tower distance groups
data["distance_group"] = pd.cut(
    data["tower_1_distance_m"],
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

# Analyse each fold
results = []

for fold in sorted(data["fold"].unique()):

    group = data[data["fold"] == fold]

    print("\n" + "=" * 70)
    print(f"FOLD {fold}")
    print("=" * 70)

    print("\nERROR BY TOWER DISTANCE")

    distance_results = group.groupby(
        "distance_group",
        observed=True
    )[list(f"{name}_error" for name in models)].mean()

    print(distance_results.round(2).to_string())

    print("\nERROR CORRELATION WITH TERRAIN FEATURES")

    terrain_features = [
        "tower_1_distance_m",
        "tower_1_terrain_roughness_m",
        "tower_1_knife_edge_diffraction_loss_db",
        "tower_1_fresnel_clearance_fraction"
    ]

    for name in models:

        error_column = f"{name}_error"

        print(f"\n{name}:")

        for feature in terrain_features:

            correlation = group[error_column].corr(
                group[feature]
            )

            print(
                f"{feature}: {correlation:.4f}"
                if pd.notna(correlation)
                else f"{feature}: unavailable"
            )

    # Store results
    for distance, row in distance_results.iterrows():

        result = {
            "fold": fold,
            "distance_group": distance,
            "rows": len(group[
                group["distance_group"] == distance
            ])
        }

        for model in models:
            result[f"{model}_MAE"] = row[f"{model}_error"]

        results.append(result)

# Save output
output = PROJECT / "outputs" / "error_vs_distance_analysis.csv"

pd.DataFrame(results).to_csv(output, index=False)

print("\n" + "=" * 70)
print("ERROR VS DISTANCE ANALYSIS COMPLETED")
print("Saved to:", output)