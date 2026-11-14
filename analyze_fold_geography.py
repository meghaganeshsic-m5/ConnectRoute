
import pandas as pd
import numpy as np
from pathlib import Path

# File paths
PROJECT = Path(__file__).resolve().parent

prediction_file = PROJECT / "outputs" / "experimental_model_predictions.parquet"
feature_file = PROJECT / "data" / "processed" / "salzburg_propagation_features.parquet"

# Load datasets
pred = pd.read_parquet(prediction_file)
features = pd.read_parquet(feature_file)

print("Prediction shape:", pred.shape)
print("Feature shape:", features.shape)

# Check row index validity
if pred["row_index"].min() < 0 or pred["row_index"].max() >= len(features):
    raise ValueError("Prediction row_index is outside feature dataset.")

# Align original feature rows with predictions
geo = features.iloc[pred["row_index"].to_numpy()].reset_index(drop=True)

# Verify spatial fold mapping
if not np.array_equal(
    pred["fold"].to_numpy(),
    geo["spatial_fold"].to_numpy()
):
    raise ValueError(
        "Fold mismatch! row_index does not align with spatial_fold."
    )

print("\nSUCCESS: All prediction rows match their original spatial folds.")

# Select geographical and terrain columns.
# Do not include rsrp_dbm here because predictions already contain it.
columns = [
    "latitude",
    "longitude",
    "ele",
    "tower_1_distance_m",
    "tower_1_terrain_roughness_m",
    "tower_1_fresnel_clearance_fraction",
    "tower_1_knife_edge_diffraction_loss_db"
]

# Combine predictions and selected features
data = pd.concat(
    [
        pred.reset_index(drop=True),
        geo[columns].reset_index(drop=True)
    ],
    axis=1
)

# Check for duplicate column names
if data.columns.duplicated().any():
    raise ValueError("Duplicate column names detected.")

# Analyse each fold
results = []

for fold in sorted(data["fold"].unique()):

    group = data[data["fold"] == fold]

    print("\n" + "=" * 65)
    print(f"FOLD {fold}")
    print("=" * 65)

    print("Rows:", len(group))

    print("\nGEOGRAPHICAL DISTRIBUTION")

    print(
        "Latitude range:",
        group["latitude"].min(),
        "to",
        group["latitude"].max()
    )

    print(
        "Longitude range:",
        group["longitude"].min(),
        "to",
        group["longitude"].max()
    )

    print("\nTOWER DISTANCE")
    print(group["tower_1_distance_m"].describe().round(2))

    print("\nTERRAIN FEATURES")

    terrain_cols = [
        "ele",
        "tower_1_terrain_roughness_m",
        "tower_1_fresnel_clearance_fraction",
        "tower_1_knife_edge_diffraction_loss_db"
    ]

    print(group[terrain_cols].describe().round(2))

    print("\nMODEL ERROR")

    for model in [
        "physics_baseline_rsrp_dbm",
        "pure_lightgbm_rsrp_dbm",
        "hybrid_lightgbm_rsrp_dbm",
        "hybrid_xgboost_rsrp_dbm"
    ]:

        error = group[model] - group["rsrp_dbm"]

        print(
            model,
            "| MAE:", round(error.abs().mean(), 2),
            "| RMSE:", round(np.sqrt((error ** 2).mean()), 2),
            "| Bias:", round(error.mean(), 2)
        )

    # Save fold summary
    row = {
        "fold": fold,
        "rows": len(group)
    }

    for col in columns:
        row[f"{col}_mean"] = group[col].mean()
        row[f"{col}_median"] = group[col].median()

    results.append(row)

# Save results
output = PROJECT / "outputs" / "fold_geography_analysis.csv"

pd.DataFrame(results).to_csv(output, index=False)

print("\n" + "=" * 65)
print("GEOGRAPHICAL ANALYSIS COMPLETED")
print("Saved to:", output)
