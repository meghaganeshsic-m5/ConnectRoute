import pandas as pd
from pathlib import Path

# =====================================================
# 1. FILE PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent

FEATURES_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "salzburg_propagation_features.parquet"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT / "outputs"
    / "experimental_model_predictions.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "outputs"
    / "rsrp_distribution_by_fold.csv"
)


# =====================================================
# 2. LOAD DATASETS
# =====================================================

print("Loading datasets...")

features = pd.read_parquet(FEATURES_FILE)
predictions = pd.read_parquet(PREDICTIONS_FILE)

print("Features shape:", features.shape)
print("Predictions shape:", predictions.shape)


# =====================================================
# 3. DEFINE MODEL COLUMNS
# =====================================================

model_columns = {
    "Physics": "physics_baseline_rsrp_dbm",
    "LightGBM": "pure_lightgbm_rsrp_dbm",
    "Hybrid LightGBM": "hybrid_lightgbm_rsrp_dbm",
    "XGBoost": "hybrid_xgboost_rsrp_dbm"
}


# =====================================================
# 4. CHECK REQUIRED COLUMNS
# =====================================================

required_prediction_columns = [
    "row_index",
    "fold"
] + list(model_columns.values())

required_feature_columns = [
    "latitude",
    "longitude",
    "rsrp_dbm"
]

for column in required_prediction_columns:

    if column not in predictions.columns:
        raise ValueError(
            f"Missing prediction column: {column}"
        )

for column in required_feature_columns:

    if column not in features.columns:
        raise ValueError(
            f"Missing feature column: {column}"
        )


# =====================================================
# 5. MERGE DATASETS
# =====================================================

df = predictions.merge(
    features[required_feature_columns],
    left_on="row_index",
    right_index=True,
    suffixes=("", "_feature"),
    validate="one_to_one"
)

# Actual measured RSRP from original feature dataset
df["actual_rsrp"] = df["rsrp_dbm_feature"]

# Use fold from prediction dataset
df["fold_number"] = df["fold"]

print("\nMerged dataset shape:", df.shape)


# =====================================================
# 6. CLASSIFY SIGNAL STRENGTH
# =====================================================

def classify_signal(value):

    if pd.isna(value):
        return "Unknown"

    elif value >= -80:
        return "Strong"

    elif value >= -100:
        return "Moderate"

    else:
        return "Weak"


df["signal_category"] = df["actual_rsrp"].apply(
    classify_signal
)


# =====================================================
# 7. ANALYSE EACH FOLD
# =====================================================

results = []

folds = sorted(df["fold_number"].dropna().unique())

for fold in folds:

    fold_df = df[
        df["fold_number"] == fold
    ].copy()

    actual = fold_df["actual_rsrp"].dropna()

    if len(actual) == 0:
        continue

    print("\n" + "=" * 70)
    print(f"FOLD {fold}")
    print("=" * 70)

    print(f"Observations: {len(fold_df)}")

    # -------------------------------------------------
    # Actual RSRP statistics
    # -------------------------------------------------

    print("\nACTUAL RSRP DISTRIBUTION")

    print(f"Mean:    {actual.mean():.2f} dBm")
    print(f"Median:  {actual.median():.2f} dBm")
    print(f"Std:     {actual.std():.2f} dB")
    print(f"Minimum: {actual.min():.2f} dBm")
    print(f"Maximum: {actual.max():.2f} dBm")

    # -------------------------------------------------
    # Signal category distribution
    # -------------------------------------------------

    print("\nSIGNAL CATEGORY DISTRIBUTION")

    category_counts = (
        fold_df["signal_category"].value_counts()
    )

    total = len(fold_df)

    percentages = {}

    for category in ["Strong", "Moderate", "Weak"]:

        count = category_counts.get(category, 0)

        percentage = (
            count / total * 100
            if total > 0 else 0
        )

        percentages[category] = percentage

        print(
            f"{category:10s}: "
            f"{count:7d} "
            f"({percentage:6.2f}%)"
        )

    # -------------------------------------------------
    # Model prediction distributions
    # -------------------------------------------------

    print("\nMODEL PREDICTION DISTRIBUTION")

    for model_name, column in model_columns.items():

        predicted = fold_df[column].dropna()

        if len(predicted) == 0:
            continue

        predicted_mean = predicted.mean()
        predicted_median = predicted.median()

        print(
            f"{model_name:20s} "
            f"Mean: {predicted_mean:8.2f} dBm | "
            f"Median: {predicted_median:8.2f} dBm"
        )

        results.append({

            "fold": fold,

            "model": model_name,

            "observations": len(fold_df),

            "actual_mean": actual.mean(),

            "actual_median": actual.median(),

            "actual_std": actual.std(),

            "actual_min": actual.min(),

            "actual_max": actual.max(),

            "strong_percentage": percentages["Strong"],

            "moderate_percentage": percentages["Moderate"],

            "weak_percentage": percentages["Weak"],

            "predicted_mean": predicted_mean,

            "predicted_median": predicted_median

        })


# =====================================================
# 8. SAVE RESULTS
# =====================================================

result_df = pd.DataFrame(results)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =====================================================
# 9. COMPLETION
# =====================================================

print("\n" + "=" * 70)

print("RSRP DISTRIBUTION ANALYSIS COMPLETED")

print("Total folds analysed:", len(folds))

print("\nResults saved to:")
print(OUTPUT_FILE)

print("=" * 70)