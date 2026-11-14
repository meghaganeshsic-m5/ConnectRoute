import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================
# 1. FILE PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent

PREDICTIONS_FILE = (
    PROJECT_ROOT / "outputs"
    / "experimental_model_predictions.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "outputs"
    / "error_by_signal_category.csv"
)


# =====================================================
# 2. LOAD DATASET
# =====================================================

print("Loading prediction dataset...")

df = pd.read_parquet(PREDICTIONS_FILE)

print("Dataset shape:", df.shape)


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

required_columns = [
    "fold",
    "rsrp_dbm"
] + list(model_columns.values())

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )


# =====================================================
# 5. CLASSIFY ACTUAL SIGNAL STRENGTH
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


df["signal_category"] = df["rsrp_dbm"].apply(
    classify_signal
)


# =====================================================
# 6. DEFINE SIGNAL CATEGORIES
# =====================================================

categories = [
    "Strong",
    "Moderate",
    "Weak"
]

folds = sorted(df["fold"].dropna().unique())

results = []


# =====================================================
# 7. CALCULATE ERRORS
# =====================================================

for fold in folds:

    fold_df = df[df["fold"] == fold]

    print("\n" + "=" * 75)
    print(f"FOLD {fold}")
    print("=" * 75)

    for category in categories:

        category_df = fold_df[
            fold_df["signal_category"] == category
        ]

        actual = category_df["rsrp_dbm"]

        count = len(category_df)

        print(f"\nSIGNAL CATEGORY: {category}")
        print(f"Observations: {count}")

        if count == 0:
            print("No observations in this category.")
            continue

        for model_name, column in model_columns.items():

            valid = category_df[
                ["rsrp_dbm", column]
            ].dropna()

            y_actual = valid["rsrp_dbm"]
            y_predicted = valid[column]

            if len(valid) == 0:
                continue

            errors = y_predicted - y_actual

            mae = np.mean(np.abs(errors))

            rmse = np.sqrt(
                np.mean(errors ** 2)
            )

            bias = np.mean(errors)

            print(
                f"{model_name:20s} "
                f"MAE: {mae:7.2f} dB | "
                f"RMSE: {rmse:7.2f} dB | "
                f"Bias: {bias:+7.2f} dB"
            )

            results.append({

                "fold": fold,

                "signal_category": category,

                "model": model_name,

                "observations": len(valid),

                "MAE": mae,

                "RMSE": rmse,

                "Bias": bias

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

print("\n" + "=" * 75)

print("SIGNAL CATEGORY ERROR ANALYSIS COMPLETED")

print("\nTotal folds analysed:", len(folds))

print("\nResults saved to:")
print(OUTPUT_FILE)

print("=" * 75)