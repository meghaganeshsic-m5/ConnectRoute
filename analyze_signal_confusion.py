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

CONFUSION_OUTPUT = (
    PROJECT_ROOT / "outputs"
    / "signal_confusion_matrices.csv"
)

METRICS_OUTPUT = (
    PROJECT_ROOT / "outputs"
    / "signal_confusion_metrics.csv"
)


# =====================================================
# 2. LOAD DATA
# =====================================================

print("Loading prediction dataset...")

df = pd.read_parquet(PREDICTIONS_FILE)

print("Dataset shape:", df.shape)


# =====================================================
# 3. MODEL COLUMNS
# =====================================================

models = {
    "Physics": "physics_baseline_rsrp_dbm",
    "LightGBM": "pure_lightgbm_rsrp_dbm",
    "Hybrid LightGBM": "hybrid_lightgbm_rsrp_dbm",
    "XGBoost": "hybrid_xgboost_rsrp_dbm"
}

required_columns = [
    "fold",
    "rsrp_dbm"
] + list(models.values())

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")


# =====================================================
# 4. SIGNAL CLASSIFICATION
# =====================================================

categories = ["Strong", "Moderate", "Weak"]


def classify_signal(value):

    if pd.isna(value):
        return np.nan

    if value >= -80:
        return "Strong"

    elif value >= -100:
        return "Moderate"

    else:
        return "Weak"


df["actual_category"] = df["rsrp_dbm"].apply(
    classify_signal
)


# =====================================================
# 5. CONFUSION MATRIX ANALYSIS
# =====================================================

confusion_results = []
metric_results = []

folds = sorted(df["fold"].dropna().unique())

for fold in folds:

    fold_df = df[df["fold"] == fold]

    print("\n" + "=" * 75)
    print(f"FOLD {fold}")
    print("=" * 75)

    for model_name, prediction_column in models.items():

        valid_df = fold_df[
            ["actual_category", prediction_column]
        ].dropna().copy()

        valid_df["predicted_category"] = (
            valid_df[prediction_column].apply(classify_signal)
        )

        valid_df = valid_df.dropna(
            subset=["predicted_category"]
        )

        actual = valid_df["actual_category"]
        predicted = valid_df["predicted_category"]

        # Confusion matrix
        matrix = pd.crosstab(
            actual,
            predicted
        ).reindex(
            index=categories,
            columns=categories,
            fill_value=0
        )

        print(f"\nMODEL: {model_name}")
        print("\nConfusion Matrix:")
        print(matrix)

        # Save confusion matrix in long format
        for actual_category in categories:

            for predicted_category in categories:

                confusion_results.append({

                    "fold": fold,
                    "model": model_name,
                    "actual_category": actual_category,
                    "predicted_category": predicted_category,
                    "count": int(
                        matrix.loc[
                            actual_category,
                            predicted_category
                        ]
                    )

                })

        # Calculate classification metrics
        total = matrix.to_numpy().sum()

        correct = sum(
            matrix.loc[category, category]
            for category in categories
        )

        accuracy = correct / total if total else 0

        class_recalls = []
        class_precisions = []
        class_f1_scores = []

        for category in categories:

            true_positive = matrix.loc[
                category, category
            ]

            actual_count = matrix.loc[
                category
            ].sum()

            predicted_count = matrix[
                category
            ].sum()

            recall = (
                true_positive / actual_count
                if actual_count else 0
            )

            precision = (
                true_positive / predicted_count
                if predicted_count else 0
            )

            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0
            )

            class_recalls.append(recall)
            class_precisions.append(precision)
            class_f1_scores.append(f1)

        balanced_accuracy = np.mean(class_recalls)

        macro_f1 = np.mean(class_f1_scores)

        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")

        metric_results.append({

            "fold": fold,
            "model": model_name,
            "observations": total,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1

        })


# =====================================================
# 6. SAVE RESULTS
# =====================================================

confusion_df = pd.DataFrame(confusion_results)

metrics_df = pd.DataFrame(metric_results)

confusion_df.to_csv(
    CONFUSION_OUTPUT,
    index=False
)

metrics_df.to_csv(
    METRICS_OUTPUT,
    index=False
)


# =====================================================
# 7. FINAL SUMMARY
# =====================================================

print("\n" + "=" * 75)
print("SIGNAL CONFUSION ANALYSIS COMPLETED")
print("=" * 75)

print("\nClassification Metrics:")
print(metrics_df.to_string(index=False))

print("\nConfusion matrices saved to:")
print(CONFUSION_OUTPUT)

print("\nClassification metrics saved to:")
print(METRICS_OUTPUT)