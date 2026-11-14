
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "salzburg_propagation_features.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    print("\nLoading training dataset...")

    df = pd.read_parquet(DATA_FILE)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    return df


# ==========================================
# SELECT NUMERIC FEATURES
# ==========================================

def select_features(df):

    numeric_df = df.select_dtypes(
        include=["number", "bool"]
    ).copy()

    numeric_df = numeric_df.replace(
        [np.inf, -np.inf], np.nan
    )

    # Columns that must not be treated as
    # predictive features.

    excluded_columns = [
        "rsrp_dbm",
        "cell_id",
        "pci",
        "netmode",
        "line",
        "trip",
        "predecessor",
        "predecessor2",
        "spatial_block",
        "spatial_fold",
        "easting_m",
        "northing_m",
        "latitude",
        "longitude",
        "chainage_m"
    ]

    feature_columns = [
        col for col in numeric_df.columns
        if col not in excluded_columns
    ]

    return numeric_df, feature_columns


# ==========================================
# CORRELATION ANALYSIS
# ==========================================

def calculate_correlations(df, feature_columns):

    print("\nCalculating feature correlations...")

    results = []

    for feature in feature_columns:

        valid = df[
            [feature, "rsrp_dbm"]
        ].dropna()

        if len(valid) < 2:
            continue

        if valid[feature].nunique() < 2:
            continue

        pearson = valid[feature].corr(
            valid["rsrp_dbm"],
            method="pearson"
        )

        spearman = valid[feature].corr(
            valid["rsrp_dbm"],
            method="spearman"
        )

        results.append({
            "Feature": feature,
            "Pearson_Correlation": pearson,
            "Spearman_Correlation": spearman,
            "Absolute_Pearson": abs(pearson),
            "Available_Rows": len(valid),
            "Missing_Percentage": (
                df[feature].isna().mean() * 100
            )
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Absolute_Pearson",
        ascending=False
    )

    return results_df


# ==========================================
# GROUP FEATURES BY TYPE
# ==========================================

def categorize_features(results):

    def get_category(name):

        if name in [
            "sinr", "rsrq", "rssi", "signal"
        ]:
            return "Measured Signal Quality"

        if any(x in name for x in [
            "terrain",
            "fresnel",
            "diffraction",
            "elevation"
        ]):
            return "Terrain Features"

        if any(x in name for x in [
            "path_loss",
            "pathloss"
        ]):
            return "Propagation Model Features"

        if "distance" in name:
            return "Distance Features"

        if "bearing" in name:
            return "Bearing Features"

        if "samples" in name or "range" in name:
            return "Tower Metadata"

        return "Other Features"

    results = results.copy()

    results["Category"] = results[
        "Feature"
    ].apply(get_category)

    return results


# ==========================================
# FEATURE CORRELATION CHART
# ==========================================

def generate_chart(results):

    print("\nGenerating correlation chart...")

    top_features = results.head(20).copy()

    top_features = top_features.sort_values(
        "Pearson_Correlation"
    )

    plt.figure(figsize=(11, 8))

    plt.barh(
        top_features["Feature"],
        top_features["Pearson_Correlation"]
    )

    plt.xlabel("Pearson Correlation with RSRP")
    plt.ylabel("Feature")
    plt.title("Top 20 Features Correlated with RSRP")

    plt.axvline(
        x=0,
        linestyle="--"
    )

    plt.grid(
        axis="x",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "feature_correlation.png",
        dpi=300
    )

    plt.close()


# ==========================================
# FEATURE REDUNDANCY ANALYSIS
# ==========================================

def feature_redundancy_analysis(df, feature_columns):

    print("\nChecking feature redundancy...")

    selected = df[feature_columns].copy()

    selected = selected.replace(
        [np.inf, -np.inf], np.nan
    )

    # Remove features with too many missing values
    selected = selected.loc[
        :,
        selected.isna().mean() < 0.30
    ]

    # Correlation needs complete values.
    selected = selected.dropna(axis=0)

    # Avoid analysing an excessively large matrix.
    if selected.empty or selected.shape[1] < 2:
        print("Not enough complete data for redundancy analysis.")
        return

    corr_matrix = selected.corr().abs()

    upper_triangle = corr_matrix.where(
        np.triu(
            np.ones(corr_matrix.shape),
            k=1
        ).astype(bool)
    )

    redundant_pairs = []

    for column in upper_triangle.columns:

        for index in upper_triangle.index:

            value = upper_triangle.loc[index, column]

            if pd.notna(value) and value > 0.90:

                redundant_pairs.append({
                    "Feature_1": index,
                    "Feature_2": column,
                    "Absolute_Correlation": value
                })

    redundancy_df = pd.DataFrame(
        redundant_pairs
    )

    if not redundancy_df.empty:

        redundancy_df = redundancy_df.sort_values(
            "Absolute_Correlation",
            ascending=False
        )

    print("\nHighly correlated feature pairs (> 0.90):")

    if redundancy_df.empty:
        print("No highly correlated pairs found.")
    else:
        print(redundancy_df.head(30).to_string(index=False))

    redundancy_df.to_csv(
        OUTPUT_DIR / "feature_redundancy.csv",
        index=False
    )


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data()

    numeric_df, feature_columns = select_features(df)

    results = calculate_correlations(
        numeric_df,
        feature_columns
    )

    results = categorize_features(results)

    print("\n================================")
    print("TOP 20 FEATURES")
    print("================================")

    print(
        results[
            [
                "Feature",
                "Category",
                "Pearson_Correlation",
                "Spearman_Correlation",
                "Missing_Percentage"
            ]
        ].head(20).round(4).to_string(index=False)
    )

    # Save complete correlation report.

    results.to_csv(
        OUTPUT_DIR / "feature_correlation_report.csv",
        index=False
    )

    # Save feature groups.

    category_report = results.groupby(
        "Category"
    ).agg(
        Feature_Count=("Feature", "count"),
        Mean_Absolute_Correlation=(
            "Absolute_Pearson", "mean"
        )
    ).sort_values(
        "Mean_Absolute_Correlation",
        ascending=False
    )

    category_report.to_csv(
        OUTPUT_DIR / "feature_category_summary.csv"
    )

    generate_chart(results)

    feature_redundancy_analysis(
        numeric_df,
        feature_columns
    )

    print("\n================================")
    print("FEATURE ANALYSIS COMPLETED!")
    print("================================")

    print("\nGenerated files:")
    print("1. feature_correlation_report.csv")
    print("2. feature_category_summary.csv")
    print("3. feature_redundancy.csv")
    print("4. feature_correlation.png")


if __name__ == "__main__":
    main()
