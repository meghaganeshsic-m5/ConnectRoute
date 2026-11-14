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
CHART_DIR = OUTPUT_DIR / "training_analysis_charts"


# ==========================================
# LOAD DATASET
# ==========================================

def load_data():

    print("\nLoading Salzburg training dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_parquet(DATA_FILE)

    print("Dataset loaded successfully!")
    print("Total rows:", len(df))
    print("Total columns:", len(df.columns))

    return df


# ==========================================
# BASIC DATASET INFORMATION
# ==========================================

def basic_analysis(df):

    print("\n========== DATASET INFORMATION ==========")

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst five records:")
    print(df.head())


# ==========================================
# MISSING VALUES
# ==========================================

def missing_value_analysis(df):

    print("\n========== MISSING VALUE ANALYSIS ==========")

    missing = pd.DataFrame({
        "Column": df.columns,
        "Missing_Count": df.isnull().sum().values,
        "Missing_Percentage": (
            df.isnull().mean().values * 100
        )
    })

    missing = missing.sort_values(
        by="Missing_Percentage",
        ascending=False
    )

    print(missing.head(20).to_string(index=False))

    missing.to_csv(
        OUTPUT_DIR / "missing_value_report.csv",
        index=False
    )


# ==========================================
# RSRP AND DISTANCE STATISTICS
# ==========================================

def statistical_analysis(df):

    print("\n========== RSRP STATISTICS ==========")

    print(
        df["rsrp_dbm"].describe().round(3)
    )

    print("\n========== NEAREST TOWER DISTANCE ==========")

    print(
        df["tower_1_distance_m"].describe().round(3)
    )

    stats = df[
        ["rsrp_dbm", "tower_1_distance_m"]
    ].describe().T

    stats.to_csv(
        OUTPUT_DIR / "training_statistics.csv"
    )


# ==========================================
# CORRELATION ANALYSIS
# ==========================================

def correlation_analysis(df):

    print("\n========== CORRELATION ANALYSIS ==========")

    valid = df[
        ["rsrp_dbm", "tower_1_distance_m"]
    ].replace([np.inf, -np.inf], np.nan).dropna()

    valid = valid[
        valid["tower_1_distance_m"] > 0
    ]

    valid["log_distance"] = np.log10(
        valid["tower_1_distance_m"]
    )

    raw_corr = valid[
        "rsrp_dbm"
    ].corr(valid["tower_1_distance_m"])

    log_corr = valid[
        "rsrp_dbm"
    ].corr(valid["log_distance"])

    print("RSRP vs raw distance:", round(raw_corr, 4))

    print("RSRP vs log distance:", round(log_corr, 4))


# ==========================================
# DISTANCE RANGE ANALYSIS
# ==========================================

def distance_range_analysis(df):

    print("\n========== DISTANCE RANGE ANALYSIS ==========")

    valid = df[
        ["rsrp_dbm", "tower_1_distance_m"]
    ].dropna().copy()

    bins = [0, 50, 100, 200, 300, 400, 500,
            1000, 2000, 5000, np.inf]

    labels = [
        "0-50m",
        "50-100m",
        "100-200m",
        "200-300m",
        "300-400m",
        "400-500m",
        "500-1000m",
        "1000-2000m",
        "2000-5000m",
        "Above 5000m"
    ]

    valid["Distance_Range"] = pd.cut(
        valid["tower_1_distance_m"],
        bins=bins,
        labels=labels,
        right=False
    )

    report = valid.groupby(
        "Distance_Range",
        observed=False
    ).agg(
        Measurement_Count=("rsrp_dbm", "count"),
        Mean_RSRP=("rsrp_dbm", "mean"),
        Median_RSRP=("rsrp_dbm", "median"),
        Minimum_RSRP=("rsrp_dbm", "min"),
        Maximum_RSRP=("rsrp_dbm", "max")
    ).round(3)

    print(report)

    report.to_csv(
        OUTPUT_DIR / "distance_range_analysis.csv"
    )


# ==========================================
# OUTLIER ANALYSIS
# ==========================================

def outlier_analysis(df):

    print("\n========== OUTLIER ANALYSIS ==========")

    columns = [
        "rsrp_dbm",
        "tower_1_distance_m"
    ]

    results = []

    for column in columns:

        data = df[column].dropna()

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = data[
            (data < lower) | (data > upper)
        ]

        print(f"\nColumn: {column}")
        print("Lower boundary:", round(lower, 3))
        print("Upper boundary:", round(upper, 3))
        print("Potential outliers:", len(outliers))

        results.append({
            "Column": column,
            "Lower_Boundary": lower,
            "Upper_Boundary": upper,
            "Outlier_Count": len(outliers),
            "Outlier_Percentage": (
                len(outliers) / len(data) * 100
            )
        })

    pd.DataFrame(results).to_csv(
        OUTPUT_DIR / "outlier_analysis.csv",
        index=False
    )


# ==========================================
# VISUALIZATION
# ==========================================

def generate_charts(df):

    print("\n========== GENERATING CHARTS ==========")

    CHART_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    valid = df[
        ["rsrp_dbm", "tower_1_distance_m"]
    ].replace([np.inf, -np.inf], np.nan).dropna()

    valid = valid[
        valid["tower_1_distance_m"] > 0
    ]

    # RSRP distribution

    plt.figure(figsize=(9, 5))

    plt.hist(
        valid["rsrp_dbm"],
        bins=50
    )

    plt.xlabel("RSRP (dBm)")
    plt.ylabel("Number of Measurements")
    plt.title("RSRP Distribution")
    plt.grid(alpha=0.3)

    plt.savefig(
        CHART_DIR / "rsrp_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # Tower distance distribution

    plt.figure(figsize=(9, 5))

    plt.hist(
        valid["tower_1_distance_m"],
        bins=50
    )

    plt.xlabel("Nearest Tower Distance (m)")
    plt.ylabel("Number of Measurements")
    plt.title("Tower Distance Distribution")
    plt.grid(alpha=0.3)

    plt.savefig(
        CHART_DIR / "distance_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # RSRP vs distance

    plt.figure(figsize=(9, 5))

    plt.scatter(
        valid["tower_1_distance_m"],
        valid["rsrp_dbm"],
        alpha=0.15,
        s=5
    )

    plt.xlabel("Nearest Tower Distance (m)")
    plt.ylabel("RSRP (dBm)")
    plt.title("RSRP vs Nearest Tower Distance")
    plt.grid(alpha=0.3)

    plt.savefig(
        CHART_DIR / "rsrp_vs_distance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # RSRP vs log distance

    plt.figure(figsize=(9, 5))

    plt.scatter(
        np.log10(valid["tower_1_distance_m"]),
        valid["rsrp_dbm"],
        alpha=0.15,
        s=5
    )

    plt.xlabel("Log10 of Nearest Tower Distance")
    plt.ylabel("RSRP (dBm)")
    plt.title("RSRP vs Log Distance")
    plt.grid(alpha=0.3)

    plt.savefig(
        CHART_DIR / "rsrp_vs_log_distance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("All charts generated successfully!")


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data()

    required_columns = [
        "rsrp_dbm",
        "tower_1_distance_m"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column missing: {column}"
            )

    basic_analysis(df)

    missing_value_analysis(df)

    statistical_analysis(df)

    correlation_analysis(df)

    distance_range_analysis(df)

    outlier_analysis(df)

    generate_charts(df)

    print("\n====================================")
    print("TRAINING DATA ANALYSIS COMPLETED!")
    print("====================================")

    print("\nResults saved inside:")
    print(OUTPUT_DIR)

    print("\nCharts saved inside:")
    print(CHART_DIR)


if __name__ == "__main__":
    main()