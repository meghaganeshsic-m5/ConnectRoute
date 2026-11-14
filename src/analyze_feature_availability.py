
import pandas as pd
import numpy as np

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
# FEATURE CLASSIFICATION
# ==========================================

def classify_feature(feature):

    name = feature.lower()

    # Target
    if name == "rsrp_dbm":
        return "Target Variable"

    # Direct measurements that may not be
    # available at an unmeasured route location.
    measured_features = [
        "rssi",
        "sinr",
        "rsrq",
        "signal",
        "datarate"
    ]

    if name in measured_features:
        return "Measured Signal - Exclude"

    # Identifiers and metadata
    identifier_features = [
        "cell_id",
        "pci",
        "netmode",
        "line",
        "trip",
        "file",
        "timestamp",
        "predecessor",
        "predecessor2",
        "spatial_block",
        "spatial_fold"
    ]

    if name in identifier_features:
        return "Identifier or Metadata"

    # Coordinates and location
    if name in [
        "latitude",
        "longitude",
        "easting_m",
        "northing_m",
        "chainage_m",
        "newpos"
    ]:
        return "Location Feature - Review"

    # Tower distance and bearing
    if "distance" in name or "bearing" in name:
        return "Tower Geometry - Candidate"

    # Terrain-related features
    if any(word in name for word in [
        "elevation",
        "terrain",
        "fresnel",
        "diffraction",
        "profile_sample"
    ]):
        return "Terrain - Candidate"

    # Propagation-model features
    if "path_loss" in name or "pathloss" in name:
        return "Propagation - Candidate"

    # Tower metadata
    if any(word in name for word in [
        "samples",
        "range",
        "created",
        "updated",
        "mcc",
        "net",
        "area",
        "unit"
    ]):
        return "Tower Metadata - Review"

    return "Other - Review"


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    print("\nLoading training dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    df = pd.read_parquet(DATA_FILE)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    return df


# ==========================================
# FEATURE AVAILABILITY REPORT
# ==========================================

def create_availability_report(df):

    print("\nClassifying features...")

    records = []

    for column in df.columns:

        category = classify_feature(column)

        records.append({

            "Feature": column,

            "Category": category,

            "Data_Type": str(df[column].dtype),

            "Missing_Count": df[column].isna().sum(),

            "Missing_Percentage": (
                df[column].isna().mean() * 100
            ),

            "Unique_Values": df[column].nunique()

        })

    report = pd.DataFrame(records)

    report.to_csv(
        OUTPUT_DIR / "feature_availability_report.csv",
        index=False
    )

    print("\nFEATURE CATEGORY SUMMARY")
    print("=========================")

    print(
        report.groupby("Category").size()
        .sort_values(ascending=False)
    )

    return report


# ==========================================
# CHECK TARGET-RELATED FEATURES
# ==========================================

def check_target_related_features(df):

    print("\nCHECKING TARGET-RELATED FEATURES")

    print("================================")

    target = "rsrp_dbm"

    candidate_columns = [
        "signal",
        "rssi",
        "sinr",
        "rsrq",
        "datarate"
    ]

    results = []

    for column in candidate_columns:

        if column not in df.columns:
            continue

        valid = df[
            [target, column]
        ].dropna()

        correlation = valid[
            target
        ].corr(valid[column])

        results.append({

            "Feature": column,

            "Correlation_With_RSRP": correlation,

            "Available_Rows": len(valid)

        })

    result_df = pd.DataFrame(results)

    print(result_df.round(4).to_string(index=False))

    result_df.to_csv(
        OUTPUT_DIR / "target_related_features.csv",
        index=False
    )


# ==========================================
# CHECK DUPLICATE FEATURES
# ==========================================

def check_duplicate_features(df):

    print("\nCHECKING EXACT DUPLICATE FEATURES")

    print("=================================")

    numeric = df.select_dtypes(
        include=["number", "bool"]
    )

    duplicates = []

    columns = numeric.columns.tolist()

    for i in range(len(columns)):

        col1 = columns[i]

        for j in range(i + 1, len(columns)):

            col2 = columns[j]

            if numeric[col1].equals(numeric[col2]):

                duplicates.append({

                    "Feature_1": col1,

                    "Feature_2": col2

                })

    duplicate_df = pd.DataFrame(duplicates)

    if duplicate_df.empty:

        print("No exact duplicate columns found.")

    else:

        print(
            duplicate_df.head(30).to_string(index=False)
        )

    duplicate_df.to_csv(
        OUTPUT_DIR / "exact_duplicate_features.csv",
        index=False
    )


# ==========================================
# CHECK PROPAGATION FEATURE AVAILABILITY
# ==========================================

def check_propagation_features(report):

    print("\nPROPAGATION FEATURE ANALYSIS")

    print("============================")

    propagation = report[
        report["Category"] == "Propagation - Candidate"
    ].copy()

    print(
        propagation[
            [
                "Feature",
                "Missing_Percentage",
                "Unique_Values"
            ]
        ].to_string(index=False)
    )

    propagation.to_csv(
        OUTPUT_DIR / "propagation_feature_report.csv",
        index=False
    )


# ==========================================
# MAIN
# ==========================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data()

    report = create_availability_report(df)

    check_target_related_features(df)

    check_duplicate_features(df)

    check_propagation_features(report)

    print("\n====================================")

    print("FEATURE AVAILABILITY ANALYSIS DONE!")

    print("====================================")

    print("\nGenerated files:")

    print("1. feature_availability_report.csv")

    print("2. target_related_features.csv")

    print("3. exact_duplicate_features.csv")

    print("4. propagation_feature_report.csv")


if __name__ == "__main__":
    main()
