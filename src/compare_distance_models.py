
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "serving_cell_distance_comparison.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"


# --------------------------------------------------
# MODEL EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_model(name, X_train, X_test, y_train, y_test):

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    intercept = model.intercept_
    slope = model.coef_[0]

    path_loss_exponent = -slope / 10

    results = {
        "Model": name,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Intercept": intercept,
        "Slope": slope,
        "Path_Loss_Exponent": path_loss_exponent
    }

    return results, predictions


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():

    print("\nLoading serving-cell comparison dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run compare_tower_distances.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    print("Original rows:", len(df))


    # --------------------------------------------------
    # DATA CLEANING
    # --------------------------------------------------

    required_columns = [
        "rsrp_dbm",
        "tower_1_distance_m",
        "serving_cell_distance_m",
        "easting_m",
        "northing_m"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df[required_columns].copy()

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna()

    df = df[
        (df["tower_1_distance_m"] > 0)
        & (df["serving_cell_distance_m"] > 0)
    ].copy()

    print("Valid rows:", len(df))


    # --------------------------------------------------
    # CREATE LOG-DISTANCE FEATURES
    # --------------------------------------------------

    df["nearest_log_distance"] = np.log10(
        df["tower_1_distance_m"]
    )

    df["serving_log_distance"] = np.log10(
        df["serving_cell_distance_m"]
    )

    y = df["rsrp_dbm"]


    # --------------------------------------------------
    # SPATIAL TRAIN-TEST SPLIT
    # --------------------------------------------------

    # Divide measurement locations into 500 m grid cells.
    # Measurements from the same grid cell stay together.

    grid_size = 500

    df["grid_x"] = np.floor(
        df["easting_m"] / grid_size
    )

    df["grid_y"] = np.floor(
        df["northing_m"] / grid_size
    )

    groups = (
        df["grid_x"].astype(str)
        + "_"
        + df["grid_y"].astype(str)
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_indices, test_indices = next(
        splitter.split(df, y, groups=groups)
    )

    print("\nTRAIN-TEST SPLIT")
    print("----------------")
    print("Training rows:", len(train_indices))
    print("Testing rows:", len(test_indices))
    print("Spatial grid size:", grid_size, "metres")


    # --------------------------------------------------
    # MODEL A: NEAREST TOWER DISTANCE
    # --------------------------------------------------

    print("\nTraining Model A: Nearest Tower Distance...")

    X_nearest = df[
        ["nearest_log_distance"]
    ]

    nearest_results, nearest_predictions = evaluate_model(
        "Nearest Tower",
        X_nearest.iloc[train_indices],
        X_nearest.iloc[test_indices],
        y.iloc[train_indices],
        y.iloc[test_indices]
    )


    # --------------------------------------------------
    # MODEL B: SERVING CELL DISTANCE
    # --------------------------------------------------

    print("\nTraining Model B: Serving Cell Distance...")

    X_serving = df[
        ["serving_log_distance"]
    ]

    serving_results, serving_predictions = evaluate_model(
        "Serving Cell",
        X_serving.iloc[train_indices],
        X_serving.iloc[test_indices],
        y.iloc[train_indices],
        y.iloc[test_indices]
    )


    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    results_df = pd.DataFrame([
        nearest_results,
        serving_results
    ])

    print("\nMODEL PERFORMANCE COMPARISON")
    print("============================")

    print(
        results_df[
            ["Model", "R2", "MAE", "RMSE"]
        ].round(4).to_string(index=False)
    )

    print("\nMODEL COEFFICIENTS")
    print("==================")

    print(
        results_df[
            [
                "Model",
                "Intercept",
                "Slope",
                "Path_Loss_Exponent"
            ]
        ].round(4).to_string(index=False)
    )


    # --------------------------------------------------
    # DETERMINE BETTER MODEL
    # --------------------------------------------------

    print("\nMODEL COMPARISON SUMMARY")
    print("========================")

    best_r2 = results_df.loc[
        results_df["R2"].idxmax(), "Model"
    ]

    best_mae = results_df.loc[
        results_df["MAE"].idxmin(), "Model"
    ]

    best_rmse = results_df.loc[
        results_df["RMSE"].idxmin(), "Model"
    ]

    print("Better R2:", best_r2)
    print("Lower MAE:", best_mae)
    print("Lower RMSE:", best_rmse)


    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_file = (
        OUTPUT_DIR / "distance_model_metrics.csv"
    )

    results_df.to_csv(
        metrics_file,
        index=False
    )

    predictions_df = df.iloc[test_indices][
        [
            "rsrp_dbm",
            "tower_1_distance_m",
            "serving_cell_distance_m"
        ]
    ].copy()

    predictions_df["nearest_predicted_rsrp"] = (
        nearest_predictions
    )

    predictions_df["serving_predicted_rsrp"] = (
        serving_predictions
    )

    predictions_file = (
        OUTPUT_DIR / "distance_model_test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_file,
        index=False
    )


    # --------------------------------------------------
    # COMPLETION
    # --------------------------------------------------

    print("\nEXPERIMENT COMPLETED SUCCESSFULLY!")

    print("\nSaved files:")
    print(metrics_file)
    print(predictions_file)


if __name__ == "__main__":
    main()
