
import pandas as pd
import numpy as np
from pathlib import Path
from pyproj import Transformer


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MEASUREMENT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "salzburg_propagation_features.parquet"
)

TOWER_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "opencellid_austria_lte_2021.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():

    print("\nLoading datasets...")

    measurements = pd.read_parquet(MEASUREMENT_FILE)
    towers = pd.read_parquet(TOWER_FILE)

    # Remove invalid measurement IDs
    measurements = measurements[
        measurements["cell_id"].notna()
        & (measurements["cell_id"] != 0)
    ].copy()

    measurements["cell_id"] = (
        measurements["cell_id"].astype(int)
    )

    towers["cell"] = towers["cell"].astype(int)

    # Filter valid tower records
    towers = towers[
        towers["samples"].notna()
        & (towers["samples"] >= 5)
        & towers["latitude"].notna()
        & towers["longitude"].notna()
    ].copy()

    # Exclude ambiguous cell IDs with multiple records
    counts = towers.groupby("cell")["cell"].transform("size")
    unique_towers = towers[counts == 1].copy()

    # Rename tower coordinates BEFORE merging
    unique_towers = unique_towers.rename(
        columns={
            "latitude": "tower_latitude",
            "longitude": "tower_longitude"
        }
    )

    # --------------------------------------------------
    # MATCH SERVING CELLS
    # --------------------------------------------------

    matched = measurements.merge(
        unique_towers,
        left_on="cell_id",
        right_on="cell",
        how="inner"
    )

    print("\nMatching results")
    print("----------------")
    print("Measurement rows:", len(measurements))
    print("Matched rows:", len(matched))
    print("Unmatched rows:", len(measurements) - len(matched))

    if matched.empty:
        print("No matching records found.")
        return

    # --------------------------------------------------
    # COORDINATE TRANSFORMATION
    # --------------------------------------------------

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:32633",
        always_xy=True
    )

    tower_x, tower_y = transformer.transform(
        matched["tower_longitude"].to_numpy(),
        matched["tower_latitude"].to_numpy()
    )

    # --------------------------------------------------
    # SERVING CELL DISTANCE
    # --------------------------------------------------

    matched["serving_cell_distance_m"] = np.sqrt(
        (matched["easting_m"].to_numpy() - tower_x) ** 2
        +
        (matched["northing_m"].to_numpy() - tower_y) ** 2
    )

    # Compare with nearest tower
    matched["distance_difference_m"] = (
        matched["serving_cell_distance_m"]
        - matched["tower_1_distance_m"]
    )

    # --------------------------------------------------
    # DISTANCE STATISTICS
    # --------------------------------------------------

    print("\nDISTANCE COMPARISON")
    print("-------------------")

    print("\nNearest tower distance:")
    print(
        matched["tower_1_distance_m"]
        .describe()
        .round(2)
    )

    print("\nServing cell distance:")
    print(
        matched["serving_cell_distance_m"]
        .describe()
        .round(2)
    )

    print("\nDistance difference:")
    print(
        matched["distance_difference_m"]
        .describe()
        .round(2)
    )

    # --------------------------------------------------
    # RSRP CORRELATION
    # --------------------------------------------------

    print("\nRSRP AND DISTANCE CORRELATION")
    print("-----------------------------")

    nearest_corr = matched[
        ["rsrp_dbm", "tower_1_distance_m"]
    ].corr().iloc[0, 1]

    serving_corr = matched[
        ["rsrp_dbm", "serving_cell_distance_m"]
    ].corr().iloc[0, 1]

    print("Nearest tower:", round(nearest_corr, 4))
    print("Serving cell:", round(serving_corr, 4))

    # --------------------------------------------------
    # DISTANCE COMPARISON COUNTS
    # --------------------------------------------------

    print("\nADDITIONAL ANALYSIS")
    print("-------------------")

    print(
        "Serving cell closer:",
        (
            matched["serving_cell_distance_m"]
            < matched["tower_1_distance_m"]
        ).sum()
    )

    print(
        "Serving cell farther:",
        (
            matched["serving_cell_distance_m"]
            > matched["tower_1_distance_m"]
        ).sum()
    )

    print(
        "Equal distances:",
        (
            matched["serving_cell_distance_m"]
            == matched["tower_1_distance_m"]
        ).sum()
    )

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = (
        OUTPUT_DIR
        / "serving_cell_distance_comparison.csv"
    )

    matched.to_csv(output_file, index=False)

    print("\nExperiment completed successfully!")
    print("Results saved to:", output_file)


if __name__ == "__main__":
    main()
