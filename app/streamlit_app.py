"""Illustrative route alternatives dashboard with automatic signal estimates."""

from __future__ import annotations

import json

from pathlib import Path

from typing import Any

import folium

import numpy as np

import pandas as pd

import requests

import streamlit as st

from pyproj import CRS

from connectroute.evaluate import build_benchmark_tables

from connectroute.route_predict import (

    DEFAULT_SALZBURG_FEATURES,

    DEFAULT_SALZBURG_PREDICTIONS,

    DEFAULT_TOWERS,

    fit_route_projection_model,

    predict_route_signal,

)

from connectroute.route_terrain import (

    add_route_terrain_features,

    prepare_route_dem,

)

from connectroute.routing import (

    fetch_route_alternatives,

    geocode_place,

    sample_route_coordinates,

)

st.set_page_config(page_title="ConnectRoute 2.0", page_icon="📶", layout="wide")

st.markdown(

    """

    <style>

    :root {

        --cr-ink: #14213d;

        --cr-muted: #64748b;

        --cr-line: #dbe4ef;

        --cr-blue: #1d4ed8;

        --cr-teal: #0d9488;

        --cr-surface: #f5f8fc;

    }

    .stApp {

        background:

            radial-gradient(ellipse at 86% 0%, rgba(37, 99, 235, .09), transparent 34rem),

            #f3f6fb;

        color: var(--cr-ink);

    }

    [data-testid="stHeader"] { background: rgba(243, 246, 251, .92); }

    [data-testid="stMain"] .block-container {

        max-width: 1440px;

        padding: 2rem clamp(1rem, 3vw, 3rem) 4rem;

    }

    [data-testid="stMarkdownContainer"] p { color: var(--cr-muted); }

    .cr-eyebrow {

        color: #0f766e;

        font-size: .75rem;

        font-weight: 800;

        letter-spacing: .16em;

        text-transform: uppercase;

        margin: 0 0 .65rem;

    }

    .cr-title {

        color: var(--cr-ink);

        font-size: clamp(2.1rem, 4vw, 3.35rem);

        font-weight: 800;

        letter-spacing: -.055em;

        line-height: 1.04;

        margin: 0;

    }

    .cr-subtitle {

        color: var(--cr-muted);

        font-size: 1.04rem;

        line-height: 1.6;

        margin: .8rem 0 1.35rem;

        max-width: 760px;

    }

    .cr-status {

        align-items: center;

        background: #fff8e8;

        border: 1px solid #f4dda9;

        border-radius: 12px;

        color: #805817;

        display: flex;

        font-size: .9rem;

        gap: .65rem;

        line-height: 1.45;

        margin: .75rem 0 1.25rem;

        padding: .8rem 1rem;

    }

    .cr-status-dot {

        background: #e3a328;

        border-radius: 50%;

        flex: 0 0 .55rem;

        height: .55rem;

    }

    [data-testid="stForm"] {

        background: #fff;

        border: 1px solid var(--cr-line);

        border-radius: 18px;

        box-shadow: 0 10px 28px rgba(27, 45, 79, .055);

        padding: 1.2rem 1.3rem .5rem;

    }

    [data-testid="stForm"] label {

        color: var(--cr-ink);

        font-size: .85rem;

        font-weight: 700;

    }

    [data-testid="stTextInput"] input,

    [data-testid="stSelectbox"] [role="combobox"] {

        border-radius: 10px;

        min-height: 2.8rem;

    }

    .stButton button, [data-testid="stFormSubmitButton"] button {

        background: linear-gradient(110deg, #0f766e, #115e59);

        border: 0;

        border-radius: 10px;

        color: white;

        font-weight: 750;

        min-height: 2.8rem;

        transition: transform .15s ease, box-shadow .15s ease;

        width: 100%;

    }

    .stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {

        box-shadow: 0 8px 18px rgba(15, 118, 110, .24);

        color: white;

        transform: translateY(-1px);

    }

    .cr-section-title {

        color: var(--cr-ink);

        font-size: 1.22rem;

        font-weight: 780;

        letter-spacing: -.025em;

        margin: .4rem 0 .15rem;

    }

    .cr-section-note { color: var(--cr-muted); font-size: .88rem; margin: 0 0 .8rem; }

    [data-testid="stMetric"] {

        background: #fff;

        border: 1px solid var(--cr-line);

        border-radius: 14px;

        padding: .9rem 1rem;

    }

    [data-testid="stMetricLabel"] p {

        color: var(--cr-muted);

        font-size: .78rem;

        font-weight: 650;

    }

    [data-testid="stMetricValue"] {

        color: var(--cr-ink);

        font-size: 1.55rem;

        font-weight: 780;

    }

    [data-testid="stTabs"] button { font-weight: 700; }

    [data-testid="stDataFrame"] {

        border: 1px solid var(--cr-line);

        border-radius: 12px;

        overflow: hidden;

    }


    [data-testid="stMetric"] {
        box-shadow: 0 8px 22px rgba(20, 33, 61, .045);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 26px rgba(20, 33, 61, .08);
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #0f766e;
        border-bottom-color: #0f766e;
    }
    .cr-signal-legend {
        display: flex; flex-wrap: wrap; gap: .55rem;
        margin: .55rem 0 .85rem;
    }
    .cr-legend-item {
        display: inline-flex; align-items: center; gap: .4rem;
        border: 1px solid #e2e8f0; border-radius: 999px;
        background: #fff; padding: .32rem .62rem;
        color: #334155; font-size: .78rem; font-weight: 650;
    }
    .cr-swatch { width: .62rem; height: .62rem; border-radius: 50%; display: inline-block; }
    </style>

    """,

    unsafe_allow_html=True,

)

st.markdown(

    """

    <div class="cr-eyebrow">ConnectRoute <span aria-hidden="true">/</span> Route intelligence</div>

    <h1 class="cr-title">Plan your journey.<br>See the signal ahead.</h1>

    <p class="cr-subtitle">

      Compare driving routes and explore estimated LTE signal and terrain along the way.

      Start with two places below.

    </p>

    <div class="cr-status">

      <span class="cr-status-dot"></span>

      <span><strong>Research preview:</strong> signal is an illustrative Salzburg-to-India transfer,

      not validated coverage. Terrain is shown as context, not used to adjust the signal estimate.</span>

    </div>

    """,

    unsafe_allow_html=True,

)

with st.form("route_search_form", border=False):

    start_column, destination_column, spacing_column, submit_column = st.columns(

        [1.2, 1.2, 0.75, 0.8],

        vertical_alignment="bottom",

    )

    with start_column:

        origin_text = st.text_input(

            "Starting point",

            "Bengaluru, Karnataka, India",

            placeholder="City, address, or landmark",

        )

    with destination_column:

        destination_text = st.text_input(

            "Destination",

            "Mysuru, Karnataka, India",

            placeholder="City, address, or landmark",

        )

    with spacing_column:

        sample_spacing = st.selectbox(

            "Profile detail",

            options=[50, 100, 200],

            index=1,

            format_func=lambda value: f"{value} m",

        )

    with submit_column:

        submitted = st.form_submit_button(

            "Explore routes",

            type="primary",

            icon=":material/route:",

        )

st.caption(

    "Map and routing: OpenStreetMap / OSRM · Tower records: OpenCelliD, CC BY-SA 4.0 · "

    "Elevation: Copernicus DEM GLO-30 DSM"

)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_FILES = {

    "propagation": PROJECT_ROOT / "data" / "processed" / "spatial_baseline_predictions.parquet",

    "ml": PROJECT_ROOT / "data" / "processed" / "spatial_model_predictions.parquet",

    "features": PROJECT_ROOT / "data" / "processed" / "salzburg_propagation_features.parquet",

    "summary": PROJECT_ROOT / "results" / "model_summary.json",

}

@st.cache_data(show_spinner=False, ttl=3600)

def _geocode(place: str) -> dict[str, Any]:

    return geocode_place(place)

@st.cache_data(show_spinner=False, ttl=900)

def _routes(origin: dict[str, Any], destination: dict[str, Any]) -> list[dict[str, Any]]:

    return fetch_route_alternatives(origin, destination, alternatives=3)

@st.cache_data(show_spinner=False)

def _india_towers(path: str) -> pd.DataFrame:

    return pd.read_parquet(path)

@st.cache_data(show_spinner=False)

def _terrain_profile(

    profile: pd.DataFrame,

    dem_path: str,

    crs_string: str,

) -> pd.DataFrame:

    return add_route_terrain_features(

        profile,

        Path(dem_path),

        CRS.from_user_input(crs_string),

    )

@st.cache_data(show_spinner=False)

def _route_model(

    feature_path: str,

    prediction_path: str,

) -> tuple[dict[str, float | int], dict[str, float]]:

    return fit_route_projection_model(Path(feature_path), Path(prediction_path))

@st.cache_data(show_spinner=False)

def _model_benchmark(

    propagation_path: str,

    ml_path: str,

    features_path: str,

    summary_path: str,

) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:

    propagation_columns = [

        "fold",

        "rsrp_dbm",

        "predicted_free_space_rsrp_dbm",

        "predicted_cost231_hata_rsrp_dbm",

        "predicted_hata_rural_rsrp_dbm",

        "predicted_3gpp_rma_los_rsrp_dbm",

        "predicted_3gpp_rma_nlos_rsrp_dbm",

        "predicted_3gpp_uma_los_rsrp_dbm",

        "predicted_3gpp_uma_nlos_rsrp_dbm",

        "predicted_fitted_log_distance_rsrp_dbm",

    ]

    ml_columns = [

        "fold",

        "rsrp_dbm",

        "pure_lightgbm_rsrp_dbm",

        "hybrid_lightgbm_rsrp_dbm",

        "hybrid_xgboost_rsrp_dbm",

        "quantile_50_rsrp_dbm",

    ]

    feature_columns = [

        "rsrp_dbm",

        "best_server_cost231_hata_in_domain",

        "best_server_hata_rural_in_domain",

        "best_server_3gpp_rma_los_in_domain",

        "best_server_3gpp_rma_nlos_in_domain",

        "best_server_3gpp_uma_los_in_domain",

        "best_server_3gpp_uma_nlos_in_domain",

    ]

    propagation_metrics, ml_metrics = build_benchmark_tables(

        pd.read_parquet(propagation_path, columns=propagation_columns),

        pd.read_parquet(ml_path, columns=ml_columns),

        pd.read_parquet(features_path, columns=feature_columns),

    )

    with Path(summary_path).open(encoding="utf-8") as summary_file:

        model_summary = json.load(summary_file)

    return propagation_metrics, ml_metrics, model_summary

def _signal_profiles(

    routes: list[dict[str, Any]],

    samples_by_route: list[list[dict[str, float]]],

) -> tuple[list[pd.DataFrame], dict[str, Any]]:

    if not DEFAULT_TOWERS.is_file():

        raise FileNotFoundError(

            "India LTE tower data is not installed. Run "

            r".**\.**&#x76;env\Script&#x73;**\p**ython.exe -m connectroute.tower_ingest "

            "--country-code 404 405 --output data\\\processed\\\\"

            "opencellid_india_lte_2021.parquet --summary "

            "results\\\india_tower_ingest_summary.json."

        )

    towers = _india_towers(str(DEFAULT_TOWERS))

    fitted, residuals = _route_model(

        str(DEFAULT_SALZBURG_FEATURES),

        str(DEFAULT_SALZBURG_PREDICTIONS),

    )

    profiles: list[pd.DataFrame] = []

    details: list[dict[str, Any]] = []

    for route, samples in zip(routes, samples_by_route, strict=True):

        profile, summary = predict_route_signal(samples, towers, fitted, residuals)

        profile["route_distance_m"] = float(route["distance"])

        profile["route_duration_s"] = float(route["duration"])

        profiles.append(profile)

        details.append(summary)

    return profiles, {

        "eligible_india_towers": int(

            max(item["eligible_lte_towers"] for item in details)

        ),

        "route_details": details,

        "model": fitted,

        "residual_calibration": residuals,

    }

def _terrain_profiles(

    profiles: list[pd.DataFrame],

) -> tuple[list[pd.DataFrame], dict[str, str]]:

    route_coordinates = np.vstack(

        [

            profile[["longitude", "latitude"]].to_numpy(dtype=float)

            for profile in profiles

        ]

    )

    nearest_tower_coordinates = np.vstack(

        [

            profile[["nearest_tower_longitude", "nearest_tower_latitude"]].to_numpy(

                dtype=float

            )

            for profile in profiles

        ]

    )

    projected_dem, projected_crs = prepare_route_dem(

        route_coordinates,

        nearest_tower_coordinates,

    )

    profiles = [

        _terrain_profile(profile, str(projected_dem), projected_crs.to_string())

        for profile in profiles

    ]

    return profiles, {

        "terrain_dem": "Copernicus DEM GLO-30 Public (DSM)",

        "terrain_projected_crs": projected_crs.to_string(),

    }

def _route_color(rsrp: float) -> str:

    if rsrp >= -80:

        return "#16a34a"

    if rsrp >= -100:

        return "#f59e0b"

    return "#ef4444"

with st.expander("Salzburg model benchmark", expanded=True):

    st.caption(

        "These tables compare saved out-of-fold predictions against measured Salzburg RSRP. "

        "They are separate from the illustrative Karnataka route projection below."

    )

    missing_benchmark_files = [

        path for path in BENCHMARK_FILES.values() if not path.is_file()

    ]

    if missing_benchmark_files:

        missing_names = ", ".join(

            str(path.relative_to(PROJECT_ROOT)) for path in missing_benchmark_files

        )

        st.info(

            "Saved benchmark results are unavailable. Generate the feature, propagation, "

            "and model outputs with \`experiments\\\run_pipeline.ps1\`. Missing: "

            f"{missing_names}"

        )

    else:

        benchmark_paths = tuple(

            str(BENCHMARK_FILES[key])

            for key in ("propagation", "ml", "features", "summary")

        )

        propagation_metrics, ml_metrics, model_summary = _model_benchmark(

            *benchmark_paths

        )

        propagation_tab, ml_tab = st.tabs(

            ["Propagation and Hata", "Machine learning"]

        )

        with propagation_tab:

            shown_metrics = propagation_metrics.copy()

            shown_metrics[["RMSE (dB)", "MAE (dB)", "Bias (dB)"]] = shown_metrics[

                ["RMSE (dB)", "MAE (dB)", "Bias (dB)"]

            ].round(2)

            shown_metrics["Best-server in domain (%)"] = shown_metrics[

                "Best-server in domain (%)"

            ].round(1)

            st.dataframe(shown_metrics, hide_index=True, width="stretch")

            st.warning(

                "COST-231 Hata and rural Hata are outside their stated applicability "

                "domain for all best-server Salzburg samples (0% in domain). Their "

                "RMSE values are extrapolations, not evidence that Hata works well "

                "or poorly in-domain."

            )

        with ml_tab:

            trial_count = model_summary["tuning_trials_per_model_per_outer_fold"]

            old_baseline = model_summary["physics_residual_baseline"]

            st.warning(

                f"Historical development run: {trial_count} tuning trials per model "

                f"and the old {old_baseline} physics baseline. The current model code "

                "uses a fitted log-distance baseline, so these ML scores are stale "

                "and are not a fair comparison with the propagation table. Rerun the "

                "full pipeline before selecting a model."

            )

            shown_ml_metrics = ml_metrics.round(2)

            st.dataframe(shown_ml_metrics, hide_index=True, width="stretch")

        st.caption(

            "All scores are dB error against measured RSRP. Lower RMSE/MAE is better; "

            "bias is prediction minus measurement. Spatial evaluation uses five "

            "contiguous folds with a 500 m training buffer."

        )

if submitted:

    for key in (

        "route_options",

        "route_samples",

        "signal_profiles",

        "signal_summary",

        "origin",

        "destination",

    ):

        st.session_state.pop(key, None)

    try:

        with st.spinner("Finding routes and profiling terrain…"):

            origin = _geocode(origin_text)

            destination = _geocode(destination_text)

            route_options = _routes(origin, destination)

            samples_by_route = [

                sample_route_coordinates(

                    route["geometry"]["coordinates"],

                    spacing_m=float(sample_spacing),

                )

                for route in route_options

            ]

            signal_profiles, signal_summary = _signal_profiles(

                route_options,

                samples_by_route,

            )

        st.session_state["route_options"] = route_options

        st.session_state["route_samples"] = samples_by_route

        st.session_state["signal_profiles"] = signal_profiles

        st.session_state["signal_summary"] = signal_summary

        st.session_state["origin"] = origin

        st.session_state["destination"] = destination

    except FileNotFoundError as error:

        st.error(str(error))

    except (ValueError, KeyError) as error:

        st.error(f"Could not build route alternatives: {error}")

    except requests.RequestException as error:

        st.error(

            "A route service request failed. Check your internet connection and "

            f"the public service status, then try again. Details: {error}"

        )

routes = st.session_state.get("route_options")

route_samples = st.session_state.get("route_samples")

signal_profiles = st.session_state.get("signal_profiles")

if routes and route_samples and signal_profiles:

    has_terrain = all(

        "terrain_diffraction_loss_db" in profile.columns

        for profile in signal_profiles

    )

    route_rows: list[dict[str, float | int | str]] = []

    for index, (route, samples, profile) in enumerate(

        zip(routes, route_samples, signal_profiles, strict=True),

        start=1,

    ):

        route_length = float(profile["distance_along_route_m"].iloc[-1])

        segment_lengths = np.diff(profile["distance_along_route_m"].to_numpy(dtype=float))

        segment_predictions = profile["predicted_rsrp_dbm"].to_numpy(dtype=float)

        below_threshold_length = float(

            np.sum(segment_lengths * (segment_predictions[:-1] < -110.0))

        )

        row: dict[str, float | int | str] = {

            "Route": index,

            "Distance (km)": float(route["distance"]) / 1000,

            "Travel time (min)": float(route["duration"]) / 60,

            "Sampled points": len(samples),

            "Mean estimated RSRP (dBm)": float(profile["predicted_rsrp_dbm"].mean()),

            "Estimated route below -110 dBm (%)": (

                100 * below_threshold_length / route_length if route_length else 0

            ),

            "Median nearest tower (km)": float(

                profile["nearest_tower_distance_m"].median() / 1000

            ),

        }

        if has_terrain:

            row["Median terrain diffraction loss (dB)"] = float(

                profile["terrain_diffraction_loss_db"].median()

            )

            row["Route samples in possible valley terrain (%)"] = float(

                100

                * profile["terrain_context"]

                .eq("local low ground (possible valley)")

                .mean()

            )

        route_rows.append(row)

    route_table = pd.DataFrame(route_rows)

    st.markdown('<div class="cr-section-title">Your route options</div>', unsafe_allow_html=True)

    st.markdown(

        '<p class="cr-section-note">Choose a route to inspect its signal profile. '

        'Terrain context can be calculated separately.</p>',

        unsafe_allow_html=True,

    )

    with st.expander(f"Compare all {len(routes)} routes", expanded=len(routes) > 1):

        display_columns = [

            "Route",

            "Distance (km)",

            "Travel time (min)",

            "Mean estimated RSRP (dBm)",

        ]

        if has_terrain:

            display_columns.append("Median terrain diffraction loss (dB)")

        display_table = route_table[display_columns].sort_values("Travel time (min)")

        st.dataframe(display_table, hide_index=True, width="stretch")

    selected_route_index = st.selectbox(

        "Route to inspect",

        options=list(range(len(routes))),

        format_func=lambda index: (

            f"Route {index + 1} · {route_rows[index]['Distance (km)']:.1f} km · "

            f"{route_rows[index]['Travel time (min)']:.0f} min"

        ),

        label_visibility="collapsed",

        key="selected_route_index",

    )

    selected_route = routes[selected_route_index]

    selected_samples = route_samples[selected_route_index]

    selected_profile = signal_profiles[selected_route_index]

    summary_row = route_rows[selected_route_index]

    range_fraction = float(

        selected_profile["outside_salzburg_tower_distance_range"].mean()

    )

    mean_rsrp = float(selected_profile["predicted_rsrp_dbm"].mean())

    st.markdown(

        f'<div class="cr-section-title">Route {selected_route_index + 1}</div>'

        f'<p class="cr-section-note">{st.session_state["origin"]["display_name"]} '

        f'<span aria-hidden="true">→</span> {st.session_state["destination"]["display_name"]}</p>',

        unsafe_allow_html=True,

    )

    metric_columns = st.columns(4)

    metric_columns[0].metric("Distance", f"{summary_row['Distance (km)']:.1f} km")

    metric_columns[1].metric("Drive time", f"{summary_row['Travel time (min)']:.0f} min")

    metric_columns[2].metric("Avg. estimated RSRP", f"{mean_rsrp:.1f} dBm")

    metric_columns[3].metric(

        "Median nearest tower",

        f"{summary_row['Median nearest tower (km)']:.1f} km",

    )

    # Count predicted samples in each signal-strength band.
    signal_values = selected_profile["predicted_rsrp_dbm"].astype(float)
    strong_count = int((signal_values >= -80).sum())
    moderate_count = int(((signal_values < -80) & (signal_values >= -100)).sum())
    weak_count = int((signal_values < -100).sum())
    total_count = max(len(signal_values), 1)

    st.markdown("#### Signal quality breakdown")
    quality_columns = st.columns(3)
    quality_columns[0].metric(
        "Strong signal", f"{strong_count} points",
        f"{strong_count / total_count:.0%} of route samples",
    )
    quality_columns[1].metric(
        "Moderate signal", f"{moderate_count} points",
        f"{moderate_count / total_count:.0%} of route samples",
    )
    quality_columns[2].metric(
        "Weak signal", f"{weak_count} points",
        f"{weak_count / total_count:.0%} of route samples",
    )
    st.caption(
        f"Predicted range: {signal_values.min():.1f} to {signal_values.max():.1f} dBm. "
        "A more negative dBm value indicates a weaker signal."
    )

    map_column, profile_column = st.columns([1.2, 0.8], gap="large")

    center = [

        float(selected_profile["latitude"].mean()),

        float(selected_profile["longitude"].mean()),

    ]

    route_map = folium.Map(location=center, zoom_start=8, control_scale=True)

    for route_index, (route, profile) in enumerate(

        zip(routes, signal_profiles, strict=True)

    ):

        if route_index == selected_route_index:

            for point_index in range(len(profile) - 1):

                first = profile.iloc[point_index]

                second = profile.iloc[point_index + 1]

                mean_signal = (

                    float(first["predicted_rsrp_dbm"])

                    + float(second["predicted_rsrp_dbm"])

                ) / 2

                folium.PolyLine(

                    [

                        (float(first["latitude"]), float(first["longitude"])),

                        (float(second["latitude"]), float(second["longitude"])),

                    ],

                    color=_route_color(mean_signal),

                    weight=7,

                    opacity=0.9,

                    tooltip=(

                        f"Estimated RSRP {mean_signal:.1f} dBm"

                        + (

                            f" · terrain diffraction "

                            f"{first['terrain_diffraction_loss_db']:.1f} dB"

                            if has_terrain

                            else ""

                        )

                    ),

                ).add_to(route_map)

            for point_index in range(0, len(profile), max(1, len(profile) // 80)):

                record = profile.iloc[point_index]

                folium.CircleMarker(

                    location=(record["latitude"], record["longitude"]),

                    radius=4,

                    color=_route_color(float(record["predicted_rsrp_dbm"])),

                    fill=True,

                    fill_opacity=0.85,

                    tooltip=(

                        f"Estimated RSRP {record['predicted_rsrp_dbm']:.1f} dBm"

                        + (

                            f" · elevation {record['receiver_elevation_m']:.0f} m · "

                            f"{record['terrain_context']}"

                            if has_terrain

                            else ""

                        )

                    ),

                ).add_to(route_map)

        else:

            alternate_coordinates = [

                (float(latitude), float(longitude))

                for longitude, latitude in route["geometry"]["coordinates"]

            ]

            folium.PolyLine(

                alternate_coordinates,

                color="#9aa8bb",

                weight=3,

                opacity=0.55,

                tooltip=f"Alternative route {route_index + 1}",

            ).add_to(route_map)

    start_point = selected_profile.iloc[0]

    end_point = selected_profile.iloc[-1]

    folium.Marker(

        [start_point["latitude"], start_point["longitude"]],

        tooltip="Start",

        icon=folium.Icon(color="green", icon="play"),

    ).add_to(route_map)

    folium.Marker(

        [end_point["latitude"], end_point["longitude"]],

        tooltip="Destination",

        icon=folium.Icon(color="red", icon="flag"),

    ).add_to(route_map)

    route_map.fit_bounds(

        [

            [

                float(selected_profile["latitude"].min()),

                float(selected_profile["longitude"].min()),

            ],

            [

                float(selected_profile["latitude"].max()),

                float(selected_profile["longitude"].max()),

            ],

        ],

        padding=(24, 24),

    )

    with map_column:

        with st.container(border=True):

            st.markdown("**Signal along the route**")

            st.markdown(
                """
                <div class="cr-signal-legend">
                  <span class="cr-legend-item"><i class="cr-swatch" style="background:#16a34a"></i>Strong ≥ −80 dBm</span>
                  <span class="cr-legend-item"><i class="cr-swatch" style="background:#f59e0b"></i>Moderate −100 to −80 dBm</span>
                  <span class="cr-legend-item"><i class="cr-swatch" style="background:#ef4444"></i>Weak &lt; −100 dBm</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.iframe(

                route_map.get_root().render(),

                width="stretch",

                height=540,

                tab_index=0,

            )

    with profile_column:

        with st.container(border=True):

            st.markdown("**Route profile**")

            signal_tab, terrain_tab = st.tabs(["Signal", "Terrain"])

            with signal_tab:

                st.line_chart(

                    selected_profile,

                    x="distance_along_route_m",

                    y=[

                        "predicted_rsrp_dbm",

                        "estimated_lower_90_dbm",

                        "estimated_upper_90_dbm",

                    ],

                    x_label="Distance along route (m)",

                    y_label="Estimated RSRP (dBm)",

                    height=290,

                    width="stretch",

                )

                st.caption(

                    "Bounds are empirical Salzburg residual bands, not local "

                    "confidence intervals. "

                    f"{range_fraction:.0%} of points are outside the Salzburg "

                    "tower-distance range."

                )

            with terrain_tab:

                if has_terrain:

                    low_ground_fraction = float(

                        selected_profile["terrain_context"]

                        .eq("local low ground (possible valley)")

                        .mean()

                    )

                    st.line_chart(

                        selected_profile,

                        x="distance_along_route_m",

                        y="receiver_elevation_m",

                        x_label="Distance along route (m)",

                        y_label="Surface elevation (m)",

                        height=220,

                        width="stretch",

                    )

                    st.line_chart(

                        selected_profile,

                        x="distance_along_route_m",

                        y="terrain_diffraction_loss_db",

                        x_label="Distance along route (m)",

                        y_label="Terrain diffraction (dB)",

                        height=220,

                        width="stretch",

                    )

                    st.caption(

                        f"{low_ground_fraction:.0%} of samples are at least 10 m below "

                        "their 500 m local terrain mean (possible low ground, not a "

                        "validated valley classification). Terrain does not adjust RSRP."

                    )

                else:

                    st.caption(

                        "Terrain is optional and may take longer on first use while "

                        "Copernicus elevation data is prepared."

                    )

                    if st.button(

                        "Calculate terrain context",

                        key=f"calculate_terrain_{selected_route_index}",

                        icon=":material/landscape:",

                    ):

                        try:

                            with st.spinner("Preparing terrain profiles…"):

                                terrain_profiles, terrain_summary = _terrain_profiles(

                                    signal_profiles

                                )

                            st.session_state["signal_profiles"] = terrain_profiles

                            st.session_state["signal_summary"].update(terrain_summary)

                            st.rerun()

                        except (OSError, ValueError) as error:

                            st.error(f"Could not calculate terrain context: {error}")

    st.info(

        "Research preview: the RSRP estimate uses a Salzburg-fitted distance model "

        "with 2021 India LTE tower records. Terrain is displayed as context only; "

        "adding these features did not improve buffered Salzburg validation. "

        "This is not validated coverage for Karnataka."

    )

    with st.expander("How this estimate is produced"):

        st.write(

            "The route is sampled at the selected spacing. Each sample is matched "

            "to its nearest eligible local LTE record; a distance-only regression "

            "fitted to Salzburg measurements estimates RSRP. Elevation, terrain "

            "diffraction, roughness, and a local low-ground indicator are calculated "

            "from Copernicus GLO-30, but are not used to modify signal. OpenStreetMap "

            "and OSRM provide geocoding and driving routes."

        )

        st.caption(

            f"Using {st.session_state['signal_summary']['eligible_india_towers']:,} "

            "eligible LTE tower records. "

            + (

                "Terrain source: Copernicus DEM GLO-30 Public (DSM)."

                if has_terrain

                else "Terrain profile is not loaded."

            )

        )

    st.download_button(

        "Download route alternatives (GeoJSON)",

        data=json.dumps(

            {

                "type": "FeatureCollection",

                "features": [

                    {

                        "type": "Feature",

                        "properties": {

                            "route_index": index,

                            "distance_m": route["distance"],

                            "duration_s": route["duration"],

                            "mean_estimated_rsrp_dbm": float(

                                signal_profiles[index - 1]["predicted_rsrp_dbm"].mean()

                            ),

                            "median_terrain_diffraction_loss_db": (

                                float(

                                    signal_profiles[index - 1][

                                        "terrain_diffraction_loss_db"

                                    ].median()

                                )

                                if has_terrain

                                else None

                            ),

                        },

                        "geometry": route["geometry"],

                    }

                    for index, route in enumerate(routes, start=1)

                ],

            },

            indent=2,

        ),

        file_name="connectroute_alternatives.geojson",

        mime="application/geo+json",

    )
