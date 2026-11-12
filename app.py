
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(
    page_title="Shipment ETA Prediction",
    page_icon="🚢",
    layout="wide",
)

@st.cache_data
def load_data():
    shipments = pd.read_csv(DATA_DIR / "shipments.csv")
    journeys = pd.read_csv(DATA_DIR / "journeys.csv")
    predictions = pd.read_csv(DATA_DIR / "oof_predictions.csv")

    return shipments, journeys, predictions


shipments, journeys, predictions = load_data()

st.title("🚢 Shipment ETA & Delay Prediction")
st.caption("Synthetic shipment data | Stage-based delay prediction")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    [
        "Overview",
        "Shipment Explorer",
        "Model Evaluation",
        "Data Explorer",
    ],
)

# Shared calculations
total_shipments = len(shipments)
late_shipments = int((shipments["final_delay_h"] > 0).sum())
on_time_shipments = total_shipments - late_shipments
average_delay = shipments["final_delay_h"].mean()

if page == "Overview":
    st.header("Project Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Shipments", total_shipments)
    c2.metric("Late Shipments", late_shipments)
    c3.metric("On-time / Early", on_time_shipments)
    c4.metric("Average Delay", f"{average_delay:.2f} h")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Shipment Status")

        status = pd.DataFrame({
            "Status": ["Late", "On-time / Early"],
            "Shipments": [late_shipments, on_time_shipments],
        }).set_index("Status")

        st.bar_chart(status)

    with right:
        st.subheader("Delay Distribution")

        st.histogram if False else None
        st.bar_chart(
            shipments["final_delay_h"]
            .round()
            .value_counts()
            .sort_index()
            .rename_axis("Delay (hours)")
            .rename("Shipments")
        )

    st.subheader("Shipments by Route")

    route_counts = (
        shipments.groupby("route_id")
        .size()
        .sort_values(ascending=False)
        .rename("Shipments")
    )

    st.bar_chart(route_counts)

    st.subheader("Recent Shipment Records")
    st.dataframe(shipments.head(10), use_container_width=True)


elif page == "Shipment Explorer":
    st.header("Shipment Explorer")

    shipment_ids = sorted(shipments["shipment_id"].unique())

    selected_id = st.selectbox(
        "Select Shipment",
        shipment_ids,
    )

    shipment = shipments[
        shipments["shipment_id"] == selected_id
    ].iloc[0]

    prediction = predictions[
        predictions["shipment_id"] == selected_id
    ]

    if prediction.empty:
        st.warning("No out-of-fold prediction found for this shipment.")
    else:
        prediction = prediction.iloc[0]

        actual = float(prediction["actual_delay_h"])
        structured = float(prediction["structured_h"])
        flat = float(prediction["flat_h"])

        st.subheader(f"Shipment: {selected_id}")

        c1, c2, c3 = st.columns(3)

        c1.metric("Actual Delay", f"{actual:.2f} h")
        c2.metric(
            "Structured Prediction",
            f"{structured:.2f} h",
            delta=f"Error: {abs(actual - structured):.2f} h",
            delta_color="inverse",
        )
        c3.metric(
            "Flat Ridge Prediction",
            f"{flat:.2f} h",
            delta=f"Error: {abs(actual - flat):.2f} h",
            delta_color="inverse",
        )

        st.divider()

        st.subheader("Shipment Information")

        details = {
            "Route": shipment["route_id"],
            "Origin": shipment["origin"],
            "Destination": shipment["destination"],
            "Scheduled Arrival": shipment["scheduled_arrival_utc"],
            "Actual Arrival": shipment["actual_arrival_utc"],
            "Customs Observed": shipment["customs_observed"],
        }

        st.table(
            pd.DataFrame(
                details.items(),
                columns=["Field", "Value"],
            )
        )

        st.subheader("Prediction Comparison")

        comparison = pd.DataFrame({
            "Prediction": [
                "Actual",
                "Training Mean",
                "Flat Ridge",
                "Structured Propagation",
            ],
            "Delay (hours)": [
                actual,
                prediction["mean_h"],
                flat,
                structured,
            ],
        }).set_index("Prediction")

        st.bar_chart(comparison)

        st.subheader("Journey Stages")

        shipment_journeys = journeys[
            journeys["shipment_id"] == selected_id
        ].sort_values("stage_order")

        display_columns = [
            "stage",
            "planned_duration_h",
            "scheduled_duration_h",
            "congestion_index",
            "actual_duration_h",
            "is_observed",
            "carried_in_h",
            "carried_out_h",
        ]

        st.dataframe(
            shipment_journeys[display_columns],
            use_container_width=True,
        )


elif page == "Model Evaluation":
    st.header("Model Evaluation")
    st.write("All metrics below are calculated from the saved out-of-fold predictions.")

    model_columns = {
        "Zero-delay Baseline": "schedule_h",
        "Training Mean": "mean_h",
        "Flat Ridge": "flat_h",
        "Structured Propagation": "structured_h",
    }

    results = []

    for model_name, column in model_columns.items():
        errors = predictions[column] - predictions["actual_delay_h"]

        results.append({
            "Model": model_name,
            "MAE (hours)": errors.abs().mean(),
            "Bias (hours)": errors.mean(),
        })

    metrics = pd.DataFrame(results)

    st.subheader("Model Performance")
    st.dataframe(
        metrics.round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("MAE Comparison")

    chart = metrics.set_index("Model")[["MAE (hours)"]]
    st.bar_chart(chart)

    st.subheader("Prediction vs Actual")

    selected_model = st.selectbox(
        "Select model",
        list(model_columns.keys()),
        index=3,
    )

    selected_column = model_columns[selected_model]

    plot_data = predictions[
        ["actual_delay_h", selected_column]
    ].rename(columns={
        "actual_delay_h": "Actual",
        selected_column: "Predicted",
    })

    st.scatter_chart(
        plot_data,
        x="Actual",
        y="Predicted",
    )

    st.subheader("Fold-wise Performance")

    fold_results = []

    for fold, group in predictions.groupby("fold"):
        errors = (
            group["structured_h"] - group["actual_delay_h"]
        )

        fold_results.append({
            "Fold": fold,
            "Shipments": len(group),
            "MAE (hours)": errors.abs().mean(),
        })

    st.dataframe(
        pd.DataFrame(fold_results).round(2),
        use_container_width=True,
        hide_index=True,
    )


elif page == "Data Explorer":
    st.header("Data Explorer")

    dataset = st.selectbox(
        "Choose dataset",
        ["Shipments", "Journeys", "OOF Predictions"],
    )

    datasets = {
        "Shipments": shipments,
        "Journeys": journeys,
        "OOF Predictions": predictions,
    }

    selected_data = datasets[dataset]

    st.write(f"Records: {len(selected_data)}")
    st.write(f"Columns: {len(selected_data.columns)}")

    search = st.text_input(
        "Search records by shipment ID"
    )

    if search:
        selected_data = selected_data[
            selected_data["shipment_id"]
            .astype(str)
            .str.contains(search, case=False, na=False)
        ]

    st.dataframe(
        selected_data,
        use_container_width=True,
        height=500,
    )

    st.download_button(
        "Download displayed CSV",
        data=selected_data.to_csv(index=False),
        file_name=f"{dataset.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
    