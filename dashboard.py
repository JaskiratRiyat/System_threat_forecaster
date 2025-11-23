import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


@st.cache_data(show_spinner=True)
def load_data(path: str, sample_frac: float = 1.0, random_state: int = 42) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)
    return df


@st.cache_resource(show_spinner=True)
def train_simple_model(df: pd.DataFrame):
    feature_cols = [
        "IsSecureBootEnabled",
        "IsGamer",
        "ProcessorCoreCount",
        "TotalPhysicalRAMMB",
        "PrimaryDiskCapacityMB",
        "RegionIdentifier",
    ]

    # Keep only rows where all required columns + target are present
    cols = feature_cols + ["target"]
    df_model = df[cols].dropna()

    X = df_model[feature_cols]
    y = df_model["target"]

    # Simple train/validation split just for sanity (we won't show metrics here)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000)),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe, feature_cols


def main():
    st.set_page_config(
        page_title="System Threat Forecaster Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("System Threat Forecaster ")
    st.markdown(
        """A visual and interactive dashboard built on top of the **System Threat Forecaster** Kaggle dataset.

This Dasboard is mainly used for:
- Introducing the dataset and target variable.
- Showing key patterns in system configuration and threat labels.
- Demonstrating a simple machine learning–based risk prediction.
        """
    )

    # Sidebar controls
    st.sidebar.header("Configuration")
    sample_frac = st.sidebar.slider(
        "Sample fraction for faster loading",
        min_value=0.1,
        max_value=1.0,
        value=0.4,
        step=0.1,
        help="Use a subset of the data to keep the dashboard responsive.",
    )

    st.sidebar.info(
        "This dashboard reads `train.csv` from the project folder. Make sure it is in the same directory as `dashboard.py`."
    )

    with st.spinner("Loading data..."):
        df = load_data("train.csv", sample_frac=sample_frac)

    if "target" not in df.columns:
        st.error("Column 'target' not found in train.csv. Please ensure you are using the competition's train file.")
        return

    # Basic overview
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows (sample)", f"{len(df):,}")
    with col2:
        st.metric("Columns", f"{df.shape[1]}")
    with col3:
        positive_rate = df["target"].mean()
        st.metric("Threat rate (target=1)", f"{positive_rate*100:.2f}%")
    with col4:
        st.metric("Unique Countries", df["CountryID"].nunique() if "CountryID" in df.columns else "N/A")

    st.markdown("---")

    tab_overview, tab_eda, tab_risk = st.tabs([
        "Target Distribution",
        "EDA Highlights",
        "Interactive Risk Demo",
    ])

    # Tab 1: Target distribution
    with tab_overview:
        st.subheader("Target Distribution")
        target_counts = df["target"].value_counts().rename(index={0: "No Threat", 1: "Threat"})
        fig_target = px.bar(
            target_counts,
            x=target_counts.index,
            y=target_counts.values,
            color=target_counts.index,
            text=target_counts.values,
            labels={"x": "Class", "y": "Count"},
            title="Threat vs No Threat (sample)",
            color_discrete_sequence=["#2E86DE", "#E74C3C"],
        )
        fig_target.update_traces(textposition="outside")
        st.plotly_chart(fig_target, use_container_width=True)

        st.write(
            """The dataset is approximately balanced between machines **with** and **without** detected threats.
This makes metrics like Accuracy and ROC AUC meaningful for evaluation."""
        )

    # Tab 2: EDA highlights
    with tab_eda:
        st.subheader("EDA Highlights")
        st.markdown(
            """Below are a few high-level patterns connecting system configuration and threat labels."""
        )

        eda_col1, eda_col2 = st.columns(2)

        # Secure Boot vs threat
        if "IsSecureBootEnabled" in df.columns:
            with eda_col1:
                boot_stats = (
                    df.groupby("IsSecureBootEnabled")["target"]
                    .mean()
                    .reset_index()
                    .replace({"IsSecureBootEnabled": {0: "Disabled", 1: "Enabled"}})
                )
                fig_boot = px.bar(
                    boot_stats,
                    x="IsSecureBootEnabled",
                    y="target",
                    color="IsSecureBootEnabled",
                    labels={"IsSecureBootEnabled": "Secure Boot", "target": "Threat rate"},
                    title="Secure Boot vs Threat Rate",
                    color_discrete_sequence=["#E74C3C", "#27AE60"],
                )
                fig_boot.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig_boot, use_container_width=True)
        else:
            with eda_col1:
                st.info("Column 'IsSecureBootEnabled' not available in this sample.")

        # IsGamer vs threat
        if "IsGamer" in df.columns:
            with eda_col2:
                gamer_stats = (
                    df.groupby("IsGamer")["target"]
                    .mean()
                    .reset_index()
                    .replace({"IsGamer": {0: "Non-gamer", 1: "Gamer"}})
                )
                fig_gamer = px.bar(
                    gamer_stats,
                    x="IsGamer",
                    y="target",
                    color="IsGamer",
                    labels={"IsGamer": "Profile", "target": "Threat rate"},
                    title="Gamer Profile vs Threat Rate",
                    color_discrete_sequence=["#2980B9", "#8E44AD"],
                )
                fig_gamer.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig_gamer, use_container_width=True)
        else:
            with eda_col2:
                st.info("Column 'IsGamer' not available in this sample.")

        st.markdown("---")

        if "RegionIdentifier" in df.columns:
            top_regions = (
                df.groupby("RegionIdentifier")["target"].mean().sort_values(ascending=False).head(10)
            )
            fig_region = px.bar(
                top_regions,
                x=top_regions.index.astype(str),
                y=top_regions.values,
                labels={"x": "RegionIdentifier", "y": "Threat rate"},
                title="Top 10 Regions by Threat Rate (sample)",
                color=top_regions.values,
                color_continuous_scale="Reds",
            )
            fig_region.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_region, use_container_width=True)
        else:
            st.info("Column 'RegionIdentifier' not available in this sample.")

    # Tab 3: Simple interactive risk demo
    with tab_risk:
        st.subheader("Interactive Threat Risk Demo")
        st.markdown(
            """This is a **simple Logistic Regression model** trained on a few key features
from the training data. It is not the same as your full Kaggle model but serves as
an intuitive demo during your presentation."""
        )

        try:
            with st.spinner("Training simple model (only first time)..."):
                model, feature_cols = train_simple_model(df)
        except Exception as e:
            st.error(f"Could not train demo model: {e}")
            return

        # Default values are based on medians / common values from the dataset
        demo_col1, demo_col2, demo_col3 = st.columns(3)

        with demo_col1:
            is_secure_boot = st.selectbox("Secure Boot Enabled", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            is_gamer = st.selectbox("Is Gamer", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with demo_col2:
            proc_cores = int(df["ProcessorCoreCount"].median()) if "ProcessorCoreCount" in df.columns else 4
            processor_core_count = st.slider(
                "Processor Core Count",
                min_value=1,
                max_value=32,
                value=proc_cores,
            )
            ram_med = int(df["TotalPhysicalRAMMB"].median()) if "TotalPhysicalRAMMB" in df.columns else 8000
            total_ram = st.slider(
                "Total Physical RAM (MB)",
                min_value=1024,
                max_value=65536,
                value=ram_med,
                step=512,
            )
        with demo_col3:
            disk_med = int(df["PrimaryDiskCapacityMB"].median()) if "PrimaryDiskCapacityMB" in df.columns else 256000
            disk_cap = st.slider(
                "Primary Disk Capacity (MB)",
                min_value=20000,
                max_value=2048000,
                value=disk_med,
                step=20000,
            )
            region_med = int(df["RegionIdentifier"].median()) if "RegionIdentifier" in df.columns else 7
            region_id = st.slider(
                "Region Identifier",
                min_value=int(df["RegionIdentifier"].min()) if "RegionIdentifier" in df.columns else 1,
                max_value=int(df["RegionIdentifier"].max()) if "RegionIdentifier" in df.columns else 15,
                value=region_med,
            )

        # Build input for the model
        input_array = np.array(
            [
                [
                    is_secure_boot,
                    is_gamer,
                    processor_core_count,
                    total_ram,
                    disk_cap,
                    region_id,
                ]
            ]
        )

        proba = model.predict_proba(input_array)[0, 1]

        st.markdown("---")
        risk_col1, risk_col2 = st.columns([2, 1])
        with risk_col1:
            st.metric("Predicted Threat Probability", f"{proba*100:.2f}%")
        with risk_col2:
            if proba < 0.3:
                risk_level = "Low"
                color = "#27AE60"
            elif proba < 0.7:
                risk_level = "Medium"
                color = "#F1C40F"
            else:
                risk_level = "High"
                color = "#E74C3C"

            st.markdown(
                f"<div style='padding: 0.75rem; border-radius: 0.5rem; background-color: {color}; color: white; text-align: center; font-size: 1.2rem;'>Risk Level: <b>{risk_level}</b></div>",
                unsafe_allow_html=True,
            )

        st.caption(
            "This demo model is intentionally simple and trained only on a subset of features. I have specifically kept it simple to show the future scope and deployability of the model of our application for the report "
            "Credit for the app goes to Jaskirat, Sugam, Sarthak and Tarun."
        )


if __name__ == "__main__":
    main()
