from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from data_generator import generate_geojson, generate_households
from risk_scorer import explain_row, score_dataframe, train_logistic_model

st.set_page_config(page_title="Stunting Risk Heatmap Dashboard", layout="wide")

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
HOUSEHOLDS = DATA_DIR / "households.csv"
GEOJSON_FILE = DATA_DIR / "districts.geojson"
GOLD = DATA_DIR / "gold_stunting_flag.csv"


@st.cache_data
 def load_data():
    if HOUSEHOLDS.exists():
        df = pd.read_csv(HOUSEHOLDS)
    else:
        df = generate_households()
    if GOLD.exists():
        gold = pd.read_csv(GOLD)
        df = df.merge(gold, on="household_id", how="left", suffixes=("", "_gold"))
        if "gold_stunting_flag" in df.columns:
            df["stunting_flag"] = df["gold_stunting_flag"].fillna(df.get("stunting_flag", 0)).astype(int)
    if GEOJSON_FILE.exists():
        geo = json.loads(GEOJSON_FILE.read_text())
    else:
        geo = generate_geojson()
    scored = score_dataframe(df)
    return scored, geo


def main():
    st.title("Stunting Risk Heatmap Dashboard")
    st.caption("Tier 1 AIMS KTT challenge: household risk scoring, district choropleth, and printable A4 brief.")

    df, geo = load_data()
    if "stunting_flag" in df.columns:
        _, metrics = train_logistic_model(df.dropna(subset=["stunting_flag"]))
    else:
        metrics = {}

    col1, col2, col3 = st.columns(3)
    with col1:
        districts = ["All"] + sorted(df["district"].dropna().unique().tolist())
        district = st.selectbox("District", districts)
    with col2:
        threshold = st.slider("Risk threshold", 0.1, 0.9, 0.5, 0.05)
    with col3:
        st.metric("Households", f"{len(df):,}")

    filtered = df.copy()
    if district != "All":
        filtered = filtered[filtered["district"] == district]

    filtered = filtered.copy()
    filtered["high_risk"] = (filtered["risk_score"] >= threshold).astype(int)

    st.subheader("Sector-level heatmap")
    agg = filtered.groupby(["district", "sector"], as_index=False).agg(
        risk_score=("risk_score", "mean"),
        households=("household_id", "count"),
        high_risk=("high_risk", "sum"),
    )

    fig = px.choropleth(
        agg,
        geojson=geo,
        locations="district",
        featureidkey="properties.district",
        color="risk_score",
        color_continuous_scale="Reds",
        scope="africa",
        hover_data={"sector": True, "households": True, "high_risk": True, "risk_score": ":.2f"},
    )
    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top high-risk households")
    display_cols = ["household_id", "district", "sector", "risk_score", "avg_meal_count", "water_source", "sanitation_tier", "income_band", "children_under5"]
    st.dataframe(filtered.sort_values("risk_score", ascending=False)[display_cols].head(20), use_container_width=True)

    st.subheader("One household explanation")
    chosen = st.selectbox("Choose household", filtered.sort_values("risk_score", ascending=False).head(50)["household_id"].tolist())
    row = filtered[filtered["household_id"] == chosen].iloc[0]
    st.write({"household_id": chosen, "risk_score": round(float(row["risk_score"]), 3), "top_drivers": explain_row(row)})

    if metrics:
        st.subheader("Model metrics on labelled rows")
        st.write(metrics)


if __name__ == "__main__":
    main()
