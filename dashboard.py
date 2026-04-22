from __future__ import annotations

import ast
import json
from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.components.v1 import html

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "data"

st.set_page_config(page_title="Stunting Risk Heatmap Dashboard", layout="wide")

@st.cache_data
def load_inputs():
    scored = pd.read_csv(ARTIFACT_DIR / "scored_households.csv")
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
    geojson = json.loads((DATA_DIR / "districts.geojson").read_text())
    return scored, metrics, geojson


def parse_drivers(value):
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
        return ", ".join(parsed)
    except Exception:
        return str(value)


scored, metrics, district_geojson = load_inputs()

st.title("Stunting Risk Heatmap Dashboard")
st.caption("Tier 1 baseline: household scoring, district map, sector tables, printable A4 pages")

left, right = st.columns([1, 1])
with left:
    district_choice = st.selectbox("District", ["All"] + sorted(scored["district"].unique().tolist()))
with right:
    threshold = st.slider("Risk threshold", 0.0, 1.0, float(metrics["threshold"]), 0.01)

filtered = scored.copy()
if district_choice != "All":
    filtered = filtered[filtered["district"] == district_choice]
filtered = filtered[filtered["risk_score"] >= threshold]

summary = (
    filtered.groupby(["district", "sector"], as_index=False)
    .agg(high_risk_households=("household_id", "count"), mean_risk=("risk_score", "mean"))
    .sort_values(["district", "mean_risk"], ascending=[True, False])
)

district_summary = (
    filtered.groupby("district", as_index=False)
    .agg(high_risk_households=("household_id", "count"), mean_risk=("risk_score", "mean"))
)
if district_summary.empty:
    district_summary = (
        scored.groupby("district", as_index=False)
        .agg(high_risk_households=("household_id", "count"), mean_risk=("risk_score", "mean"))
    )

c1, c2, c3 = st.columns(3)
c1.metric("Threshold", f"{threshold:.2f}")
c2.metric("CV ROC AUC", f"{metrics['cv_roc_auc']:.3f}")
c3.metric("CV F1", f"{metrics['cv_f1']:.3f}")

fig = px.choropleth_mapbox(
    district_summary,
    geojson=district_geojson,
    featureidkey="properties.district",
    locations="district",
    color="mean_risk",
    hover_name="district",
    hover_data={"high_risk_households": True, "mean_risk": ':.3f'},
    mapbox_style="carto-positron",
    center={"lat": -1.94, "lon": 30.06},
    zoom=9,
    opacity=0.65,
)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.subheader("District choropleth")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Sector summary")
st.dataframe(summary, use_container_width=True, hide_index=True)

st.subheader("Top high-risk households")
detail = filtered.sort_values("risk_score", ascending=False).head(20).copy()
if not detail.empty:
    detail["top_drivers"] = detail["top_drivers"].map(parse_drivers)
    st.dataframe(
        detail[["anon_household_id", "district", "sector", "risk_score", "top_drivers"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No households match the selected threshold.")

st.subheader("Printable pages")
pdf_files = sorted((ROOT / "printable").glob("*.pdf"))
if pdf_files:
    st.write("Generated sector PDFs:")
    for pdf in pdf_files[:10]:
        st.write(f"- {pdf.name}")
