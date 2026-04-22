# aims-ktt-hackathon

## Stunting Risk Heatmap Dashboard for Community Health Decision-Making

### Overview
This project builds a simple data-driven tool to identify households at high risk of childhood stunting in Rwanda. It supports local leaders (Umudugudu chiefs, Abunzi) by providing both an interactive dashboard and printable reports for low-tech environments.

The system computes a household-level risk score using key indicators such as nutrition, sanitation, water access, income, and number of children under five. These scores are then aggregated to produce a sector-level heatmap for decision-making.

---

### Features
- Household-level **stunting risk scoring**
- Sector-level **choropleth heatmap**
- Interactive dashboard with **filters (district, risk threshold)**
- Printable **A4 reports** for community leaders
- Designed for **low-bandwidth and offline use**

---

### Data

The challenge brief listed synthetic input files as provided materials, but they were not available in my packet at build time, so I recreated them from the published generator specification in the brief.

The dataset includes:
- `households.csv`: 2,500 households with socio-economic and geographic data
- `gold_stunting_flag.csv`: 300 labeled households
- `districts.geojson`: simplified district boundaries

---

## Project Structure
