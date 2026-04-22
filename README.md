# Stunting Risk Heatmap Dashboard

Tier 1 AIMS KTT Fellowship hackathon solution for a simple, explainable stunting-risk dashboard and printable A4 briefing pages for local leaders.

## What this repo contains
- `risk_scorer.py` — per-household risk scoring with a clear `score(household)` function.
- `dashboard.py` — Streamlit dashboard with district filter and risk-threshold slider.
- `generate_printables.py` — creates the printable A4 PDF pages for chiefs / Umudugudu leaders.
- `data/` — synthetic generator-ready data files or your provided challenge files.
- `printable/` — generated PDF output.

## Run in 2 commands
```bash
pip install -r requirements.txt
python generate_printables.py && streamlit run dashboard.py
```

## Notes
- If `data/households.csv` and `data/districts.geojson` exist, the app uses them.
- If they are missing, the repo falls back to synthetic generation.
- The printable PDF anonymises all households by ID only.

## 4-minute video checklist
- Camera on for intro and outro.
- Live code walk through `risk_scorer.py::score`.
- Live terminal output from `streamlit run dashboard.py`.
- Open the printable PDF and explain it as if briefing a village chief.
- Answer the three required questions aloud.
