# S2.T1.2 - Stunting Risk Heatmap Dashboard

Tier 1 baseline submission for the AIMS KTT Fellowship hackathon.

## What this does
- Computes a per-household stunting risk score from synthetic NISR-style household data.
- Aggregates risk by district and sector.
- Ships a Streamlit dashboard with a district filter and risk-threshold slider.
- Exports printable A4 sector pages with top-10 anonymised high-risk households and their top drivers.

## Important note on the data
The challenge brief listed synthetic input files as provided materials, but they were not available in my packet at build time, so I recreated them from the published generator specification in the brief.

## Repo structure
- `risk_scorer.py` - model training, scoring, calibration, top-driver explanations.
- `dashboard.py` - Streamlit dashboard.
- `prepare_submission.py` - generates scored outputs and printable PDFs.
- `generate_synthetic_data.py` - regenerates the synthetic source data.
- `data/` - households CSV, gold labels CSV, district GeoJSON.
- `printable/` - generated PDF pages.
- `artifacts/` - scored household file, metrics, manifest.

## Run in 2 commands
```bash
pip install -r requirements.txt
python prepare_submission.py && streamlit run dashboard.py
```

## Expected outputs
- `artifacts/scored_households.csv`
- `artifacts/metrics.json`
- `printable/*.pdf`

## Model choice
I used a logistic regression baseline because the challenge is Tier 1, CPU-only, and the labelled set is small (300 households). The score is explainable enough for a live defense and easy to calibrate with a threshold.

## Product adaptation
The printable A4 pages are designed for low-connectivity local workflows:
- The district data officer prints one page per sector every month.
- The village chief receives only anonymised IDs and top drivers, not names.
- The chief annotates home visits, nutrition kit needs, WASH gaps, and referrals by pen.
- Annotated sheets return to the district review meeting, where severe cases are escalated through the health system.
- Bilingual headers (Kinyarwanda / English) and symbols can support mixed literacy settings.

## Video URL
https://youtu.be/isJoP1kusDo


## Known limitations
- The district polygons are simplified synthetic geometry.
- The labelled set is small and synthetic, so the model is intended as a triage baseline rather than a clinical decision tool.
- A production deployment would require validated local baselines and field-tested escalation rules.
