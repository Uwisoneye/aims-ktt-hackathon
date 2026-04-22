import json
from pathlib import Path
import hashlib
import math
import random

import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

OUTDIR = Path('/mnt/data/stunting_synth_data')
OUTDIR.mkdir(parents=True, exist_ok=True)

# Five simplified districts with rough centroids in Rwanda.
DISTRICTS = {
    'Nyarugenge': {'lat': -1.9500, 'lon': 30.0600, 'urban_prob': 0.90},
    'Gasabo':      {'lat': -1.9000, 'lon': 30.1200, 'urban_prob': 0.70},
    'Kicukiro':    {'lat': -1.9700, 'lon': 30.1000, 'urban_prob': 0.75},
    'Musanze':     {'lat': -1.5000, 'lon': 29.6300, 'urban_prob': 0.35},
    'Nyagatare':   {'lat': -1.3000, 'lon': 30.3300, 'urban_prob': 0.25},
}

SECTORS = {
    'Nyarugenge': ['Kimisagara', 'Nyamirambo', 'Mageragere', 'Muhima'],
    'Gasabo': ['Kacyiru', 'Remera', 'Kimironko', 'Bumbogo'],
    'Kicukiro': ['Gikondo', 'Kanombe', 'Kagarama', 'Nyarugunga'],
    'Musanze': ['Muhoza', 'Cyuve', 'Kinigi', 'Shingiro'],
    'Nyagatare': ['Rwimiyaga', 'Karangazi', 'Mimuri', 'Matimba'],
}

WATER_SOURCES = ['piped', 'public_tap', 'protected_well', 'surface_water']
SANITATION_TIERS = [1, 2, 3]  # 1 poor, 3 good
INCOME_BANDS = ['low', 'lower_middle', 'middle', 'upper']

# Household counts sum to 2500.
DISTRICT_COUNTS = {
    'Nyarugenge': 550,
    'Gasabo': 600,
    'Kicukiro': 500,
    'Musanze': 400,
    'Nyagatare': 450,
}
assert sum(DISTRICT_COUNTS.values()) == 2500


def stable_anon_id(raw_id: str) -> str:
    digest = hashlib.sha1(raw_id.encode('utf-8')).hexdigest()[:10].upper()
    return f'HH-{digest}'


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def sample_household(district: str, idx: int) -> dict:
    meta = DISTRICTS[district]
    sector = random.choice(SECTORS[district])
    urban = np.random.rand() < meta['urban_prob']

    # Spatial spread: urban points are tighter around the centroid.
    spread = 0.02 if urban else 0.05
    lat = np.random.normal(meta['lat'], spread)
    lon = np.random.normal(meta['lon'], spread)

    # Socioeconomic generation tuned so city districts have somewhat better conditions.
    if urban:
        income_probs = [0.18, 0.34, 0.30, 0.18]
        water_probs = [0.48, 0.30, 0.17, 0.05]
        sanitation_probs = [0.10, 0.40, 0.50]
        meal_mean = 2.55
    else:
        income_probs = [0.40, 0.33, 0.20, 0.07]
        water_probs = [0.14, 0.22, 0.34, 0.30]
        sanitation_probs = [0.35, 0.42, 0.23]
        meal_mean = 2.05

    # District-specific pressure adjustments to create visible variation.
    district_meal_adj = {
        'Nyarugenge': 0.10,
        'Gasabo': 0.05,
        'Kicukiro': 0.02,
        'Musanze': -0.05,
        'Nyagatare': -0.12,
    }[district]

    income_band = np.random.choice(INCOME_BANDS, p=income_probs)
    water_source = np.random.choice(WATER_SOURCES, p=water_probs)
    sanitation_tier = int(np.random.choice(SANITATION_TIERS, p=sanitation_probs))
    children_under5 = int(np.random.choice([0, 1, 2, 3, 4], p=[0.15, 0.33, 0.27, 0.17, 0.08]))

    avg_meal_count = clip(np.random.normal(meal_mean + district_meal_adj, 0.45), 1.0, 3.5)
    avg_meal_count = round(float(avg_meal_count), 1)

    # True stunting propensity: logistic score + noise, targeting ~22% prevalence overall.
    z = -1.85
    z += 0.95 * max(0, 2.0 - avg_meal_count)
    z += {'piped': -0.45, 'public_tap': -0.10, 'protected_well': 0.25, 'surface_water': 0.75}[water_source]
    z += {1: 0.60, 2: 0.10, 3: -0.35}[sanitation_tier]
    z += {'low': 0.65, 'lower_middle': 0.20, 'middle': -0.15, 'upper': -0.45}[income_band]
    z += 0.18 * max(children_under5 - 1, 0)
    z += {'Nyarugenge': -0.10, 'Gasabo': -0.05, 'Kicukiro': 0.00, 'Musanze': 0.12, 'Nyagatare': 0.20}[district]
    z += np.random.normal(0, 0.28)
    p = 1.0 / (1.0 + math.exp(-z))
    stunting_flag = int(np.random.rand() < p)

    household_id = f'{district[:3].upper()}-{idx:05d}'
    return {
        'household_id': household_id,
        'lat': round(float(lat), 6),
        'lon': round(float(lon), 6),
        'district': district,
        'sector': sector,
        'children_under5': children_under5,
        'avg_meal_count': avg_meal_count,
        'water_source': water_source,
        'sanitation_tier': sanitation_tier,
        'income_band': income_band,
        'true_stunting_flag': stunting_flag,
        'true_risk_probability': round(float(p), 4),
        'urban_rural': 'urban' if urban else 'rural',
        'anon_household_id': stable_anon_id(household_id),
    }


def make_geojson(path: Path):
    features = []
    for district, meta in DISTRICTS.items():
        lat = meta['lat']
        lon = meta['lon']
        half_lon = 0.11 if district in {'Musanze', 'Nyagatare'} else 0.08
        half_lat = 0.08 if district in {'Musanze', 'Nyagatare'} else 0.06
        coords = [
            [lon - half_lon, lat - half_lat],
            [lon + half_lon, lat - half_lat],
            [lon + half_lon, lat + half_lat],
            [lon - half_lon, lat + half_lat],
            [lon - half_lon, lat - half_lat],
        ]
        features.append({
            'type': 'Feature',
            'properties': {'district': district},
            'geometry': {'type': 'Polygon', 'coordinates': [coords]},
        })
    geojson = {'type': 'FeatureCollection', 'features': features}
    with path.open('w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2)


def main():
    rows = []
    counter = 1
    for district, count in DISTRICT_COUNTS.items():
        for _ in range(count):
            rows.append(sample_household(district, counter))
            counter += 1

    df = pd.DataFrame(rows)

    # Adjust intercept if prevalence drifts too far due to randomness.
    prevalence = df['true_stunting_flag'].mean()

    households_cols = [
        'household_id', 'lat', 'lon', 'district', 'sector', 'children_under5',
        'avg_meal_count', 'water_source', 'sanitation_tier', 'income_band'
    ]
    df[households_cols].to_csv(OUTDIR / 'households.csv', index=False)

    # Balanced labeled subset: 150 positives, 150 negatives when available.
    pos = df[df['true_stunting_flag'] == 1].sample(n=150, random_state=RANDOM_SEED)
    neg = df[df['true_stunting_flag'] == 0].sample(n=150, random_state=RANDOM_SEED)
    gold = pd.concat([pos, neg], axis=0).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    gold[['household_id', 'true_stunting_flag']].rename(columns={'true_stunting_flag': 'stunting_flag'}).to_csv(
        OUTDIR / 'gold_stunting_flag.csv', index=False
    )

    make_geojson(OUTDIR / 'districts.geojson')

    summary = {
        'rows': int(len(df)),
        'overall_true_prevalence': round(float(prevalence), 4),
        'district_prevalence': df.groupby('district')['true_stunting_flag'].mean().round(4).to_dict(),
        'sample_sectors': {k: v for k, v in list(SECTORS.items())[:2]},
    }
    with (OUTDIR / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('Wrote files to', OUTDIR)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
