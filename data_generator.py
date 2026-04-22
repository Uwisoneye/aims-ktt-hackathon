from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

DISTRICTS = ["Nyarugenge", "Gasabo", "Kicukiro", "Huye", "Rubavu"]
SECTOR_MAP = {
    "Nyarugenge": ["Muhima", "Kimisagara", "Nyarugenge"],
    "Gasabo": ["Kacyiru", "Kimironko", "Remera"],
    "Kicukiro": ["Kanombe", "Gahanga", "Niboye"],
    "Huye": ["Tumba", "Ngoma", "Rusatira"],
    "Rubavu": ["Gisenyi", "Nyamyumba", "Kanama"],
}
WATER = ["piped", "protected_well", "unprotected_well", "surface"]
SANITATION = ["improved", "shared", "unimproved", "open_defecation"]
INCOME = ["high", "middle", "low", "very_low"]


def _district_center(district: str):
    centers = {
        "Nyarugenge": (-1.95, 30.05),
        "Gasabo": (-1.92, 30.08),
        "Kicukiro": (-1.97, 30.10),
        "Huye": (-2.60, 29.74),
        "Rubavu": (-1.68, 29.25),
    }
    return centers[district]


def generate_households(n: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        district = rng.choice(DISTRICTS, p=[0.22, 0.30, 0.20, 0.14, 0.14])
        sector = rng.choice(SECTOR_MAP[district])
        lat0, lon0 = _district_center(district)
        lat = lat0 + rng.normal(0, 0.03)
        lon = lon0 + rng.normal(0, 0.03)
        if district in ["Huye", "Rubavu"]:
            income_band = rng.choice(INCOME, p=[0.05, 0.20, 0.35, 0.40])
        else:
            income_band = rng.choice(INCOME, p=[0.15, 0.35, 0.30, 0.20])
        meal_count = int(rng.choice([1, 2, 3, 4], p=[0.20, 0.35, 0.30, 0.15]))
        water_source = rng.choice(WATER, p=[0.42, 0.25, 0.20, 0.13])
        sanitation_tier = rng.choice(SANITATION, p=[0.35, 0.25, 0.25, 0.15])
        children_under5 = int(rng.choice([0, 1, 2, 3], p=[0.36, 0.35, 0.20, 0.09]))

        linear = -1.3
        linear += {1: 0.50, 2: 0.20, 3: 0.00, 4: -0.10}[meal_count]
        linear += {"piped": 0.00, "protected_well": 0.15, "unprotected_well": 0.32, "surface": 0.45}[water_source]
        linear += {"improved": 0.00, "shared": 0.12, "unimproved": 0.30, "open_defecation": 0.50}[sanitation_tier]
        linear += {"high": 0.00, "middle": 0.08, "low": 0.20, "very_low": 0.35}[income_band]
        linear += min(children_under5, 3) * 0.12
        linear += 0.10 if district in ["Huye", "Rubavu"] else 0.00
        linear += rng.normal(0, 0.35)
        flag = int(1 / (1 + math.exp(-linear)) > 0.5)

        rows.append({
            "household_id": f"H{i+1:05d}",
            "lat": lat,
            "lon": lon,
            "district": district,
            "sector": sector,
            "children_under5": children_under5,
            "avg_meal_count": meal_count,
            "water_source": water_source,
            "sanitation_tier": sanitation_tier,
            "income_band": income_band,
            "stunting_flag": flag,
        })
    return pd.DataFrame(rows)


def generate_geojson() -> dict:
    # Simple, clean fallback polygons if the supplied file is missing.
    polys = {
        "Nyarugenge": [
            [29.98, -1.98], [30.02, -1.98], [30.02, -1.94], [29.98, -1.94], [29.98, -1.98]
        ],
        "Gasabo": [
            [30.03, -1.96], [30.12, -1.96], [30.12, -1.88], [30.03, -1.88], [30.03, -1.96]
        ],
        "Kicukiro": [
            [30.02, -2.00], [30.12, -2.00], [30.12, -1.95], [30.02, -1.95], [30.02, -2.00]
        ],
        "Huye": [
            [29.70, -2.66], [29.78, -2.66], [29.78, -2.56], [29.70, -2.56], [29.70, -2.66]
        ],
        "Rubavu": [
            [29.20, -1.73], [29.30, -1.73], [29.30, -1.64], [29.20, -1.64], [29.20, -1.73]
        ],
    }
    features = []
    for district, coords in polys.items():
        features.append({
            "type": "Feature",
            "properties": {"district": district},
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })
    return {"type": "FeatureCollection", "features": features}


if __name__ == "__main__":
    out = Path("data")
    out.mkdir(exist_ok=True, parents=True)
    df = generate_households()
    df.to_csv(out / "households.csv", index=False)
    df.sample(300, random_state=42)[["household_id", "stunting_flag"]].rename(columns={"stunting_flag": "gold_stunting_flag"}).to_csv(out / "gold_stunting_flag.csv", index=False)
    print("Generated data/households.csv and data/gold_stunting_flag.csv")
