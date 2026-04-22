from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


# Reasonable Rwanda-style defaults for a small Tier 1 challenge.
WATER_RISK = {
    "piped": 0.00,
    "protected_well": 0.12,
    "unprotected_well": 0.28,
    "surface": 0.40,
    "other": 0.20,
}

SANITATION_RISK = {
    "improved": 0.00,
    "shared": 0.12,
    "unimproved": 0.28,
    "open_defecation": 0.45,
}

INCOME_RISK = {
    "high": 0.00,
    "middle": 0.10,
    "low": 0.24,
    "very_low": 0.38,
}

MEAL_RISK = {
    # More meals = lower risk
    1: 0.40,
    2: 0.20,
    3: 0.00,
    4: -0.05,
}


def _norm_choice(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "other"
    return str(value).strip().lower().replace(" ", "_")


def score(household: Dict[str, object] | pd.Series) -> float:
    """Return a 0-1 risk score for a single household.

    The score is intentionally simple and explainable for live defense.
    """
    if isinstance(household, pd.Series):
        h = household.to_dict()
    else:
        h = dict(household)

    meal_count = int(h.get("avg_meal_count", h.get("meal_count", 2)) or 2)
    water = _norm_choice(h.get("water_source", "other"))
    sanitation = _norm_choice(h.get("sanitation_tier", "other"))
    income = _norm_choice(h.get("income_band", "other"))
    children = int(h.get("children_under5", 0) or 0)

    # Logistic-style linear score.
    linear = -1.1
    linear += MEAL_RISK.get(meal_count, 0.10)
    linear += WATER_RISK.get(water, 0.20)
    linear += SANITATION_RISK.get(sanitation, 0.20)
    linear += INCOME_RISK.get(income, 0.20)
    linear += min(children, 4) * 0.10

    return float(1 / (1 + math.exp(-linear)))


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["risk_score"] = out.apply(score, axis=1)
    out["risk_label"] = (out["risk_score"] >= 0.5).astype(int)
    return out


def explain_row(household: pd.Series) -> List[Tuple[str, float]]:
    """Return the top three drivers for a household."""
    drivers = []
    meal_count = int(household.get("avg_meal_count", 2) or 2)
    water = _norm_choice(household.get("water_source", "other"))
    sanitation = _norm_choice(household.get("sanitation_tier", "other"))
    income = _norm_choice(household.get("income_band", "other"))
    children = int(household.get("children_under5", 0) or 0)

    drivers.append((f"Meal pattern ({meal_count}/day)", abs(MEAL_RISK.get(meal_count, 0.10))))
    drivers.append((f"Water source ({water})", abs(WATER_RISK.get(water, 0.20))))
    drivers.append((f"Sanitation ({sanitation})", abs(SANITATION_RISK.get(sanitation, 0.20))))
    drivers.append((f"Income band ({income})", abs(INCOME_RISK.get(income, 0.20))))
    drivers.append((f"Under-5 children ({children})", min(children, 4) * 0.10))

    drivers.sort(key=lambda x: x[1], reverse=True)
    return drivers[:3]


def train_logistic_model(labeled_df: pd.DataFrame) -> Tuple[object, Dict[str, float]]:
    """Train a tiny logistic regression on the provided labels if present."""
    if "stunting_flag" not in labeled_df.columns:
        return None, {}

    df = labeled_df.copy()
    df = score_dataframe(df)

    feature_cols = [
        "avg_meal_count",
        "children_under5",
        "water_source",
        "sanitation_tier",
        "income_band",
        "district",
    ]
    df = pd.get_dummies(df[feature_cols + ["stunting_flag"]], columns=["water_source", "sanitation_tier", "income_band", "district"], drop_first=True)
    X = df.drop(columns=["stunting_flag"])
    y = df["stunting_flag"].astype(int)

    if y.nunique() < 2:
        return None, {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 3),
        "f1": round(float(f1_score(y_test, preds)), 3),
        "auc": round(float(roc_auc_score(y_test, prob)), 3),
    }
    return model, metrics
