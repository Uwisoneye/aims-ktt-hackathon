from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

NUMERIC_FEATURES = ["children_under5", "avg_meal_count"]
CATEGORICAL_FEATURES = ["district", "sector", "water_source", "sanitation_tier", "income_band"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RISK_REASONS = {
    "avg_meal_count": lambda v: (2.1 - float(v)) * 1.7 if float(v) < 2.1 else 0.0,
    "children_under5": lambda v: max(float(v) - 2, 0.0) * 0.45,
    "water_source": {
        "surface": 1.6,
        "unprotected_well": 1.2,
        "public_tap": 0.5,
        "piped": 0.0,
        "borehole": 0.2,
    },
    "sanitation_tier": {
        "open_defecation": 1.6,
        "pit_basic": 0.8,
        "pit_improved": 0.4,
        "flush": 0.0,
    },
    "income_band": {
        "very_low": 1.4,
        "low": 0.9,
        "lower_middle": 0.4,
        "middle": 0.0,
    },
}


@dataclass
class TrainingArtifacts:
    pipeline: Pipeline
    threshold: float
    cv_roc_auc: float
    cv_f1: float


class RiskScorer:
    def __init__(self, pipeline: Pipeline, threshold: float = 0.5) -> None:
        self.pipeline = pipeline
        self.threshold = threshold

    def predict_proba(self, households: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(households[FEATURES])[:, 1]

    def score(self, household: pd.Series | Dict) -> Dict:
        df = pd.DataFrame([dict(household)])
        risk_score = float(self.predict_proba(df)[0])
        reasons = explain_top_drivers(df.iloc[0])
        return {
            "risk_score": round(risk_score, 4),
            "risk_flag": int(risk_score >= self.threshold),
            "top_drivers": reasons,
        }


def build_pipeline() -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("prep", preprocessor), ("model", model)])


def select_threshold(y_true: np.ndarray, probas: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, probas)
    if len(thresholds) == 0:
        return 0.5
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / np.clip(precisions[:-1] + recalls[:-1], 1e-9, None)
    idx = int(np.nanargmax(f1_scores))
    return float(thresholds[idx])


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    households = pd.read_csv(DATA_DIR / "households.csv")
    gold = pd.read_csv(DATA_DIR / "gold_stunting_flag.csv")
    return households, gold


def train_model() -> Tuple[RiskScorer, TrainingArtifacts, pd.DataFrame, pd.DataFrame]:
    households, gold = load_data()
    labelled = households.merge(gold, on="household_id", how="inner")
    X = labelled[FEATURES]
    y = labelled["stunting_flag"].astype(int)

    base_pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probas = cross_val_predict(base_pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    threshold = select_threshold(y.to_numpy(), cv_probas)
    cv_preds = (cv_probas >= threshold).astype(int)
    cv_roc_auc = float(roc_auc_score(y, cv_probas))
    cv_f1 = float(f1_score(y, cv_preds))

    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)
    scorer = RiskScorer(final_pipeline, threshold=threshold)
    artifacts = TrainingArtifacts(final_pipeline, threshold, cv_roc_auc, cv_f1)
    return scorer, artifacts, households, labelled


def explain_top_drivers(row: pd.Series) -> List[str]:
    contributions = {}
    contributions["Low meal count"] = RISK_REASONS["avg_meal_count"](row.get("avg_meal_count", 0))
    contributions["Many under-5 children"] = RISK_REASONS["children_under5"](row.get("children_under5", 0))
    water = str(row.get("water_source", ""))
    sanitation = str(row.get("sanitation_tier", ""))
    income = str(row.get("income_band", ""))
    contributions[f"Water: {water}"] = RISK_REASONS["water_source"].get(water, 0.0)
    contributions[f"Sanitation: {sanitation}"] = RISK_REASONS["sanitation_tier"].get(sanitation, 0.0)
    contributions[f"Income: {income}"] = RISK_REASONS["income_band"].get(income, 0.0)
    ranked = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, score in ranked if score > 0][:3]


def anonymize_household_id(hid: str) -> str:
    suffix = str(hid).split("-")[-1][-4:]
    return f"HH-{suffix}"


def score_households() -> Tuple[pd.DataFrame, Dict]:
    scorer, artifacts, households, labelled = train_model()
    scored = households.copy()
    scored["risk_score"] = scorer.predict_proba(households)
    scored["risk_flag"] = (scored["risk_score"] >= artifacts.threshold).astype(int)
    scored["anon_household_id"] = scored["household_id"].map(anonymize_household_id)
    scored["top_drivers"] = scored.apply(explain_top_drivers, axis=1)
    metrics = {
        "threshold": round(float(artifacts.threshold), 4),
        "cv_roc_auc": round(float(artifacts.cv_roc_auc), 4),
        "cv_f1": round(float(artifacts.cv_f1), 4),
        "n_households": int(len(households)),
        "n_labelled": int(len(labelled)),
    }
    return scored, metrics


def save_artifacts() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    scored, metrics = score_households()
    scored.to_csv(ARTIFACT_DIR / "scored_households.csv", index=False)
    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    save_artifacts()
