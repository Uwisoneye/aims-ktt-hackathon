from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from risk_scorer import score_households

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
PRINTABLE_DIR = ROOT / "printable"


def _driver_text(drivers) -> str:
    if isinstance(drivers, str):
        try:
            parsed = eval(drivers, {"__builtins__": {}}, {})
            if isinstance(parsed, list):
                return ", ".join(parsed)
        except Exception:
            return drivers
    if isinstance(drivers, list):
        return ", ".join(drivers)
    return str(drivers)


def build_printable_pdf(df: pd.DataFrame, district: str, sector: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )
    styles = getSampleStyleSheet()
    title = styles["Title"]
    title.fontName = "Helvetica-Bold"
    title.fontSize = 16
    subtitle = styles["BodyText"]
    subtitle.fontSize = 10
    note_style = ParagraphStyle(
        "note",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
        spaceAfter=4,
    )

    story: List = []
    story.append(Paragraph(f"Umudugudu Monthly Risk Page - {district} / {sector}", title))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "Audience: village chief and sector social affairs focal point. "
        "Privacy: anonymized household IDs only; no names, phone numbers, or exact addresses.",
        subtitle,
    ))
    story.append(Spacer(1, 3 * mm))

    high_risk = int((df["risk_flag"] == 1).sum())
    avg_risk = float(df["risk_score"].mean()) if not df.empty else 0.0
    summary_table = Table(
        [["Metric", "Value"],
         ["District", district],
         ["Sector", sector],
         ["Households in sector", str(len(df))],
         ["High-risk households", str(high_risk)],
         ["Average risk score", f"{avg_risk:.3f}"]],
        colWidths=[55 * mm, 105 * mm],
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9ead3")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph(
        "Review loop: district nutrition officer prints this page monthly, hands it to the village chief, "
        "collects handwritten follow-up notes during the sector meeting, and escalates severe cases to MINISANTE via the district hospital nutrition focal point.",
        note_style,
    ))
    story.append(Spacer(1, 2 * mm))

    top10 = df.sort_values("risk_score", ascending=False).head(10).copy()
    rows = [["Anon ID", "Risk", "Top drivers"]]
    for _, row in top10.iterrows():
        rows.append([
            str(row["anon_household_id"]),
            f"{float(row['risk_score']):.3f}",
            _driver_text(row["top_drivers"]),
        ])

    table = Table(rows, colWidths=[25 * mm, 18 * mm, 125 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#cfe2f3")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEADING", (0, 0), (-1, -1), 11),
    ]))
    story.append(table)
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "Suggested field use: circle households for home visit, write date of visit in pen, and mark urgent referral if edema, repeated illness, or severe food insecurity is reported.",
        note_style,
    ))

    doc.build(story)


def main() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    PRINTABLE_DIR.mkdir(exist_ok=True)

    scored, metrics = score_households()
    scored.to_csv(ARTIFACT_DIR / "scored_households.csv", index=False)
    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    created = []
    grouped = scored.groupby(["district", "sector"], sort=True)
    for (district, sector), group in grouped:
        filename = f"{district}_{sector}".replace(" ", "_") + ".pdf"
        out_path = PRINTABLE_DIR / filename
        build_printable_pdf(group, district, sector, out_path)
        created.append(filename)

    manifest = {
        "printable_count": len(created),
        "printable_files": created,
        "metrics": metrics,
    }
    with open(ARTIFACT_DIR / "prepare_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
