from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from data_generator import generate_households
from risk_scorer import explain_row, score_dataframe

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PRINT_DIR = ROOT / "printable"
PRINT_DIR.mkdir(exist_ok=True, parents=True)


def make_page(sector_name: str, sector_df: pd.DataFrame, pdf: PdfPages):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    top = sector_df.sort_values("risk_score", ascending=False).head(10)
    lines = [
        f"UMUDUGUDU PAGE: {sector_name}",
        "",
        f"Review date: Monthly risk brief",
        f"Total households in sector sample: {len(sector_df)}",
        f"High-risk threshold: 0.50",
        "",
        "Top 10 anonymised households",
        "ID     Risk   Drivers",
        "---------------------------------------------------------------",
    ]
    for _, row in top.iterrows():
        drivers = "; ".join([d for d, _ in explain_row(row)])
        lines.append(f"{row['household_id']:<7} {row['risk_score']:.2f}  {drivers}")
    lines += [
        "",
        "Privacy note: household names are removed; only household IDs are shown.",
        "Escalation loop: chief -> sector officer -> district M&E / MINISANTE if severe.",
    ]

    y = 0.95
    for i, line in enumerate(lines):
        ax.text(0.05, y, line, fontsize=12 if i == 0 else 10, family="monospace", va="top")
        y -= 0.04 if i == 0 else 0.03
    pdf.savefig(fig)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_DIR / "households.csv") if (DATA_DIR / "households.csv").exists() else generate_households()
    df = score_dataframe(df)
    with PdfPages(PRINT_DIR / "umudugudu_pages.pdf") as pdf:
        for sector in df["sector"].dropna().unique()[:5]:
            make_page(sector, df[df["sector"] == sector].copy(), pdf)
    print(f"Saved {PRINT_DIR / 'umudugudu_pages.pdf'}")


if __name__ == "__main__":
    main()
