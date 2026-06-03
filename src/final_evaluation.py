"""Generate final Pin-To-Place evaluation and recommendation notes.

Run from the project root:
    python -m src.final_evaluation

Inputs expected in data/processed:
    ground_truth_combined.csv
    manual_review_pilot.csv
    multipin_visual_review.csv
    machine_visual_review.csv

Outputs:
    data/processed/final_evaluation_summary.txt
    docs/final_recommendation.md
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
DOCS = PROJECT_ROOT / "docs"

GROUND_TRUTH_PATH = PROCESSED / "ground_truth_combined.csv"
MANUAL_REVIEW_PATH = PROCESSED / "manual_review_pilot.csv"
MULTIPIN_REVIEW_PATH = PROCESSED / "multipin_visual_review.csv"
MACHINE_REVIEW_PATH = PROCESSED / "machine_visual_review.csv"

SUMMARY_OUTPUT = PROCESSED / "final_evaluation_summary.txt"
RECOMMENDATION_OUTPUT = DOCS / "final_recommendation.md"


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def pct(numerator: int | float, denominator: int | float) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{100 * numerator / denominator:.1f}%"


def value_counts_block(df: pd.DataFrame, column: str) -> list[str]:
    if df.empty or column not in df.columns:
        return [f"{column}: unavailable"]

    counts = df[column].fillna("missing").value_counts(dropna=False)
    return [f"{column}:", counts.to_string()]


def metric_summary(df: pd.DataFrame, column: str) -> list[str]:
    if df.empty or column not in df.columns:
        return [f"{column}: unavailable"]

    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return [f"{column}: unavailable"]

    return [
        f"{column}:",
        f"  count: {len(values)}",
        f"  mean: {values.mean():.2f}",
        f"  median: {values.median():.2f}",
        f"  p90: {np.percentile(values, 90):.2f}",
        f"  p95: {np.percentile(values, 95):.2f}",
        f"  max: {values.max():.2f}",
    ]


def segment_summary(df: pd.DataFrame, segment_col: str, metric_col: str) -> pd.DataFrame:
    if df.empty or segment_col not in df.columns or metric_col not in df.columns:
        return pd.DataFrame()

    work = df[[segment_col, metric_col]].copy()
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    work = work.dropna(subset=[metric_col])

    if work.empty:
        return pd.DataFrame()

    grouped = work.groupby(segment_col)[metric_col]
    return grouped.agg(
        rows="count",
        mean="mean",
        median="median",
        p90=lambda s: np.percentile(s, 90),
        p95=lambda s: np.percentile(s, 95),
        max="max",
    ).reset_index()


def summarize_manual_review(df: pd.DataFrame) -> list[str]:
    lines = ["Manual Review"]

    if df.empty:
        return lines + ["manual_review_pilot.csv: missing"]

    rows = len(df)
    lines.append(f"Rows reviewed: {rows}")

    if "manual_should_move" in df.columns:
        should_move = int(truthy(df["manual_should_move"]).sum())
        lines.append(f"Manual should-move rate: {should_move}/{rows} ({pct(should_move, rows)})")

    if "manual_needs_multi_pin" in df.columns:
        needs_multi = int(truthy(df["manual_needs_multi_pin"]).sum())
        lines.append(f"Manual multi-pin rate: {needs_multi}/{rows} ({pct(needs_multi, rows)})")

    if "manual_review_status" in df.columns:
        privacy = int((df["manual_review_status"].astype(str) == "privacy_sensitive").sum())
        lines.append(f"Privacy-sensitive rate: {privacy}/{rows} ({pct(privacy, rows)})")
        lines.extend(value_counts_block(df, "manual_review_status"))

    return lines


def summarize_multipin_review(df: pd.DataFrame) -> list[str]:
    lines = ["Multi-Pin Proxy Review"]

    if df.empty:
        return lines + ["multipin_visual_review.csv: missing"]

    rows = len(df)
    lines.append(f"Rows reviewed: {rows}")

    if "visual_review_needs_human" in df.columns:
        needs_human = int(truthy(df["visual_review_needs_human"]).sum())
        lines.append(f"Rows originally needing human review: {needs_human}/{rows} ({pct(needs_human, rows)})")

    if "visual_review_priority" in df.columns:
        high = int((df["visual_review_priority"].astype(str) == "high").sum())
        lines.append(f"High-priority rows: {high}/{rows} ({pct(high, rows)})")

    lines.extend(value_counts_block(df, "visual_review_status"))

    return lines


def summarize_machine_review(df: pd.DataFrame) -> list[str]:
    lines = ["Machine Visual Review"]

    if df.empty:
        return lines + ["machine_visual_review.csv: missing"]

    rows = len(df)
    lines.append(f"Rows reviewed: {rows}")
    lines.append("This is machine-assisted validation, not human ground truth.")

    if "machine_should_accept_without_human" in df.columns:
        accepted = int(truthy(df["machine_should_accept_without_human"]).sum())
        lines.append(f"Accepted without human review: {accepted}/{rows} ({pct(accepted, rows)})")

    lines.extend(value_counts_block(df, "machine_visual_status"))
    lines.extend(value_counts_block(df, "machine_pedestrian_entry_correct"))
    lines.extend(value_counts_block(df, "machine_vehicle_entry_correct"))
    lines.extend(metric_summary(df, "machine_confidence"))

    if "machine_visual_status" in df.columns and "name" in df.columns:
        unresolved = df[df["machine_visual_status"].isin(["wrong_target", "needs_human_review", "ambiguous"])]
        if not unresolved.empty:
            lines.extend(["Unresolved or rejected machine-review rows:"])
            for _, row in unresolved.iterrows():
                name = row.get("name", "unknown")
                status = row.get("machine_visual_status", "unknown")
                confidence = row.get("machine_confidence", "unknown")
                reason = str(row.get("machine_reasoning", "")).strip()
                lines.append(f"- {name}: {status}, confidence={confidence}")
                if reason:
                    lines.append(f"  reason: {reason}")

    return lines


def build_final_summary() -> str:
    gt = read_optional_csv(GROUND_TRUTH_PATH)
    manual = read_optional_csv(MANUAL_REVIEW_PATH)
    multipin = read_optional_csv(MULTIPIN_REVIEW_PATH)
    machine = read_optional_csv(MACHINE_REVIEW_PATH)

    lines = [
        "Pin-To-Place Final Evaluation Summary",
        "",
        "Ground Truth",
        f"Rows: {len(gt)}",
    ]

    lines.extend(metric_summary(gt, "offset_haversine_m"))
    lines.extend(metric_summary(gt, "arrival_cost_m"))

    for segment_col in ["tier_label", "place_complexity", "pin_ambiguity", "category_primary"]:
        seg = segment_summary(gt, segment_col, "arrival_cost_m")
        if not seg.empty:
            lines.extend(["", f"Arrival cost by {segment_col}", seg.round(2).to_string(index=False)])

    lines.extend(["", *summarize_manual_review(manual)])
    lines.extend(["", *summarize_multipin_review(multipin)])
    lines.extend(["", *summarize_machine_review(machine)])

    lines.extend([
        "",
        "Final Interpretation",
        "Median offset is not the right primary success metric because most original pins already do not move.",
        "The useful signal is concentrated in p90/p95 arrival cost, manual should-move rows, privacy-sensitive rows, ambiguous rows, wrong-target rows, and multi-pin rows.",
        "The Gemini-based machine visual review accepted 3 of 5 high-priority unresolved rows, rejected 1 as a wrong target, and left 1 requiring review.",
        "The recommended production path is conservative task-aware pinning: keep stable pins where they are already adequate, move only high-confidence failures, and represent shared, pedestrian, and vehicle arrival explicitly where one coordinate is insufficient.",
    ])

    return "\n".join(lines) + "\n"


def build_recommendation_markdown(summary_text: str) -> str:
    markdown = """# Final Recommendation

## Recommendation

Pin-To-Place should not ship as a simple global pin-repositioning system.

The project should ship as a task-aware pin evaluation framework that decides when a place pin should stay fixed, when it should move, when it should be flagged for privacy or ambiguity, and when a place needs shared or multiple arrival targets.

## Why

The baseline median offset is already near zero, so optimizing median movement would reward the system for doing almost nothing. The meaningful errors appear in the tail: high p90/p95 arrival cost, ambiguous open spaces, complex multi-tenant places, privacy-sensitive locations, and places where pedestrian and vehicle arrival targets differ.

## Machine Visual Review

Because manual review was unavailable for the remaining high-priority rows, unresolved cases were evaluated using a Gemini-based machine visual review layer over satellite imagery.

This review accepted 3 of 5 high-priority unresolved rows, rejected 1 as a wrong target, and left 1 requiring human review. These labels are treated as machine-assisted validation, not final human ground truth.

## Production Rule

Use a conservative hierarchy:

1. Keep the current Overture pin when the place is low-ambiguity and arrival cost is low.
2. Move the pin only when the ground-truth target is high-confidence and arrival cost is meaningfully reduced.
3. Flag privacy-sensitive, ambiguous, and machine-rejected rows instead of forcing a coordinate.
4. Use shared, pedestrian, vehicle, delivery, and accessible arrival targets where one coordinate is insufficient.
5. Require human review for high-priority rows before treating proxy or machine labels as final ground truth.

## Final Project Position

Pin-To-Place is strongest as a validation and routing-aware pin schema project, not as a universal pin-moving project. Its contribution is showing when a single coordinate is adequate, when it is harmful, and when the place needs a more explicit arrival model.

## Generated Evaluation Notes

```text
"""
    markdown += summary_text.strip()
    markdown += """
"""
    return markdown
def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    summary = build_final_summary()
    SUMMARY_OUTPUT.write_text(summary, encoding="utf-8")
    RECOMMENDATION_OUTPUT.write_text(build_recommendation_markdown(summary), encoding="utf-8")

    print(f"Saved: {SUMMARY_OUTPUT}")
    print(f"Saved: {RECOMMENDATION_OUTPUT}")
if __name__ == "__main__":
    main()