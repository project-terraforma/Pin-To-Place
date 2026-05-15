"""
Generate combined ground truth and manual review queues.

Run from project root:
    python -m src.review_queue
"""

from pathlib import Path

import pandas as pd

from src.features import place_complexity, pin_ambiguity, should_move_rule
from src.metrics import task_aware_report, segmented_task_report


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_ground_truth_batches() -> pd.DataFrame:
    files = sorted(PROCESSED.glob("ground_truth_*.csv"))

    if not files:
        raise FileNotFoundError("No ground_truth_*.csv files found in data/processed")

    dfs = [pd.read_csv(path) for path in files]
    return pd.concat(dfs, ignore_index=True)


def enrich_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["place_complexity"] = df.apply(place_complexity, axis=1)
    df["pin_ambiguity"] = df.apply(pin_ambiguity, axis=1)
    df["should_move"] = df.apply(should_move_rule, axis=1)

    return df


def make_review_queues(df: pd.DataFrame) -> None:
    high_offset = df[df["offset_haversine_m"] >= 30].copy()
    low_confidence = df[df["gt_confidence"] < 0.6].copy()
    multi_tenant = df[df["tier_label"] == "multi_tenant"].copy()
    movable = df[df["should_move"]].copy()

    zero_source = df[df["offset_haversine_m"] == 0].copy()
    zero_sample = zero_source.sample(
        n=min(150, len(zero_source)),
        random_state=42,
    )

    high_offset.to_csv(PROCESSED / "review_high_offset.csv", index=False)
    low_confidence.to_csv(PROCESSED / "review_low_confidence.csv", index=False)
    multi_tenant.to_csv(PROCESSED / "review_multi_tenant.csv", index=False)
    movable.to_csv(PROCESSED / "review_should_move.csv", index=False)
    zero_sample.to_csv(PROCESSED / "review_zero_offset_sample.csv", index=False)


def write_summary(df: pd.DataFrame) -> None:
    lines = []
    lines.append("Pin-To-Place Ground Truth Audit Summary")
    lines.append("")
    lines.append("Overall")
    lines.append(str(task_aware_report(df)))
    lines.append("")
    lines.append("By tier")
    lines.append(segmented_task_report(df, "tier_label").to_string(index=False))
    lines.append("")
    lines.append("By place complexity")
    lines.append(segmented_task_report(df, "place_complexity").to_string(index=False))
    lines.append("")
    lines.append("By ambiguity")
    lines.append(segmented_task_report(df, "pin_ambiguity").to_string(index=False))

    (PROCESSED / "ground_truth_audit_summary.txt").write_text("\n".join(lines))


def main() -> None:
    df = load_ground_truth_batches()
    df = enrich_ground_truth(df)

    combined_path = PROCESSED / "ground_truth_combined.csv"
    df.to_csv(combined_path, index=False)

    make_review_queues(df)
    write_summary(df)

    print(f"Saved combined dataset: {combined_path}")
    print("Saved review queues and audit summary.")


if __name__ == "__main__":
    main()
