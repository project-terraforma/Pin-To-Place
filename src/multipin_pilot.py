"""
Build the multi-pin analysis dataset from the ground truth.

Selects all rows in ground_truth_combined.csv that have both a car_entry and a
pedestrian_entry coordinate from the dual-annotation pipeline. These are rows
where GPT-4o already identified two meaningfully different arrival targets, so
no manual coordinate entry is needed.

Run from the project root:
    python -m src.multipin_pilot
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import haversine_meters

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"

GT_PATH = PROCESSED / "ground_truth_combined.csv"
MULTIPIN_OUTPUT = PROCESSED / "multipin_pilot.csv"
MULTIPIN_SUMMARY = PROCESSED / "multipin_pilot_summary.txt"


def build_multipin_pilot() -> pd.DataFrame:
    if not GT_PATH.exists():
        raise FileNotFoundError(f"Missing ground truth: {GT_PATH}")

    df = pd.read_csv(GT_PATH)

    has_car = df["car_entry_lat"].notna() & df["car_entry_lon"].notna()
    has_ped = df["pedestrian_entry_lat"].notna() & df["pedestrian_entry_lon"].notna()
    dual = df[has_car & has_ped].copy().reset_index(drop=True)

    dual["car_pedestrian_distance_m"] = dual.apply(
        lambda r: haversine_meters(
            r["pedestrian_entry_lat"], r["pedestrian_entry_lon"],
            r["car_entry_lat"], r["car_entry_lon"],
        ),
        axis=1,
    )

    dual["current_to_pedestrian_m"] = dual.apply(
        lambda r: haversine_meters(
            r["current_lat"], r["current_lon"],
            r["pedestrian_entry_lat"], r["pedestrian_entry_lon"],
        ),
        axis=1,
    )

    dual["current_to_car_m"] = dual.apply(
        lambda r: haversine_meters(
            r["current_lat"], r["current_lon"],
            r["car_entry_lat"], r["car_entry_lon"],
        ),
        axis=1,
    )

    preferred = [
        "id", "name", "category_primary", "region",
        "tier_label", "sub_tier_label", "place_complexity", "pin_ambiguity",
        "current_lat", "current_lon",
        "gt_lat", "gt_lon", "gt_confidence", "gt_reasoning",
        "pedestrian_entry_lat", "pedestrian_entry_lon", "pedestrian_entry_confidence",
        "car_entry_lat", "car_entry_lon", "car_entry_confidence", "car_entry_reasoning",
        "entry_mode",
        "cv_disagreement_m",
        "car_pedestrian_distance_m",
        "current_to_pedestrian_m",
        "current_to_car_m",
        "offset_haversine_m",
        "full_address",
    ]
    existing = [c for c in preferred if c in dual.columns]
    remaining = [c for c in dual.columns if c not in existing]
    return dual[existing + remaining]


def write_summary(df: pd.DataFrame) -> None:
    sep = df["car_pedestrian_distance_m"]
    lines = [
        "Multi-Pin Dataset Summary",
        "",
        f"Rows: {len(df)}",
        "",
        "Source: ground_truth_combined.csv — rows with both car_entry and pedestrian_entry",
        "from the dual-annotation pipeline. No manual coordinate entry required.",
        "",
        "Car/pedestrian separation distance (m)",
        sep.describe().to_string(),
        f"pct separation >= 10m:  {(sep >= 10).mean() * 100:.1f}%",
        f"pct separation >= 25m:  {(sep >= 25).mean() * 100:.1f}%",
        f"pct separation >= 50m:  {(sep >= 50).mean() * 100:.1f}%",
        "",
    ]

    for col in ["tier_label", "place_complexity", "category_primary"]:
        if col in df.columns:
            lines += [f"{col} counts", df[col].value_counts(dropna=False).to_string(), ""]

    MULTIPIN_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = build_multipin_pilot()
    df.to_csv(MULTIPIN_OUTPUT, index=False)
    write_summary(df)

    print(f"Saved: {MULTIPIN_OUTPUT}")
    print(f"Saved: {MULTIPIN_SUMMARY}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
