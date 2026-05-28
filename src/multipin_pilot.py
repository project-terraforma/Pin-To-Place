"""
Create a multi-pin pilot from manually reviewed rows.

Run from the project root:
    python -m src.multipin_pilot
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"

MANUAL_REVIEW_PATH = PROCESSED / "manual_review_pilot.csv"
MULTIPIN_OUTPUT = PROCESSED / "multipin_pilot.csv"
MULTIPIN_SUMMARY = PROCESSED / "multipin_pilot_summary.txt"

MULTIPIN_COLUMNS = [
    "pedestrian_entry_lat",
    "pedestrian_entry_lon",
    "pedestrian_entry_confidence",
    "vehicle_entry_lat",
    "vehicle_entry_lon",
    "vehicle_entry_confidence",
    "delivery_entry_lat",
    "delivery_entry_lon",
    "delivery_entry_confidence",
    "accessible_entry_lat",
    "accessible_entry_lon",
    "accessible_entry_confidence",
    "multipin_notes",
]


def _is_true(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def build_multipin_pilot() -> pd.DataFrame:
    if not MANUAL_REVIEW_PATH.exists():
        raise FileNotFoundError(f"Missing manual review pilot: {MANUAL_REVIEW_PATH}")

    df = pd.read_csv(MANUAL_REVIEW_PATH)
    if "manual_needs_multi_pin" not in df.columns:
        raise ValueError("manual_review_pilot.csv is missing manual_needs_multi_pin")

    multipin = df[df["manual_needs_multi_pin"].apply(_is_true)].copy()
    multipin = multipin.reset_index(drop=True)

    for col in MULTIPIN_COLUMNS:
        if col not in multipin.columns:
            multipin[col] = ""

    preferred_cols = [
        "pilot_source",
        "id",
        "name",
        "category_primary",
        "region",
        "tier_label",
        "place_complexity",
        "pin_ambiguity",
        "manual_review_status",
        "manual_should_move",
        "manual_primary_pin_type",
        "manual_needs_multi_pin",
        "current_lat",
        "current_lon",
        "gt_lat",
        "gt_lon",
        "gt_confidence",
        "offset_haversine_m",
        "full_address",
        "gt_reasoning",
        *MULTIPIN_COLUMNS,
        "manual_notes",
    ]

    existing = [col for col in preferred_cols if col in multipin.columns]
    remaining = [col for col in multipin.columns if col not in existing]
    return multipin[existing + remaining]


def write_summary(multipin: pd.DataFrame) -> None:
    lines = [
        "Multi-Pin Pilot Summary",
        "",
        f"Rows: {len(multipin)}",
        "",
        "Purpose",
        "Rows in this file were marked manual_needs_multi_pin=true during the manual review pilot.",
        "Fill separate pedestrian, vehicle, delivery, and accessible pin targets where applicable.",
        "",
    ]

    for col in ["tier_label", "category_primary", "place_complexity", "pin_ambiguity", "manual_primary_pin_type"]:
        if col in multipin.columns:
            lines.extend([
                f"{col} counts",
                multipin[col].value_counts(dropna=False).to_string(),
                "",
            ])

    lines.extend([
        "Columns to fill",
        ", ".join(MULTIPIN_COLUMNS),
        "",
        "Guidance",
        "- pedestrian_entry: storefront, lobby, door, or walking entrance.",
        "- vehicle_entry: driveway, parking lot entrance, gate, or drop-off access point.",
        "- delivery_entry: loading/service/delivery access if visible or inferable.",
        "- accessible_entry: ramp/accessible path/curb-cut connected entrance if visible.",
        "- leave non-applicable coordinates blank and explain in multipin_notes.",
    ])

    MULTIPIN_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    multipin = build_multipin_pilot()
    multipin.to_csv(MULTIPIN_OUTPUT, index=False)
    write_summary(multipin)

    print(f"Saved: {MULTIPIN_OUTPUT}")
    print(f"Saved: {MULTIPIN_SUMMARY}")
    print(f"Rows: {len(multipin)}")


if __name__ == "__main__":
    main()
