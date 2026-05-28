"""
Create and summarize a balanced manual review pilot.

Run from project root:
    python -m src.manual_review
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"

PILOT_OUTPUT = PROCESSED / "manual_review_pilot.csv"
PILOT_SUMMARY = PROCESSED / "manual_review_pilot_summary.txt"

REVIEW_COLUMNS = [
    "manual_review_status",
    "manual_should_move",
    "manual_primary_pin_type",
    "manual_needs_multi_pin",
    "manual_notes",
]

REVIEW_STATUS_VALUES = [
    "accepted",
    "wrong_target",
    "ambiguous",
    "bad_imagery",
    "privacy_sensitive",
    "needs_more_context",
]

PRIMARY_PIN_TYPE_VALUES = [
    "current",
    "pedestrian_entry",
    "vehicle_entry",
    "delivery_entry",
    "accessible_entry",
    "area_centroid",
    "unknown",
]


def _read_review_file(filename: str) -> pd.DataFrame:
    path = PROCESSED / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing review queue: {path}")
    return pd.read_csv(path)


def _sample_queue(
    df: pd.DataFrame,
    n: int,
    source_label: str,
    random_state: int,
) -> pd.DataFrame:
    sampled = df.sample(n=min(n, len(df)), random_state=random_state).copy()
    sampled["pilot_source"] = source_label
    return sampled


def build_manual_review_pilot(
    high_offset_n: int = 40,
    low_confidence_n: int = 30,
    multi_tenant_n: int = 25,
    zero_offset_n: int = 25,
    random_state: int = 42,
) -> pd.DataFrame:
    parts = [
        _sample_queue(
            _read_review_file("review_high_offset.csv"),
            high_offset_n,
            "high_offset",
            random_state,
        ),
        _sample_queue(
            _read_review_file("review_low_confidence.csv"),
            low_confidence_n,
            "low_confidence",
            random_state + 1,
        ),
        _sample_queue(
            _read_review_file("review_multi_tenant.csv"),
            multi_tenant_n,
            "multi_tenant",
            random_state + 2,
        ),
        _sample_queue(
            _read_review_file("review_zero_offset_sample.csv"),
            zero_offset_n,
            "zero_offset_sample",
            random_state + 3,
        ),
    ]

    pilot = pd.concat(parts, ignore_index=True)
    pilot = pilot.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

    for col in REVIEW_COLUMNS:
        if col not in pilot.columns:
            pilot[col] = ""

    preferred_cols = [
        "pilot_source",
        "id",
        "name",
        "category_primary",
        "region",
        "tier_label",
        "place_complexity",
        "pin_ambiguity",
        "gt_confidence",
        "offset_haversine_m",
        "arrival_cost_m",
        "should_move",
        "current_lat",
        "current_lon",
        "gt_lat",
        "gt_lon",
        "full_address",
        "gt_reasoning",
        *REVIEW_COLUMNS,
    ]
    existing = [col for col in preferred_cols if col in pilot.columns]
    remaining = [col for col in pilot.columns if col not in existing]
    return pilot[existing + remaining]


def write_summary(pilot: pd.DataFrame) -> None:
    lines = [
        "Manual Review Pilot Summary",
        "",
        f"Rows: {len(pilot)}",
        "",
        "Pilot source counts",
        pilot["pilot_source"].value_counts(dropna=False).to_string(),
        "",
        "Allowed manual_review_status values",
        ", ".join(REVIEW_STATUS_VALUES),
        "",
        "Allowed manual_primary_pin_type values",
        ", ".join(PRIMARY_PIN_TYPE_VALUES),
        "",
        "Manual columns to fill",
        ", ".join(REVIEW_COLUMNS),
    ]

    for col in ["tier_label", "place_complexity", "pin_ambiguity", "should_move"]:
        if col in pilot.columns:
            lines.extend(["", f"{col} counts", pilot[col].value_counts(dropna=False).to_string()])

    PILOT_SUMMARY.write_text("\n".join(lines) + "\n")


def main() -> None:
    pilot = build_manual_review_pilot()
    pilot.to_csv(PILOT_OUTPUT, index=False)
    write_summary(pilot)

    print(f"Saved: {PILOT_OUTPUT}")
    print(f"Saved: {PILOT_SUMMARY}")
    print(f"Rows: {len(pilot)}")


if __name__ == "__main__":
    main()
