"""Build a full-dataset queue of likely multi-pin candidates.

Run:
    python -m src.multipin_candidate_queue

Optional:
    set MULTIPIN_QUEUE_LIMIT=250

Input:
    data/processed/ground_truth_combined.csv

Output:
    data/processed/multipin_candidate_queue.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = PROCESSED / "ground_truth_combined.csv"
OUTPUT_PATH = PROCESSED / "multipin_candidate_queue.csv"

DEFAULT_LIMIT = int(os.environ.get("MULTIPIN_QUEUE_LIMIT", "250"))


CATEGORY_KEYWORDS = {
    "park",
    "campground",
    "rv park",
    "shopping center",
    "mall",
    "plaza",
    "school",
    "university",
    "college",
    "hospital",
    "airport",
    "stadium",
    "arena",
    "resort",
    "campus",
    "industrial",
    "warehouse",
    "distribution",
    "trail",
    "marina",
    "terminal",
    "station",
    "hotel",
    "apartment",
    "complex",
}

NAME_KEYWORDS = {
    "park",
    "rv",
    "campground",
    "camp",
    "mall",
    "center",
    "centre",
    "plaza",
    "campus",
    "school",
    "hospital",
    "airport",
    "terminal",
    "gate",
    "resort",
    "trail",
    "marina",
    "stadium",
    "arena",
    "warehouse",
    "distribution",
    "apartments",
    "shopping",
    "marketplace",
}


def norm(value: object) -> str:
    return str(value or "").strip().lower()


def has_keyword(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def numeric(row: pd.Series, column: str, default: float = 0.0) -> float:
    try:
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def score_row(row: pd.Series) -> tuple[int, list[str]]:
    score = 0
    reasons = []

    name = norm(row.get("name"))
    category = norm(row.get("category_primary"))
    address = norm(row.get("full_address"))
    complexity = norm(row.get("place_complexity"))
    ambiguity = norm(row.get("pin_ambiguity"))

    combined_text = " ".join([name, category, address])

    offset = numeric(row, "offset_haversine_m")

    if has_keyword(category, CATEGORY_KEYWORDS):
        score += 4
        reasons.append("multi_pin_category")

    if has_keyword(name, NAME_KEYWORDS):
        score += 3
        reasons.append("multi_pin_name_signal")

    if has_keyword(combined_text, {"entrance", "gate", "terminal", "campus", "parking", "lot"}):
        score += 2
        reasons.append("access_in_text")

    if complexity == "high":
        score += 4
        reasons.append("high_complexity")
    elif complexity == "medium":
        score += 2
        reasons.append("medium_complexity")

    if ambiguity == "high":
        score += 4
        reasons.append("high_ambiguity")
    elif ambiguity == "medium":
        score += 2
        reasons.append("medium_ambiguity")

    if offset >= 50:
        score += 4
        reasons.append("offset_ge_50m")
    elif offset >= 20:
        score += 3
        reasons.append("offset_ge_20m")
    elif offset >= 10:
        score += 1
        reasons.append("offset_ge_10m")

    if has_keyword(combined_text, {"park", "campground", "rv", "campus", "mall", "airport"}):
        score += 2
        reasons.append("large_place_signal")

    return score, reasons


def ensure_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    for column in [
        "current_lat",
        "current_lon",
        "gt_lat",
        "gt_lon",
        "pedestrian_entry_lat",
        "pedestrian_entry_lon",
        "vehicle_entry_lat",
        "vehicle_entry_lon",
        "car_entry_lat",
        "car_entry_lon",
        "entry_mode",
    ]:
        if column not in output.columns:
            output[column] = ""

    if "car_entry_lat" in output.columns and "gt_lat" in output.columns:
        output["car_entry_lat"] = output["car_entry_lat"].where(output["car_entry_lat"].astype(str).str.strip() != "", output["gt_lat"])
    if "car_entry_lon" in output.columns and "gt_lon" in output.columns:
        output["car_entry_lon"] = output["car_entry_lon"].where(output["car_entry_lon"].astype(str).str.strip() != "", output["gt_lon"])

    output["entry_mode"] = output["entry_mode"].where(output["entry_mode"].astype(str).str.strip() != "", "candidate_unknown")

    return output


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    scored = []

    for _, row in df.iterrows():
        score, reasons = score_row(row)
        row_dict = row.to_dict()
        row_dict["multipin_candidate_score"] = score
        row_dict["multipin_candidate_reasons"] = ";".join(reasons)
        row_dict["multipin_candidate"] = score >= 5
        scored.append(row_dict)

    output = pd.DataFrame(scored)
    output = ensure_review_columns(output)

    output = output.sort_values(
        by=["multipin_candidate_score", "offset_haversine_m"],
        ascending=[False, False],
        na_position="last",
    )

    output = output.head(DEFAULT_LIMIT)
    output.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows queued: {len(output)}")
    print(output["multipin_candidate_score"].describe().to_string())


if __name__ == "__main__":
    main()