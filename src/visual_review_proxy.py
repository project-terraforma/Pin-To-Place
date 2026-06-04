"""
Automated proxy review for the multi-pin pilot.

This does not replace true satellite/street-level visual validation. It fills a
consistent automated review layer from the existing manual review labels,
LLM ground-truth confidence, pin type, and pin-distance relationships.

Run from the project root:
    python -m src.visual_review_proxy
"""

from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd

from src.metrics import haversine_meters


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"

MULTIPIN_PATH = PROCESSED / "multipin" / "multipin_pilot.csv"
OUTPUT_PATH = PROCESSED / "multipin" / "multipin_visual_review.csv"
SUMMARY_PATH = PROCESSED / "multipin" / "multipin_visual_review_summary.txt"


def _map_link(lat, lon) -> str:
    if pd.isna(lat) or pd.isna(lon):
        return ""
    return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"


def _search_link(row) -> str:
    query = " ".join([str(row.get("name", "")), str(row.get("full_address", ""))])
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(query)}"


def _has_pair(row, lat_col: str, lon_col: str) -> bool:
    return pd.notna(row.get(lat_col)) and pd.notna(row.get(lon_col))


def _distance(row, a_lat: str, a_lon: str, b_lat: str, b_lon: str) -> float:
    if not all(pd.notna(row.get(col)) for col in [a_lat, a_lon, b_lat, b_lon]):
        return np.nan
    return haversine_meters(row[a_lat], row[a_lon], row[b_lat], row[b_lon])


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def classify_proxy_review(row) -> dict:
    tier = str(row.get("tier_label", ""))
    complexity = str(row.get("place_complexity", ""))
    primary = str(row.get("manual_primary_pin_type", ""))
    conf = float(row.get("gt_confidence", 0) or 0)

    has_ped = _has_pair(row, "pedestrian_entry_lat", "pedestrian_entry_lon")
    has_vehicle = _has_pair(row, "car_entry_lat", "car_entry_lon")
    ped_vehicle_distance = row.get("car_pedestrian_distance_m", np.nan)

    result = {
        "visual_review_status": "proxy_reviewed",
        "visual_pedestrian_entry_correct": "unknown",
        "visual_vehicle_entry_correct": "unknown",
        "visual_delivery_entry_visible": "unknown",
        "visual_accessible_entry_visible": "unknown",
        "visual_review_priority": "medium",
        "visual_review_needs_human": "true",
        "visual_notes": "",
    }

    notes = ["Automated proxy review; not satellite/street-level validation."]

    if tier == "open_space":
        if has_vehicle and conf >= 0.8:
            result["visual_vehicle_entry_correct"] = "likely"
            result["visual_review_status"] = "proxy_car_accepted"
            result["visual_review_priority"] = "medium"
            notes.append("Open-space row uses LLM ground truth as car/main-access proxy.")
        else:
            result["visual_review_status"] = "needs_human_review"
            result["visual_review_priority"] = "high"
            notes.append("Open-space row lacks a confident car/main-access proxy.")

        result["visual_pedestrian_entry_correct"] = "unknown"
        notes.append("Pedestrian entry for open-space rows requires imagery.")

    elif tier == "multi_tenant" or complexity == "multi_tenant":
        if has_ped and conf >= 0.8:
            result["visual_pedestrian_entry_correct"] = "likely"
            result["visual_review_status"] = "proxy_pedestrian_accepted"
            notes.append("Multi-tenant row uses LLM ground truth as pedestrian/storefront proxy.")
        else:
            result["visual_review_status"] = "needs_human_review"
            result["visual_review_priority"] = "high"
            notes.append("Multi-tenant row lacks a confident pedestrian/storefront proxy.")

        if has_vehicle:
            result["visual_vehicle_entry_correct"] = "weak_proxy"
            notes.append("Car entry is a weak proxy and needs imagery.")
        else:
            result["visual_vehicle_entry_correct"] = "unknown"

    elif primary == "car_entry":
        if has_vehicle and conf >= 0.8:
            result["visual_vehicle_entry_correct"] = "likely"
            result["visual_review_status"] = "proxy_car_accepted"
            notes.append("Primary manual pin type is car_entry and proxy exists.")

    elif primary == "pedestrian_entry":
        if has_ped and conf >= 0.8:
            result["visual_pedestrian_entry_correct"] = "likely"
            result["visual_review_status"] = "proxy_pedestrian_accepted"
            notes.append("Primary manual pin type is pedestrian_entry and proxy exists.")

    if pd.notna(ped_vehicle_distance):
        if ped_vehicle_distance >= 25:
            result["visual_review_priority"] = "high"
            notes.append(f"Pedestrian and vehicle proxy pins differ by {ped_vehicle_distance:.1f}m.")
        elif ped_vehicle_distance >= 10:
            notes.append(f"Pedestrian and vehicle proxy pins differ by {ped_vehicle_distance:.1f}m.")

    if not has_ped and not has_vehicle:
        result["visual_review_status"] = "needs_human_review"
        result["visual_review_priority"] = "high"
        notes.append("No pedestrian or vehicle proxy coordinate exists.")

    if result["visual_review_priority"] == "medium" and result["visual_review_status"].startswith("proxy_"):
        result["visual_review_needs_human"] = "false"

    result["visual_notes"] = " ".join(notes)
    return result


def main() -> None:
    if not MULTIPIN_PATH.exists():
        raise FileNotFoundError(f"Missing multi-pin pilot: {MULTIPIN_PATH}")

    df = pd.read_csv(MULTIPIN_PATH)

    df["google_search_url"] = df.apply(_search_link, axis=1)
    df["current_pin_url"] = df.apply(lambda r: _map_link(r.get("current_lat"), r.get("current_lon")), axis=1)
    df["gt_pin_url"] = df.apply(lambda r: _map_link(r.get("gt_lat"), r.get("gt_lon")), axis=1)
    df["pedestrian_entry_url"] = df.apply(
        lambda r: _map_link(r.get("pedestrian_entry_lat"), r.get("pedestrian_entry_lon")),
        axis=1,
    )
    df["car_entry_url"] = df.apply(
        lambda r: _map_link(r.get("car_entry_lat"), r.get("car_entry_lon")),
        axis=1,
    )

    df["has_pedestrian_entry"] = df.apply(
        lambda r: _has_pair(r, "pedestrian_entry_lat", "pedestrian_entry_lon"),
        axis=1,
    )
    df["has_car_entry"] = df.apply(
        lambda r: _has_pair(r, "car_entry_lat", "car_entry_lon"),
        axis=1,
    )
    df["has_delivery_entry"] = df.apply(
        lambda r: _has_pair(r, "delivery_entry_lat", "delivery_entry_lon"),
        axis=1,
    )
    df["has_accessible_entry"] = df.apply(
        lambda r: _has_pair(r, "accessible_entry_lat", "accessible_entry_lon"),
        axis=1,
    )

    df["car_pedestrian_distance_m"] = df.apply(
        lambda r: _distance(
            r,
            "pedestrian_entry_lat",
            "pedestrian_entry_lon",
            "car_entry_lat",
            "car_entry_lon",
        ),
        axis=1,
    )

    review = df.apply(classify_proxy_review, axis=1, result_type="expand")
    for col in review.columns:
        df[col] = review[col]

    preferred = [
        "name",
        "category_primary",
        "tier_label",
        "place_complexity",
        "pin_ambiguity",
        "manual_review_status",
        "manual_primary_pin_type",
        "manual_needs_multi_pin",
        "visual_review_status",
        "visual_review_priority",
        "visual_review_needs_human",
        "visual_pedestrian_entry_correct",
        "visual_vehicle_entry_correct",
        "car_pedestrian_distance_m",
        "full_address",
        "google_search_url",
        "current_pin_url",
        "gt_pin_url",
        "pedestrian_entry_url",
        "car_entry_url",
        "visual_notes",
        "multipin_notes",
    ]
    cols = [col for col in preferred if col in df.columns] + [col for col in df.columns if col not in preferred]
    df = df[cols]
    df.to_csv(OUTPUT_PATH, index=False)

    lines = [
        "Multi-Pin Visual Proxy Review Summary",
        "",
        f"Rows: {len(df)}",
        "",
        "Important",
        "This is an automated proxy review. It does not replace true visual validation.",
        "",
        "visual_review_status counts",
        df["visual_review_status"].value_counts(dropna=False).to_string(),
        "",
        "visual_review_priority counts",
        df["visual_review_priority"].value_counts(dropna=False).to_string(),
        "",
        "visual_review_needs_human counts",
        df["visual_review_needs_human"].value_counts(dropna=False).to_string(),
        "",
        "Pin coverage",
        df[["has_pedestrian_entry", "has_car_entry", "has_delivery_entry", "has_accessible_entry"]].sum().to_string(),
        "",
        "Car/pedestrian proxy distance",
        df["car_pedestrian_distance_m"].describe().to_string(),
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Saved: {SUMMARY_PATH}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
