"""Machine visual review for unresolved multi-pin rows using Gemini.

Run from the project root:
    python -m src.machine_visual_review

Required:
    set GEMINI_API_KEY=...

Optional:
    set MACHINE_REVIEW_MODEL=gemini-2.5-flash
    set MACHINE_REVIEW_ALL=1

Inputs:
    data/processed/multipin_visual_review.csv

Outputs:
    data/processed/machine_visual_review.csv
    data/processed/machine_visual_review_summary.txt
    data/processed/machine_review_images/*.jpg

This is machine-assisted validation, not human ground truth.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests
from google import genai
from google.genai import types


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
IMAGE_DIR = PROCESSED / "machine_review_images"

INPUT_PATH = PROCESSED / "multipin_visual_review.csv"
OUTPUT_PATH = PROCESSED / "machine_visual_review.csv"
SUMMARY_PATH = PROCESSED / "machine_visual_review_summary.txt"

MODEL = os.environ.get("MACHINE_REVIEW_MODEL", "gemini-2.5-flash")
REVIEW_ONLY_HUMAN_NEEDED = os.environ.get("MACHINE_REVIEW_ALL", "").lower() not in {"1", "true", "yes"}

AUTO_ACCEPT_CONFIDENCE = 0.85
AMBIGUOUS_CONFIDENCE = 0.55


PIN_STYLE = {
    "current": {"color": "red", "label": "Current Overture pin"},
    "gt": {"color": "yellow", "label": "LLM ground-truth / primary pin"},
    "shared": {"color": "magenta", "label": "Shared arrival/access pin"},
    "pedestrian": {"color": "cyan", "label": "Pedestrian entry pin"},
    "vehicle": {"color": "lime", "label": "Vehicle/car entry pin"},
}


REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "machine_visual_status": {
            "type": "string",
            "enum": [
                "accepted",
                "likely_correct",
                "ambiguous",
                "wrong_target",
                "privacy_sensitive",
                "needs_human_review",
            ],
        },
        "machine_pedestrian_entry_correct": {
            "type": "string",
            "enum": ["yes", "likely", "no", "unknown", "not_applicable"],
        },
        "machine_vehicle_entry_correct": {
            "type": "string",
            "enum": ["yes", "likely", "no", "unknown", "not_applicable"],
        },
        "machine_confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "machine_should_accept_without_human": {
            "type": "boolean",
        },
        "machine_reasoning": {
            "type": "string",
        },
    },
    "required": [
        "machine_visual_status",
        "machine_pedestrian_entry_correct",
        "machine_vehicle_entry_correct",
        "machine_confidence",
        "machine_should_accept_without_human",
        "machine_reasoning",
    ],
}


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def has_coord(row: pd.Series, lat_col: str, lon_col: str) -> bool:
    return pd.notna(row.get(lat_col)) and pd.notna(row.get(lon_col)) and str(row.get(lat_col)).strip() != "" and str(row.get(lon_col)).strip() != ""


def effective_review_targets(row: pd.Series) -> dict[str, tuple[str, str]]:
    entry_mode = str(row.get("entry_mode", "")).strip().lower()

    if entry_mode == "shared":
        if has_coord(row, "car_entry_lat", "car_entry_lon"):
            return {"shared": ("car_entry_lat", "car_entry_lon")}
        if has_coord(row, "gt_lat", "gt_lon"):
            return {"shared": ("gt_lat", "gt_lon")}

    targets = {}

    if has_coord(row, "pedestrian_entry_lat", "pedestrian_entry_lon"):
        targets["pedestrian"] = ("pedestrian_entry_lat", "pedestrian_entry_lon")
    elif has_coord(row, "gt_lat", "gt_lon"):
        targets["pedestrian"] = ("gt_lat", "gt_lon")

    if has_coord(row, "vehicle_entry_lat", "vehicle_entry_lon"):
        targets["vehicle"] = ("vehicle_entry_lat", "vehicle_entry_lon")
    elif has_coord(row, "car_entry_lat", "car_entry_lon"):
        targets["vehicle"] = ("car_entry_lat", "car_entry_lon")

    return targets


def row_center(row: pd.Series) -> tuple[float, float]:
    candidates = [
        ("current_lat", "current_lon"),
        ("gt_lat", "gt_lon"),
        ("pedestrian_entry_lat", "pedestrian_entry_lon"),
        ("vehicle_entry_lat", "vehicle_entry_lon"),
        ("car_entry_lat", "car_entry_lon"),
    ]

    lats = []
    lons = []
    for lat_col, lon_col in candidates:
        if has_coord(row, lat_col, lon_col):
            lats.append(float(row[lat_col]))
            lons.append(float(row[lon_col]))

    if not lats or not lons:
        raise ValueError("Row has no usable coordinates.")

    return sum(lats) / len(lats), sum(lons) / len(lons)


def bbox_around(lat: float, lon: float, meters: float = 180) -> tuple[float, float, float, float]:
    lat_delta = meters / 111_320
    lon_delta = meters / (111_320 * max(0.2, abs(math.cos(math.radians(lat)))))
    return lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta


def fetch_esri_image(row: pd.Series, output_path: Path) -> tuple[Path, tuple[float, float, float, float]]:
    lat, lon = row_center(row)
    bbox = bbox_around(lat, lon)

    url = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
    params = {
        "bbox": ",".join(str(x) for x in bbox),
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": "1024,1024",
        "format": "jpg",
        "f": "image",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path, bbox


def draw_review_image(
    row: pd.Series,
    raw_image_path: Path,
    bbox: tuple[float, float, float, float],
    output_path: Path,
) -> Path:
    west, south, east, north = bbox
    image = plt.imread(raw_image_path)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=140)
    ax.imshow(image, extent=[west, east, south, north])
    ax.set_axis_off()

    base_pins = [
        ("current", "current_lat", "current_lon"),
        ("gt", "gt_lat", "gt_lon"),
    ]

    for pin_name, lat_col, lon_col in base_pins:
        if has_coord(row, lat_col, lon_col):
            style = PIN_STYLE[pin_name]
            ax.scatter(
                float(row[lon_col]),
                float(row[lat_col]),
                s=150,
                c=style["color"],
                edgecolors="black",
                linewidths=1.5,
                label=style["label"],
            )

    for pin_name, (lat_col, lon_col) in effective_review_targets(row).items():
        if has_coord(row, lat_col, lon_col):
            style = PIN_STYLE[pin_name]
            ax.scatter(
                float(row[lon_col]),
                float(row[lat_col]),
                s=220,
                c=style["color"],
                edgecolors="black",
                linewidths=2.0,
                marker="*",
                label=style["label"],
            )

    title = " | ".join(
        part
        for part in [
            str(row.get("name", "")),
            str(row.get("category_primary", "")),
            f"entry_mode={row.get('entry_mode', '')}",
        ]
        if part and part != "nan"
    )
    ax.set_title(title[:140], fontsize=9)
    ax.legend(loc="lower left", fontsize=8)

    fig.tight_layout(pad=0)
    fig.savefig(output_path, format="jpg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return output_path


def machine_review_prompt(row: pd.Series) -> str:
    entry_mode = str(row.get("entry_mode", "")).strip().lower()
    targets = effective_review_targets(row)

    if entry_mode == "shared":
        target_text = """
This row has entry_mode=shared.

The magenta star is the proposed shared arrival/access pin. Judge whether it is visually plausible as a shared pedestrian/vehicle access point, such as a main entrance, driveway, access road, parking entrance, park access, RV park entrance, or campground access.
""".strip()
    else:
        visible_targets = ", ".join(targets.keys()) if targets else "none"
        target_text = f"""
This row has entry_mode={entry_mode or "unknown"}.

Visible proposed target types: {visible_targets}.

Judge whether the proposed pedestrian and vehicle/car pins are visually plausible for real-world arrival.
""".strip()

    return f"""
You are reviewing a place-pin validation image for a mapping dataset.

Place:
- name: {row.get("name", "")}
- category: {row.get("category_primary", "")}
- address: {row.get("full_address", "")}
- tier: {row.get("tier_label", "")}
- complexity: {row.get("place_complexity", "")}
- ambiguity: {row.get("pin_ambiguity", "")}

Pins shown:
- red circle = current Overture pin
- yellow circle = existing LLM ground-truth / primary pin
- magenta star = proposed shared arrival/access pin, if present
- cyan star = proposed pedestrian entry pin, if present
- green star = proposed vehicle/car entry pin, if present

{target_text}

Rules:
- Do not overclaim when satellite imagery is unclear.
- Vehicle/car entry usually means driveway, parking lot entrance, gate, access road, or drop-off access.
- Pedestrian entry usually means storefront, lobby entrance, front door, path-connected entrance, or obvious access point.
- For parks, RV parks, campgrounds, and open spaces, main access, road access, or parking entry can be valid.
- If the image cannot support a conclusion, mark it ambiguous or needs_human_review.
- Use privacy_sensitive if the row appears to point to a private residence or sensitive location.

For entry_mode=shared:
- Set machine_pedestrian_entry_correct and machine_vehicle_entry_correct to likely/yes if the shared pin plausibly works for both.
- Set them to unknown if the imagery is unclear.
- Set them to no if the shared pin is visibly wrong.
""".strip()


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    return json.loads(text)


def call_gemini_vision(row: pd.Series, image_path: Path) -> dict[str, Any]:
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("Missing GEMINI_API_KEY environment variable.")

    client = genai.Client()

    image_part = types.Part.from_bytes(
        data=image_path.read_bytes(),
        mime_type="image/jpeg",
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            machine_review_prompt(row),
            image_part,
        ],
        config={
            "temperature": 0,
            "response_mime_type": "application/json",
            "response_json_schema": REVIEW_SCHEMA,
        },
    )

    return parse_json_response(response.text or "{}")


def normalize_review(review: dict[str, Any]) -> dict[str, Any]:
    confidence = float(review.get("machine_confidence", 0) or 0)
    confidence = max(0.0, min(1.0, confidence))

    status = str(review.get("machine_visual_status", "needs_human_review")).strip()

    if confidence >= AUTO_ACCEPT_CONFIDENCE and status in {"accepted", "likely_correct"}:
        accept_without_human = True
    else:
        accept_without_human = False

    if confidence < AMBIGUOUS_CONFIDENCE and status not in {"wrong_target", "privacy_sensitive"}:
        status = "needs_human_review"

    return {
        "machine_visual_status": status,
        "machine_pedestrian_entry_correct": review.get("machine_pedestrian_entry_correct", "unknown"),
        "machine_vehicle_entry_correct": review.get("machine_vehicle_entry_correct", "unknown"),
        "machine_confidence": confidence,
        "machine_should_accept_without_human": accept_without_human,
        "machine_reasoning": review.get("machine_reasoning", ""),
    }


def fallback_review(error: Exception) -> dict[str, Any]:
    return {
        "machine_visual_status": "needs_human_review",
        "machine_pedestrian_entry_correct": "unknown",
        "machine_vehicle_entry_correct": "unknown",
        "machine_confidence": 0.0,
        "machine_should_accept_without_human": False,
        "machine_reasoning": f"Machine review failed: {error}",
    }


def rows_to_review(df: pd.DataFrame) -> pd.DataFrame:
    if not REVIEW_ONLY_HUMAN_NEEDED:
        return df.copy()

    if "visual_review_needs_human" not in df.columns:
        return df.copy()

    return df[df["visual_review_needs_human"].apply(truthy)].copy()


def write_summary(df: pd.DataFrame) -> None:
    lines = [
        "Machine Visual Review Summary",
        "",
        "Important",
        "This is machine-assisted validation, not human ground truth.",
        "",
        f"Rows reviewed: {len(df)}",
        "",
    ]

    for col in [
        "machine_visual_status",
        "machine_should_accept_without_human",
        "machine_pedestrian_entry_correct",
        "machine_vehicle_entry_correct",
    ]:
        if col in df.columns:
            lines.extend([col, df[col].value_counts(dropna=False).to_string(), ""])

    if "machine_confidence" in df.columns:
        lines.extend(["machine_confidence", df["machine_confidence"].describe().to_string(), ""])

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_PATH}")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(INPUT_PATH)
    review_df = rows_to_review(source).reset_index(drop=True)

    if review_df.empty:
        print("No rows require machine review.")
        return

    reviewed_rows = []

    for i, row in review_df.iterrows():
        place_id = str(row.get("id", f"row_{i}"))
        safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in place_id)

        raw_image = IMAGE_DIR / f"{i:03d}_{safe_id}_raw.jpg"
        marked_image = IMAGE_DIR / f"{i:03d}_{safe_id}_marked.jpg"

        print(f"[{i + 1}/{len(review_df)}] Reviewing {row.get('name', place_id)}")

        try:
            raw_path, bbox = fetch_esri_image(row, raw_image)
            draw_review_image(row, raw_path, bbox, marked_image)
            review = normalize_review(call_gemini_vision(row, marked_image))
            time.sleep(1.0)
        except Exception as exc:
            review = fallback_review(exc)

        output_row = row.to_dict()
        output_row["machine_review_model"] = MODEL
        output_row["machine_review_provider"] = "gemini"
        output_row["machine_review_image"] = str(marked_image)
        output_row.update(review)
        reviewed_rows.append(output_row)

        pd.DataFrame(reviewed_rows).to_csv(OUTPUT_PATH, index=False)

    output = pd.DataFrame(reviewed_rows)
    output.to_csv(OUTPUT_PATH, index=False)
    write_summary(output)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Saved: {SUMMARY_PATH}")
    print(f"Saved images in: {IMAGE_DIR}")


if __name__ == "__main__":
    main()