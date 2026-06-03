"""Gemini visual review for full-dataset multi-pin candidates.

Run:
    python -m src.multipin_candidate_review

Required:
    set GEMINI_API_KEY=...

Optional:
    set MACHINE_REVIEW_MODEL=gemini-2.5-flash
    set MULTIPIN_REVIEW_LIMIT=100

Input:
    data/processed/multipin_candidate_queue.csv

Outputs:
    data/processed/multipin_candidate_machine_review.csv
    data/processed/multipin_candidate_machine_review_summary.txt
    data/processed/multipin_candidate_review_images/*.jpg
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
IMAGE_DIR = PROCESSED / "multipin_candidate_review_images"

INPUT_PATH = PROCESSED / "multipin_candidate_queue.csv"
OUTPUT_PATH = PROCESSED / "multipin_candidate_machine_review.csv"
SUMMARY_PATH = PROCESSED / "multipin_candidate_machine_review_summary.txt"

MODEL = os.environ.get("MACHINE_REVIEW_MODEL", "gemini-2.5-flash")
LIMIT = int(os.environ.get("MULTIPIN_REVIEW_LIMIT", "100"))


REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "multipin_need_status": {
            "type": "string",
            "enum": [
                "single_pin_ok",
                "shared_access_pin",
                "needs_pedestrian_vehicle_split",
                "needs_delivery_pin",
                "needs_accessible_pin",
                "needs_multiple_pins",
                "ambiguous",
                "privacy_sensitive",
                "needs_human_review",
            ],
        },
        "machine_confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "recommended_pin_schema": {
            "type": "string",
        },
        "should_prioritize_for_human_review": {
            "type": "boolean",
        },
        "machine_reasoning": {
            "type": "string",
        },
    },
    "required": [
        "multipin_need_status",
        "machine_confidence",
        "recommended_pin_schema",
        "should_prioritize_for_human_review",
        "machine_reasoning",
    ],
}


def norm(value: object) -> str:
    return str(value or "").strip()


def has_coord(row: pd.Series, lat_col: str, lon_col: str) -> bool:
    return (
        pd.notna(row.get(lat_col))
        and pd.notna(row.get(lon_col))
        and str(row.get(lat_col)).strip() != ""
        and str(row.get(lon_col)).strip() != ""
    )


def row_center(row: pd.Series) -> tuple[float, float]:
    candidates = [
        ("current_lat", "current_lon"),
        ("gt_lat", "gt_lon"),
        ("car_entry_lat", "car_entry_lon"),
    ]

    lats = []
    lons = []

    for lat_col, lon_col in candidates:
        if has_coord(row, lat_col, lon_col):
            lats.append(float(row[lat_col]))
            lons.append(float(row[lon_col]))

    if not lats:
        raise ValueError("Row has no usable coordinates.")

    return sum(lats) / len(lats), sum(lons) / len(lons)


def bbox_around(lat: float, lon: float, meters: float = 240) -> tuple[float, float, float, float]:
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


def draw_candidate_image(
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

    if has_coord(row, "current_lat", "current_lon"):
        ax.scatter(
            float(row["current_lon"]),
            float(row["current_lat"]),
            s=160,
            c="red",
            edgecolors="black",
            linewidths=1.5,
            label="Current Overture pin",
        )

    if has_coord(row, "gt_lat", "gt_lon"):
        ax.scatter(
            float(row["gt_lon"]),
            float(row["gt_lat"]),
            s=180,
            c="yellow",
            edgecolors="black",
            linewidths=1.5,
            label="LLM ground-truth / candidate pin",
        )

    title = " | ".join(
        part
        for part in [
            norm(row.get("name")),
            norm(row.get("category_primary")),
            f"score={row.get('multipin_candidate_score', '')}",
        ]
        if part
    )

    ax.set_title(title[:140], fontsize=9)
    ax.legend(loc="lower left", fontsize=8)

    fig.tight_layout(pad=0)
    fig.savefig(output_path, format="jpg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return output_path


def review_prompt(row: pd.Series) -> str:
    return f"""
You are reviewing a satellite image for a mapping dataset.

The question is not whether the exact pin is perfect. The question is whether this place likely needs more than one arrival/access pin.

Place:
- name: {row.get("name", "")}
- category: {row.get("category_primary", "")}
- address: {row.get("full_address", "")}
- complexity: {row.get("place_complexity", "")}
- ambiguity: {row.get("pin_ambiguity", "")}
- candidate score: {row.get("multipin_candidate_score", "")}
- candidate reasons: {row.get("multipin_candidate_reasons", "")}

Pins shown:
- red circle = current Overture pin
- yellow circle = LLM ground-truth / candidate place pin

Decide the best pin schema:
- single_pin_ok: one normal place pin is enough.
- shared_access_pin: one shared arrival/access pin is better than a centroid/storefront pin.
- needs_pedestrian_vehicle_split: pedestrian and vehicle arrivals differ materially.
- needs_delivery_pin: delivery/service/loading access is likely distinct.
- needs_accessible_pin: accessible entrance is likely distinct and important.
- needs_multiple_pins: multiple specialized pins are likely needed, but type is uncertain.
- ambiguous: imagery is unclear.
- privacy_sensitive: avoid precise public repositioning.
- needs_human_review: high-risk or uncertain case.

Rules:
- Be conservative.
- Do not infer multiple pins just because the place is large.
- Favor multi-pin labels only when the image or place type suggests meaningfully different access targets.
- Parks, RV parks, campuses, shopping centers, hospitals, airports, schools, industrial sites, and large complexes often need shared or multiple access pins.
- Private residences and sensitive facilities should be flagged instead of forced into a precise pin.

Return only valid JSON matching the schema.
""".strip()


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    return json.loads(text)


def call_gemini(row: pd.Series, image_path: Path) -> dict[str, Any]:
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("Missing GEMINI_API_KEY environment variable.")

    client = genai.Client()

    image_part = types.Part.from_bytes(
        data=image_path.read_bytes(),
        mime_type="image/jpeg",
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[review_prompt(row), image_part],
        config={
            "temperature": 0,
            "response_mime_type": "application/json",
            "response_json_schema": REVIEW_SCHEMA,
        },
    )

    return parse_json_response(response.text or "{}")


def fallback_review(error: Exception) -> dict[str, Any]:
    return {
        "multipin_need_status": "needs_human_review",
        "machine_confidence": 0.0,
        "recommended_pin_schema": "unknown",
        "should_prioritize_for_human_review": True,
        "machine_reasoning": f"Machine review failed: {error}",
    }


def normalize_review(review: dict[str, Any]) -> dict[str, Any]:
    confidence = float(review.get("machine_confidence", 0) or 0)
    confidence = max(0.0, min(1.0, confidence))

    status = str(review.get("multipin_need_status", "needs_human_review")).strip()

    if confidence < 0.45 and status not in {"privacy_sensitive"}:
        status = "needs_human_review"

    prioritize = bool(review.get("should_prioritize_for_human_review", False))
    if status in {"ambiguous", "privacy_sensitive", "needs_human_review"}:
        prioritize = True

    return {
        "multipin_need_status": status,
        "machine_confidence": confidence,
        "recommended_pin_schema": review.get("recommended_pin_schema", "unknown"),
        "should_prioritize_for_human_review": prioritize,
        "machine_reasoning": review.get("machine_reasoning", ""),
    }


def write_summary(df: pd.DataFrame) -> None:
    lines = [
        "Multi-Pin Candidate Machine Review Summary",
        "",
        "Important",
        "This is machine-assisted screening, not human ground truth.",
        "",
        f"Rows reviewed: {len(df)}",
        "",
    ]

    for col in [
        "multipin_need_status",
        "recommended_pin_schema",
        "should_prioritize_for_human_review",
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

    source = pd.read_csv(INPUT_PATH).head(LIMIT).reset_index(drop=True)
    reviewed_rows = []

    for i, row in source.iterrows():
        place_id = str(row.get("id", f"row_{i}"))
        safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in place_id)

        raw_image = IMAGE_DIR / f"{i:03d}_{safe_id}_raw.jpg"
        marked_image = IMAGE_DIR / f"{i:03d}_{safe_id}_marked.jpg"

        print(f"[{i + 1}/{len(source)}] Reviewing {row.get('name', place_id)}")

        try:
            raw_path, bbox = fetch_esri_image(row, raw_image)
            draw_candidate_image(row, raw_path, bbox, marked_image)
            review = normalize_review(call_gemini(row, marked_image))
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