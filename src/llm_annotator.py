"""
LLM vision-based ground truth annotation.
Uses GPT-4o or Claude to examine satellite imagery and identify correct pin locations.
Supports single-pin annotation (all tiers) and dual-label annotation (Tier 1/2 only).
"""

import os
import json
import re
import base64
import logging
from pathlib import Path
from dataclasses import dataclass, field

from src.features import get_tier
from src.metrics import haversine_meters

logger = logging.getLogger(__name__)


def _parse_json(content: str) -> dict:
    """Extract a JSON object from an LLM response regardless of surrounding text."""
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON object found in LLM response: {content[:200]!r}")


def _encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class AnnotationResult:
    place_id: str
    gt_lat: float | None
    gt_lon: float | None
    confidence: float
    reasoning: str
    model: str
    raw_response: dict | None = None


@dataclass
class DualAnnotationResult:
    """Holds both pedestrian-entry and car-entry pin annotations for Tier 1/2 places."""
    place_id: str
    # Pedestrian entry (primary pin for Tier 1/2)
    pedestrian_lat: float | None
    pedestrian_lon: float | None
    pedestrian_confidence: float
    pedestrian_reasoning: str
    pedestrian_model: str
    # Car entry (driveway / parking lot entrance)
    car_lat: float | None
    car_lon: float | None
    car_confidence: float
    car_reasoning: str
    car_model: str
    # 'both' | 'car_only' | 'shared'
    entry_mode: str = "both"


# ── Prompts ────────────────────────────────────────────────────────────────────

ANNOTATION_PROMPTS = {
    1: """You are a geospatial analyst placing a map pin for a {category} business.

## Place Information
- **Name:** {name}
- **Category:** {category}
- **Address:** {address}

## Task
The red marker shows the current pin. Identify the PEDESTRIAN ENTRY pin location.

The pin should mark the main customer entrance facing the street — the storefront
door or lobby entry a visitor on foot would walk to. Do NOT place the pin at the
building center, parking lot, or side/rear entrance.

Steps:
1. Find the building matching the address.
2. Identify the street-facing facade with the primary entrance (look for awnings, signage, door openings).
3. Place the pin at that entrance, as close to the door as the imagery allows.
4. If the current pin is already at the entrance, return the center coordinates.

Respond with ONLY this JSON (no markdown):
{{
    "pixel_x": <x pixel from left edge, 0-{tile_size}>,
    "pixel_y": <y pixel from top edge, 0-{tile_size}>,
    "confidence": <0.0 to 1.0>,
    "reasoning": "<brief explanation>"
}}
Image is {tile_size}x{tile_size} px. Current pin is at center ({center},{center}).""",

    2: """You are a geospatial analyst placing a map pin for a {category} located
inside a shared building or complex (strip mall, office suite, or shopping center).

## Place Information
- **Name:** {name}
- **Category:** {category}
- **Address:** {address}

## Task
The red marker shows the current pin. Identify the PEDESTRIAN ENTRY pin location.

The pin should mark the specific unit's storefront entrance — NOT the building's
overall center and NOT the shared parking lot entrance.

Steps:
1. Identify the multi-tenant building or complex.
2. Find the specific unit for "{name}" within the building by its address portion.
3. Place the pin at the visible entrance of that individual unit.
4. If the current pin is already at the correct unit entrance, return the center coordinates.

Respond with ONLY this JSON (no markdown):
{{
    "pixel_x": <x pixel from left edge, 0-{tile_size}>,
    "pixel_y": <y pixel from top edge, 0-{tile_size}>,
    "confidence": <0.0 to 1.0>,
    "reasoning": "<brief explanation>"
}}
Image is {tile_size}x{tile_size} px. Current pin is at center ({center},{center}).""",

    3: """You are a geospatial analyst placing a map pin for a {category} — an open
space, outdoor area, or campground.

## Place Information
- **Name:** {name}
- **Category:** {category}
- **Address:** {address}

## Task
The red marker shows the current pin. Identify the CORRECT pin location.

For open spaces, the pin should mark the main access point — a parking lot entrance,
gate, or trailhead where visitors first arrive. If no clear access point exists,
place the pin at the visual center of the area.

Steps:
1. Locate the open space or park area.
2. Look for a parking lot entrance, gate, or road entry serving as the main access.
3. If multiple access points exist, choose the most prominent.
4. If no access point is identifiable, use area center.

Respond with ONLY this JSON (no markdown):
{{
    "pixel_x": <x pixel from left edge, 0-{tile_size}>,
    "pixel_y": <y pixel from top edge, 0-{tile_size}>,
    "confidence": <0.0 to 1.0>,
    "reasoning": "<brief explanation>"
}}
Image is {tile_size}x{tile_size} px. Current pin is at center ({center},{center}).""",

    4: """You are a geospatial analyst placing a map pin for a {category} that has
no dedicated physical building (home business, mobile service, or address-only entry).

## Place Information
- **Name:** {name}
- **Category:** {category}
- **Address:** {address}

## Task
The red marker shows the current pin.

Steps:
1. Check whether a visible standalone commercial building exists at this address.
2. If yes (clear storefront/commercial structure), place the pin at the entrance as
   you would for a standard commercial place.
3. If no dedicated commercial structure is visible (residential building, empty lot,
   or home-based business), return the center coordinates and set confidence below 0.4.

Respond with ONLY this JSON (no markdown):
{{
    "pixel_x": <x pixel from left edge, 0-{tile_size}>,
    "pixel_y": <y pixel from top edge, 0-{tile_size}>,
    "confidence": <0.0 to 1.0>,
    "reasoning": "<brief explanation>"
}}
Image is {tile_size}x{tile_size} px. Current pin is at center ({center},{center}).""",
}


CAR_ENTRY_PROMPTS = {
    1: """You are a geospatial analyst identifying where a driver would enter a property.

## Place Information
- **Name:** {name}
- **Category:** {category}
- **Address:** {address}

## Task
The red marker shows the current pin. Identify the CAR ENTRY point.

The car entry pin should mark where a driver turns off the road to reach this place:
the driveway cut, parking lot entrance, or drop-off lane entry — NOT the building door.

Steps:
1. Find the property matching the address.
2. Look for the driveway entrance or parking lot entry from the road (curb cut, gate, lot opening).
3. If multiple vehicle entries exist, pick the primary / most visible one from the road.
4. If the place has no parking or driveway (pedestrian-only urban storefront), return
   the center and set confidence < 0.3.

Respond with ONLY this JSON (no markdown):
{{
    "pixel_x": <x pixel from left edge, 0-{tile_size}>,
    "pixel_y": <y pixel from top edge, 0-{tile_size}>,
    "confidence": <0.0 to 1.0>,
    "reasoning": "<brief explanation>"
}}
Image is {tile_size}x{tile_size} px. Current pin is at center ({center},{center}).""",

    2: """You are a geospatial analyst identifying where a driver would enter a shared property.

## Place Information
- **Name:** {name}
- **Category:** {category}
- **Address:** {address}

## Task
The red marker shows the current pin. Identify the CAR ENTRY point.

For a multi-tenant building, the car entry is the shared parking lot entrance
nearest to this unit — NOT the individual unit door or storefront.

Steps:
1. Identify the shared building or complex.
2. Find the parking lot entry from the nearest road that serves this tenant's section.
3. If multiple lot entrances exist, choose the one closest to this unit's address.

Respond with ONLY this JSON (no markdown):
{{
    "pixel_x": <x pixel from left edge, 0-{tile_size}>,
    "pixel_y": <y pixel from top edge, 0-{tile_size}>,
    "confidence": <0.0 to 1.0>,
    "reasoning": "<brief explanation>"
}}
Image is {tile_size}x{tile_size} px. Current pin is at center ({center},{center}).""",
}


# ── Low-level API helpers ──────────────────────────────────────────────────────

def _call_openai_with_prompt(
    image_path: Path,
    prompt: str,
    model: str,
    tile_size: int,
    run_label: str = "ground_truth_annotation",
) -> dict | None:
    """Make a single OpenAI vision call. Returns parsed JSON dict or None on failure."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed")
        return None

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    b64_image = _encode_image(image_path)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
            temperature=0.1,
        )
        from src.cost_tracker import log_usage
        usage = response.usage
        log_usage(model, usage.prompt_tokens, usage.completion_tokens, run_label=run_label)
        return _parse_json(response.choices[0].message.content.strip())
    except Exception as e:
        logger.warning(f"OpenAI call failed ({model}): {e}")
        return None


def _call_anthropic_with_prompt(
    image_path: Path,
    prompt: str,
    model: str,
    tile_size: int,
    run_label: str = "ground_truth_annotation",
) -> dict | None:
    """Make a single Anthropic vision call. Returns parsed JSON dict or None on failure."""
    try:
        from anthropic import Anthropic
    except ImportError:
        logger.error("anthropic package not installed")
        return None

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    b64_image = _encode_image(image_path)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64_image,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        from src.cost_tracker import log_usage
        log_usage(model, response.usage.input_tokens, response.usage.output_tokens, run_label=run_label)
        return _parse_json(response.content[0].text.strip())
    except Exception as e:
        logger.warning(f"Anthropic call failed ({model}): {e}")
        return None


def _call_provider(
    image_path: Path,
    prompt: str,
    provider: str,
    model: str,
    tile_size: int,
    run_label: str = "ground_truth_annotation",
) -> dict | None:
    """Route to the appropriate provider. Returns parsed JSON dict or None."""
    if provider == "openai":
        return _call_openai_with_prompt(image_path, prompt, model, tile_size, run_label)
    elif provider == "anthropic":
        return _call_anthropic_with_prompt(image_path, prompt, model, tile_size, run_label)
    else:
        raise ValueError(f"Unknown provider: {provider!r}")


# ── Public annotation functions ────────────────────────────────────────────────

def annotate_with_openai(image_path: Path, name: str, category: str,
                          address: str, tile_size: int = 640,
                          model: str = "gpt-4o", tier: int = 1) -> AnnotationResult:
    center = tile_size // 2
    prompt = ANNOTATION_PROMPTS[tier].format(
        name=name, category=category, address=address,
        tile_size=tile_size, center=center,
    )
    result = _call_openai_with_prompt(image_path, prompt, model, tile_size)
    if result is None:
        return AnnotationResult("", None, None, 0.0, "openai call failed", model)
    return AnnotationResult(
        place_id="",
        gt_lat=None, gt_lon=None,
        confidence=result.get("confidence", 0.5),
        reasoning=result.get("reasoning", ""),
        model=model,
        raw_response={"pixel_x": result["pixel_x"], "pixel_y": result["pixel_y"]},
    )


def annotate_with_anthropic(image_path: Path, name: str, category: str,
                             address: str, tile_size: int = 640,
                             model: str = "claude-sonnet-4-6", tier: int = 1) -> AnnotationResult:
    center = tile_size // 2
    prompt = ANNOTATION_PROMPTS[tier].format(
        name=name, category=category, address=address,
        tile_size=tile_size, center=center,
    )
    result = _call_anthropic_with_prompt(image_path, prompt, model, tile_size)
    if result is None:
        return AnnotationResult("", None, None, 0.0, "anthropic call failed", model)
    return AnnotationResult(
        place_id="",
        gt_lat=None, gt_lon=None,
        confidence=result.get("confidence", 0.5),
        reasoning=result.get("reasoning", ""),
        model=model,
        raw_response={"pixel_x": result["pixel_x"], "pixel_y": result["pixel_y"]},
    )


def annotate_place(image_path: Path, name: str, category: str, address: str,
                   lat_center: float, lon_center: float,
                   tile_size: int = 640, zoom: int = 18,
                   provider: str = "openai",
                   model: str | None = None,
                   tier: int | None = None) -> AnnotationResult:
    """
    Single-pin annotation pipeline: LLM pixel prediction → lat/lon conversion.
    tier: 1-4 per taxonomy; auto-derived from category if None.
    """
    if tier is None:
        tier, _ = get_tier(category)

    if provider == "openai":
        result = annotate_with_openai(image_path, name, category, address, tile_size,
                                      model=model or "gpt-4o-mini", tier=tier)
    elif provider == "anthropic":
        result = annotate_with_anthropic(image_path, name, category, address, tile_size,
                                         model=model or "claude-sonnet-4-6", tier=tier)
    else:
        raise ValueError(f"Unknown provider: {provider!r}")

    if result.raw_response and "pixel_x" in result.raw_response:
        from src.satellite_fetcher import pixel_to_latlon
        result.gt_lat, result.gt_lon = pixel_to_latlon(
            lat_center, lon_center,
            result.raw_response["pixel_x"],
            result.raw_response["pixel_y"],
            size=tile_size, zoom=zoom,
        )

    return result


def annotate_place_dual(
    image_path: Path,
    name: str,
    category: str,
    address: str,
    lat_center: float,
    lon_center: float,
    sub_tier: str,
    tile_size: int = 640,
    zoom: int = 18,
    provider: str = "openai",
    model: str | None = None,
) -> DualAnnotationResult:
    """
    Dual-label annotation for Tier 1/2 places: pedestrian entry + car entry.

    sub_tier controls which prompts are used and whether the car entry is expected
    to differ meaningfully from the pedestrian entry.

    Entry mode rules:
    - '1a' (drive-up only): car_only — no pedestrian approach expected
    - All other Tier 1/2 sub-tiers: both — two separate LLM calls
    - If distance(car, pedestrian) < 5m after annotation: collapsed to 'shared'
    """
    from src.satellite_fetcher import pixel_to_latlon

    tier_int = int(sub_tier[0])  # "1a" → 1, "2b" → 2
    _model = model or ("gpt-4o-mini" if provider == "openai" else "claude-sonnet-4-6")
    center = tile_size // 2

    # ── Drive-up-only: use car entry prompt, no pedestrian entry ──────────────
    if sub_tier == "1a":
        car_prompt = CAR_ENTRY_PROMPTS[1].format(
            name=name, category=category, address=address,
            tile_size=tile_size, center=center,
        )
        raw = _call_provider(image_path, car_prompt, provider, _model, tile_size)
        car_lat, car_lon = None, None
        if raw and "pixel_x" in raw:
            car_lat, car_lon = pixel_to_latlon(
                lat_center, lon_center, raw["pixel_x"], raw["pixel_y"],
                size=tile_size, zoom=zoom,
            )
        return DualAnnotationResult(
            place_id="",
            pedestrian_lat=None, pedestrian_lon=None,
            pedestrian_confidence=0.0, pedestrian_reasoning="drive_up_only",
            pedestrian_model=_model,
            car_lat=car_lat, car_lon=car_lon,
            car_confidence=raw.get("confidence", 0.0) if raw else 0.0,
            car_reasoning=raw.get("reasoning", "") if raw else "",
            car_model=_model,
            entry_mode="car_only",
        )

    # ── Standard dual-label: both calls fire simultaneously ───────────────────
    from concurrent.futures import ThreadPoolExecutor

    ped_prompt = ANNOTATION_PROMPTS[tier_int].format(
        name=name, category=category, address=address,
        tile_size=tile_size, center=center,
    )
    car_prompt_tier = min(tier_int, 2)
    car_prompt = CAR_ENTRY_PROMPTS[car_prompt_tier].format(
        name=name, category=category, address=address,
        tile_size=tile_size, center=center,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        ped_future = pool.submit(_call_provider, image_path, ped_prompt, provider, _model, tile_size)
        car_future = pool.submit(_call_provider, image_path, car_prompt, provider, _model, tile_size)
        ped_raw = ped_future.result()
        car_raw = car_future.result()

    ped_lat, ped_lon = None, None
    if ped_raw and "pixel_x" in ped_raw:
        ped_lat, ped_lon = pixel_to_latlon(
            lat_center, lon_center, ped_raw["pixel_x"], ped_raw["pixel_y"],
            size=tile_size, zoom=zoom,
        )

    car_lat, car_lon = None, None
    if car_raw and "pixel_x" in car_raw:
        car_lat, car_lon = pixel_to_latlon(
            lat_center, lon_center, car_raw["pixel_x"], car_raw["pixel_y"],
            size=tile_size, zoom=zoom,
        )

    # Collapse to 'shared' when entries are within 5m of each other
    entry_mode = "both"
    if ped_lat is not None and car_lat is not None:
        if haversine_meters(ped_lat, ped_lon, car_lat, car_lon) < 5:
            entry_mode = "shared"

    return DualAnnotationResult(
        place_id="",
        pedestrian_lat=ped_lat, pedestrian_lon=ped_lon,
        pedestrian_confidence=ped_raw.get("confidence", 0.0) if ped_raw else 0.0,
        pedestrian_reasoning=ped_raw.get("reasoning", "") if ped_raw else "",
        pedestrian_model=_model,
        car_lat=car_lat, car_lon=car_lon,
        car_confidence=car_raw.get("confidence", 0.0) if car_raw else 0.0,
        car_reasoning=car_raw.get("reasoning", "") if car_raw else "",
        car_model=_model,
        entry_mode=entry_mode,
    )
