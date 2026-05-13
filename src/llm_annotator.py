"""
LLM vision-based ground truth annotation.
Uses GPT-4o or Claude to examine satellite imagery and identify correct pin locations.
"""

import os
import json
import re
import base64
import logging
from pathlib import Path
from dataclasses import dataclass

from src.features import get_tier

logger = logging.getLogger(__name__)


def _parse_json(content: str) -> dict:
    """Extract a JSON object from an LLM response regardless of surrounding text."""
    content = content.strip()
    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # Find first {...} block (handles markdown code fences and extra explanation)
    match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON object found in LLM response: {content[:200]!r}")


@dataclass
class AnnotationResult:
    place_id: str
    gt_lat: float | None
    gt_lon: float | None
    confidence: float
    reasoning: str
    model: str
    raw_response: dict | None = None


ANNOTATION_PROMPTS = {
    1: """You are a geospatial analyst placing a map pin for a {category} business.

## Place Information
- **Name:** {name}
- **Category:** {category}
- **Address:** {address}

## Task
The red marker shows the current pin. Identify the CORRECT pin location.

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
The red marker shows the current pin. Identify the CORRECT pin location.

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


def _encode_image(image_path: Path) -> str:
    """Encode image to base64 for API calls."""
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def annotate_with_openai(image_path: Path, name: str, category: str,
                          address: str, tile_size: int = 640,
                          model: str = "gpt-4o", tier: int = 1) -> AnnotationResult:
    """Use OpenAI's vision API to annotate a place."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed")
        return AnnotationResult("", None, None, 0.0, "openai not installed", model)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    center = tile_size // 2
    prompt = ANNOTATION_PROMPTS[tier].format(
        name=name, category=category, address=address,
        tile_size=tile_size, center=center,
    )
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
        log_usage(model, usage.prompt_tokens, usage.completion_tokens, run_label="ground_truth_annotation")

        content = response.choices[0].message.content.strip()
        result = _parse_json(content)
        return AnnotationResult(
            place_id="",
            gt_lat=None,  # Will be filled by caller after pixel->latlon conversion
            gt_lon=None,
            confidence=result.get("confidence", 0.5),
            reasoning=result.get("reasoning", ""),
            model=model,
            raw_response={"pixel_x": result["pixel_x"], "pixel_y": result["pixel_y"]},
        )
    except Exception as e:
        logger.warning(f"OpenAI annotation failed: {e}")
        return AnnotationResult("", None, None, 0.0, str(e), model)


def annotate_with_anthropic(image_path: Path, name: str, category: str,
                             address: str, tile_size: int = 640,
                             model: str = "claude-sonnet-4-6", tier: int = 1) -> AnnotationResult:
    """Use Anthropic's vision API to annotate a place."""
    try:
        from anthropic import Anthropic
    except ImportError:
        logger.error("anthropic package not installed")
        return AnnotationResult("", None, None, 0.0, "anthropic not installed", model)

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    center = tile_size // 2
    prompt = ANNOTATION_PROMPTS[tier].format(
        name=name, category=category, address=address,
        tile_size=tile_size, center=center,
    )
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
        log_usage(model, response.usage.input_tokens, response.usage.output_tokens, run_label="ground_truth_annotation")

        content = response.content[0].text.strip()
        result = _parse_json(content)
        return AnnotationResult(
            place_id="",
            gt_lat=None,
            gt_lon=None,
            confidence=result.get("confidence", 0.5),
            reasoning=result.get("reasoning", ""),
            model=model,
            raw_response={"pixel_x": result["pixel_x"], "pixel_y": result["pixel_y"]},
        )
    except Exception as e:
        logger.warning(f"Anthropic annotation failed: {e}")
        return AnnotationResult("", None, None, 0.0, str(e), model)


def annotate_place(image_path: Path, name: str, category: str, address: str,
                   lat_center: float, lon_center: float,
                   tile_size: int = 640, zoom: int = 18,
                   provider: str = "openai",
                   model: str | None = None,
                   tier: int | None = None) -> AnnotationResult:
    """
    Full annotation pipeline: get LLM pixel prediction, convert to lat/lon.
    model overrides the default for the chosen provider.
    tier: 1-4 per taxonomy; if None, auto-derived from category.
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
        raise ValueError(f"Unknown provider: {provider}")

    if result.raw_response and "pixel_x" in result.raw_response:
        from src.satellite_fetcher import pixel_to_latlon
        result.gt_lat, result.gt_lon = pixel_to_latlon(
            lat_center, lon_center,
            result.raw_response["pixel_x"],
            result.raw_response["pixel_y"],
            size=tile_size, zoom=zoom,
        )

    return result
