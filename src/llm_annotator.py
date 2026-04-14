"""
LLM vision-based ground truth annotation.
Uses GPT-4o or Claude to examine satellite imagery and identify correct pin locations.
"""

import os
import json
import base64
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AnnotationResult:
    place_id: str
    gt_lat: float | None
    gt_lon: float | None
    confidence: float
    reasoning: str
    model: str
    raw_response: dict | None = None


ANNOTATION_PROMPT = """You are a geospatial analyst determining the correct pin location for a place on a map.

## Place Information
- **Name:** {name}
- **Category:** {category}
- **Address:** {address}

## Task
Look at this satellite imagery tile. The red marker shows where the pin is currently placed.

Determine the CORRECT location where this place's pin should be. Consider:
1. For businesses/shops: the main entrance or storefront facing the street
2. For hotels/resorts: the main lobby entrance
3. For restaurants/cafes: the primary customer entrance
4. For parks/campgrounds: the main access point or center of the area
5. For offices/professional services: the building entrance

## Response Format
Respond with ONLY a JSON object (no markdown, no explanation outside the JSON):
{{
    "pixel_x": <x pixel from left edge of image, 0-{tile_size}>,
    "pixel_y": <y pixel from top edge of image, 0-{tile_size}>,
    "confidence": <0.0 to 1.0>,
    "reasoning": "<brief explanation of why you chose this location>"
}}

The image is {tile_size}x{tile_size} pixels. The current pin is at the center ({center},{center}).
If you believe the current pin location is already correct, return pixel_x={center}, pixel_y={center}.
"""


def _encode_image(image_path: Path) -> str:
    """Encode image to base64 for API calls."""
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def annotate_with_openai(image_path: Path, name: str, category: str,
                          address: str, tile_size: int = 640,
                          model: str = "gpt-4o") -> AnnotationResult:
    """Use OpenAI's vision API to annotate a place."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed")
        return AnnotationResult("", None, None, 0.0, "openai not installed", model)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    center = tile_size // 2
    prompt = ANNOTATION_PROMPT.format(
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
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
            temperature=0.1,
        )
        content = response.choices[0].message.content.strip()
        # Parse JSON from response
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content)
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
                             model: str = "claude-sonnet-4-20250514") -> AnnotationResult:
    """Use Anthropic's vision API to annotate a place."""
    try:
        from anthropic import Anthropic
    except ImportError:
        logger.error("anthropic package not installed")
        return AnnotationResult("", None, None, 0.0, "anthropic not installed", model)

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    center = tile_size // 2
    prompt = ANNOTATION_PROMPT.format(
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
        content = response.content[0].text.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content)
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
                   provider: str = "openai") -> AnnotationResult:
    """
    Full annotation pipeline: get LLM pixel prediction, convert to lat/lon.
    """
    from src.satellite_fetcher import GoogleStaticMapFetcher

    if provider == "openai":
        result = annotate_with_openai(image_path, name, category, address, tile_size)
    elif provider == "anthropic":
        result = annotate_with_anthropic(image_path, name, category, address, tile_size)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if result.raw_response and "pixel_x" in result.raw_response:
        fetcher = GoogleStaticMapFetcher(zoom=zoom, size=tile_size)
        gt_lat, gt_lon = fetcher.pixel_to_latlon(
            lat_center, lon_center,
            result.raw_response["pixel_x"],
            result.raw_response["pixel_y"],
        )
        result.gt_lat = gt_lat
        result.gt_lon = gt_lon

    return result
