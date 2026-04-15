"""
Method 3: LLM-Augmented Contextual Reasoning for pin repositioning.
Uses LLM vision + place context to reason about correct pin placement.
"""

import os
import json
import base64
import logging
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from src.metrics import haversine_meters

logger = logging.getLogger(__name__)


REPOSITION_PROMPT = """You are a geospatial expert repositioning a place pin on a map for maximum accuracy.

## Place Information
- **Name:** {name}
- **Category:** {category}
- **Address:** {address}
- **Current confidence score:** {confidence}
- **Number of data sources:** {source_count}

## Geocoded Positions
{geocode_info}

## Task
Look at this satellite imagery tile. The red marker shows the CURRENT pin position.

Your job is to determine the BEST position for this place's pin. Reason step-by-step:

1. **Identify the building** — Which structure in the image matches this place?
2. **Consider the place type** — For a {category}:
   - If it's a restaurant/shop: pin should be near the customer entrance facing the street
   - If it's a hotel/resort: pin should be at the main lobby entrance
   - If it's an office/professional service: pin at the building entrance
   - If it's a park/campground: pin at the main access point or center
   - If it's in a strip mall or multi-tenant building: pin at the specific unit's entrance
3. **Evaluate the current pin** — Is it already well-placed, or should it move?
4. **Consider the geocoded positions** — Do they agree? Which seems most accurate?

## Response Format
Respond with ONLY a JSON object:
{{
    "pixel_x": <x pixel from left edge, 0-{tile_size}>,
    "pixel_y": <y pixel from top edge, 0-{tile_size}>,
    "confidence": <0.0 to 1.0>,
    "reasoning": "<step-by-step explanation>",
    "should_move": <true/false — whether the pin should move from current position>,
    "estimated_improvement_m": <estimated meters of improvement, 0 if should not move>
}}

Image is {tile_size}x{tile_size} pixels. Current pin is at center ({center},{center}).
"""


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


@dataclass
class RepositionResult:
    place_id: str
    new_lat: float | None
    new_lon: float | None
    confidence: float
    should_move: bool
    reasoning: str
    model: str


def reposition_single_openai(
    image_path: Path, name: str, category: str, address: str,
    confidence: float, source_count: int,
    geocode_info: str, lat_center: float, lon_center: float,
    tile_size: int = 640, zoom: int = 18,
    model: str = "gpt-4o",
) -> RepositionResult:
    """Reposition a single place using OpenAI vision."""
    try:
        from openai import OpenAI
    except ImportError:
        return RepositionResult("", None, None, 0.0, False, "openai not installed", model)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    center = tile_size // 2
    prompt = REPOSITION_PROMPT.format(
        name=name, category=category, address=address,
        confidence=confidence, source_count=source_count,
        geocode_info=geocode_info, tile_size=tile_size, center=center,
    )
    b64_image = _encode_image(image_path)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64_image}", "detail": "high",
                    }},
                ],
            }],
            max_tokens=800,
            temperature=0.1,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content)

        # Convert pixel to lat/lon
        from src.satellite_fetcher import GoogleStaticMapFetcher
        fetcher = GoogleStaticMapFetcher(zoom=zoom, size=tile_size)
        new_lat, new_lon = fetcher.pixel_to_latlon(
            lat_center, lon_center, result["pixel_x"], result["pixel_y"]
        )

        return RepositionResult(
            place_id="",
            new_lat=new_lat, new_lon=new_lon,
            confidence=result.get("confidence", 0.5),
            should_move=result.get("should_move", True),
            reasoning=result.get("reasoning", ""),
            model=model,
        )
    except Exception as e:
        logger.warning(f"OpenAI repositioning failed: {e}")
        return RepositionResult("", None, None, 0.0, False, str(e), model)


def reposition_single_anthropic(
    image_path: Path, name: str, category: str, address: str,
    confidence: float, source_count: int,
    geocode_info: str, lat_center: float, lon_center: float,
    tile_size: int = 640, zoom: int = 18,
    model: str = "claude-sonnet-4-6",
) -> RepositionResult:
    """Reposition a single place using Anthropic vision."""
    try:
        from anthropic import Anthropic
    except ImportError:
        return RepositionResult("", None, None, 0.0, False, "anthropic not installed", model)

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    center = tile_size // 2
    prompt = REPOSITION_PROMPT.format(
        name=name, category=category, address=address,
        confidence=confidence, source_count=source_count,
        geocode_info=geocode_info, tile_size=tile_size, center=center,
    )
    b64_image = _encode_image(image_path)

    try:
        response = client.messages.create(
            model=model, max_tokens=800,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": b64_image,
                    }},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        content = response.content[0].text.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content)

        from src.satellite_fetcher import GoogleStaticMapFetcher
        fetcher = GoogleStaticMapFetcher(zoom=zoom, size=tile_size)
        new_lat, new_lon = fetcher.pixel_to_latlon(
            lat_center, lon_center, result["pixel_x"], result["pixel_y"]
        )

        return RepositionResult(
            place_id="",
            new_lat=new_lat, new_lon=new_lon,
            confidence=result.get("confidence", 0.5),
            should_move=result.get("should_move", True),
            reasoning=result.get("reasoning", ""),
            model=model,
        )
    except Exception as e:
        logger.warning(f"Anthropic repositioning failed: {e}")
        return RepositionResult("", None, None, 0.0, False, str(e), model)


def reposition_with_llm(
    df: pd.DataFrame,
    tiles_dir: Path | None = None,
    geocode_results_map: dict | None = None,
    provider: str = "openai",
    max_places: int | None = None,
) -> pd.DataFrame:
    """
    Reposition places using LLM contextual reasoning.

    Adds columns: llm_lat, llm_lon, llm_confidence, llm_should_move, llm_reasoning
    """
    from src.satellite_fetcher import TILES_DIR
    tiles_dir = tiles_dir or TILES_DIR

    results = []
    process_df = df.head(max_places) if max_places else df

    for idx, row in process_df.iterrows():
        place_id = row["id"]
        tile_path = tiles_dir / f"{place_id}.png"

        if not tile_path.exists():
            results.append({
                "llm_lat": None, "llm_lon": None,
                "llm_confidence": 0.0, "llm_should_move": False,
                "llm_reasoning": "no_tile",
            })
            continue

        # Build geocode info string
        geocode_info = "No geocoded positions available."
        if geocode_results_map and place_id in geocode_results_map:
            geo_lines = []
            for gr in geocode_results_map[place_id]:
                geo_lines.append(f"- {gr.get('source', 'unknown')}: "
                                 f"lat={gr.get('lat'):.6f}, lon={gr.get('lon'):.6f}")
            if geo_lines:
                geocode_info = "\n".join(geo_lines)

        reposition_fn = (reposition_single_openai if provider == "openai"
                         else reposition_single_anthropic)

        result = reposition_fn(
            image_path=tile_path,
            name=row.get("name") or "Unknown",
            category=row.get("category_primary") or "place",
            address=row.get("full_address", ""),
            confidence=row.get("confidence", 0.5),
            source_count=row.get("source_count", 1),
            geocode_info=geocode_info,
            lat_center=row["lat"],
            lon_center=row["lon"],
        )
        result.place_id = place_id

        results.append({
            "llm_lat": result.new_lat,
            "llm_lon": result.new_lon,
            "llm_confidence": result.confidence,
            "llm_should_move": result.should_move,
            "llm_reasoning": result.reasoning,
        })

        if (idx + 1) % 10 == 0:
            logger.info(f"LLM repositioned {idx + 1}/{len(process_df)} places")

    result_df = pd.DataFrame(results, index=process_df.index)
    return pd.concat([process_df, result_df], axis=1)
