"""
Method 3: LLM-Augmented Contextual Reasoning for pin repositioning.
Uses LLM vision + place context to reason about correct pin placement.
"""

import os
import json
import re
import base64
import logging
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from src.metrics import haversine_meters
from src.features import get_tier

logger = logging.getLogger(__name__)


_REPOSITION_SHARED = """## Place Information
- **Name:** {name}
- **Category:** {category}
- **Address:** {address}
- **Current confidence score:** {confidence}
- **Number of data sources:** {source_count}

## Geocoded Positions
{geocode_info}

{tier_task}

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

Image is {tile_size}x{tile_size} pixels. Current pin is at center ({center},{center})."""

_REPOSITION_TIER_TASKS = {
    1: """## Task — Standard Commercial Place
You are a geospatial expert repositioning a pin for a {category} business.
Look at the satellite tile. The red marker shows the CURRENT pin.

1. Identify which building matches this place.
2. Find the main customer entrance facing the street (awnings, signage, doorway).
3. Evaluate whether the current pin is already at the entrance.
4. Consider whether geocoded positions confirm the building identity.

The ideal pin is at the street-facing storefront door — not the building center or parking lot.""",

    2: """## Task — Multi-Tenant Building
You are a geospatial expert repositioning a pin for a {category} in a shared building.
Look at the satellite tile. The red marker shows the CURRENT pin.

1. Identify the multi-tenant building (strip mall, office suite, shopping center).
2. Find the specific unit for this place within the building.
3. Pin should be at the individual unit's entrance — NOT the building center or shared parking entry.
4. Use geocoded positions to help identify the unit's address portion.""",

    3: """## Task — Open Space / Outdoor Area
You are a geospatial expert repositioning a pin for a {category} open space.
Look at the satellite tile. The red marker shows the CURRENT pin.

1. Identify the open space or park boundary.
2. Find the main access point: parking lot entrance, gate, or trailhead.
3. If no clear access point is visible, use the area's visual center.
4. Do not place the pin inside a building unless it is the main visitor entry structure.""",

    4: """## Task — No Dedicated Building
You are a geospatial expert repositioning a pin for a {category} with no dedicated building.
Look at the satellite tile. The red marker shows the CURRENT pin.

1. Check whether a visible standalone commercial structure exists at this address.
2. If yes, pin at the entrance (treat as Tier 1).
3. If no commercial building is visible, the current pin is likely the best estimate —
   set should_move=false and confidence below 0.4.""",
}


def _build_reposition_prompt(tier: int, **kwargs) -> str:
    tier_task = _REPOSITION_TIER_TASKS[tier].format(**kwargs)
    return _REPOSITION_SHARED.format(tier_task=tier_task, **kwargs)


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
    model: str = "gpt-4o", tier: int = 1,
) -> RepositionResult:
    """Reposition a single place using OpenAI vision."""
    try:
        from openai import OpenAI
    except ImportError:
        return RepositionResult("", None, None, 0.0, False, "openai not installed", model)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    center = tile_size // 2
    prompt = _build_reposition_prompt(
        tier, name=name, category=category, address=address,
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
        from src.cost_tracker import log_usage
        from src.satellite_fetcher import pixel_to_latlon
        usage = response.usage
        log_usage(model, usage.prompt_tokens, usage.completion_tokens, run_label="llm_repositioning")

        content = response.choices[0].message.content.strip()
        match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        result = json.loads(match.group(0) if match else content)

        new_lat, new_lon = pixel_to_latlon(
            lat_center, lon_center, result["pixel_x"], result["pixel_y"], tile_size, zoom
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
    model: str = "claude-sonnet-4-6", tier: int = 1,
) -> RepositionResult:
    """Reposition a single place using Anthropic vision."""
    try:
        from anthropic import Anthropic
    except ImportError:
        return RepositionResult("", None, None, 0.0, False, "anthropic not installed", model)

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    center = tile_size // 2
    prompt = _build_reposition_prompt(
        tier, name=name, category=category, address=address,
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
        from src.cost_tracker import log_usage
        from src.satellite_fetcher import pixel_to_latlon
        log_usage(model, response.usage.input_tokens, response.usage.output_tokens, run_label="llm_repositioning")

        content = response.content[0].text.strip()
        match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        result = json.loads(match.group(0) if match else content)

        new_lat, new_lon = pixel_to_latlon(
            lat_center, lon_center, result["pixel_x"], result["pixel_y"], tile_size, zoom
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

        # Use tier from ground truth if available, otherwise derive from category
        if "tier" in row and row["tier"] is not None:
            tier = int(row["tier"])
        else:
            tier, _ = get_tier(row.get("category_primary"))

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
            tier=tier,
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
