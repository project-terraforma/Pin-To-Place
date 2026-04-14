"""
Ground truth construction pipeline.
Orchestrates stratified sampling, satellite tile fetching, LLM annotation,
and cross-validation with multi-geocoder consensus.
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.data_loader import load_places
from src.satellite_fetcher import GoogleStaticMapFetcher
from src.llm_annotator import annotate_place
from src.geocoder import MultiGeocoder
from src.metrics import haversine_meters

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def stratified_sample(df: pd.DataFrame, n: int = 750,
                       region_col: str = "region",
                       category_col: str = "category_primary",
                       random_state: int = 42) -> pd.DataFrame:
    """
    Select a stratified sample of places.
    Stratifies by region and category proportionally, with minimum 1 per stratum.
    """
    # Create strata
    df = df.copy()
    df["_stratum"] = df[region_col].fillna("UNK") + "_" + df[category_col].fillna("UNK")

    # Proportional allocation with minimum 1 per stratum
    stratum_counts = df["_stratum"].value_counts()
    total = len(df)
    allocation = {}
    for stratum, count in stratum_counts.items():
        allocation[stratum] = max(1, int(round(count / total * n)))

    # Adjust to hit target n
    allocated_total = sum(allocation.values())
    if allocated_total > n:
        # Remove from largest strata
        for stratum in sorted(allocation, key=allocation.get, reverse=True):
            if allocated_total <= n:
                break
            if allocation[stratum] > 1:
                allocation[stratum] -= 1
                allocated_total -= 1

    sampled = []
    rng = np.random.RandomState(random_state)
    for stratum, target_n in allocation.items():
        stratum_df = df[df["_stratum"] == stratum]
        actual_n = min(target_n, len(stratum_df))
        sampled.append(stratum_df.sample(n=actual_n, random_state=rng))

    result = pd.concat(sampled).drop(columns=["_stratum"])
    logger.info(f"Sampled {len(result)} places from {len(stratum_counts)} strata (target: {n})")
    return result


def build_ground_truth(
    df: pd.DataFrame | None = None,
    sample_n: int = 750,
    provider: str = "openai",
    google_maps_key: str | None = None,
    output_path: Path | None = None,
    max_places: int | None = None,
) -> pd.DataFrame:
    """
    Full ground truth construction pipeline:
    1. Stratified sample
    2. Fetch satellite tiles
    3. LLM vision annotation
    4. Save results

    Args:
        df: Input DataFrame (loads from parquet if None)
        sample_n: Number of places to sample
        provider: LLM provider ("openai" or "anthropic")
        google_maps_key: Google Maps API key for satellite tiles
        output_path: Where to save results
        max_places: Limit processing (for testing)
    """
    if df is None:
        df = load_places()

    output_path = output_path or (PROCESSED_DIR / "ground_truth.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Stratified sample
    sample_df = stratified_sample(df, n=sample_n)
    if max_places:
        sample_df = sample_df.head(max_places)

    # Step 2: Fetch satellite tiles
    fetcher = GoogleStaticMapFetcher(api_key=google_maps_key)

    results = []
    for idx, row in sample_df.iterrows():
        place_id = row["id"]
        logger.info(f"Processing {place_id}: {row['name']} ({row['category_primary']})")

        # Fetch tile
        tile_path = fetcher.fetch_tile(row["lat"], row["lon"], place_id)
        if tile_path is None:
            logger.warning(f"Skipping {place_id}: no tile fetched")
            continue

        # Step 3: LLM annotation
        annotation = annotate_place(
            image_path=tile_path,
            name=row["name"] or "Unknown",
            category=row["category_primary"] or "place",
            address=row["full_address"],
            lat_center=row["lat"],
            lon_center=row["lon"],
            provider=provider,
        )

        results.append({
            "id": place_id,
            "name": row["name"],
            "category_primary": row["category_primary"],
            "region": row["region"],
            "current_lat": row["lat"],
            "current_lon": row["lon"],
            "gt_lat": annotation.gt_lat,
            "gt_lon": annotation.gt_lon,
            "gt_confidence": annotation.confidence,
            "gt_reasoning": annotation.reasoning,
            "gt_model": annotation.model,
            "full_address": row["full_address"],
        })

    result_df = pd.DataFrame(results)

    # Compute offset
    valid = result_df.dropna(subset=["gt_lat", "gt_lon"])
    if len(valid) > 0:
        result_df.loc[valid.index, "offset_m"] = valid.apply(
            lambda r: haversine_meters(r["current_lat"], r["current_lon"], r["gt_lat"], r["gt_lon"]),
            axis=1,
        )

    result_df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(result_df)} ground truth records to {output_path}")

    return result_df


def cross_validate_with_geocoders(
    gt_df: pd.DataFrame,
    n_samples: int = 100,
    google_key: str | None = None,
    mapbox_key: str | None = None,
) -> pd.DataFrame:
    """
    Cross-validate LLM ground truth against multi-geocoder consensus.
    Returns a DataFrame with geocoder results and agreement metrics.
    """
    sample = gt_df.dropna(subset=["gt_lat", "gt_lon"]).sample(
        n=min(n_samples, len(gt_df)), random_state=42
    )

    multi_geocoder = MultiGeocoder(google_key=google_key, mapbox_key=mapbox_key)
    results = []

    for _, row in sample.iterrows():
        geocode_results = multi_geocoder.geocode_all(row["full_address"])

        geo_positions = [(r.lat, r.lon, r.source) for r in geocode_results if r.lat is not None]
        if not geo_positions:
            continue

        # Compute consensus (median position)
        consensus_lat = np.median([p[0] for p in geo_positions])
        consensus_lon = np.median([p[1] for p in geo_positions])

        # Distance from LLM ground truth to geocoder consensus
        llm_vs_consensus = haversine_meters(
            row["gt_lat"], row["gt_lon"], consensus_lat, consensus_lon
        )

        # Distance from current pin to geocoder consensus
        current_vs_consensus = haversine_meters(
            row["current_lat"], row["current_lon"], consensus_lat, consensus_lon
        )

        results.append({
            "id": row["id"],
            "name": row["name"],
            "gt_lat": row["gt_lat"],
            "gt_lon": row["gt_lon"],
            "consensus_lat": consensus_lat,
            "consensus_lon": consensus_lon,
            "llm_vs_consensus_m": round(llm_vs_consensus, 2),
            "current_vs_consensus_m": round(current_vs_consensus, 2),
            "n_geocoders": len(geo_positions),
            "geocoder_sources": [p[2] for p in geo_positions],
        })

    return pd.DataFrame(results)
