"""
Method 1: Multi-Geocoder Ensemble repositioning.
Queries multiple geocoding services and computes a consensus position.
"""

import logging

import numpy as np
import pandas as pd

from src.geocoder import MultiGeocoder, GeocodingResult
from src.metrics import haversine_meters

logger = logging.getLogger(__name__)


def compute_consensus(results: list[GeocodingResult],
                      agreement_radius_m: float = 25.0) -> dict:
    """
    Compute a consensus position from multiple geocoding results.

    Strategy:
    1. If all results agree within agreement_radius_m, use weighted average.
    2. If there's disagreement, use the median position.
    3. Flag when results diverge significantly.

    Returns dict with lat, lon, confidence, method, agreement_m, n_sources.
    """
    valid = [r for r in results if r.lat is not None]
    if not valid:
        return {"lat": None, "lon": None, "confidence": 0.0,
                "method": "none", "agreement_m": None, "n_sources": 0}

    if len(valid) == 1:
        r = valid[0]
        return {"lat": r.lat, "lon": r.lon, "confidence": r.confidence,
                "method": f"single_{r.source}", "agreement_m": 0.0, "n_sources": 1}

    lats = [r.lat for r in valid]
    lons = [r.lon for r in valid]
    weights = [r.confidence for r in valid]

    # Compute pairwise max distance to measure agreement
    max_dist = 0.0
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            d = haversine_meters(lats[i], lons[i], lats[j], lons[j])
            max_dist = max(max_dist, d)

    if max_dist <= agreement_radius_m:
        # Good agreement — use confidence-weighted average
        total_w = sum(weights)
        if total_w == 0:
            total_w = len(weights)
            weights = [1.0] * len(weights)
        avg_lat = sum(w * lat for w, lat in zip(weights, lats)) / total_w
        avg_lon = sum(w * lon for w, lon in zip(weights, lons)) / total_w
        return {
            "lat": avg_lat, "lon": avg_lon,
            "confidence": min(1.0, sum(weights) / len(weights) * 1.2),
            "method": "weighted_average",
            "agreement_m": round(max_dist, 2),
            "n_sources": len(valid),
        }
    else:
        # Disagreement — use median (more robust to outliers)
        med_lat = float(np.median(lats))
        med_lon = float(np.median(lons))
        return {
            "lat": med_lat, "lon": med_lon,
            "confidence": max(0.3, sum(weights) / len(weights) * 0.7),
            "method": "median",
            "agreement_m": round(max_dist, 2),
            "n_sources": len(valid),
        }


def reposition_with_ensemble(
    df: pd.DataFrame,
    google_key: str | None = None,
    mapbox_key: str | None = None,
    agreement_radius_m: float = 25.0,
) -> pd.DataFrame:
    """
    Reposition all places using multi-geocoder ensemble.

    Adds columns: ensemble_lat, ensemble_lon, ensemble_confidence,
                  ensemble_method, ensemble_agreement_m
    """
    multi_geocoder = MultiGeocoder(google_key=google_key, mapbox_key=mapbox_key)

    results = []
    for idx, row in df.iterrows():
        address = row.get("full_address", "")
        if not address:
            results.append({
                "ensemble_lat": None, "ensemble_lon": None,
                "ensemble_confidence": 0.0, "ensemble_method": "no_address",
                "ensemble_agreement_m": None,
            })
            continue

        geocode_results = multi_geocoder.geocode_all(address)
        consensus = compute_consensus(geocode_results, agreement_radius_m)

        results.append({
            "ensemble_lat": consensus["lat"],
            "ensemble_lon": consensus["lon"],
            "ensemble_confidence": consensus["confidence"],
            "ensemble_method": consensus["method"],
            "ensemble_agreement_m": consensus["agreement_m"],
        })

        if (idx + 1) % 100 == 0:
            logger.info(f"Geocoded {idx + 1}/{len(df)} places")

    result_df = pd.DataFrame(results, index=df.index)
    return pd.concat([df, result_df], axis=1)
