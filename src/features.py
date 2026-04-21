"""
Feature extraction for ML candidate ranking (Method 2).
Generates candidate positions and extracts features for each.
"""

import logging
from math import radians, cos

import numpy as np
import pandas as pd

from src.metrics import haversine_meters

logger = logging.getLogger(__name__)

# Category groups for one-hot encoding
CATEGORY_GROUPS = {
    "food": ["restaurant", "fast_food_restaurant", "pizza_restaurant", "mexican_restaurant",
             "american_restaurant", "coffee_shop", "cafe", "bakery", "bar"],
    "lodging": ["hotel", "accommodation", "resort", "campground", "motel"],
    "retail": ["shopping", "convenience_store", "discount_store", "mobile_phone_store",
               "grocery_store", "clothing_store"],
    "services": ["professional_services", "lawyer", "printing_services", "self_storage_facility",
                 "day_care_preschool", "event_planning"],
    "health": ["pharmacy", "health_and_medical", "dentist", "doctor"],
    "auto": ["gas_station", "auto_repair", "car_dealer"],
}


def categorize_place(category: str | None) -> str:
    """Map a specific category to a broader group."""
    if not category:
        return "other"
    for group, categories in CATEGORY_GROUPS.items():
        if category in categories or any(c in (category or "") for c in categories):
            return group
    return "other"


def generate_candidates(row: dict,
                        geocode_results: list[dict] | None = None,
                        building_centroids: list[tuple[float, float]] | None = None,
                        ) -> list[dict]:
    """
    Generate candidate positions for a place.

    Each candidate is a dict with:
        lat, lon, source (str describing where this candidate came from)
    """
    candidates = []

    # Candidate 0: current pin position (baseline)
    candidates.append({
        "lat": row["lat"], "lon": row["lon"],
        "source": "current_pin",
    })

    # Candidates from geocoding results
    if geocode_results:
        for gr in geocode_results:
            if gr.get("lat") is not None:
                candidates.append({
                    "lat": gr["lat"], "lon": gr["lon"],
                    "source": f"geocode_{gr.get('source', 'unknown')}",
                })

    # Candidates from building centroids
    if building_centroids:
        for i, (blat, blon) in enumerate(building_centroids):
            candidates.append({
                "lat": blat, "lon": blon,
                "source": f"building_{i}",
            })

    return candidates


def extract_candidate_features(candidate: dict, row: dict,
                                all_candidates: list[dict]) -> dict:
    """
    Extract features for a single candidate position.
    """
    clat, clon = candidate["lat"], candidate["lon"]
    plat, plon = row["lat"], row["lon"]

    # Distance from current pin
    dist_from_pin = haversine_meters(plat, plon, clat, clon)

    # Distance from geocoded positions (if any)
    geocode_dists = []
    for c in all_candidates:
        if c["source"].startswith("geocode_"):
            geocode_dists.append(haversine_meters(clat, clon, c["lat"], c["lon"]))
    avg_geocode_dist = np.mean(geocode_dists) if geocode_dists else -1.0

    # Distance from building centroids (if any)
    building_dists = []
    for c in all_candidates:
        if c["source"].startswith("building_"):
            building_dists.append(haversine_meters(clat, clon, c["lat"], c["lon"]))
    min_building_dist = min(building_dists) if building_dists else -1.0
    n_buildings = len(building_dists)

    # Category features
    category_group = categorize_place(row.get("category_primary"))

    features = {
        "dist_from_pin_m": dist_from_pin,
        "avg_geocode_dist_m": avg_geocode_dist,
        "min_building_dist_m": min_building_dist,
        "n_buildings_nearby": n_buildings,
        "n_geocoders": len(geocode_dists),
        "confidence": row.get("confidence", 0.5),
        "source_count": row.get("source_count", 1),
        "is_current_pin": 1 if candidate["source"] == "current_pin" else 0,
        "is_geocode": 1 if candidate["source"].startswith("geocode_") else 0,
        "is_building": 1 if candidate["source"].startswith("building_") else 0,
        # Category one-hot
        "cat_food": 1 if category_group == "food" else 0,
        "cat_lodging": 1 if category_group == "lodging" else 0,
        "cat_retail": 1 if category_group == "retail" else 0,
        "cat_services": 1 if category_group == "services" else 0,
        "cat_health": 1 if category_group == "health" else 0,
        "cat_auto": 1 if category_group == "auto" else 0,
    }

    return features


def build_training_data(gt_df: pd.DataFrame,
                         geocode_results_map: dict | None = None,
                         building_centroids_map: dict | None = None,
                         ) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build training data for the candidate ranking model.

    For each ground-truth place, generates candidates and labels
    the candidate closest to the ground truth as positive (1).

    Returns (features_df, labels_series).
    """
    all_features = []
    all_labels = []

    for _, row in gt_df.iterrows():
        place_id = row["id"]
        gt_lat, gt_lon = row.get("gt_lat"), row.get("gt_lon")

        if gt_lat is None or gt_lon is None:
            continue

        geocode_results = geocode_results_map.get(place_id, []) if geocode_results_map else []
        building_centroids = building_centroids_map.get(place_id, []) if building_centroids_map else []

        candidates = generate_candidates(row, geocode_results, building_centroids)
        if not candidates:
            continue

        # Find closest candidate to ground truth
        dists_to_gt = [
            haversine_meters(c["lat"], c["lon"], gt_lat, gt_lon)
            for c in candidates
        ]
        best_idx = int(np.argmin(dists_to_gt))

        for i, candidate in enumerate(candidates):
            features = extract_candidate_features(candidate, row, candidates)
            features["place_id"] = place_id
            features["candidate_lat"] = candidate["lat"]
            features["candidate_lon"] = candidate["lon"]
            features["candidate_source"] = candidate["source"]
            all_features.append(features)
            all_labels.append(1 if i == best_idx else 0)

    features_df = pd.DataFrame(all_features)
    labels = pd.Series(all_labels, name="label")

    logger.info(f"Built training data: {len(features_df)} candidates, "
                f"{labels.sum()} positive, {(~labels.astype(bool)).sum()} negative")

    return features_df, labels
