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


TIER_MAP: dict[str, int] = {
    # Tier 1 — Standard Commercial (standalone building, single entrance)
    "accommodation": 1, "american_restaurant": 1, "antique_store": 1,
    "appliance_store": 1, "auto_parts_and_supply_store": 1, "bagel_shop": 1,
    "bakery": 1, "bar": 1, "bar_and_grill_restaurant": 1, "barbecue_restaurant": 1,
    "bed_and_breakfast": 1, "bicycle_shop": 1, "bike_repair_maintenance": 1,
    "bookstore": 1, "boutique": 1, "breakfast_and_brunch_restaurant": 1,
    "bridal_shop": 1, "bubble_tea": 1, "burger_restaurant": 1, "butcher_shop": 1,
    "cafe": 1, "cannabis_dispensary": 1, "caribbean_restaurant": 1,
    "chicken_restaurant": 1, "chicken_wings_restaurant": 1, "chinese_restaurant": 1,
    "clothing_store": 1, "coffee_shop": 1, "convenience_store": 1,
    "cosmetic_and_beauty_supplies": 1, "cuban_restaurant": 1, "delicatessen": 1,
    "department_store": 1, "desserts": 1, "diner": 1, "discount_store": 1,
    "distillery": 1, "donuts": 1, "drugstore": 1, "dry_cleaning": 1,
    "electronics": 1, "eyewear_and_optician": 1, "fast_food_restaurant": 1,
    "florist": 1, "flowers_and_gifts_shop": 1, "frozen_yoghurt_shop": 1,
    "furniture_store": 1, "gift_shop": 1, "grocery_store": 1, "gun_and_ammo": 1,
    "hardware_store": 1, "hair_supply_stores": 1, "hawaiian_restaurant": 1,
    "health_food_store": 1, "home_decor": 1, "home_goods_store": 1,
    "home_improvement_store": 1, "hostel": 1, "hotel": 1,
    "ice_cream_shop": 1, "indian_restaurant": 1, "inn": 1, "irish_pub": 1,
    "it_service_and_computer_repair": 1, "italian_restaurant": 1,
    "japanese_restaurant": 1, "jewelry_store": 1, "korean_restaurant": 1,
    "laundromat": 1, "liquor_store": 1, "lodge": 1, "lounge": 1,
    "lumber_store": 1, "mattress_store": 1, "meat_shop": 1, "medical_supply": 1,
    "mediterranean_restaurant": 1, "mexican_restaurant": 1,
    "mobile_phone_accessories": 1, "mobile_phone_repair": 1, "mobile_phone_store": 1,
    "motel": 1, "musical_instrument_store": 1, "nursery_and_gardening": 1,
    "outdoor_gear": 1, "paint_store": 1, "party_supply": 1, "pawn_shop": 1,
    "pet_store": 1, "pharmacy": 1, "pizza_restaurant": 1, "pub": 1,
    "resort": 1, "restaurant": 1, "retail": 1, "sandwich_shop": 1,
    "seafood_restaurant": 1, "self_storage_facility": 1, "shoe_store": 1,
    "smoothie_juice_bar": 1, "specialty_grocery_store": 1, "sporting_goods": 1,
    "sports_bar": 1, "steakhouse": 1, "storage_facility": 1, "supermarket": 1,
    "sushi_restaurant": 1, "tea_room": 1, "texmex_restaurant": 1,
    "thai_restaurant": 1, "thrift_store": 1, "tire_shop": 1, "tobacco_shop": 1,
    "turkish_restaurant": 1, "used_vintage_and_consignment": 1,
    "vegetarian_restaurant": 1, "vietnamese_restaurant": 1,
    "vitamins_and_supplements": 1, "wholesale_store": 1, "wine_bar": 1,
    "winery": 1,
    # Tier 2 — Multi-Tenant (unit within shared building or complex)
    "building_supply_store": 2, "child_care_and_day_care": 2,
    "day_care_preschool": 2, "hot_tubs_and_pools": 2, "laboratory": 2,
    "rental_kiosks": 2, "shipping_center": 2, "shopping": 2,
    "shopping_center": 2, "venue_and_event_space": 2,
    # Tier 3 — Open Spaces (parks, campgrounds, outdoor access points)
    "cabin": 3, "campground": 3, "cottage": 3, "farmers_market": 3,
    "food_stand": 3, "food_truck": 3, "funeral_services_and_cemeteries": 3,
    "holiday_rental_home": 3, "pop_up_shop": 3, "rv_park": 3, "taxidermist": 3,
    # Tier 4 — No Building (home business, mobile, service-only)
    "advertising_agency": 4, "bail_bonds_service": 4, "bankruptcy_law": 4,
    "business_law": 4, "caterer": 4, "commercial_printer": 4,
    "construction_services": 4, "courier_and_delivery_services": 4,
    "criminal_defense_law": 4, "data_recovery": 4, "divorce_and_family_law": 4,
    "employment_agencies": 4, "engineering_services": 4, "event_photography": 4,
    "event_planning": 4, "food_delivery_service": 4, "garbage_collection_service": 4,
    "graphic_designer": 4, "immigration_law": 4, "internet_service_provider": 4,
    "it_consultant": 4, "janitorial_services": 4, "lawyer": 4, "legal_services": 4,
    "life_coach": 4, "marketing_agency": 4, "notary_public": 4, "online_shop": 4,
    "personal_injury_law": 4, "pest_control_service": 4, "printing_services": 4,
    "professional_services": 4, "public_relations": 4, "security_services": 4,
    "septic_services": 4, "sewing_and_alterations": 4, "shredding_services": 4,
    "software_development": 4, "telecommunications": 4, "videographer": 4,
    "web_designer": 4, "wedding_planning": 4, "wills_trusts_and_probate": 4,
    "writing_service": 4,
}

TIER_LABELS: dict[int, str] = {
    1: "standard_commercial",
    2: "multi_tenant",
    3: "open_space",
    4: "no_building",
}


def get_tier(category: str | None) -> tuple[int, str]:
    """Return (tier_int, tier_label) for a given Overture Maps category_primary.
    Unknown/unmapped categories default to Tier 1 (standard commercial)."""
    tier = TIER_MAP.get((category or "").lower().strip(), 1)
    return tier, TIER_LABELS[tier]


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


def place_complexity(row) -> str:
    """
    Classify how difficult a place is likely to be for pin placement.

    simple: standalone/single obvious target
    multi_tenant: suite/unit/shared-building cases
    complex: large properties, campuses, resorts, plazas, open spaces
    """
    category = str(row.get("category_primary", "")).lower()
    name = str(row.get("name", "")).lower()
    address = str(row.get("full_address", "")).lower()

    complex_keywords = [
        "center", "centre", "plaza", "mall", "campus", "resort",
        "complex", "village", "marketplace", "casino", "park",
        "fairgrounds", "grounds", "station", "terminal",
    ]

    if any(keyword in name for keyword in complex_keywords):
        return "complex"

    if category in {
        "campground", "resort", "shopping_center", "hospital",
        "university", "rv_park", "golf_course", "fairgrounds",
        "airport", "stadium", "theme_park",
    }:
        return "complex"

    if "suite" in address or " ste " in address or " unit " in address or "#" in address:
        return "multi_tenant"

    return "simple"


def pin_ambiguity(row) -> str:
    """
    Estimate whether a place has one obvious pin or multiple plausible pin targets.
    """
    complexity = row.get("place_complexity") or place_complexity(row)
    tier = row.get("tier_label")
    category = str(row.get("category_primary", "")).lower()

    if tier == "no_building":
        return "high"

    if complexity == "complex":
        return "high"

    if complexity == "multi_tenant":
        return "medium"

    if category in {"hotel", "resort", "campground", "shopping_center"}:
        return "medium"

    return "low"


SUB_TIER_MAP: dict[str, str] = {
    # 1a — Drive-up only (car entry only, no meaningful pedestrian approach)
    "gas_station": "1a", "car_wash": "1a", "auto_repair": "1a",
    "car_dealer": "1a", "tire_shop": "1a",
    # 1b — Large standalone (car and pedestrian entries meaningfully far apart)
    "hotel": "1b", "motel": "1b", "accommodation": "1b", "resort": "1b",
    "lodge": "1b", "hostel": "1b", "inn": "1b", "bed_and_breakfast": "1b",
    "department_store": "1b", "grocery_store": "1b", "supermarket": "1b",
    "wholesale_store": "1b", "home_improvement_store": "1b",
    "furniture_store": "1b", "self_storage_facility": "1b",
    # 2a — Strip mall / retail center tenant
    "shopping_center": "2a", "shopping": "2a",
    # 2b — Office suite / professional building
    # Note: professional_services, lawyer, legal_services are Tier 4 in TIER_MAP
    # (no dedicated building) so they never reach this lookup — omitted intentionally.
    "dentist": "2b", "doctor": "2b", "health_and_medical": "2b",
    "day_care_preschool": "2b", "child_care_and_day_care": "2b",
    "laboratory": "2b", "shipping_center": "2b",
    # 3a — Bounded outdoor (small area, single access point)
    "farmers_market": "3a", "food_stand": "3a", "food_truck": "3a",
    "pop_up_shop": "3a", "funeral_services_and_cemeteries": "3a",
    # 3b — Large outdoor (campground, rural, extended access zones)
    "campground": "3b", "rv_park": "3b", "cabin": "3b",
    "cottage": "3b", "holiday_rental_home": "3b",
}

SUB_TIER_LABELS: dict[str, str] = {
    "1a": "drive_up_only",
    "1b": "large_standalone",
    "1c": "small_urban_storefront",
    "2a": "strip_mall_tenant",
    "2b": "office_suite",
    "2c": "other_multi_tenant",
    "3a": "bounded_outdoor",
    "3b": "large_outdoor",
    "4":  "no_building",
}


def get_sub_tier(category: str | None, name: str | None = None) -> tuple[str, str]:
    """Return (sub_tier_code, sub_tier_label) refining the 4-tier hierarchy.

    Sub-tiers encode which pin types to assign and how far apart car/pedestrian
    entries are expected to be, which drives the dual-label annotation strategy.
    """
    tier, _ = get_tier(category)
    cat = (category or "").lower().strip()
    name_lower = (name or "").lower()

    if tier == 4:
        return "4", SUB_TIER_LABELS["4"]

    sub = SUB_TIER_MAP.get(cat)
    if sub:
        return sub, SUB_TIER_LABELS[sub]

    if tier == 1:
        large_keywords = [
            "hotel", "inn", "lodge", "suites", "resort", "motel",
            "walmart", "target", "costco", "home depot", "lowe",
        ]
        if any(w in name_lower for w in large_keywords):
            return "1b", SUB_TIER_LABELS["1b"]
        return "1c", SUB_TIER_LABELS["1c"]

    if tier == 2:
        return "2c", SUB_TIER_LABELS["2c"]

    if tier == 3:
        if cat in {"campground", "rv_park", "cabin", "cottage", "holiday_rental_home"}:
            return "3b", SUB_TIER_LABELS["3b"]
        return "3a", SUB_TIER_LABELS["3a"]

    return str(tier), f"tier_{tier}"


def should_move_rule(row) -> bool:
    """
    Conservative rule for whether a pin should be considered movable.

    This protects already-good pins and avoids training a model that moves
    low-confidence or intentionally approximate records.
    """
    tier = row.get("tier_label")
    confidence = row.get("gt_confidence", 0)
    offset = row.get("offset_haversine_m", 0)

    if tier == "no_building":
        return False

    if confidence < 0.6:
        return False

    if offset > 10:
        return True

    if tier in {"open_space", "multi_tenant"} and offset > 5:
        return True

    return False
