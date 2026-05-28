"""
Ground truth construction pipeline.
Orchestrates stratified sampling, satellite tile fetching, LLM annotation,
and cross-validation with multi-geocoder consensus.

New in this version:
- sub_tier column on every row (from get_sub_tier)
- dual_label=True: Tier 1/2 places get car_entry + pedestrian_entry coordinates
- inter_annotator_n: first N Tier-1 places in the batch are re-annotated with a
  second provider for inter-annotator agreement (set to 50 in the first batch only)
- llm_workers: number of parallel annotation threads (default 3, ~3x speedup)
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import numpy as np

from src.data_loader import load_places
from src.satellite_fetcher import GoogleStaticMapFetcher, MapboxStaticFetcher, ESRIStaticFetcher
from src.llm_annotator import annotate_place, annotate_place_dual
from src.geocoder import MultiGeocoder
from src.metrics import haversine_meters, euclidean_meters, manhattan_meters
from src.features import get_tier, get_sub_tier

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

COLUMNS = [
    "id", "name", "category_primary", "region",
    "tier", "tier_label",
    "sub_tier", "sub_tier_label",
    "current_lat", "current_lon",
    "gt_lat", "gt_lon", "gt_confidence", "gt_reasoning", "gt_model",
    "pedestrian_entry_lat", "pedestrian_entry_lon", "pedestrian_entry_confidence",
    "car_entry_lat", "car_entry_lon", "car_entry_confidence", "car_entry_reasoning",
    "entry_mode",
    "cv_gt_lat", "cv_gt_lon", "cv_confidence", "cv_model", "cv_disagreement_m",
    "full_address",
    "offset_haversine_m", "offset_euclidean_m", "offset_manhattan_m",
]


def stratified_sample(df: pd.DataFrame, n: int = 750,
                       region_col: str = "region",
                       category_col: str = "category_primary",
                       random_state: int = 42) -> pd.DataFrame:
    df = df.copy()
    df["_stratum"] = df[region_col].fillna("UNK") + "_" + df[category_col].fillna("UNK")

    stratum_counts = df["_stratum"].value_counts()
    total = len(df)
    allocation = {}
    for stratum, count in stratum_counts.items():
        allocation[stratum] = max(1, int(round(count / total * n)))

    allocated_total = sum(allocation.values())
    if allocated_total > n:
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
    if len(result) > n:
        result = result.sample(n=n, random_state=random_state)
    print(f"Sampled {len(result)} places from {len(stratum_counts)} strata (target: {n})")
    return result


def _annotate_one(
    row: pd.Series,
    tile_path: Path,
    provider: str,
    model: str,
    dual_label: bool,
    cv_ids: frozenset,
    cv_provider: str,
    cv_model: str,
    idx: int,
    n_total: int,
    tile_size: int = 512,
) -> dict | None:
    """
    Annotate a single place. Returns a record dict or None on failure.
    This function is thread-safe — it does not mutate any shared state.
    """
    place_id = row["id"]
    tier_int, tier_label = get_tier(row.get("category_primary"))
    sub_tier, sub_tier_label = get_sub_tier(row.get("category_primary"), row.get("name"))

    print(f"[{idx}/{n_total}] {row['name']} ({row['category_primary']}, {row['region']}) "
          f"[{sub_tier}: {sub_tier_label}]")

    ped_lat = ped_lon = ped_conf = None
    ped_reasoning = ""
    car_lat = car_lon = car_conf = None
    car_reasoning = ""
    entry_mode = None
    gt_lat = gt_lon = gt_conf = None
    gt_reasoning = ""
    gt_model_name = model

    try:
        if dual_label and tier_int in {1, 2}:
            dual = annotate_place_dual(
                image_path=tile_path,
                name=row["name"] or "Unknown",
                category=row["category_primary"] or "place",
                address=row["full_address"],
                lat_center=row["lat"],
                lon_center=row["lon"],
                sub_tier=sub_tier,
                tile_size=tile_size,
                provider=provider,
                model=model,
            )
            ped_lat, ped_lon = dual.pedestrian_lat, dual.pedestrian_lon
            ped_conf = dual.pedestrian_confidence
            car_lat, car_lon = dual.car_lat, dual.car_lon
            car_conf = dual.car_confidence
            car_reasoning = dual.car_reasoning
            entry_mode = dual.entry_mode
            gt_lat, gt_lon = ped_lat, ped_lon
            gt_conf = ped_conf
            gt_reasoning = dual.pedestrian_reasoning
            gt_model_name = dual.pedestrian_model
        else:
            single = annotate_place(
                image_path=tile_path,
                name=row["name"] or "Unknown",
                category=row["category_primary"] or "place",
                address=row["full_address"],
                lat_center=row["lat"],
                lon_center=row["lon"],
                tile_size=tile_size,
                provider=provider,
                model=model,
                tier=tier_int,
            )
            gt_lat, gt_lon = single.gt_lat, single.gt_lon
            gt_conf = single.confidence
            gt_reasoning = single.reasoning
            gt_model_name = single.model
            if tier_int == 3:
                car_lat, car_lon, car_conf = gt_lat, gt_lon, gt_conf
                entry_mode = "shared"

    except Exception as e:
        print(f"  [ERROR] {place_id} annotation failed: {e}")
        return None

    if gt_lat is None:
        print(f"  [WARN] {place_id} — no coordinates returned: {gt_reasoning}")

    # Inter-annotator CV
    cv_gt_lat = cv_gt_lon = cv_conf = None
    cv_model_name = None
    cv_disagreement = None

    if place_id in cv_ids and gt_lat is not None:
        try:
            print(f"  [CV] Re-annotating {place_id} with {cv_provider}/{cv_model}...")
            cv = annotate_place(
                image_path=tile_path,
                name=row["name"] or "Unknown",
                category=row["category_primary"] or "place",
                address=row["full_address"],
                lat_center=row["lat"],
                lon_center=row["lon"],
                tile_size=tile_size,
                provider=cv_provider,
                model=cv_model,
                tier=tier_int,
            )
            cv_gt_lat, cv_gt_lon = cv.gt_lat, cv.gt_lon
            cv_conf = cv.confidence
            cv_model_name = cv.model
            if cv_gt_lat is not None:
                cv_disagreement = round(
                    haversine_meters(gt_lat, gt_lon, cv_gt_lat, cv_gt_lon), 2
                )
                print(f"  [CV] Disagreement: {cv_disagreement}m")
        except Exception as e:
            print(f"  [CV ERROR] {place_id}: {e}")

    # Offset metrics
    offset_hav = offset_euc = offset_man = None
    if gt_lat is not None:
        offset_hav = haversine_meters(row["lat"], row["lon"], gt_lat, gt_lon)
        offset_euc = euclidean_meters(row["lat"], row["lon"], gt_lat, gt_lon)
        offset_man = manhattan_meters(row["lat"], row["lon"], gt_lat, gt_lon)

    return {
        "id": place_id,
        "name": row["name"],
        "category_primary": row["category_primary"],
        "region": row["region"],
        "tier": tier_int,
        "tier_label": tier_label,
        "sub_tier": sub_tier,
        "sub_tier_label": sub_tier_label,
        "current_lat": row["lat"],
        "current_lon": row["lon"],
        "gt_lat": gt_lat,
        "gt_lon": gt_lon,
        "gt_confidence": gt_conf,
        "gt_reasoning": gt_reasoning,
        "gt_model": gt_model_name,
        "pedestrian_entry_lat": ped_lat,
        "pedestrian_entry_lon": ped_lon,
        "pedestrian_entry_confidence": ped_conf,
        "car_entry_lat": car_lat,
        "car_entry_lon": car_lon,
        "car_entry_confidence": car_conf,
        "car_entry_reasoning": car_reasoning,
        "entry_mode": entry_mode,
        "cv_gt_lat": cv_gt_lat,
        "cv_gt_lon": cv_gt_lon,
        "cv_confidence": cv_conf,
        "cv_model": cv_model_name,
        "cv_disagreement_m": cv_disagreement,
        "full_address": row["full_address"],
        "offset_haversine_m": offset_hav,
        "offset_euclidean_m": offset_euc,
        "offset_manhattan_m": offset_man,
    }


def build_ground_truth(
    df: pd.DataFrame | None = None,
    sample_n: int = 750,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    mapbox_key: str | None = None,
    google_maps_key: str | None = None,
    output_path: Path | None = None,
    max_places: int | None = None,
    tile_workers: int = 8,
    batch_start: int | None = None,
    batch_end: int | None = None,
    dual_label: bool = True,
    inter_annotator_n: int = 0,
    cv_provider: str = "anthropic",
    cv_model: str = "claude-haiku-4-5-20251001",
    llm_workers: int = 3,
    tile_size: int = 512,
) -> pd.DataFrame:
    """
    Full ground truth construction pipeline.

    Args:
        provider / model: Primary annotator (all places).
        dual_label: Tier 1/2 places get both car_entry and pedestrian_entry.
        inter_annotator_n: How many Tier-1 places to cross-validate with a second
            provider. Set to 50 on the FIRST batch only; 0 for all others.
            Automatically skipped when max_places < inter_annotator_n (test runs).
        cv_provider / cv_model: Second annotator for inter-annotator check.
        llm_workers: Parallel threads for LLM annotation (default 3).
            Raise to 5 if your API rate limits allow; lower to 1 to debug.
    """
    import os
    if df is None:
        df = load_places()

    # Step 1: Select places
    if batch_start is not None or batch_end is not None:
        start = batch_start or 0
        end = min(batch_end or len(df), len(df))
        sample_df = df.iloc[start:end].copy()
        batch_label = f"{start}_{end - 1}"
        print(f"Batch mode: rows {start}–{end - 1} ({len(sample_df)} places)")
        if output_path is None:
            output_path = PROCESSED_DIR / f"ground_truth_{batch_label}.parquet"
    else:
        sample_n = min(sample_n, 750)
        sample_df = stratified_sample(df, n=sample_n)
        batch_label = "sampled"
        if output_path is None:
            output_path = PROCESSED_DIR / "ground_truth.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if max_places:
        sample_df = sample_df.head(max_places)

    # Step 2: Inter-annotator CV place IDs
    # Skip CV when this is a small test run (max_places set and < inter_annotator_n)
    effective_cv_n = inter_annotator_n
    if max_places and max_places < inter_annotator_n:
        effective_cv_n = 0
        print(f"Skipping inter-annotator CV (test run: max_places={max_places} < {inter_annotator_n})")

    cv_ids: frozenset = frozenset()
    if effective_cv_n > 0:
        tier1_ids = [
            row["id"] for _, row in sample_df.iterrows()
            if get_tier(row.get("category_primary"))[0] == 1
        ][:effective_cv_n]
        cv_ids = frozenset(tier1_ids)
        print(f"Inter-annotator CV: {len(cv_ids)} places will be re-annotated "
              f"with {cv_provider}/{cv_model}")

    # Step 3: Pick satellite tile fetcher
    _google_key = google_maps_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    _mapbox_key = mapbox_key or os.environ.get("MAPBOX_API_KEY")
    if _google_key:
        fetcher = GoogleStaticMapFetcher(api_key=_google_key, size=tile_size)
        logger.info("Using Google Maps for satellite tiles")
    elif _mapbox_key:
        fetcher = MapboxStaticFetcher(api_key=_mapbox_key, size=tile_size)
        logger.info("Using Mapbox for satellite tiles")
    else:
        fetcher = ESRIStaticFetcher(size=tile_size)
        logger.info("Using ESRI World Imagery (no key required)")

    # Step 4: Fetch tiles in parallel
    print(f"Fetching {len(sample_df)} tiles with {tile_workers} workers...")
    tile_map: dict[str, Path] = {}

    def _fetch(row):
        return row["id"], fetcher.fetch_tile(row["lat"], row["lon"], row["id"])

    n_tiles = len(sample_df)
    with ThreadPoolExecutor(max_workers=tile_workers) as pool:
        futures = {pool.submit(_fetch, row): row["id"] for _, row in sample_df.iterrows()}
        for i, future in enumerate(as_completed(futures), 1):
            place_id, path = future.result()
            if path:
                tile_map[place_id] = path
                print(f"  [TILE {i}/{n_tiles}] {place_id} ✓")
            else:
                print(f"  [TILE {i}/{n_tiles}] {place_id} FAILED")

    print(f"Tiles ready: {len(tile_map)}/{len(sample_df)}")

    # Step 5: LLM annotation — parallel across places
    csv_path = output_path.with_suffix(".csv")
    rows_to_annotate = [
        (idx + 1, row)
        for idx, (_, row) in enumerate(sample_df.iterrows())
        if row["id"] in tile_map
    ]
    n_total = len(rows_to_annotate)

    results: list[dict] = []
    results_lock = threading.Lock()

    print(f"\nAnnotating {n_total} places with {llm_workers} parallel workers "
          f"({provider}/{model})...")

    def _submit(idx_row):
        idx, row = idx_row
        return _annotate_one(
            row=row,
            tile_path=tile_map[row["id"]],
            provider=provider,
            model=model,
            dual_label=dual_label,
            cv_ids=cv_ids,
            cv_provider=cv_provider,
            cv_model=cv_model,
            idx=idx,
            n_total=n_total,
            tile_size=tile_size,
        )

    with ThreadPoolExecutor(max_workers=llm_workers) as pool:
        futures = {pool.submit(_submit, item): item[1]["id"] for item in rows_to_annotate}
        for future in as_completed(futures):
            record = future.result()
            if record is not None:
                with results_lock:
                    results.append(record)
                    pd.DataFrame(results, columns=COLUMNS).to_csv(csv_path, index=False)

    if not results:
        print("No places successfully annotated — check tile fetching and API keys")
        result_df = pd.DataFrame(columns=COLUMNS)
        result_df.to_parquet(output_path, index=False)
        return result_df

    result_df = pd.DataFrame(results, columns=COLUMNS)
    result_df.to_parquet(output_path, index=False)
    result_df.to_csv(csv_path, index=False)

    valid = result_df.dropna(subset=["gt_lat", "gt_lon"])
    print(f"\nDone. Annotated {len(valid)}/{len(result_df)} places with valid coordinates.")
    print(f"Saved to {output_path} and {csv_path}")

    # Inter-annotator summary
    if effective_cv_n > 0:
        cv_rows = result_df.dropna(subset=["cv_disagreement_m"])
        if len(cv_rows) > 0:
            median_dis = cv_rows["cv_disagreement_m"].median()
            mean_dis = cv_rows["cv_disagreement_m"].mean()
            kr_met = "✓ KR MET" if median_dis < 15 else "✗ KR NOT MET"
            print(f"\n── Inter-Annotator Agreement (OKR 1 KR3) ──────────────────")
            print(f"  Places cross-validated : {len(cv_rows)}")
            print(f"  Median disagreement    : {median_dis:.1f}m  (target: < 15m)  {kr_met}")
            print(f"  Mean disagreement      : {mean_dis:.1f}m")
            print(f"  Primary  : {provider}/{model}")
            print(f"  Secondary: {cv_provider}/{cv_model}")
            print(f"────────────────────────────────────────────────────────────")

    return result_df


def cross_validate_with_geocoders(
    gt_df: pd.DataFrame,
    n_samples: int = 100,
    google_key: str | None = None,
    mapbox_key: str | None = None,
) -> pd.DataFrame:
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

        consensus_lat = np.median([p[0] for p in geo_positions])
        consensus_lon = np.median([p[1] for p in geo_positions])
        llm_vs_consensus = haversine_meters(
            row["gt_lat"], row["gt_lon"], consensus_lat, consensus_lon
        )
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
