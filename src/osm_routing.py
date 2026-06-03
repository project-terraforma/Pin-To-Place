"""
OSM-based routing for arrival cost computation via OSRM.

Uses the OSRM public routing API instead of Overpass/osmnx. OSRM has all OSM
roads pre-computed and answers route queries in milliseconds with a simple HTTP
GET — no graph downloads, no Overpass dependency.

Free, no API key required. Public demo server: router.project-osrm.org

Only rows where offset_haversine_m > 0 need routing — zero-offset rows are
skipped (routing cost is trivially 0). Results are cached to a CSV so re-runs
only compute new/missing rows.

Run from the project root:
    python -m src.osm_routing
"""

import logging
import time
from pathlib import Path

import requests
import pandas as pd

from src.metrics import haversine_meters

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"

GT_PATH      = PROCESSED / "ground_truth_combined.csv"
CACHE_PATH   = PROCESSED / "osm_routing_cache.csv"
OUTPUT_PATH  = PROCESSED / "osm_arrival_cost.csv"
SUMMARY_PATH = PROCESSED / "osm_arrival_cost_summary.txt"

OSRM_BASE    = "http://router.project-osrm.org/route/v1/foot"
TIMEOUT      = 10   # seconds per request
DELAY        = 1.0  # seconds between requests
MAX_RETRIES  = 3


def route_one(row: pd.Series) -> dict:
    """
    Compute pedestrian network distance via OSRM for a single place.
    Returns dict with walk_m, routing_penalty_m, routing_method.
    """
    place_id = str(row.get("id", row.name))
    cur_lat, cur_lon = row["current_lat"], row["current_lon"]
    gt_lat,  gt_lon  = row["gt_lat"],      row["gt_lon"]
    straight = round(haversine_meters(cur_lat, cur_lon, gt_lat, gt_lon), 2)

    url = f"{OSRM_BASE}/{cur_lon},{cur_lat};{gt_lon},{gt_lat}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params={"overview": "false"}, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            if data.get("code") != "Ok" or not data.get("routes"):
                return {
                    "id": place_id, "walk_m": straight,
                    "straight_line_m": straight, "routing_penalty_m": 0.0,
                    "routing_method": f"fallback (OSRM code={data.get('code')})",
                }

            walk_m  = round(data["routes"][0]["distance"], 2)
            penalty = round(max(0.0, walk_m - straight), 2)
            return {
                "id": place_id, "walk_m": walk_m,
                "straight_line_m": straight, "routing_penalty_m": penalty,
                "routing_method": "osrm",
            }

        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(5 * attempt)
            else:
                logger.warning("Routing failed for %s after %d attempts: %s",
                               place_id, MAX_RETRIES, e)
                return {
                    "id": place_id, "walk_m": straight,
                    "straight_line_m": straight, "routing_penalty_m": 0.0,
                    "routing_method": f"fallback ({type(e).__name__})",
                }


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        cache_df = pd.read_csv(CACHE_PATH)
        # Only keep successful rows — don't reuse old failures
        good = cache_df[cache_df["routing_method"].isin(["osrm", "osmnx", "osmnx_same_node"])]
        return {str(r["id"]): r.to_dict() for _, r in good.iterrows()}
    return {}


def _save_cache(cache: dict) -> None:
    pd.DataFrame(list(cache.values())).to_csv(CACHE_PATH, index=False)


def compute_routing(df: pd.DataFrame) -> pd.DataFrame:
    cache = _load_cache()

    needs_routing  = df[df["offset_haversine_m"] > 0].copy()
    already_cached = {str(i) for i in needs_routing["id"].astype(str) if str(i) in cache}
    to_compute     = needs_routing[~needs_routing["id"].astype(str).isin(already_cached)]

    print(f"Total rows:          {len(df)}")
    print(f"Zero-offset (skip):  {(df['offset_haversine_m'] == 0).sum()}")
    print(f"Needs routing:       {len(needs_routing)}")
    print(f"Already cached:      {len(already_cached)}")
    print(f"To compute now:      {len(to_compute)}")

    rows_list = list(to_compute.iterrows())
    for i, (_, row) in enumerate(rows_list):
        place_id = str(row["id"])
        if place_id in cache:
            continue
        result = route_one(row)
        cache[place_id] = result
        if (i + 1) % 10 == 0 or (i + 1) == len(rows_list):
            _save_cache(cache)
            print(f"  {i + 1}/{len(rows_list)} done")
        time.sleep(DELAY)

    _save_cache(cache)

    routing_df = pd.DataFrame(list(cache.values()))
    zero_ids   = df[df["offset_haversine_m"] == 0]["id"].astype(str)
    zero_rows  = pd.DataFrame({
        "id": zero_ids, "walk_m": 0.0, "straight_line_m": 0.0,
        "routing_penalty_m": 0.0, "routing_method": "zero_offset",
    })

    all_routing = pd.concat([routing_df, zero_rows], ignore_index=True)
    all_routing["id"] = all_routing["id"].astype(str)

    result = df.copy()
    result["id"] = result["id"].astype(str)
    result = result.merge(
        all_routing[["id", "walk_m", "routing_penalty_m", "routing_method"]],
        on="id", how="left",
    )
    result["osm_arrival_cost_m"] = result["walk_m"].fillna(result["offset_haversine_m"])
    return result


def write_summary(df: pd.DataFrame) -> None:
    routed   = df[df["routing_method"] == "osrm"]
    fallback = df[df["routing_method"].str.startswith("fallback", na=False)]
    penalty  = df["routing_penalty_m"].fillna(0)
    old_cost = df["offset_haversine_m"]
    new_cost = df["osm_arrival_cost_m"]

    lines = [
        "OSM Arrival Cost Summary (OSRM)",
        "",
        f"Total rows:            {len(df)}",
        f"Routed via OSRM:       {len(routed)}",
        f"Fallback (haversine):  {len(fallback)}",
        f"Zero-offset skipped:   {(df['routing_method'] == 'zero_offset').sum()}",
        "",
        "Routing penalty > 0 (extra walking due to real obstacles)",
        penalty[penalty > 0].describe().round(2).to_string(),
        f"Places with any penalty:  {(penalty > 0).sum()}",
        f"Penalty >= 10m:           {(penalty >= 10).sum()}",
        f"Penalty >= 25m:           {(penalty >= 25).sum()}",
        "",
        "p90/p95 comparison",
        f"  haversine p90:  {old_cost.quantile(.9):.2f}m   OSRM p90:  {new_cost.quantile(.9):.2f}m",
        f"  haversine p95:  {old_cost.quantile(.95):.2f}m   OSRM p95:  {new_cost.quantile(.95):.2f}m",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = pd.read_csv(GT_PATH)
    result = compute_routing(df)
    result.to_csv(OUTPUT_PATH, index=False)
    write_summary(result)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
