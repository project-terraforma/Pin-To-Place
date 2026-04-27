"""
Offset measurement and evaluation metrics for pin placement accuracy.
"""

from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance on a sphere. Most accurate for geographic coordinates."""
    R = 6_371_000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def euclidean_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Flat-plane (Euclidean) distance in meters.
    Projects degree differences to meters using a local approximation at the
    midpoint latitude. Fast and accurate for short distances (< 1km).
    """
    mid_lat = radians((lat1 + lat2) / 2)
    meters_per_deg_lat = 111_320
    meters_per_deg_lon = 111_320 * cos(mid_lat)
    dy = (lat2 - lat1) * meters_per_deg_lat
    dx = (lon2 - lon1) * meters_per_deg_lon
    return sqrt(dx ** 2 + dy ** 2)


def manhattan_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Manhattan (taxicab) distance in meters — sum of north/south and east/west
    components. Approximates travel distance on a grid street network.
    """
    mid_lat = radians((lat1 + lat2) / 2)
    meters_per_deg_lat = 111_320
    meters_per_deg_lon = 111_320 * cos(mid_lat)
    dy = abs(lat2 - lat1) * meters_per_deg_lat
    dx = abs(lon2 - lon1) * meters_per_deg_lon
    return dx + dy


def compute_offsets(df: pd.DataFrame,
                    current_lat: str = "lat",
                    current_lon: str = "lon",
                    gt_lat: str = "gt_lat",
                    gt_lon: str = "gt_lon") -> pd.Series:
    """Compute Haversine offset in meters between current and ground-truth positions."""
    return df.apply(
        lambda r: haversine_meters(r[current_lat], r[current_lon], r[gt_lat], r[gt_lon]),
        axis=1,
    )


def offset_report(offsets: pd.Series) -> dict:
    """Generate summary statistics for a series of offset distances (in meters)."""
    return {
        "count": len(offsets),
        "mean_m": round(offsets.mean(), 2),
        "median_m": round(offsets.median(), 2),
        "std_m": round(offsets.std(), 2),
        "p90_m": round(np.percentile(offsets, 90), 2),
        "p95_m": round(np.percentile(offsets, 95), 2),
        "max_m": round(offsets.max(), 2),
        "pct_within_10m": round((offsets <= 10).mean() * 100, 1),
        "pct_within_25m": round((offsets <= 25).mean() * 100, 1),
        "pct_within_50m": round((offsets <= 50).mean() * 100, 1),
        "pct_within_100m": round((offsets <= 100).mean() * 100, 1),
        "pct_within_250m": round((offsets <= 250).mean() * 100, 1),
    }


def regression_rate(baseline_offsets: pd.Series, new_offsets: pd.Series) -> float:
    """
    Compute regression rate: percentage of places where the new method
    moved the pin further from ground truth than the baseline.
    """
    regressions = (new_offsets > baseline_offsets).sum()
    return round(regressions / len(baseline_offsets) * 100, 2)


def improvement_summary(baseline_offsets: pd.Series, new_offsets: pd.Series) -> dict:
    """Compare a new method's offsets against the baseline."""
    baseline_report = offset_report(baseline_offsets)
    new_report = offset_report(new_offsets)

    return {
        "baseline": baseline_report,
        "new_method": new_report,
        "median_improvement_m": round(baseline_report["median_m"] - new_report["median_m"], 2),
        "median_improvement_pct": round(
            (baseline_report["median_m"] - new_report["median_m"]) / baseline_report["median_m"] * 100, 1
        ) if baseline_report["median_m"] > 0 else 0,
        "mean_improvement_m": round(baseline_report["mean_m"] - new_report["mean_m"], 2),
        "regression_rate_pct": regression_rate(baseline_offsets, new_offsets),
    }


def segmented_report(df: pd.DataFrame, offset_col: str, segment_col: str,
                     top_n: int | None = None) -> pd.DataFrame:
    """Generate offset reports segmented by a categorical column."""
    results = []
    groups = df.groupby(segment_col)
    for name, group in groups:
        report = offset_report(group[offset_col])
        report["segment"] = name
        report["n"] = len(group)
        results.append(report)

    result_df = pd.DataFrame(results).sort_values("median_m", ascending=False)
    if top_n:
        result_df = result_df.head(top_n)
    return result_df


if __name__ == "__main__":
    # Quick unit test for haversine
    # NYC to LA: ~3,944 km
    dist = haversine_meters(40.7128, -74.0060, 34.0522, -118.2437)
    print(f"NYC to LA: {dist / 1000:.0f} km (expected ~3944 km)")

    # Same point: should be 0
    dist0 = haversine_meters(40.7128, -74.0060, 40.7128, -74.0060)
    print(f"Same point: {dist0:.6f} m (expected 0)")

    # ~111 m (1 degree lat ~111km, so 0.001 degree ~ 111m)
    dist1 = haversine_meters(40.0, -74.0, 40.001, -74.0)
    print(f"0.001 degree lat: {dist1:.1f} m (expected ~111 m)")
