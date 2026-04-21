"""
Data loader for Overture Maps place data.
Decodes WKB geometries, flattens nested fields, and provides deduplication.
"""

import struct
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PARQUET = DATA_DIR / "raw" / "project_d_samples.parquet"


def decode_wkb_point(wkb: bytes) -> tuple[float, float] | None:
    """Decode a WKB-encoded Point geometry to (longitude, latitude)."""
    if not isinstance(wkb, bytes) or len(wkb) != 21:
        return None
    byte_order = wkb[0]
    fmt = "<" if byte_order == 1 else ">"
    geom_type = struct.unpack(f"{fmt}I", wkb[1:5])[0]
    if geom_type != 1:  # Not a Point
        return None
    lon, lat = struct.unpack(f"{fmt}dd", wkb[5:21])
    return lon, lat


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the Haversine distance between two points in meters."""
    R = 6_371_000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _safe_get(obj, *keys, default=None):
    """Safely navigate nested dicts/lists."""
    for key in keys:
        if obj is None:
            return default
        try:
            obj = obj[key]
        except (KeyError, IndexError, TypeError):
            return default
    return obj


def load_places(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load the Overture places parquet file and return a flattened DataFrame.

    Columns added:
        lon, lat           - decoded from WKB geometry
        name               - names.primary
        category_primary   - categories.primary
        category_alternate - categories.alternate (list)
        country, region, locality, freeform, postcode - from addresses[0]
        source_datasets    - list of source dataset names
        source_count       - number of sources
    """
    path = Path(path) if path else RAW_PARQUET
    df = pd.read_parquet(path)

    # Decode geometry
    coords = df["geometry"].apply(decode_wkb_point)
    df["lon"] = coords.apply(lambda c: c[0] if c else None)
    df["lat"] = coords.apply(lambda c: c[1] if c else None)

    # Flatten names
    df["name"] = df["names"].apply(lambda x: _safe_get(x, "primary"))

    # Flatten categories
    df["category_primary"] = df["categories"].apply(lambda x: _safe_get(x, "primary"))
    df["category_alternate"] = df["categories"].apply(
        lambda x: list(_safe_get(x, "alternate", default=[])) if _safe_get(x, "alternate") is not None else []
    )

    # Flatten first address
    df["country"] = df["addresses"].apply(lambda x: _safe_get(x, 0, "country"))
    df["region"] = df["addresses"].apply(lambda x: _safe_get(x, 0, "region"))
    df["locality"] = df["addresses"].apply(lambda x: _safe_get(x, 0, "locality"))
    df["freeform"] = df["addresses"].apply(lambda x: _safe_get(x, 0, "freeform"))
    df["postcode"] = df["addresses"].apply(lambda x: _safe_get(x, 0, "postcode"))

    # Flatten sources
    df["source_datasets"] = df["sources"].apply(
        lambda x: [s.get("dataset", "") for s in x] if isinstance(x, list) else []
    )
    df["source_count"] = df["source_datasets"].apply(len)

    # Build full address string for geocoding
    df["full_address"] = df.apply(
        lambda r: ", ".join(
            filter(None, [r["freeform"], r["locality"], r["region"], r["postcode"], r["country"]])
        ),
        axis=1,
    )

    return df


def find_near_duplicates(df: pd.DataFrame, name_col: str = "name",
                          max_distance_m: float = 50.0) -> pd.DataFrame:
    """
    Identify near-duplicate places: same name (case-insensitive) within max_distance_m.

    Returns a DataFrame of duplicate pairs with columns:
        idx_a, idx_b, name, distance_m
    """
    duplicates = []
    name_groups = df.dropna(subset=[name_col]).groupby(
        df[name_col].str.lower().str.strip()
    )

    for _, group in name_groups:
        if len(group) < 2:
            continue
        idxs = group.index.tolist()
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                dist = haversine_meters(
                    df.loc[a, "lat"], df.loc[a, "lon"],
                    df.loc[b, "lat"], df.loc[b, "lon"],
                )
                if dist <= max_distance_m:
                    duplicates.append({
                        "idx_a": a,
                        "idx_b": b,
                        "name": df.loc[a, name_col],
                        "distance_m": round(dist, 2),
                    })

    return pd.DataFrame(duplicates)


if __name__ == "__main__":
    df = load_places()
    print(f"Loaded {len(df)} places")
    print(f"Lat range: {df['lat'].min():.4f} to {df['lat'].max():.4f}")
    print(f"Lon range: {df['lon'].min():.4f} to {df['lon'].max():.4f}")
    print(f"Null lats: {df['lat'].isna().sum()}")
    print(f"\nTop 10 categories:")
    print(df["category_primary"].value_counts().head(10))
    print(f"\nTop 10 regions:")
    print(df["region"].value_counts().head(10))

    dupes = find_near_duplicates(df)
    print(f"\nNear-duplicates found: {len(dupes)}")
    if len(dupes) > 0:
        print(dupes.head(10))
