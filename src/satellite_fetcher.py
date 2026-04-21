"""
Fetch satellite imagery tiles centered on place locations.
Supports Google Maps Static API and Mapbox Static Images API.
"""

import os
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

TILES_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "tiles"


class GoogleStaticMapFetcher:
    """Fetch satellite tiles via Google Maps Static API."""

    BASE_URL = "https://maps.googleapis.com/maps/api/staticmap"

    def __init__(self, api_key: str | None = None, size: int = 640, zoom: int = 18):
        self.api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY", "")
        self.size = size  # pixels (max 640 for free tier)
        self.zoom = zoom  # 18 = ~250m x 250m at mid-latitudes

    def fetch_tile(self, lat: float, lon: float, place_id: str,
                   output_dir: Path | None = None) -> Path | None:
        """Fetch a satellite tile and save to disk. Returns the file path."""
        if not self.api_key:
            logger.warning("Google Maps API key not set")
            return None

        output_dir = output_dir or TILES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{place_id}.png"

        if output_path.exists():
            return output_path

        try:
            resp = requests.get(
                self.BASE_URL,
                params={
                    "center": f"{lat},{lon}",
                    "zoom": self.zoom,
                    "size": f"{self.size}x{self.size}",
                    "maptype": "satellite",
                    "key": self.api_key,
                    # Mark the current pin location
                    "markers": f"color:red|{lat},{lon}",
                },
                timeout=15,
            )
            resp.raise_for_status()
            output_path.write_bytes(resp.content)
            logger.info(f"Saved tile for {place_id} to {output_path}")
            return output_path
        except Exception as e:
            logger.warning(f"Failed to fetch tile for {place_id}: {e}")
            return None

    def tile_bounds(self, lat: float, lon: float) -> dict:
        """
        Approximate the geographic bounds of a tile.
        At zoom 18, each pixel covers ~0.6m at the equator.
        Returns dict with n/s/e/w lat/lon bounds and meters_per_pixel.
        """
        import math
        meters_per_pixel = 156543.03392 * math.cos(math.radians(lat)) / (2 ** self.zoom)
        half_size_m = (self.size / 2) * meters_per_pixel
        # Approximate degrees
        dlat = half_size_m / 111_320
        dlon = half_size_m / (111_320 * math.cos(math.radians(lat)))
        return {
            "north": lat + dlat,
            "south": lat - dlat,
            "east": lon + dlon,
            "west": lon - dlon,
            "meters_per_pixel": meters_per_pixel,
            "tile_width_m": half_size_m * 2,
        }

    def pixel_to_latlon(self, lat_center: float, lon_center: float,
                        pixel_x: float, pixel_y: float) -> tuple[float, float]:
        """
        Convert pixel offset from tile center to lat/lon.
        pixel_x: positive = right (east), pixel_y: positive = down (south)
        """
        import math
        meters_per_pixel = 156543.03392 * math.cos(math.radians(lat_center)) / (2 ** self.zoom)
        dx_m = (pixel_x - self.size / 2) * meters_per_pixel
        dy_m = (self.size / 2 - pixel_y) * meters_per_pixel  # flip Y axis
        new_lat = lat_center + dy_m / 111_320
        new_lon = lon_center + dx_m / (111_320 * math.cos(math.radians(lat_center)))
        return new_lat, new_lon


class MapboxStaticFetcher:
    """Fetch satellite tiles via Mapbox Static Images API."""

    BASE_URL = "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static"

    def __init__(self, api_key: str | None = None, size: int = 640, zoom: int = 18):
        self.api_key = api_key or os.environ.get("MAPBOX_API_KEY", "")
        self.size = size
        self.zoom = zoom

    def fetch_tile(self, lat: float, lon: float, place_id: str,
                   output_dir: Path | None = None) -> Path | None:
        if not self.api_key:
            logger.warning("Mapbox API key not set")
            return None

        output_dir = output_dir or TILES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{place_id}.png"

        if output_path.exists():
            return output_path

        try:
            # pin marker
            marker = f"pin-s+ff0000({lon},{lat})"
            url = f"{self.BASE_URL}/{marker}/{lon},{lat},{self.zoom}/{self.size}x{self.size}"
            resp = requests.get(
                url,
                params={"access_token": self.api_key},
                timeout=15,
            )
            resp.raise_for_status()
            output_path.write_bytes(resp.content)
            logger.info(f"Saved tile for {place_id} to {output_path}")
            return output_path
        except Exception as e:
            logger.warning(f"Failed to fetch Mapbox tile for {place_id}: {e}")
            return None


if __name__ == "__main__":
    # Example: show tile bounds for a point in Miami
    fetcher = GoogleStaticMapFetcher()
    bounds = fetcher.tile_bounds(25.7754, -80.1891)
    print(f"Tile bounds for Miami point:")
    for k, v in bounds.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
