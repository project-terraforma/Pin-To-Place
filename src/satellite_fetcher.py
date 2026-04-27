"""
Fetch satellite imagery tiles centered on place locations.
Supports ESRI World Imagery (free, no key), Mapbox, and Google Maps Static API.
"""

import io
import math
import os
import time
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

TILES_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "tiles"


def _get_with_retry(url: str, params: dict, timeout: int = 30,
                    retries: int = 3, backoff: float = 2.0) -> requests.Response:
    """GET with exponential backoff on timeout or 5xx errors."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt == retries - 1:
                raise
            wait = backoff ** attempt
            logger.warning(f"Request failed ({e}), retrying in {wait:.0f}s...")
            time.sleep(wait)
        except requests.HTTPError as e:
            if resp.status_code < 500 or attempt == retries - 1:
                raise
            time.sleep(backoff ** attempt)
    raise RuntimeError("Unreachable")


def _is_valid_image(path: Path) -> bool:
    """Return True only if the file exists and is a readable image."""
    if not path.exists():
        return False
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        path.unlink(missing_ok=True)  # delete the corrupt file
        return False


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

        if _is_valid_image(output_path):
            return output_path

        try:
            resp = _get_with_retry(
                self.BASE_URL,
                params={
                    "center": f"{lat},{lon}",
                    "zoom": self.zoom,
                    "size": f"{self.size}x{self.size}",
                    "maptype": "satellite",
                    "key": self.api_key,
                    "markers": f"color:red|{lat},{lon}",
                },
            )
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
        return pixel_to_latlon(lat_center, lon_center, pixel_x, pixel_y, self.size, self.zoom)


def pixel_to_latlon(lat_center: float, lon_center: float,
                    pixel_x: float, pixel_y: float,
                    size: int = 640, zoom: int = 18) -> tuple[float, float]:
    """
    Convert a pixel position in a static map tile to lat/lon.
    Works for both Google and Mapbox tiles since the math only depends on zoom/size.
    pixel_x: 0 = left edge, positive = right (east)
    pixel_y: 0 = top edge, positive = down (south)
    """
    import math
    meters_per_pixel = 156543.03392 * math.cos(math.radians(lat_center)) / (2 ** zoom)
    dx_m = (pixel_x - size / 2) * meters_per_pixel
    dy_m = (size / 2 - pixel_y) * meters_per_pixel  # flip Y: down in pixels = south
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

        if _is_valid_image(output_path):
            return output_path

        try:
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

    def pixel_to_latlon(self, lat_center: float, lon_center: float,
                        px: float, py: float) -> tuple[float, float]:
        return pixel_to_latlon(lat_center, lon_center, px, py, self.size, self.zoom)


class ESRIStaticFetcher:
    """
    Fetch satellite tiles from ESRI World Imagery. No API key required.
    Stitches a 3x3 grid of 256px XYZ tiles into a single cropped image.
    """

    TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    TILE_PX = 256

    def __init__(self, size: int = 640, zoom: int = 18):
        self.size = size
        self.zoom = zoom

    def _to_tile(self, lat: float, lon: float) -> tuple[float, float]:
        n = 2 ** self.zoom
        fx = (lon + 180.0) / 360.0 * n
        lat_r = math.radians(lat)
        fy = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
        return fx, fy

    def fetch_tile(self, lat: float, lon: float, place_id: str,
                   output_dir: Path | None = None) -> Path | None:
        from PIL import Image, ImageDraw

        output_dir = output_dir or TILES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{place_id}.png"
        if _is_valid_image(output_path):
            return output_path

        try:
            fx, fy = self._to_tile(lat, lon)
            cx, cy = int(fx), int(fy)

            # Fetch 3x3 grid of tiles → 768x768 stitched image
            radius = 1
            grid = 2 * radius + 1
            stitched = Image.new("RGB", (grid * self.TILE_PX, grid * self.TILE_PX))

            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    url = self.TILE_URL.format(z=self.zoom, y=cy + dy, x=cx + dx)
                    resp = requests.get(url, timeout=10,
                                        headers={"User-Agent": "PinToPlace/1.0"})
                    resp.raise_for_status()
                    tile = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    stitched.paste(tile, ((dx + radius) * self.TILE_PX,
                                          (dy + radius) * self.TILE_PX))

            # Pixel position of (lat, lon) within the stitched image
            pin_px = int((fx - (cx - radius)) * self.TILE_PX)
            pin_py = int((fy - (cy - radius)) * self.TILE_PX)

            # Crop self.size x self.size centered on the pin
            half = self.size // 2
            left = max(0, pin_px - half)
            top  = max(0, pin_py - half)
            right  = min(stitched.width,  left + self.size)
            bottom = min(stitched.height, top  + self.size)
            cropped = stitched.crop((left, top, right, bottom))

            # Pad to exact size if the crop hit an edge
            if cropped.size != (self.size, self.size):
                canvas = Image.new("RGB", (self.size, self.size))
                canvas.paste(cropped)
                cropped = canvas

            # Draw red circle marker at pin location
            marker_x = pin_px - left
            marker_y = pin_py - top
            draw = ImageDraw.Draw(cropped)
            r = 6
            draw.ellipse([marker_x - r, marker_y - r, marker_x + r, marker_y + r],
                         fill="red", outline="white", width=2)

            cropped.save(output_path)
            logger.info(f"Saved ESRI tile for {place_id} to {output_path}")
            return output_path

        except Exception as e:
            logger.warning(f"Failed to fetch ESRI tile for {place_id}: {e}")
            return None

    def pixel_to_latlon(self, lat_center: float, lon_center: float,
                        px: float, py: float) -> tuple[float, float]:
        return pixel_to_latlon(lat_center, lon_center, px, py, self.size, self.zoom)


if __name__ == "__main__":
    # Example: show tile bounds for a point in Miami
    fetcher = GoogleStaticMapFetcher()
    bounds = fetcher.tile_bounds(25.7754, -80.1891)
    print(f"Tile bounds for Miami point:")
    for k, v in bounds.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
