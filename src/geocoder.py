"""
Multi-source geocoding module.
Supports Nominatim (free), Google Maps, and Mapbox geocoding APIs.
"""

import os
import time
import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


@dataclass
class GeocodingResult:
    lat: float | None
    lon: float | None
    source: str
    confidence: float
    raw_response: dict | None = None


class NominatimGeocoder:
    """Free OSM-based geocoder. Rate limited to 1 request/second."""

    BASE_URL = "https://nominatim.openstreetmap.org/search"

    def __init__(self, user_agent: str = "PinToPlace/1.0"):
        self.user_agent = user_agent
        self._last_request_time = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_request_time = time.time()

    def geocode(self, address: str) -> GeocodingResult:
        self._rate_limit()
        try:
            resp = requests.get(
                self.BASE_URL,
                params={"q": address, "format": "json", "limit": 1},
                headers={"User-Agent": self.user_agent},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data:
                return GeocodingResult(
                    lat=float(data[0]["lat"]),
                    lon=float(data[0]["lon"]),
                    source="nominatim",
                    confidence=float(data[0].get("importance", 0.5)),
                    raw_response=data[0],
                )
        except Exception as e:
            logger.warning(f"Nominatim geocoding failed for '{address}': {e}")
        return GeocodingResult(lat=None, lon=None, source="nominatim", confidence=0.0)


class GoogleGeocoder:
    """Google Maps Geocoding API."""

    BASE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY", "")

    def geocode(self, address: str) -> GeocodingResult:
        if not self.api_key:
            logger.warning("Google Maps API key not set")
            return GeocodingResult(lat=None, lon=None, source="google", confidence=0.0)
        try:
            resp = requests.get(
                self.BASE_URL,
                params={"address": address, "key": self.api_key},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("results"):
                loc = data["results"][0]["geometry"]["location"]
                location_type = data["results"][0]["geometry"].get("location_type", "")
                confidence_map = {
                    "ROOFTOP": 0.95,
                    "RANGE_INTERPOLATED": 0.7,
                    "GEOMETRIC_CENTER": 0.6,
                    "APPROXIMATE": 0.4,
                }
                return GeocodingResult(
                    lat=loc["lat"],
                    lon=loc["lng"],
                    source="google",
                    confidence=confidence_map.get(location_type, 0.5),
                    raw_response=data["results"][0],
                )
        except Exception as e:
            logger.warning(f"Google geocoding failed for '{address}': {e}")
        return GeocodingResult(lat=None, lon=None, source="google", confidence=0.0)


class MapboxGeocoder:
    """Mapbox Geocoding API."""

    BASE_URL = "https://api.mapbox.com/geocoding/v5/mapbox.places"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("MAPBOX_API_KEY", "")

    def geocode(self, address: str) -> GeocodingResult:
        if not self.api_key:
            logger.warning("Mapbox API key not set")
            return GeocodingResult(lat=None, lon=None, source="mapbox", confidence=0.0)
        try:
            url = f"{self.BASE_URL}/{requests.utils.quote(address)}.json"
            resp = requests.get(
                url,
                params={"access_token": self.api_key, "limit": 1},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("features"):
                coords = data["features"][0]["geometry"]["coordinates"]
                relevance = data["features"][0].get("relevance", 0.5)
                return GeocodingResult(
                    lat=coords[1],
                    lon=coords[0],
                    source="mapbox",
                    confidence=relevance,
                    raw_response=data["features"][0],
                )
        except Exception as e:
            logger.warning(f"Mapbox geocoding failed for '{address}': {e}")
        return GeocodingResult(lat=None, lon=None, source="mapbox", confidence=0.0)


class MultiGeocoder:
    """Geocode using multiple services and return all results."""

    def __init__(self, google_key: str | None = None, mapbox_key: str | None = None):
        self.geocoders = [NominatimGeocoder()]
        if google_key or os.environ.get("GOOGLE_MAPS_API_KEY"):
            self.geocoders.append(GoogleGeocoder(google_key))
        if mapbox_key or os.environ.get("MAPBOX_API_KEY"):
            self.geocoders.append(MapboxGeocoder(mapbox_key))

    def geocode_all(self, address: str) -> list[GeocodingResult]:
        """Return results from all configured geocoders."""
        results = []
        for geocoder in self.geocoders:
            result = geocoder.geocode(address)
            if result.lat is not None:
                results.append(result)
        return results


if __name__ == "__main__":
    # Test with Nominatim only (no API keys needed)
    geocoder = NominatimGeocoder()
    result = geocoder.geocode("261 NE 1st St, Miami, FL 33132, US")
    print(f"Nominatim result: lat={result.lat}, lon={result.lon}, confidence={result.confidence}")
