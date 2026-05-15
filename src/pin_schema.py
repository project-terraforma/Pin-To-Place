"""
Shared schema for task-aware place pins.
"""

from dataclasses import dataclass


PIN_TYPES = {
    "current": "Original Overture Maps pin.",
    "pedestrian_entry": "Where a person on foot enters the place.",
    "vehicle_entry": "Where a driver turns into the property.",
    "delivery_entry": "Likely service or delivery access point.",
    "accessible_entry": "Best accessible entrance or curb-ramp-connected entry.",
    "building_centroid": "Geometric center of the building footprint.",
}


ARRIVAL_MODES = {
    "pedestrian": "Person arriving on foot.",
    "vehicle": "Person arriving by car.",
    "delivery": "Delivery or service access.",
    "accessible": "Person needing accessible entrance/path.",
}


AMBIGUITY_LEVELS = {
    "low": "One obvious correct pin target.",
    "medium": "Multiple plausible targets, but one likely primary.",
    "high": "No clearly dominant target or place boundary is unclear.",
}


@dataclass
class PinTarget:
    place_id: str
    pin_type: str
    lat: float | None
    lon: float | None
    confidence: float
    reasoning: str = ""
