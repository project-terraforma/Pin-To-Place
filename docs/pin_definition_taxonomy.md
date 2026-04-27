# Pin Location Definition & Taxonomy

## Overview

This document evaluates candidate definitions for where an Overture Maps place pin should be positioned and recommends a category-aware hierarchical approach. It also defines **mode-of-transport pin variants** — separate car-entry and pedestrian-entry pin targets — which are labeled in the ground truth dataset for all Tier 1 and Tier 2 places.

---

## Candidate Definitions

### 1. Building Centroid

**Definition:** The geometric center (centroid) of the building footprint polygon.

| Criterion | Score |
|-----------|-------|
| User usefulness | Medium - Gives general building location but may land in courtyard or interior wall |
| Consistency | High - Deterministic computation from geometry |
| Labelability | High - Automated from building footprints |
| Scalability | High - Requires only building polygons (widely available via OSM/Overture) |

**Best for:** Single-tenant standalone buildings, warehouses, single-family homes.

**Fails for:** Strip malls (centroid is shared space), L-shaped buildings (centroid may be outside footprint), multi-story complexes (centroid ignores vertical).

---

### 2. Rooftop Centroid

**Definition:** The visual center of the rooftop as seen from satellite imagery.

| Criterion | Score |
|-----------|-------|
| User usefulness | Medium - Similar to building centroid |
| Consistency | High - Deterministic if rooftop segmentation is available |
| Labelability | Medium - Requires satellite imagery processing |
| Scalability | Medium - Depends on roof segmentation models or manual labeling |

**Best for:** Residential addresses, simple rectangular buildings.

**Fails for:** Same cases as building centroid, plus buildings with complex/overlapping roofs.

---

### 3. Main Entrance / Storefront

**Definition:** The primary customer-facing entrance of the place.

| Criterion | Score |
|-----------|-------|
| User usefulness | Very High - Exactly where users want to navigate to |
| Consistency | Low - Subjective (which entrance is "main"?), changes with renovations |
| Labelability | Low - Requires street-level imagery or manual annotation |
| Scalability | Low - Difficult to automate at global scale |

**Best for:** Restaurants, retail shops, hotels (lobby entrance), offices.

**Fails for:** Places without a clear single entrance (parks, campgrounds, multi-entrance buildings).

---

### 4. Nearest Road-Facing Point

**Definition:** The closest point on the building footprint polygon to the nearest road segment.

| Criterion | Score |
|-----------|-------|
| User usefulness | High - Good proxy for where you'd approach from the road |
| Consistency | High - Deterministic from geometry + road network |
| Labelability | High - Automated from building footprints + road data |
| Scalability | High - Both data sources widely available |

**Best for:** Places accessible from a single road, drive-up businesses (gas stations, fast food).

**Fails for:** Corner buildings (may pick wrong road), places accessed from parking lots or interior paths, multi-road frontage.

---

### 5. Parcel Centroid

**Definition:** The geometric center of the land parcel containing the place.

| Criterion | Score |
|-----------|-------|
| User usefulness | Low - Often far from the building, especially for large parcels |
| Consistency | High - Deterministic |
| Labelability | Medium - Parcel data availability varies significantly |
| Scalability | Medium - US has good parcel data, most other countries do not |

**Best for:** Large properties (farms, estates), vacant land.

**Fails for:** Urban areas (parcel may encompass multiple buildings), inconsistent parcel boundaries.

---

### 6. Address Geocode

**Definition:** The lat/lon returned by geocoding the place's street address.

| Criterion | Score |
|-----------|-------|
| User usefulness | Medium - Depends on geocoder quality |
| Consistency | Low - Different geocoders return different positions |
| Labelability | Very High - Just needs an address string |
| Scalability | Very High - Works anywhere there's an address |

**Best for:** Fallback when no other data is available, rural addresses.

**Fails for:** Geocoder-specific biases (road centerline snapping, interpolated positions), PO boxes, vague addresses.

---

## Comparative Summary

| Definition | User Usefulness | Consistency | Labelability | Scalability | Overall |
|---|---|---|---|---|---|
| Building Centroid | Medium | High | High | High | Strong baseline |
| Rooftop Centroid | Medium | High | Medium | Medium | Marginal over building |
| Main Entrance | Very High | Low | Low | Low | Ideal but impractical |
| Road-Facing Point | High | High | High | High | Best automated proxy |
| Parcel Centroid | Low | High | Medium | Medium | Niche use only |
| Address Geocode | Medium | Low | Very High | Very High | Universal fallback |

---

## Recommended Approach: Category-Aware Hierarchical Definition

Rather than a single definition for all places, we recommend a **hierarchical approach** that adapts to the place type:

### Tier 1: Standard Commercial Places
(restaurants, shops, offices, hotels)
- **Primary target:** Nearest road-facing point on building footprint
- **Refinement:** LLM-assisted identification of main entrance when satellite imagery is available
- **Fallback:** Building centroid
- **Mode-of-transport variants:** Label both a car-entry pin (road-facing driveway or parking lot entrance) and a pedestrian-entry pin (sidewalk-facing storefront or accessible entrance) — see section below

### Tier 2: Multi-Tenant Buildings
(strip malls, office suites, shopping centers)
- **Primary target:** Storefront entrance identified via LLM vision
- **Fallback:** Nearest road-facing point, biased toward the address-matching section of the building
- **Mode-of-transport variants:** Label both a car-entry pin (shared parking lot entry nearest the unit) and a pedestrian-entry pin (unit storefront entrance accessible from sidewalk)

### Tier 3: Open Spaces & Outdoor Places
(parks, campgrounds, golf courses, fairgrounds)
- **Primary target:** Main access point (parking lot entrance, gate)
- **Fallback:** Parcel/area centroid
- **Mode-of-transport variants:** A single pin is sufficient for most open spaces; label separately only when there is a clearly distinct vehicle gate vs. pedestrian path entrance

### Tier 4: No Building / No Geometry
(home businesses, mobile services, vague addresses)
- **Primary target:** Best available geocode (multi-geocoder consensus)
- **No fallback** — flag as low-confidence
- **Mode-of-transport variants:** Not applicable

### Why this approach?

1. **No single definition works for all place types.** A gas station and a park have fundamentally different spatial semantics.
2. **Automation-first.** Tier 1 (road-facing point) and Tier 4 (geocode) are fully automated. LLM assistance in Tiers 1-2 adds quality where it matters most.
3. **Measurable.** Each tier's definition is specific enough to measure inter-annotator agreement.
4. **Progressive quality.** Start with the automated baseline, improve with LLM refinement where budget allows.

---

## Mode-of-Transport Pin Variants

For Tier 1 and Tier 2 places, the ground truth dataset labels **two pin targets per place** rather than one. This supports the Quickest Path feature (OKR 2.3), which measures cost-of-arrival separately for drivers and pedestrians.

### Car-Entry Pin

**Definition:** The point at which a driver arriving by road would first enter the property — typically the driveway cut, parking lot entrance, or drop-off lane.

| Property | Notes |
|---|---|
| Target on satellite image | Edge of the road where the driveway or parking lot begins; the curb cut for vehicle access |
| Typical offset from storefront | 10–50m for standalone buildings; 20–100m for large parking lots |
| LLM annotation instruction | "Identify where a car would turn off the road to reach this place. Mark the driveway entrance or parking lot entry, not the building door." |
| Automated proxy | Nearest road-facing point on the parcel boundary (not building footprint) |

**Examples:**
- Fast food restaurant → entrance to the drive-through/parking lot from the street
- Hotel → driveway or porte-cochère entry point from the road
- Strip mall unit → shared parking lot entrance from the nearest road

---

### Pedestrian-Entry Pin

**Definition:** The point a person on foot would walk to in order to enter the place — typically the front door, storefront, or the accessible path from the nearest sidewalk.

| Property | Notes |
|---|---|
| Target on satellite image | The building entrance facing the street or sidewalk; accessible ramp or door visible from above |
| Typical offset from car-entry pin | 5–80m depending on parking lot depth |
| LLM annotation instruction | "Identify where a pedestrian approaching on foot from the nearest sidewalk would enter this place. Look for a visible entrance, accessible ramp, or doorway. Note any visible curb cuts or sidewalk connections leading to the entrance." |
| Automated proxy | Nearest road-facing point on the building footprint (existing method) |

**Examples:**
- Restaurant → front door facing the sidewalk or pedestrian path
- Hotel → lobby entrance accessible from the sidewalk (may differ from porte-cochère)
- Strip mall unit → storefront door

**Note on satellite resolution:** At zoom level 18 (~0.6m/pixel), individual curb cuts and door handles are often only a few pixels wide. LLM confidence on pedestrian-entry pins should be treated as lower-reliability than car-entry pins for small buildings. Flag annotations with confidence < 0.6 for human review.

---

### When the Two Pins Coincide

For some place types, the car-entry and pedestrian-entry pins are effectively the same point and do not need to be labeled separately:

- **Drive-up only** (gas stations, car washes, drive-throughs with no interior) → use car-entry pin only; mark `pedestrian_entry: null`
- **Pedestrian-only** (small urban storefronts with no parking, transit stations) → use pedestrian-entry pin only; mark `car_entry: null`
- **Single shared entrance** (small standalone building where the parking lot abuts the front door, < 5m separation) → label once and copy to both fields

---

### Ground Truth Schema

The ground truth parquet (`data/processed/ground_truth.parquet`) stores both pin variants per place:

| Column | Type | Description |
|---|---|---|
| `car_entry_lat` / `car_entry_lon` | float | Car-entry pin coordinates |
| `car_entry_confidence` | float 0–1 | LLM confidence for car-entry label |
| `pedestrian_entry_lat` / `pedestrian_entry_lon` | float | Pedestrian-entry pin coordinates |
| `pedestrian_entry_confidence` | float 0–1 | LLM confidence for pedestrian-entry label |
| `entry_mode` | str | `both` / `car_only` / `pedestrian_only` / `shared` |
| `gt_lat` / `gt_lon` | float | Primary pin (pedestrian-entry for Tier 1–2, car-entry for Tier 3, geocode for Tier 4) — used for all existing offset metrics |
