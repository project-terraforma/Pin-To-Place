# Pin Location Definition & Taxonomy

## Overview

This document evaluates candidate definitions for where an Overture Maps place pin should be positioned, and recommends a category-aware hierarchical approach.

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

## Recommended Approach: Category-Aware Hierarchical Definition

Rather than a single definition for all places, we recommend a **hierarchical approach** that adapts to the place type:

### Tier 1: Standard Commercial Places
(restaurants, shops, offices, hotels)
- **Primary target:** Nearest road-facing point on building footprint
- **Refinement:** LLM-assisted identification of main entrance when satellite imagery is available
- **Fallback:** Building centroid

### Tier 2: Multi-Tenant Buildings
(strip malls, office suites, shopping centers)
- **Primary target:** Storefront entrance identified via LLM vision
- **Fallback:** Nearest road-facing point, biased toward the address-matching section of the building

### Tier 3: Open Spaces & Outdoor Places
(parks, campgrounds, golf courses, fairgrounds)
- **Primary target:** Main access point (parking lot entrance, gate)
- **Fallback:** Parcel/area centroid

### Tier 4: No Building / No Geometry
(home businesses, mobile services, vague addresses)
- **Primary target:** Best available geocode (multi-geocoder consensus)
- **No fallback** — flag as low-confidence

### Why this approach?

1. **No single definition works for all place types.** A gas station and a park have fundamentally different spatial semantics.
2. **Automation-first.** Tier 1 (road-facing point) and Tier 4 (geocode) are fully automated. LLM assistance in Tiers 1-2 adds quality where it matters most.
3. **Measurable.** Each tier's definition is specific enough to measure inter-annotator agreement.
4. **Progressive quality.** Start with the automated baseline, improve with LLM refinement where budget allows.
