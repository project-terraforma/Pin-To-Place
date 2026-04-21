# Pin-To-Place: Project Plan

## Location & Positional Accuracy for Overture Maps

**Team:** Aaron J. Yam & Shivani Belambe | Project Terraforma @ UCSC

---

## 1. Problem Statement

Overture Maps releases a new global map every month, but place pins may not consistently represent the most useful real-world location of a place. A pin could be offset from the building centroid, storefront entrance, rooftop center, or other operationally useful point. Because the "correct" target itself is ambiguous, the project must first define the standard before it can measure error and improve placement quality.

## 2. Objectives

1. Establish a practical and defensible definition of a correct pin location
2. Build a ground-truth dataset of 500-1,000 labeled places from the provided ~3,425-place sample
3. Measure current spatial offset in Overture place data
4. Prototype modeling approaches to reposition place pins more accurately
5. Produce a recommendation for future production work

## 3. Dataset Overview

| Property | Value |
|----------|-------|
| **Records** | 3,425 Overture places |
| **Geography** | US-only, all 50 states (top: CA, TX, FL, NY, NC) |
| **Columns** | id, geometry (WKB point), bbox, type, version, sources, names, categories, confidence, websites, socials, emails, phones, brand, addresses |
| **Categories** | 100+ types — top: hotel (285), professional_services (181), accommodation (86), campground (74), lawyer (65), fast_food_restaurant (59) |
| **Confidence** | Mean 0.86, range 0.20-1.00 |
| **Primary source** | Meta/Facebook |

## 4. Approach

### Phase 1: Data Ingestion, Cleaning & EDA

- Load parquet data, decode WKB point geometries to lat/lon
- Flatten nested fields (names.primary, categories.primary, addresses)
- Identify and flag near-duplicates (same name within 50m, same address with different coordinates)
- Exploratory analysis: geographic distribution, category breakdown, confidence histogram, source analysis, null patterns
- Classify places by density tier (urban / suburban / rural)

**Output:** `notebooks/01_eda.ipynb`, `src/data_loader.py`

### Phase 2: Pin Location Definition & Taxonomy

Evaluate candidate pin definitions informed by EDA findings:

| Definition | Description | Pros | Cons |
|---|---|---|---|
| **Building Centroid** | Geometric center of building footprint | Consistent, automatable, widely available | May land in courtyard/interior; useless for strip malls |
| **Rooftop Centroid** | Center of rooftop footprint | Satellite-visible | Same issues as building centroid |
| **Main Entrance / Storefront** | Street-facing entrance | Most useful for navigation | Hard to label at scale; ambiguous for multi-entrance |
| **Nearest Road-Facing Point** | Closest point on footprint to nearest road | Good entrance proxy, automatable | May pick back alley, not main entrance |
| **Parcel Centroid** | Center of land parcel | Consistent | Often far from building; data inconsistent |
| **Address Geocode** | Lat/lon from geocoding street address | Universally available | Varies by geocoder quality; often road centerline |

**Recommended approach:** Category-aware hierarchical definition:
- Standard commercial places -> building centroid with road-facing refinement
- Multi-tenant (strip malls, office suites) -> storefront entrance (LLM-assisted)
- Open spaces (parks, campgrounds) -> parcel/area centroid
- Fallback -> best available geocode

**Output:** `docs/pin_definition_taxonomy.md`

### Phase 3: Ground Truth Construction (500-1,000 places)

**Strategy:** Use vision-capable LLMs (GPT-4o / Claude) to examine satellite imagery and identify correct pin locations at scale.

**Pipeline:**
1. **Stratified sampling** — select 750-1,000 places stratified by region, category, and density tier
2. **Fetch satellite imagery tiles** — via Google Maps Static API or Mapbox (~250m x 250m tiles at high zoom)
3. **LLM vision annotation** — prompt a vision LLM with:
   - Satellite image tile with current pin marked
   - Place name, address, category
   - Instructions to identify building entrance, storefront, or most appropriate location
   - Return coordinates as pixel offset from center
4. **Convert pixel offsets to lat/lon** using tile geographic bounds
5. **Confidence scoring** — LLM provides confidence; low-confidence places flagged for human review
6. **Cross-validation** — for ~100 places, compare LLM annotations against multi-geocoder consensus (Nominatim + Google + Mapbox)
7. **Inter-annotator agreement** — run 50 places through LLM twice (or two different LLMs) to measure consistency

**Output:** `data/processed/ground_truth.parquet`, `notebooks/02_ground_truth.ipynb`

### Phase 4: Baseline Offset Measurement

Compute the distance between current Overture pins and ground-truth locations:

- **Distance metrics:** Haversine distance (meters) — mean, median, p90, p95
- **Threshold accuracy:** % within 10m, 25m, 50m, 100m, 250m
- **Regression rate:** % of places where repositioning moves pin further from ground truth
- **Segmentation:** by category, region, urban/suburban/rural, confidence score bucket

**Output:** `notebooks/03_baseline_offset.ipynb`, `src/metrics.py`

### Phase 5: Prototype Repositioning Methods

#### Method 1: Multi-Geocoder Ensemble
- Query 3+ geocoding services (Nominatim, Google Maps Geocoding API, Mapbox) for each place's address
- Compute consensus position: weighted average or median
- Accept only when services agree within configurable radius (e.g., 25m)

#### Method 2: ML Candidate Ranking (XGBoost / Random Forest)
- Generate candidate positions: geocoded positions from multiple services, OSM building centroids, nearest-road-facing point, current pin
- Extract features per candidate:
  - Distance from current pin / geocoded consensus
  - Building area, perimeter, count in vicinity
  - Category encoding (one-hot)
  - Overture confidence score, source count
  - Road proximity, whether candidate is road-facing point
- Train ranking model on ground-truth dataset (80/20 stratified split)

#### Method 3: LLM-Augmented Contextual Reasoning
- Prompt LLM with place metadata + satellite imagery + building footprints + road network
- LLM reasons about building layout, entrance locations, and place type
- Returns repositioned lat/lon with explanation
- Most valuable for hard cases: multi-tenant buildings, strip malls, complexes

**Output:** `notebooks/04_repositioning.ipynb`, `src/geocoder_ensemble.py`, `src/candidate_ranker.py`, `src/llm_repositioner.py`

### Phase 6: Evaluation & Recommendation

**Comparison across all methods:**
- Same metrics as Phase 4 (mean/median/p90/p95, threshold accuracy)
- Regression rate per method
- Per-category and per-region performance
- Improvement over baseline (absolute and relative)
- Failure case analysis with examples
- Cost analysis (API calls, compute time)

**Success criteria:**
- At least one method reduces median offset by >= 30% vs. current baseline
- Regression rate below 10%
- Works across >= 80% of category types

**Output:** `notebooks/05_evaluation.ipynb`, `docs/recommendation.md`

## 5. Project Structure

```
Pin-To-Place/
├── data/
│   ├── raw/                        # Original parquet (3,425 places)
│   └── processed/                  # Ground truth, features, results
├── src/
│   ├── data_loader.py              # Data loading & WKB parsing
│   ├── geocoder.py                 # Multi-source geocoding
│   ├── geocoder_ensemble.py        # Method 1: geocoder ensemble
│   ├── satellite_fetcher.py        # Fetch satellite imagery tiles
│   ├── llm_annotator.py            # LLM vision ground truth annotation
│   ├── ground_truth.py             # Ground truth orchestration
│   ├── metrics.py                  # Offset metrics & evaluation
│   ├── features.py                 # Feature extraction for ML
│   ├── candidate_ranker.py         # Method 2: ML candidate ranking
│   └── llm_repositioner.py         # Method 3: LLM reasoning
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_ground_truth.ipynb       # Ground truth construction
│   ├── 03_baseline_offset.ipynb    # Baseline offset measurement
│   ├── 04_repositioning.ipynb      # Repositioning prototypes
│   └── 05_evaluation.ipynb         # Final evaluation & comparison
├── docs/
│   ├── project_plan.md             # This file
│   ├── pin_definition_taxonomy.md  # Pin location definitions
│   └── recommendation.md           # Final recommendation memo
├── requirements.txt
└── README.md
```

## 6. Key Dependencies & APIs

| Dependency | Purpose | Cost |
|---|---|---|
| Google Maps Static API | Satellite imagery tiles | ~$2 / 1,000 requests |
| Google Maps Geocoding API | High-quality geocoding | ~$5 / 1,000 requests |
| Mapbox Geocoding | Second geocoder source | Free tier: 100k/month |
| Nominatim (OSM) | Free geocoder | Free, 1 req/sec limit |
| OpenAI API (GPT-4o) / Anthropic API (Claude) | LLM vision for ground truth + repositioning | Variable |
| OSM / osmnx | Building footprints, road network | Free |

**Python packages:** pandas, pyarrow, geopandas, shapely, geopy, scikit-learn, xgboost, matplotlib, folium, osmnx, openai, anthropic

## 7. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| No universal definition of correct pin | Category-specific definitions with hierarchy of preferred targets |
| LLM vision annotations may be inconsistent | Inter-annotator agreement checks; multi-geocoder cross-validation; confidence scoring |
| Feature coverage varies by region | Design minimal-geometry baseline; layer richer features where available |
| Average error improves but outliers remain | Track p90/p95 and regression rate, not just mean |
| API costs exceed budget | Start with 50-place subset; estimate full-scale costs before scaling |
| OSM building footprint gaps in rural areas | Report coverage stats per region; fall back to geocode-based methods |

## 8. Verification Plan

1. **Data loading** — verify all 3,425 records load with valid US coordinates
2. **Deduplication** — check for and report near-duplicate counts
3. **Satellite tiles** — visually verify 10 fetched tiles show correct area
4. **LLM ground truth** — compare LLM annotations vs. multi-geocoder consensus on 100 places
5. **Inter-annotator** — run 50 places through LLM twice, report median disagreement
6. **Metrics** — unit test haversine against known distances
7. **Repositioning** — verify each method reduces median offset vs. baseline
8. **End-to-end** — run full pipeline on 50-place subset first, then scale
