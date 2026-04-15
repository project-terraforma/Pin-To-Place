# Objectives and Key Results

---

## Objective #1

Establish a defensible definition and ground truth dataset for correct place pin locations across Overture Maps data.

### Key Results

1. Finalize a category-aware pin location taxonomy covering all 4 place tiers (commercial, multi-tenant, open space, no-geometry) in `docs/pin_definition_taxonomy.md`
2. Produce a stratified ground truth dataset of **750 labeled places** annotated via LLM vision across all 50 US states
3. Achieve inter-annotator agreement with a median LLM disagreement of **< 15m** across 50 cross-validated places

---

## Objective #2

Deliver a repositioning pipeline that measurably improves place pin accuracy over the current Overture Maps baseline.

### Key Results

1. Measure the current baseline offset (mean, median, p90) for all 750 ground truth places segmented by category and region
2. Prototype and evaluate 3 repositioning methods (geocoder ensemble, ML ranker, LLM reasoning) and achieve a **≥ 30% reduction in median offset** vs. baseline with a regression rate **< 10%**
3. Publish a production recommendation in `docs/recommendation.md` identifying the best method(s) by category type and cost-per-place
