# Objectives and Key Results

---

## Objective #1

Establish a defensible definition and ground truth dataset for correct place pin locations across Overture Maps data.

### Key Results

1. Finalize a category-aware pin location taxonomy covering all 4 place tiers (commercial, multi-tenant, open space, no-geometry) in `docs/pin_definition_taxonomy.md`, extended to include **mode-of-transport pin variants** — distinct car-entry and pedestrian-entry pin targets where applicable (e.g., parking lot entrance vs. sidewalk/storefront entrance)
2. Produce a stratified ground truth dataset of **750 labeled places** annotated via LLM vision across all 50 US states, with car-entry and pedestrian-entry coordinates labeled separately for all Tier 1 and Tier 2 commercial places
3. Achieve inter-annotator agreement with a median LLM disagreement of **< 15m** across 50 cross-validated places

---

## Objective #2

Surpass the current Project Terraforma precision in geometric place-pin placement through specialized neural architecture.

### Key Results

1. Pivot from aggregate error reduction to baseline protection: Maintain the 0.0m median baseline offset while ensuring the candidate ranker's regression rate remains strictly below 1% across all test sets.
2. Reach a training loss of < 0.05 (normalized MSE) and validation accuracy of > 88% on the Overture Maps dataset.
3. Successfully integrate the "Quickest Path" feature, ensuring it reduces the calculated "cost of arrival" error by at least 15% in top-3 measured results compared to baseline pin data — measured separately for car-arrival (road-facing entry) and pedestrian-arrival (sidewalk/accessible entry) paths, with LLM satellite annotation enriched to detect visible pedestrian infrastructure (sidewalks, curb cuts, accessible paths) near candidate pin locations.
