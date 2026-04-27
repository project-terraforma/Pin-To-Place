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

1. Reduce median spatial offset error by 20% compared to the current Project Terraforma benchmark across the n=750 ground truth dataset.
2. Reach a training loss of < 0.05 (normalized MSE) and validation accuracy of > 88% on the Overture Maps dataset.
3. Successfully integrate the "Quickest Path" feature, ensuring it reduces the calculated "cost of arrival" error by at least 15% in top-3 measured results compared to baseline pin data — measured separately for car-arrival (road-facing entry) and pedestrian-arrival (sidewalk/accessible entry) paths, with LLM satellite annotation enriched to detect visible pedestrian infrastructure (sidewalks, curb cuts, accessible paths) near candidate pin locations.
