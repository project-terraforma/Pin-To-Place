# Recommendation

## Summary

Pin-To-Place should evolve from single-coordinate positional accuracy into task-aware place pinning.

The project results suggest that one universal pin is not sufficient for all places. Some places are well represented by the current Overture pin, while others need different targets depending on the arrival task: pedestrian entry, vehicle entry, delivery access, accessible entry, privacy-preserving approximate location, or area centroid.

## Evidence So Far

### Baseline Dataset

- Total records: `3,425` places (6,836 after dual-label expansion)
- Baseline median offset: `0.0m`
- Mean offset: `2.69m`
- p90 offset: `20.02m`
- p95 offset: `23.34m`
- Exact no-move rate: `88.5%`
- Places marked as likely movable: `675`

The baseline median offset is already `0.0m`, so median improvement is not a useful success metric. Future evaluation should focus on p90/p95 error, regression rate, arrival cost, manual review status, and whether a place needs multiple pins.

### Task-Aware Evaluation

Task-aware evaluation introduced:

- `place_complexity`
- `pin_ambiguity`
- `should_move`
- `arrival_cost_m`

Arrival-cost scoring showed that raw distance alone does not capture usefulness. Open spaces and complex places had substantially worse arrival-cost profiles than simple commercial places.

- Overall arrival-cost p95: `25.34m`
- Open-space median arrival cost: `25.0m`
- Open-space p95 arrival cost: `48.34m`; 100% of open-space places have arrival cost >10m
- Complex-place median arrival cost: `10.0m`
- Complex-place p95 arrival cost: `47.91m`

This suggests that campgrounds, parks, resorts, shopping centers, and other complex places are where single-coordinate pinning struggles most.

### Manual Review Pilot

A 103-row manual review pilot was built from high-risk queues:

- high-offset examples: 23
- low-confidence examples: 30
- multi-tenant examples: 25
- zero-offset control examples: 25

Tier breakdown: 36 no-building, 35 standard-commercial, 26 multi-tenant, 6 open-space.

Complexity breakdown: 90 simple, 9 complex, 4 multi-tenant.

Ambiguity breakdown: 53 low, 44 high, 6 medium.

`should_move` from automated evaluation: 23 true, 80 false.

Multi-pin need (based on complexity/ambiguity criteria): 9 flagged true.

Note: `manual_review_status`, `manual_should_move`, and `manual_primary_pin_type` columns are not yet filled with human labels. The pilot file is ready for visual review.

### Multi-Pin Proxy Review Findings

A 9-row multi-pin pilot was created from places flagged `manual_needs_multi_pin = true`: 5 open-space/RV parks, 2 multi-tenant complexes, 2 multi-tenant standard commercial. All 9 are either `complex` or `multi_tenant` complexity with `high` or `medium` ambiguity.

The automated proxy review found:

- `4` rows with pedestrian-entry proxy labels (`proxy_pedestrian_accepted`)
- `0` rows with vehicle-entry proxy labels (none had vehicle entry coordinates in the LLM ground truth)
- `5` rows flagged `needs_human_review` (high priority)
- `4` rows accepted by proxy review (medium priority)

Rows requiring human visual review (high priority):
- Chateau Burg RV Resort
- D'Iberville Memorial Park
- Arrowhead RV Park
- Plantation Place Dallas RV Park
- Winn Creek RV Park

Rows accepted by proxy review (medium priority):
- World Market
- Southgate Shopping Center
- FUM Child Learning Center
- Hicks Flooring Carpet One Floor & Home

No pedestrian/vehicle separation distance is available yet because vehicle entry coordinates have not been labeled for any pilot row. Adding vehicle-entry coordinates for the 9 pilot rows is the immediate next step.

These results are proxy labels derived from existing LLM ground truth. They should be treated as workflow validation, not final ground truth. The next validation step is human visual review of the five high-priority RV park / open-space rows.

## Main Finding

Place pin quality is not only a coordinate accuracy problem.

Some places should move to a more useful customer entrance or access point. Some should not move because the current pin is already useful. Some are privacy-sensitive and should avoid over-precise correction. Others are ambiguous and may require multiple task-specific pins.

This supports the central project claim:

> A single place pin is not always sufficient. The best pin depends on the arrival task.

## Recommended Production Direction

1. Preserve current pins when they are already useful.
2. Move pins only when there is strong evidence of a better arrival target.
3. Treat privacy-sensitive and no-building places conservatively.
4. Use multiple pin types for ambiguous places where one coordinate is insufficient.
5. Prioritize open-space, complex, and multi-tenant places for deeper review.
6. Measure success with p90/p95 offset, arrival cost, regression rate, manual review status, and multi-pin need rate.

## Next Steps

1. **Fill human labels in `manual_review_pilot.csv`**: open the file and complete `manual_review_status`, `manual_should_move`, `manual_primary_pin_type`, and `manual_notes` for each of the 103 rows. This is the most urgent gap — nothing downstream has real human labels yet.
2. **Add vehicle-entry coordinates for the 9 multi-pin pilot rows** in `multipin_pilot.csv`. All 9 are physical places (RV parks, shopping centers, daycares) where a separate vehicle/parking entrance exists. Without these, pedestrian/vehicle separation distances cannot be computed.
3. **Visually review the 5 high-priority open-space rows** (Chateau Burg RV Resort, D'Iberville Memorial Park, Arrowhead RV Park, Plantation Place Dallas RV Park, Winn Creek RV Park) using satellite/street-level imagery to assign true pedestrian and vehicle pin coordinates.
4. **Build a regression-safe `should_move` classifier** using the completed human labels from step 1. Features (`place_complexity`, `pin_ambiguity`, `tier_label`, `arrival_cost_m`, `cv_disagreement_m`) are already in the dataset.
5. **Expand selectively** into open spaces, resorts, hotels, and shopping centers once the pilot ground truth is trusted.

## Final Recommendation

Do not expand the dataset randomly yet. The immediate priority is filling human labels into the existing 103-row pilot, then adding vehicle-entry coordinates for the 9 multi-pin rows. Once those are complete, the separation distances can be computed and a `should_move` classifier can be trained on trusted labels rather than proxy heuristics.
