# Presentation Brief

## Project Thesis

A single place pin is not always sufficient. The best pin depends on the arrival task: pedestrian entry, vehicle entry, delivery access, accessible entry, privacy-sensitive approximation, or open-space access point.

## Dataset

- `3,425` Overture Maps place records
- US-only sample
- Categories include hotels, professional services, accommodations, campgrounds, restaurants, retail, and service businesses
- Full LLM-generated ground truth exists for all records

## Baseline Findings

- Baseline median offset: `0.0m`
- Baseline mean offset: `2.69m`
- Baseline p90 offset: `20.02m`
- Baseline p95 offset: `23.34m`
- Exact no-move rate: `88.5%`
- Likely movable places: `675`

The median offset being `0.0m` means median improvement is not the right objective. A production system should avoid moving already-good pins and focus on p90/p95 error, arrival cost, ambiguity, and regression safety.

## Task-Aware Evaluation

The project added:

- `place_complexity`
- `pin_ambiguity`
- `should_move`
- `arrival_cost_m`

Arrival-cost findings:

- Overall arrival-cost p95: `25.34m`
- Open-space median arrival cost: `25.0m`; 100% of open-space places exceed 10m arrival cost
- Open-space p95 arrival cost: `48.34m`
- Complex-place median arrival cost: `10.0m`
- Complex-place p95 arrival cost: `47.91m`

Open spaces and complex places are where single-coordinate pinning struggles most.

## Manual Review Pilot

A `103`-row pilot was built from high-offset, low-confidence, multi-tenant, and zero-offset control examples (23 / 30 / 25 / 25 respectively).

Tier breakdown: 36 no-building, 35 standard-commercial, 26 multi-tenant, 6 open-space.

Automated evaluation flags: 23 rows `should_move = true`, 9 rows flagged for multi-pin review.

Human labels (`manual_review_status`, `manual_should_move`, `manual_primary_pin_type`) are not yet filled — the pilot is ready for visual review.

## Multi-Pin Pilot

A `9`-row multi-pin pilot was created from places flagged `manual_needs_multi_pin = true` (5 open-space/RV parks, 2 multi-tenant complexes, 2 multi-tenant standard commercial).

Proxy review results:

- `4` rows have pedestrian-entry proxy labels (`proxy_pedestrian_accepted`)
- `0` rows have vehicle-entry proxy labels (not yet labeled)
- `5` rows flagged `needs_human_review` (high priority)
- `4` rows accepted by proxy review (medium priority)

Pedestrian/vehicle separation cannot yet be computed — vehicle-entry coordinates need to be added for the 9 pilot rows.

## High-Priority Rows For Visual Review

- Chateau Burg RV Resort
- D'Iberville Memorial Park
- Arrowhead RV Park
- Plantation Place Dallas RV Park
- Winn Creek RV Park

These are open-space/RV park places where pedestrian and vehicle arrival targets are likely to be meaningfully separated but cannot be confirmed without satellite or street-level imagery.

## Recommendation

1. Preserve current pins when they are already useful.
2. Move pins only when there is strong evidence of a better arrival target.
3. Treat privacy-sensitive and no-building places conservatively.
4. Use multiple pin targets for ambiguous places where one coordinate is insufficient.
5. Prioritize open-space, complex, and multi-tenant places for deeper review.
6. Measure p90/p95 offset, arrival cost, regression rate, manual review status, and multi-pin need rate.

## Limitations

- Multi-pin labels are proxy labels, not final visual ground truth.
- Delivery and accessible-entry pins were not filled in the pilot because they require clearer visual or street-level evidence.
- The five high-priority rows still need true visual review.
- LLM annotations should be treated as candidate labels, not unquestioned truth.

## Next Research Step

1. Fill human labels into `manual_review_pilot.csv` (103 rows).
2. Add vehicle-entry coordinates for the 9 multi-pin pilot rows to enable pedestrian/vehicle separation measurement.
3. Visually validate the five high-priority open-space rows.
4. Once human labels exist, train a regression-safe `should_move` classifier and expand selectively into open spaces, hotels, resorts, shopping centers, and campgrounds.
