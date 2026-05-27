# Recommendation

## Summary

Pin-To-Place should evolve from single-coordinate positional accuracy into task-aware place pinning.

The project results suggest that one universal pin is not sufficient for all places. Some places are well represented by the current Overture pin, while others need different targets depending on the arrival task: pedestrian entry, vehicle entry, delivery access, accessible entry, privacy-preserving approximate location, or area centroid.

## Evidence So Far

### Baseline Dataset

- Total records: `3,425`
- Baseline median offset: `0.0m`
- Mean offset: `6.4m`
- p90 offset: `37.55m`
- p95 offset: `40.27m`
- Exact no-move rate: `79.6%`
- Places marked as likely movable: `675`

The baseline median offset is already `0.0m`, so median improvement is not a useful success metric. Future evaluation should focus on p90/p95 error, regression rate, arrival cost, manual review status, and whether a place needs multiple pins.

### Task-Aware Evaluation

Task-aware evaluation introduced:

- `place_complexity`
- `pin_ambiguity`
- `should_move`
- `arrival_cost_m`

Arrival-cost scoring showed that raw distance alone does not capture usefulness. Open spaces and complex places had substantially worse arrival-cost profiles than simple commercial places.

- Overall arrival-cost p95: `42.61m`
- Open-space median arrival cost: `25.0m`
- Open-space p95 arrival cost: `65.44m`
- Complex-place median arrival cost: `19.62m`
- Complex-place p95 arrival cost: `63.98m`

This suggests that campgrounds, parks, resorts, shopping centers, and other complex places are where single-coordinate pinning struggles most.

### Manual Review Pilot

A 118-row manual review pilot was built from high-risk queues:

- high-offset examples
- low-confidence examples
- multi-tenant examples
- zero-offset control examples

Pilot review status:

- `privacy_sensitive`: 38
- `accepted`: 31
- `wrong_target`: 28
- `ambiguous`: 21

Movement decision:

- `false`: 69
- `true`: 41
- `unknown`: 8

Primary pin type:

- `current`: 69
- `pedestrian_entry`: 41
- `vehicle_entry`: 8

Multi-pin need:

- `false`: 97
- `true`: 21

Key pilot rates:

- `34.7%` should move
- `17.8%` need multiple pins
- `32.2%` are privacy-sensitive

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

1. Add true multi-pin latitude/longitude labels for the 21 pilot rows marked `manual_needs_multi_pin = true`.
2. Compare pedestrian-entry and vehicle-entry targets on those rows.
3. Build a regression-safe `should_move` classifier using reviewed labels.
4. Evaluate candidate repositioning methods only on trusted, reviewed labels.
5. Expand selectively into open spaces, resorts, hotels, shopping centers, campgrounds, and other complex places.

## Final Recommendation

Do not expand the dataset randomly yet. First deepen the task-aware pilot by labeling true pedestrian and vehicle pin targets for the subset where one pin is insufficient. Once that pilot proves which categories need multiple pins, expand selectively into the categories where single-coordinate pinning fails most clearly.
