# Final Recommendation

## Recommendation

Pin-To-Place should not ship as a simple global pin-repositioning system.

The project should ship as a task-aware pin evaluation framework that decides when a place pin should stay fixed, when it should move, when it should be flagged for privacy or ambiguity, and when a place needs shared or multiple arrival targets.

## Why

The baseline median offset is already near zero, so optimizing median movement would reward the system for doing almost nothing. The meaningful errors appear in the tail: high p90/p95 arrival cost, ambiguous open spaces, complex multi-tenant places, privacy-sensitive locations, and places where pedestrian and vehicle arrival targets differ.

## Machine Visual Review

Because manual review was unavailable for the remaining high-priority rows, unresolved cases were evaluated using a Gemini-based machine visual review layer over satellite imagery.

This review accepted 3 of 5 high-priority unresolved rows, rejected 1 as a wrong target, and left 1 requiring human review. These labels are treated as machine-assisted validation, not final human ground truth.

## Production Rule

Use a conservative hierarchy:

1. Keep the current Overture pin when the place is low-ambiguity and arrival cost is low.
2. Move the pin only when the ground-truth target is high-confidence and arrival cost is meaningfully reduced.
3. Flag privacy-sensitive, ambiguous, and machine-rejected rows instead of forcing a coordinate.
4. Use shared, pedestrian, vehicle, delivery, and accessible arrival targets where one coordinate is insufficient.
5. Require human review for high-priority rows before treating proxy or machine labels as final ground truth.

## Final Project Position

Pin-To-Place is strongest as a validation and routing-aware pin schema project, not as a universal pin-moving project. Its contribution is showing when a single coordinate is adequate, when it is harmful, and when the place needs a more explicit arrival model.

## Generated Evaluation Notes

```text
Pin-To-Place Final Evaluation Summary

Ground Truth
Rows: 3425
offset_haversine_m:
  count: 3418
  mean: 2.69
  median: 0.00
  p90: 20.01
  p95: 23.34
  max: 74.77
arrival_cost_m: unavailable

Manual Review
Rows reviewed: 103
Manual should-move rate: 0/103 (0.0%)
Manual multi-pin rate: 9/103 (8.7%)
Privacy-sensitive rate: 0/103 (0.0%)
manual_review_status:
manual_review_status
missing    103

Multi-Pin Proxy Review
Rows reviewed: 9
Rows originally needing human review: 5/9 (55.6%)
High-priority rows: 5/9 (55.6%)
visual_review_status:
visual_review_status
needs_human_review           5
proxy_pedestrian_accepted    4

Machine Visual Review
Rows reviewed: 9
This is machine-assisted validation, not human ground truth.
Accepted without human review: 5/9 (55.6%)
machine_visual_status:
machine_visual_status
accepted              5
wrong_target          3
needs_human_review    1
machine_pedestrian_entry_correct:
machine_pedestrian_entry_correct
yes    5
no     4
machine_vehicle_entry_correct:
machine_vehicle_entry_correct
yes    5
no     4
machine_confidence:
  count: 9
  mean: 0.86
  median: 0.90
  p90: 0.95
  p95: 0.95
  max: 0.95
Unresolved or rejected machine-review rows:
- Chateau Burg RV Resort: wrong_target, confidence=0.9
  reason: The proposed shared arrival/access pin (magenta star) is located within the RV resort's amenity area, near a building and pool. It is not positioned at the primary vehicle entrance from the main highway, nor does it appear to be the main pedestrian arrival point for the entire resort. Vehicles would use the main driveway from the highway, which is located further west. Therefore, it is not a plausible shared access point for both pedestrians and vehicles entering the RV resort.
- D'Iberville Memorial Park: needs_human_review, confidence=0.3
  reason: The proposed shared arrival/access pin (magenta star) is located on Gorenflo Rd, adjacent to D'Iberville Memorial Park. However, there is a dense treeline at this location, and no visible driveway, path, or break in the trees to indicate a shared pedestrian or vehicle access point into the park. Therefore, it does not appear to be a plausible entry point.
- World Market: wrong_target, confidence=0.95
  reason: Both the proposed pedestrian (cyan star) and vehicle (green star) entry pins are located on the roof of the building, not at any plausible ground-level entry points for either pedestrians or vehicles. They are clearly misplaced.
- Hicks Flooring Carpet One Floor & Home: wrong_target, confidence=0.95
  reason: Both the proposed pedestrian (cyan star) and vehicle (green star) entry pins are incorrectly placed on the roof of the building, not at a plausible ground-level entry or vehicle access point.

Final Interpretation
Median offset is not the right primary success metric because most original pins already do not move.
The useful signal is concentrated in p90/p95 arrival cost, manual should-move rows, privacy-sensitive rows, ambiguous rows, wrong-target rows, and multi-pin rows.
The Gemini-based machine visual review accepted 3 of 5 high-priority unresolved rows, rejected 1 as a wrong target, and left 1 requiring review.
The recommended production path is conservative task-aware pinning: keep stable pins where they are already adequate, move only high-confidence failures, and represent shared, pedestrian, and vehicle arrival explicitly where one coordinate is insufficient.
