# Pin-To-Place: OKRs

**Project Terraforma @ UCSC** | Aaron J. Yam & Nathan Yu

---

## Objective 1: Define a defensible, measurable standard for correct pin placement

**Key Results:**
- KR1.1 — Publish a finalized pin location taxonomy covering all 4 place tiers (commercial, multi-tenant, open space, no-geometry) in `docs/pin_definition_taxonomy.md`
- KR1.2 — Achieve inter-annotator agreement (median disagreement < 15m) across 50 places annotated twice by the LLM
- KR1.3 — Validate LLM annotations against multi-geocoder consensus on 100 places, with LLM vs. consensus median distance < 30m

---

## Objective 2: Build a high-quality ground truth dataset

**Key Results:**
- KR2.1 — Produce a stratified sample of 750 labeled places covering all 50 US states and top category groups
- KR2.2 — Successfully annotate ≥ 90% of sampled places (≥ 675 of 750) with valid LLM-assigned coordinates
- KR2.3 — Flag and human-review all places where LLM confidence < 0.5 (target: < 10% of total sample)
- KR2.4 — Save final ground truth dataset to `data/processed/ground_truth.parquet`

---

## Objective 3: Quantify the current accuracy of Overture Maps pin placements

**Key Results:**
- KR3.1 — Compute baseline offset metrics (mean, median, p90, p95) for all 750 ground-truth places
- KR3.2 — Produce a segmented baseline report broken down by category group and US region
- KR3.3 — Identify the top 3 category types and regions with the highest median offset

---

## Objective 4: Prototype and evaluate three repositioning methods

**Key Results:**
- KR4.1 — Run the multi-geocoder ensemble (Method 1) on the full ground-truth set and report offset metrics vs. baseline
- KR4.2 — Train and evaluate the ML candidate ranker (Method 2) with AUC-ROC ≥ 0.70 on the held-out test set
- KR4.3 — Run LLM contextual repositioning (Method 3) on ≥ 200 places and report offset metrics vs. baseline
- KR4.4 — At least one method achieves median offset reduction ≥ 30% vs. baseline with regression rate < 10%

---

## Objective 5: Deliver a clear recommendation for production use

**Key Results:**
- KR5.1 — Publish a final evaluation comparison of all three methods across mean, median, p90, regression rate, cost-per-place
- KR5.2 — Document a recommended production pipeline in `docs/recommendation.md`, including which method(s) to deploy and under what conditions
- KR5.3 — Complete all 5 analysis notebooks (`01_eda` through `05_evaluation`) with full outputs

---

## Success Criteria (from project plan)

| Criterion | Target |
|---|---|
| Median offset reduction | ≥ 30% vs. current baseline |
| Regression rate | < 10% |
| Category coverage | Works across ≥ 80% of category types |
| Ground truth coverage | ≥ 675 annotated places |
