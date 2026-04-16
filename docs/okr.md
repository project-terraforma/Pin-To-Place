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

Develop and train a specialized Neural Network architecture to correct and reposition precise geometric locations. 

### Key Results

1. Model training through building a neural network using the given datasets from Overture maps to create a ground truth dataset consisting of a minimum of $(n=750)$.
2. Ensure the Neural Network achieves a smaller "spatial offset" (distance error) than the most updated Spatial Repositioning projects at Project Terraforma.
3. Integrate a quickest-path feature based on the pin datasets and cost of getting there through machine measurements that produces top-3 results. 
