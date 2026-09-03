# Topology-Guided Multi-Modal Semantic Reconstruction of Smart Homes from Robot and Human Trajectories

## Abstract

Smart homes increasingly generate heterogeneous sensing data, including robot cleaning maps, IoT device coordinates, and long-term human movement trajectories. However, these signals remain fragmented and rarely form a unified semantic spatial model usable for higher-level reasoning and context-aware applications.  

We present **ymj_TS**, a topology-guided multi-modal system that automatically reconstructs a structured semantic home layout from heterogeneous smart-home data sources. Our approach combines (1) complementary perception signals from robot-generated floor maps and trajectory topology, (2) a constrained large language model (LLM) correction protocol to ensure structural consistency, and (3) hierarchical behavior-aware spatial type inference.  

Evaluated on **XXX real-world households** with **XXX days of trajectory data**, our system achieves **XXX% Room Type Accuracy**, **XXX F1** for furniture detection, and significantly outperforms vision-only and LLM-only baselines. We further demonstrate that the reconstructed layout improves downstream activity recognition by **XXX%** compared to geometry-only baselines.  

Our results suggest that topology-aware fusion of ambient smart-home data enables scalable, low-cost semantic spatial reconstruction without requiring CAD models or manual annotation.

---

# 1. Introduction

Smart homes today contain multiple sensing modalities:

- Robot-generated occupancy maps  
- IoT device coordinates  
- Long-term human movement trajectories  

Despite this data abundance, most smart environments lack a structured semantic representation of indoor space. Applications such as activity recognition, anomaly detection, elder care monitoring, and energy optimization rely on semantic knowledge of room types and furniture placement. However, constructing such models typically requires:

- Manual CAD input
- Expensive 3D scanning
- Human annotation

We ask:

> Can a smart home automatically reconstruct its own semantic layout from ambient sensing data already available in consumer systems?

We introduce **ymj_TS**, a multi-modal system that reconstructs a structured YAML-based semantic layout using:

- Robot floor topology
- Smart device priors
- Human behavioral trajectories

Unlike prior vision-centric approaches, we leverage **trajectory topology as a complementary perception signal**. Furthermore, instead of allowing large language models to freely generate spatial structures, we introduce a **constrained action protocol** that limits structural edits to ensure consistency.

Our contributions are:

1. **Topology-aware dual-source object proposal** that leverages both structural shading and trajectory loops for complementary furniture detection.
2. **Constrained LLM structural correction protocol** to reduce hallucination in spatial reasoning.
3. **Behavior-aware hierarchical room type inference** tailored for smart-home context modeling.
4. A real-world evaluation across **XXX households** demonstrating improved semantic reconstruction and downstream application benefits.

---

# 2. Background and Motivation

## 2.1 Fragmented Smart Home Data

Modern households generate spatially meaningful signals:

- Robot vacuum maps encode walls and free space.
- Smart appliances encode semantic priors (e.g., refrigerator → kitchen).
- Human trajectories encode room usage patterns.

Yet these are rarely fused into a coherent semantic spatial model.

## 2.2 Limitations of Existing Methods

Vision-based reconstruction approaches:

- Require RGB or depth images.
- Often rely on large annotated datasets.
- Struggle with occlusion and sparse sensor input.

LLM-only approaches:

- Generate plausible but structurally inconsistent outputs.
- Suffer from hallucination.
- Lack geometric grounding.

Our work bridges these by combining topology, geometry, and constrained semantic reasoning.

---

# 3. System Overview

Our pipeline consists of five stages:

1. **Room segmentation from robot maps**
2. **Dual-source furniture proposal**
3. **Constrained LLM structural correction**
4. **Hierarchical room type inference**
5. **Behavior-aware furniture naming**

The output is a structured semantic layout containing:

- Room types
- Spatial extents
- Furniture positions
- Behavior annotations

---

# 4. Topology-Guided Dual-Source Furniture Detection

## 4.1 Complementary Perception Signals

We observe two distinct spatial cues:

### Hatch-Based Structural Signals

Shaded regions in robot maps often indicate large furniture.

### Trajectory Loop Topology

Human and robot trajectories form closed loops around obstacles.

We define:

Loop Cluster Count = number of independent closed trajectory clusters within a bounding box.

High loop density suggests obstacle presence even without shading.

---

## 4.2 Fusion Strategy

We generate candidate rectangles from:

- Hatch regions (high confidence)
- Trajectory-enclosed regions (complementary recall)

We apply:

- IoU-based deduplication
- Size filtering
- Topology-aware merging

Ablation shows removing trajectory loops reduces small-object recall by **XXX%**.

---

# 5. Constrained LLM Structural Correction

Instead of free-form generation, we define a restricted action space:

A = {merge, delete, adjust}

The LLM receives structured candidate descriptions and may only operate within A.

This prevents:

- Invention of non-existent furniture
- Geometrically invalid placements

Compared to unrestricted LLM prompting:

- Structural consistency improves by **XXX%**
- Hallucination rate drops from **XXX% to XXX%**

---

# 6. Behavior-Aware Hierarchical Room Inference

We integrate three priors:

1. Device-based deterministic priors (e.g., washing machine → bathroom)
2. Behavioral temporal priors (night-dominant → bedroom)
3. Connectivity and spatial scale heuristics

The layered priority framework improves Room Type Accuracy by **XXX%** over flat prompting.

---

# 7. Implementation and Deployment

We implemented ymj_TS in Python (≈XXX LOC).

Input sources:

- Consumer robot maps (PNG)
- IoT device YAML
- Trajectory JSON logs

Deployment requires no additional hardware.

Average runtime per household: **XXX seconds**.

---

# 8. Evaluation

## 8.1 Dataset

We collected data from:

- XXX households
- XXX total rooms
- XXX furniture instances
- XXX days of trajectory logs

Homes varied in:

- Area (XX–XX m²)
- Room count (X–X)
- Occupancy patterns

Ground truth layouts were manually annotated.

---

## 8.2 Metrics

- Room Type Accuracy (RTA)
- Furniture Detection F1@IoU>0.2
- Furniture Naming Accuracy
- Complete Layout Score (CLS)

---

## 8.3 Baselines

1. Vision-only (CV-only)
2. LLM direct generation
3. Free-form LLM correction
4. Ours (Constrained + Multi-modal)

---

## 8.4 Results

| Method | RTA | F1 | CLS |
|--------|-----|----|-----|
| CV-only | XXX | XXX | XXX |
| LLM-only | XXX | XXX | XXX |
| Ours | **XXX** | **XXX** | **XXX** |

---

## 8.5 Downstream Application Study

We evaluated activity recognition using:

- Geometry-only baseline
- Reconstructed semantic layout

Semantic reconstruction improved recognition F1 by **XXX%**.

---

# 9. Discussion

## 9.1 Privacy Considerations

Trajectory data were anonymized.
No RGB imagery was used.
All processing can run locally.

## 9.2 Limitations

- Performance degrades with extremely sparse trajectories.
- Multi-floor layouts are not yet supported.
- Relies on availability of robot maps.

---

# 10. Conclusion

We demonstrate that smart homes can automatically reconstruct semantic spatial layouts from ambient sensing data without manual CAD input. By leveraging topology-aware perception and constrained LLM reasoning, ymj_TS enables scalable semantic modeling for ubiquitous computing applications.

---

