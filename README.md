# Representative Frame Selection for Desmond Trajectories

![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Schr%C3%B6dinger-green)
![Method](https://img.shields.io/badge/Method-PCA%20%2B%20Clustering-purple)
![Status](https://img.shields.io/badge/Status-Research%20Workflow-orange)

An automated workflow for extracting, aligning, clustering, and selecting representative protein conformations from **Desmond molecular dynamics (MD) trajectories**.

---

## Overview

This script provides an end-to-end workflow for reducing large MD trajectory datasets into a compact and structurally meaningful set of representative frames.

The pipeline is designed to identify dominant and alternative conformational states by combining:

- structural alignment to a reference structure
- RMSD-based equilibration inspection
- PCA-based dimensionality reduction
- agglomerative clustering
- consensus-based cluster model selection

Two final representative conformations are selected:

- **REP1** — medoid of the most populated cluster
- **REP2** — a structurally distinct and temporally meaningful alternative state

---

## Key Method Choices

### 1. Alignment Strategy

All frames are aligned to a reference structure using a user-defined fitting ASL.

Default fitting selection:

`protein and backbone and not atom.ele H`

This removes global translation/rotation so that clustering focuses on internal structural variation.

### 2. Feature Representation

Two feature modes are supported:

- **`ca_distances`**  
  Pairwise distances between selected atoms  
- **`ca_xyz`**  
  Flattened Cartesian coordinates of selected atoms

Default feature selection:

`protein and atom.ptype CA`

### 3. Cluster Model Selection

The optimal number of clusters is selected using a weighted consensus of two internal validation metrics:

| Metric | Weight | Role |
|---|---:|---|
| Silhouette score | 70% | Favors well-separated clusters |
| Inverse-normalized Davies–Bouldin index | 30% | Favors compact and separated clusters |

Consensus formula:

```text
consensus_score = 0.70 * normalized_silhouette
                + 0.30 * normalized_inverse_dbi
```

### 4. Representative Selection

Representative structures are defined as **cluster medoids**.

- **REP1** is chosen from the largest cluster
- **REP2** is selected from alternative clusters using a consensus of:
  - occupancy
  - temporal continuity
  - structural difference from REP1

---

## Requirements

This script must be run inside the **Schrödinger Python environment**.

### Dependencies

- Schrödinger Python runtime
- Desmond trajectory tools (`trj2mae.py`)
- `numpy`
- `matplotlib`
- `scikit-learn`

---

## Input

The script requires:

- a **CMS file**
- a **Desmond trajectory directory**
- a **reference structure**

Default reference file:

`experimental_structure.mae`

---

## Usage

### Basic Command

```bash
$SCHRODINGER/run rep_select.py \
  --cms system-out.cms \
  --trj system_trj \
  --ref experimental_structure.mae
```

### Preview Only

```bash
$SCHRODINGER/run rep_select.py \
  --cms system-out.cms \
  --trj system_trj \
  --ref experimental_structure.mae \
  --preview-only
```

### Example with Custom Sampling

```bash
$SCHRODINGER/run rep_select.py \
  --cms system-out.cms \
  --trj system_trj \
  --ref experimental_structure.mae \
  --preview-stride 20 \
  --cluster-stride 50
```

---

## Main Options

| Argument | Description |
|---|---|
| `--cms` | Desmond CMS file |
| `--trj` | trajectory directory |
| `--ref` | reference structure file |
| `--preview-stride` | frame stride for RMSD preview |
| `--cluster-stride` | frame stride for clustering |
| `--start-frame` | clustering start frame (`auto` supported) |
| `--end-frame` | clustering end frame (`auto` and `-1` supported) |
| `--fit-asl` | ASL used for structural alignment |
| `--feature-asl` | ASL used for feature generation |
| `--feature-mode` | `ca_distances` or `ca_xyz` |
| `--k-min`, `--k-max` | clustering search range |
| `--preview-only` | generate RMSD preview only |

---

## Output

Results are written to:

`Clustering_Results/`

Typical outputs include:

- RMSD preview plot and table
- extracted raw frames
- aligned trajectory frames
- cluster medoid structures
- final **REP1** and **REP2** structures
- clustering summaries
- quality report
- temporal continuity report

---

## Output Structure

```text
Clustering_Results/
├── 0_preview_frames/
├── 1_raw_frames/
├── 2_aligned_frames/
├── 3_representatives/
├── 3_top2_representatives/
├── 0_rmsd_preview.csv
├── 0_rmsd_preview.png
├── 1_k_metrics_summary.csv
├── 1b_pca_summary.csv
├── 2_cluster_assignments.csv
├── 3_cluster_summary.csv
├── 4_quality_report.txt
├── 5_representative_ranking.csv
├── 6_representative_pair_metrics.csv
└── 7_temporal_continuity_report.csv
```

---

## Notes

- The script must be executed with the **Schrödinger runtime**
- If the runtime is not already active, the script attempts to relaunch itself automatically
- Consistent ASL atom identity and ordering are required across extracted frames
- Large atom selections may increase runtime and memory usage
  
### Citation
If you use this tool in your research or publication, please cite it as follows:

İsaoğlu, M., & Durdağı, S. (2026). MD Protein Clustering Tool (Version 1.0) [Source Code]. https://github.com/DurdagiLab/md-protein-clustering-pipeline
