# Representative Frame Selection for Desmond MD Trajectories

A Python pipeline for selecting representative protein conformations from **Desmond molecular dynamics trajectories** using structural alignment, PCA-based dimensionality reduction, hierarchical clustering, and consensus-based model selection.

This workflow reduces large trajectory datasets into a compact set of **structurally meaningful representative frames** that capture both the dominant conformational state and an alternative relevant state.

## Overview

The script performs the following steps:

1. Verifies or bootstraps the **Schrödinger runtime**
2. Extracts trajectory frames from a **Desmond CMS + trajectory directory**
3. Generates an **RMSD preview** against a reference structure
4. Applies **automatic or user-defined trajectory trimming**
5. Validates **ASL consistency** and atom ordering
6. Performs a **PBC / broken-protein sanity check**
7. Aligns frames to the reference structure
8. Builds structural features
9. Applies **PCA** and **agglomerative clustering**
10. Selects the optimal number of clusters using a **weighted consensus score**
11. Identifies cluster medoids and selects **REP1** and **REP2**
12. Exports representative structures, plots, and summary reports

## Representative Selection Strategy

Two final representative structures are selected:

- **REP1**: the medoid of the **most populated cluster**
- **REP2**: a **structurally distinct** and **temporally meaningful** alternative state

### Cluster model selection

The optimal number of clusters is selected using a weighted consensus of two internal validation metrics:

- **Silhouette score** → 70%
- **Inverse-normalized Davies–Bouldin index** → 30%

```text
consensus_score = 0.70 * normalized_silhouette
                + 0.30 * normalized_inverse_dbi

Requirements

This script requires:

Schrödinger Python runtime

Desmond trajectory tools (trj2mae.py)

Python packages:

numpy

matplotlib

scikit-learn

Input Files

Required inputs:

A Desmond CMS file

A Desmond trajectory directory

A reference structure file

By default, the reference structure is expected to be:

experimental_structure.mae

in the current working directory, unless another file is provided with --ref.

Running the Script
Basic example
$SCHRODINGER/run rep_select.py \
  --cms system-out.cms \
  --trj system_trj \
  --ref experimental_structure.mae
Example with explicit preview and clustering stride
$SCHRODINGER/run rep_select.py \
  --cms system-out.cms \
  --trj system_trj \
  --ref experimental_structure.mae \
  --preview-stride 20 \
  --cluster-stride 50
Preview-only mode

This mode generates only the RMSD preview and automatic trimming suggestion:

$SCHRODINGER/run rep_select.py \
  --cms system-out.cms \
  --trj system_trj \
  --ref experimental_structure.mae \
  --preview-only
Main Command-Line Arguments
Required

--cms
Desmond -out.cms file

--trj
Desmond trajectory directory (*_trj)

Common optional arguments

--ref
Reference structure file
Default: experimental_structure.mae

--preview-stride
Frame extraction stride for RMSD preview

--cluster-stride
Frame extraction stride for clustering

--stride
Legacy shared stride argument for both preview and clustering

--start-frame
Clustering start frame (inclusive), or auto
Default: auto

--end-frame
Trajectory end frame (exclusive), -1, or auto
Default: auto

Structural selection / alignment

--extract-asl
ASL extracted from the trajectory
Default: protein

--fit-asl
ASL used for alignment
Default: protein and backbone and not atom.ele H

--feature-asl
ASL used for feature generation
Default: protein and atom.ptype CA

--feature-mode
Feature representation mode
Choices:

ca_distances

ca_xyz
Default: ca_distances

Clustering

--k-min
Minimum number of clusters
Default: 2

--k-max
Maximum number of clusters
Default: 8

--pca-variance-cutoff
PCA reduction mode:

0 < x < 1 → cumulative explained variance cutoff

x >= 1 → exact integer number of PCA components
Default: 0.90

--allow-singletons
Allows clustering solutions with singleton clusters during model selection

Auto-trimming / RMSD preview

--rolling-window
Rolling window for RMSD smoothing
Default: 5

--rmsd-std-threshold
RMSD standard deviation threshold for stability detection
Default: 0.25

--rmsd-slope-threshold
Rolling slope threshold for stability detection
Default: 0.02

--stable-windows-required
Number of consecutive stable windows required
Default: 3

PBC sanity check

--max-ca-gap
Maximum consecutive Cα gap allowed before flagging suspicion
Default: 8.0

--pbc-error-fraction
Fraction of suspicious frames required to raise an error
Default: 0.05

REP2 selection criteria

--rep2-min-occupancy-percent
Minimum occupancy required for REP2
Default: 10.0

--rep2-min-longest-run
Minimum longest temporal run required for REP2
Default: 2

--rep2-min-temporal-fraction
Minimum temporal continuity fraction required for REP2
Default: 0.25

--rep2-min-backbone-rmsd
Minimum backbone RMSD from REP1 required for REP2
Default: 0.75

Output

All results are written to:

Clustering_Results/

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
Method Summary
1. RMSD preview and trimming

The script first extracts a coarse trajectory subset and computes backbone RMSD relative to the experimental reference. A rolling mean and rolling standard deviation are used to suggest an equilibration cutoff automatically.

2. Validation steps

Before clustering, the script checks:

atom count consistency

atom identity and ordering consistency

basic PBC / broken-protein artifacts through consecutive Cα gap inspection

3. Structural feature generation

Two feature modes are available:

ca_distances
Pairwise distances between selected atoms

ca_xyz
Flattened Cartesian coordinates of selected atoms

4. PCA + clustering

Features are standardized and reduced with PCA, then clustered using:

Agglomerative clustering

Ward linkage

5. Representative selection

For each cluster, the script computes the true medoid.

REP1 is chosen from the largest cluster

REP2 is chosen from the best alternative cluster using occupancy, temporal continuity, medoid centrality, and structural difference from REP1

Notes

The script must be run with the Schrödinger runtime.

If the runtime is not active, the script attempts to relaunch itself automatically using a detected Schrödinger run executable.

The selected ASLs must produce consistent atom identities and ordering across all extracted frames.

For ca_distances, very large atom selections can become computationally expensive.
