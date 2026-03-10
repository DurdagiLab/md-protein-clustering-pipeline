

# Representative Frame Selection for Desmond Trajectories

A Python pipeline for selecting representative protein conformations from **Desmond molecular dynamics trajectories** using structural alignment, PCA-based dimensionality reduction, and hierarchical clustering.

The workflow identifies two final representative structures:

- **REP1** - the medoid of the most populated cluster
- **REP2** - a structurally distinct alternative state selected based on clustering, temporal continuity, and structural divergence

## Features

- Automatic Schrödinger runtime bootstrap
- Frame extraction from Desmond trajectories
- RMSD-based equilibration preview
- Automatic or manual trajectory trimming
- ASL consistency and atom-order validation
- Basic PBC / broken-structure sanity check
- PCA + agglomerative clustering
- Consensus-based cluster model selection
- Export of representative structures and summary reports

## Requirements

- **Schrödinger Python runtime**
- **Desmond tools** (`trj2mae.py`)
- Python packages:
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

## Input

The script requires:

- a **CMS file**
- a **Desmond trajectory directory**
- a **reference structure**

Default reference file: `experimental_structure.mae`

## Usage

### Basic run

```bash
$SCHRODINGER/run rep_select.py \
  --cms system-out.cms \
  --trj system_trj \
  --ref experimental_structure.mae
```
```Preview only
$SCHRODINGER/run rep_select.py \
  --cms system-out.cms \
  --trj system_trj \
  --ref experimental_structure.mae \
  --preview-only
```
```Example with custom stride
$SCHRODINGER/run rep_select.py \
  --cms system-out.cms \
  --trj system_trj \
  --ref experimental_structure.mae \
  --preview-stride 20 \
  --cluster-stride 50
```

### Main Options

 **--cms:** Desmond CMS file
 **--trj:** trajectory directory
 **--ref:** reference structure file
 **--preview-stride:** frame stride for RMSD preview
**--cluster-stride:** frame stride for clustering
**--start-frame:** clustering start frame (auto supported)
**--end-frame:** clustering end frame (auto and -1 supported)
**--fit-asl:** ASL used for structural alignment
**--feature-asl:** ASL used for feature generation
**--feature-mode:** ca_distances or ca_xyz
**--k-min / --k-max:** clustering range
**--preview-only:** generate RMSD preview only

### Output

Results are written to Clustering_Results/.

### Typical outputs include:

RMSD preview plot and table
Extracted and aligned frames
Cluster medoid structures
Final REP1 and REP2
Clustering summary tables
Quality and temporal continuity reports

### Notes
The script must be run with the Schrödinger runtime.

If the runtime is not active, the script attempts to relaunch itself automatically.

Consistent ASL atom identity and ordering are required across frames.

Large atom selections may increase memory and runtime costs.

### Citation
If you use this tool in your research or publication, please cite it as follows:

İsaoğlu, M., & Durdağı, S. (2026). MD Protein Clustering Tool (Version 1.0) [Source Code]. https://github.com/DurdagiLab/md-protein-clustering-pipeline
