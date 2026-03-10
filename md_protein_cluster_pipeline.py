#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MD Protein Conformational Clustering Pipeline (Desmond Trajectories)
====================================================================

Author Information
------------------
Developer: Mine Isaoglu, Ph.D.
Principal Investigator: Serdar Durdagi, Ph.D.
Affiliation: Computational Drug Design Center (HITMER), Faculty of Pharmacy,
            Bahçeşehir University, Istanbul, Turkey.
Version: March 2026

Overview
--------
This script extracts protein frames from a Desmond trajectory, evaluates their
structural stability relative to a reference structure, performs feature-based
dimensionality reduction and hierarchical clustering, and selects representative
conformations from the resulting structural ensembles.

The workflow is designed to reduce large trajectory datasets into a compact set
of structurally meaningful frames that capture both dominant and alternative
conformational states. Cluster model selection is performed using a weighted
consensus criterion that combines two complementary internal validation metrics:
the Silhouette score (70%) and the inverse-normalized Davies-Bouldin index (30%).

Two final representative structures are selected:
- REP1: the medoid of the most populated cluster
- REP2: a structurally distinct and temporally meaningful alternative state

Main workflow
-------------
1. Verify or bootstrap the Schrödinger runtime environment
2. Extract trajectory frames at user-defined intervals
3. Generate an RMSD preview against a reference structure
4. Apply automatic or user-defined trajectory trimming
5. Validate ASL consistency and atom ordering
6. Perform a basic sanity check for PBC/imaging artefacts
7. Align frames to the reference structure
8. Build structural features
9. Perform PCA and agglomerative clustering
10. Select the optimal number of clusters using a consensus score
11. Identify cluster medoids and select REP1/REP2
12. Export representative structures, plots, and summary reports

Required inputs
---------------
- Desmond CMS file
- Desmond trajectory directory
- Reference structure file

Generated outputs
-----------------
Results are written to the `Clustering_Results` directory and include:
- RMSD preview plots and tables
- raw and aligned trajectory frames
- cluster medoid structures
- final REP1 and REP2 structures
- clustering summaries and quality reports

Dependencies
------------
This script requires:
- Schrödinger Python runtime
- NumPy
- Matplotlib
- scikit-learn
"""
import os
import re
import sys
import csv
import glob
import shutil
import logging
import argparse
import subprocess

from typing import List, Tuple, Dict, Optional, Any, Union


# =============================================================================
# 0. SCHRODINGER RUNTIME BOOTSTRAP
# =============================================================================


def resolve_schrodinger_run_executable() -> str:
    """
    Resolve the Schrödinger 'run' executable robustly.

    Resolution order:
      1) $SCHRODINGER_RUN (direct executable path)
      2) $SCHRODINGER18/run
      3) $SCHRODINGER/run
      4) /opt/schrodinger2018-4/run
      5) PATH lookup via shutil.which('run')
    """
    candidates: List[str] = []

    sch_run = os.environ.get("SCHRODINGER_RUN")
    if sch_run:
        candidates.append(sch_run)

    for env_name in ("SCHRODINGER18", "SCHRODINGER"):
        root = os.environ.get(env_name)
        if root:
            candidates.append(os.path.join(root, "run"))

    candidates.append("/opt/schrodinger2018-4/run")

    which_run = shutil.which("run")
    if which_run:
        candidates.append(which_run)

    seen = set()
    unique_candidates: List[str] = []
    for cand in candidates:
        if not cand:
            continue
        norm = os.path.abspath(cand) if os.path.sep in cand else cand
        if norm in seen:
            continue
        seen.add(norm)
        unique_candidates.append(cand)

    for cand in unique_candidates:
        if os.path.sep in cand:
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return os.path.abspath(cand)
        else:
            resolved = shutil.which(cand)
            if resolved and os.access(resolved, os.X_OK):
                return resolved

    raise FileNotFoundError(
        "Schrödinger 'run' executable could not be found. Checked: "
        "$SCHRODINGER_RUN, $SCHRODINGER18/run, $SCHRODINGER/run, "
        "/opt/schrodinger2018-4/run, and PATH."
    )


SCHRODINGER_RUN_EXE: Optional[str] = None


def ensure_schrodinger_runtime() -> None:
    global SCHRODINGER_RUN_EXE

    try:
        import schrodinger  # noqa: F401
        try:
            SCHRODINGER_RUN_EXE = resolve_schrodinger_run_executable()
        except FileNotFoundError:
            SCHRODINGER_RUN_EXE = None
        return
    except ImportError:
        pass

    if os.environ.get("REP_SELECT_BOOTSTRAPPED") == "1":
        sys.stderr.write(
            "\n[ERROR] The script was relaunched with the Schrödinger runtime, "
            "but the 'schrodinger' module still could not be imported.\n"
            "This likely indicates an installation or path configuration problem.\n\n"
        )
        sys.exit(1)

    try:
        run_exe = resolve_schrodinger_run_executable()
    except FileNotFoundError as exc:
        sys.stderr.write(
            "\n[ERROR] The 'schrodinger' module could not be found.\n"
            "This script must be executed with the Schrödinger runtime.\n\n"
            "Checked locations:\n"
            "  - $SCHRODINGER_RUN\n"
            "  - $SCHRODINGER18/run\n"
            "  - $SCHRODINGER/run\n"
            "  - /opt/schrodinger2018-4/run\n"
            "  - PATH (run)\n\n"
            "Examples:\n"
            "  $SCHRODINGER18/run rep_select.py ...\n"
            "or\n"
            "  $SCHRODINGER/run rep_select.py ...\n"
            "or\n"
            "  /opt/schrodinger2018-4/run rep_select.py ...\n\n"
            "Details: %s\n\n" % str(exc)
        )
        sys.exit(1)

    SCHRODINGER_RUN_EXE = run_exe

    cmd = [run_exe, os.path.abspath(__file__)] + sys.argv[1:]
    env = os.environ.copy()
    env["REP_SELECT_BOOTSTRAPPED"] = "1"

    sys.stderr.write(
        "\n[*] Relaunching the script under the Schrödinger runtime:\n"
        "    %s\n\n" % " ".join(cmd)
    )
    sys.stderr.flush()

    rc = subprocess.call(cmd, env=env)
    sys.exit(rc)


ensure_schrodinger_runtime()


# =============================================================================
# 1. IMPORTS
# =============================================================================

try:
    import numpy as np
except ImportError:
    sys.stderr.write("\n[ERROR] numpy could not be imported.\n")
    raise

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.stderr.write("\n[ERROR] matplotlib could not be imported.\n")
    raise

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.stderr.write("\n[ERROR] Required scikit-learn modules could not be imported.\n")
    raise

try:
    from sklearn.metrics import davies_bouldin_score
    HAVE_NATIVE_DBI = True
except ImportError:
    HAVE_NATIVE_DBI = False

from schrodinger import structure
from schrodinger.structutils.analyze import evaluate_asl


# =============================================================================
# 2. LOGGING
# =============================================================================

logger = logging.getLogger("repframe")
logger.setLevel(logging.INFO)
logger.handlers = []
_handler = logging.StreamHandler(sys.stdout)
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(_handler)
logger.propagate = False


# =============================================================================
# 3. FALLBACK METRICS
# =============================================================================


def davies_bouldin_score_fallback(X: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        raise ValueError("Davies-Bouldin score requires at least two clusters.")

    centroids = []
    scatters = []

    for lab in unique_labels:
        pts = X[labels == lab]
        centroid = np.mean(pts, axis=0)
        centroids.append(centroid)

        if len(pts) == 1:
            scatter = 0.0
        else:
            scatter = np.mean(np.linalg.norm(pts - centroid, axis=1))
        scatters.append(scatter)

    centroids = np.asarray(centroids, dtype=float)
    scatters = np.asarray(scatters, dtype=float)

    diff = centroids[:, None, :] - centroids[None, :, :]
    M = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(M, np.inf)

    R = (scatters[:, None] + scatters[None, :]) / (M + 1e-12)
    D = np.max(R, axis=1)

    return float(np.mean(D))


def compute_davies_bouldin_score(X: np.ndarray, labels: np.ndarray) -> float:
    if HAVE_NATIVE_DBI:
        return float(davies_bouldin_score(X, labels))
    return davies_bouldin_score_fallback(X, labels)


# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================


def natural_sort_key(s: str) -> List[object]:
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", s)
    ]


def read_first_structure(path: str) -> Any:
    with structure.StructureReader(path) as reader:
        try:
            return next(reader)
        except StopIteration:
            raise RuntimeError("File could not be read or is empty: %s" % path)


def safe_getattr(obj: object, names: List[str], default: object = "") -> object:
    for name in names:
        if hasattr(obj, name):
            val = getattr(obj, name)
            if val is not None:
                return val
    return default


def atom_signature(atom: Any) -> Tuple[str, int, str, str, str]:
    chain = str(safe_getattr(atom, ["chain"], ""))
    resnum = safe_getattr(atom, ["resnum"], -999999)
    try:
        resnum = int(resnum)
    except Exception:
        resnum = -999999
    resname = str(safe_getattr(atom, ["pdbres", "residue_name"], ""))
    atomname = str(safe_getattr(atom, ["pdbname", "name"], ""))
    element = str(safe_getattr(atom, ["element"], ""))
    return (chain, resnum, resname, atomname, element)


def get_asl_indices_and_signature(
    st: Any,
    asl: str
) -> Tuple[List[int], List[Tuple[str, int, str, str, str]]]:
    aids = list(evaluate_asl(st, asl))
    if len(aids) == 0:
        raise ValueError("The ASL expression returned no atoms: %s" % asl)
    sig = [atom_signature(st.atom[i]) for i in aids]
    return aids, sig


def extract_xyz(st: Any, aids: List[int]) -> np.ndarray:
    return np.array([st.atom[i].xyz for i in aids], dtype=float)


def rmsd_coords(P: np.ndarray, Q: np.ndarray) -> float:
    diff = P - Q
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc

    C = np.dot(P0.T, Q0)
    V, _, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.dot(V, Wt)))
    D = np.diag([1.0, 1.0, d])
    R = np.dot(np.dot(V, D), Wt)
    t = Qc - np.dot(Pc, R)
    return R, t


def apply_transform(st: Any, R: np.ndarray, t: np.ndarray) -> None:
    xyz = np.array(st.getXYZ(), dtype=float)
    xyz2 = np.dot(xyz, R) + t
    st.setXYZ(xyz2)


def rolling_mean_std(x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    if window < 2:
        raise ValueError("rolling window must be >= 2")
    if len(x) < window:
        return np.array([]), np.array([])

    means = []
    stds = []
    for i in range(len(x) - window + 1):
        w = x[i:i + window]
        means.append(np.mean(w))
        stds.append(np.std(w, ddof=0))
    return np.array(means), np.array(stds)


def resolve_reference_structure(ref_name: str = "experimental_structure.mae") -> str:
    ref_path = os.path.abspath(ref_name)
    if not os.path.exists(ref_path):
        raise FileNotFoundError(
            "Reference structure not found: %s\n"
            "The file must be present in the working directory as '%s'."
            % (ref_path, ref_name)
        )
    return ref_path


def validate_input_paths(cms_file: str, trj_dir: str, ref_file: str) -> None:
    cms_abs = os.path.abspath(cms_file)
    trj_abs = os.path.abspath(trj_dir)
    ref_abs = os.path.abspath(ref_file)

    if not os.path.isfile(cms_abs):
        raise FileNotFoundError("CMS file not found: %s" % cms_abs)

    if not os.path.isdir(trj_abs):
        raise FileNotFoundError("Trajectory directory not found: %s" % trj_abs)

    if not os.path.isfile(ref_abs):
        raise FileNotFoundError("Reference structure not found: %s" % ref_abs)


def get_cluster_sizes(labels: np.ndarray) -> Dict[int, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def count_singleton_clusters(labels: np.ndarray) -> int:
    sizes = get_cluster_sizes(labels)
    return sum(1 for _, v in sizes.items() if v == 1)


def min_cluster_size(labels: np.ndarray) -> int:
    sizes = get_cluster_sizes(labels)
    return min(sizes.values())


def write_quality_report(path: str, lines: List[str]) -> None:
    with open(path, "w") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def normalize_array(values: List[float], higher_is_better: bool = True) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return arr
    if np.allclose(arr.max(), arr.min()):
        return np.ones_like(arr, dtype=float)
    norm = (arr - arr.min()) / (arr.max() - arr.min())
    if higher_is_better:
        return norm
    return 1.0 - norm


def contiguous_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, v in enumerate(mask.astype(bool).tolist()):
        if v and start is None:
            start = i
        if (not v) and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(mask) - 1))
    return segs


def aligned_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Pairwise RMSD after optimal superposition of P onto Q.
    """
    R, t = kabsch_align(P, Q)
    P_aligned = np.dot(P, R) + t
    return rmsd_coords(P_aligned, Q)


def strip_structure_extensions(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".mae.gz"):
        return filename[:-7]
    if lower.endswith(".maegz"):
        return filename[:-6]
    if lower.endswith(".mae"):
        return filename[:-4]
    root, _ = os.path.splitext(filename)
    return root


def infer_frame_number_from_filename(path: str) -> Optional[int]:
    base = os.path.basename(path)
    stem = strip_structure_extensions(base)

    patterns = [
        r"(?:^|[_\-.])frame[_\-]?(\d+)(?:$|[_\-.])",
        r"(?:^|[_\-.])frm[_\-]?(\d+)(?:$|[_\-.])",
        r"(?:^|[_\-.])f[_\-]?(\d+)(?:$|[_\-.])",
        r"(\d+)$"
    ]

    for pat in patterns:
        m = re.search(pat, stem, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def estimate_frame_numbers(
    n_files: int,
    start_frame: int,
    end_frame: int,
    stride: int
) -> List[int]:
    if n_files <= 0:
        return []

    if end_frame == -1:
        return [start_frame + i * stride for i in range(n_files)]

    vals = list(range(start_frame, end_frame, stride))
    return vals[:n_files]


def resolve_extracted_frame_numbers(
    files: List[str],
    start_frame: int,
    end_frame: int,
    stride: int
) -> List[int]:
    estimated = estimate_frame_numbers(
        n_files=len(files),
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride
    )

    if len(files) == 0:
        return estimated

    parsed: List[Optional[int]] = [infer_frame_number_from_filename(fp) for fp in files]

    if all(v is not None for v in parsed):
        parsed_int = [int(v) for v in parsed]

        if len(parsed_int) > 1:
            strictly_increasing = all(
                parsed_int[i] < parsed_int[i + 1]
                for i in range(len(parsed_int) - 1)
            )
            stride_matches = all(
                (parsed_int[i + 1] - parsed_int[i]) == stride
                for i in range(len(parsed_int) - 1)
            )

            if strictly_increasing and stride_matches:
                logger.info(
                    "Using trajectory frame numbers inferred from extracted filenames."
                )
                return parsed_int

            logger.warning(
                "Frame numbers parsed from filenames do not match the requested stride=%d. "
                "Falling back to estimated frame numbers based on the requested slice.",
                stride
            )
            return estimated

        logger.warning(
            "Only one extracted frame is available; actual trajectory frame numbering "
            "cannot be validated from filenames alone. Using the requested start frame."
        )
        return estimated

    logger.warning(
        "Actual trajectory frame numbers could not be inferred from extracted filenames. "
        "Using estimated frame numbers based on start/end/stride."
    )
    return estimated


# =============================================================================
# 5. FRAME EXTRACTION
# =============================================================================


def extract_frames(
    cms_file: str,
    trj_dir: str,
    out_folder: str,
    stride: int,
    extract_asl: str = "protein",
    start_frame: int = 0,
    end_frame: int = -1,
    clean_folder: bool = True
) -> Tuple[List[str], List[int]]:
    if clean_folder and os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder, exist_ok=True)

    cms_abs = os.path.abspath(cms_file)
    trj_abs = os.path.abspath(trj_dir)

    if stride <= 0:
        raise ValueError("stride must be a positive integer")
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    if end_frame != -1 and end_frame < 0:
        raise ValueError("end_frame must be -1 or >= 0")
    if end_frame != -1 and end_frame <= start_frame:
        raise ValueError("end_frame must be greater than start_frame")

    run_exe = SCHRODINGER_RUN_EXE or resolve_schrodinger_run_executable()

    if end_frame == -1:
        slice_str = "%d::%d" % (start_frame, stride)
    else:
        slice_str = "%d:%d:%d" % (start_frame, end_frame, stride)

    cmd = [
        run_exe, "trj2mae.py",
        cms_abs, trj_abs, "frame",
        "-out-format", "MAE",
        "-extract-asl", extract_asl,
        "-separate",
        "-s", slice_str
    ]

    logger.info("Frame extraction started: slice=%s", slice_str)
    process = subprocess.run(
        cmd,
        cwd=out_folder,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    if process.returncode != 0:
        logger.error("trj2mae.py failed. Command output follows:")
        logger.error(process.stdout)
        raise RuntimeError(
            "trj2mae.py failed with exit code %d" % process.returncode
        )

    files = sorted(
        glob.glob(os.path.join(out_folder, "*.mae*")),
        key=natural_sort_key
    )

    if len(files) == 0:
        logger.error("trj2mae.py did not produce any frames.")
        logger.error(process.stdout)
        raise RuntimeError("trj2mae.py did not produce any frames.")

    frame_numbers = resolve_extracted_frame_numbers(
        files=files,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride
    )

    logger.info("%d frames were extracted.", len(files))
    return files, frame_numbers


# =============================================================================
# 6. ASL / TOPOLOGY CONSISTENCY VALIDATION
# =============================================================================


def validate_asl_consistency_against_reference(
    files: List[str],
    ref_sig: List[Tuple[str, int, str, str, str]],
    asl: str,
    label: str
) -> None:
    logger.info("Validating ASL consistency for %s...", label)
    ref_n = len(ref_sig)

    for fp in files:
        st = read_first_structure(fp)
        aids, sig = get_asl_indices_and_signature(st, asl)

        if len(aids) != ref_n:
            raise RuntimeError(
                "%s atom count is inconsistent.\n"
                "File: %s\n"
                "Expected: %d, Observed: %d\n"
                "ASL: %s"
                % (label, fp, ref_n, len(aids), asl)
            )

        if sig != ref_sig:
            raise RuntimeError(
                "%s atom identity/order does not match the reference.\n"
                "File: %s\n"
                "ASL: %s\n"
                "In this case, alignment and feature calculations are not reliable."
                % (label, fp, asl)
            )

    logger.info("%s consistency validation passed.", label)


# =============================================================================
# 7. PBC / BROKEN-PROTEIN VALIDATION
# =============================================================================


def inspect_backbone_continuity(
    st: Any,
    ca_asl: str = "protein and atom.ptype CA",
    max_ca_gap: float = 8.0
) -> Tuple[bool, float, List[float]]:
    aids, _ = get_asl_indices_and_signature(st, ca_asl)
    if len(aids) < 2:
        return False, 0.0, []

    rows = []
    for aid in aids:
        atom = st.atom[aid]
        chain = str(safe_getattr(atom, ["chain"], ""))
        resnum = safe_getattr(atom, ["resnum"], None)
        try:
            resnum = int(resnum)
        except Exception:
            resnum = None
        rows.append((aid, chain, resnum))

    suspicious_gaps: List[float] = []
    xyz = extract_xyz(st, aids)

    for i in range(len(rows) - 1):
        _, chain_i, res_i = rows[i]
        _, chain_j, res_j = rows[i + 1]

        if res_i is None or res_j is None:
            continue
        if chain_i != chain_j:
            continue
        if (res_j - res_i) != 1:
            continue

        d = np.linalg.norm(xyz[i + 1] - xyz[i])
        if d > max_ca_gap:
            suspicious_gaps.append(float(d))

    max_gap = max(suspicious_gaps) if suspicious_gaps else 0.0
    return (len(suspicious_gaps) > 0), max_gap, suspicious_gaps


def run_pbc_sanity_check(
    files: List[str],
    ca_asl: str = "protein and atom.ptype CA",
    max_ca_gap: float = 8.0,
    suspicious_fraction_error: float = 0.05
) -> None:
    logger.info("Running PBC / broken-protein sanity check...")
    suspicious: List[Tuple[str, float, List[float]]] = []

    if len(files) == 0:
        logger.info("No files provided for PBC sanity check.")
        return

    first_st = read_first_structure(files[0])
    try:
        ca_aids, _ = get_asl_indices_and_signature(first_st, ca_asl)
    except Exception:
        logger.warning(
            "PBC sanity check skipped because CA ASL could not be evaluated: %s",
            ca_asl
        )
        return

    if len(ca_aids) < 2:
        logger.warning(
            "PBC sanity check skipped because fewer than two CA atoms were found: %s",
            ca_asl
        )
        return

    for fp in files:
        st = read_first_structure(fp)
        bad, max_gap, gaps = inspect_backbone_continuity(
            st, ca_asl=ca_asl, max_ca_gap=max_ca_gap
        )
        if bad:
            suspicious.append((fp, max_gap, gaps))

    frac = len(suspicious) / max(1, len(files))
    if len(suspicious) > 0:
        logger.warning(
            "%d/%d frames were flagged as suspicious (%.2f%%).",
            len(suspicious), len(files), frac * 100.0
        )

    if frac >= suspicious_fraction_error:
        first_fp, first_gap, _ = suspicious[0]
        raise RuntimeError(
            "A high probability of PBC/imaging artefacts or broken-protein behaviour was detected.\n"
            "First suspicious frame: %s\n"
            "Largest consecutive C-alpha gap: %.2f A\n"
            "It is strongly recommended to use a reimaged/repaired trajectory."
            % (first_fp, first_gap)
        )

    logger.info("PBC sanity check completed.")


# =============================================================================
# 8. RMSD PREVIEW + AUTOMATIC TRIMMING
# =============================================================================


def compute_rmsd_preview(
    raw_files: List[str],
    frame_numbers: List[int],
    ref_st: Any,
    fit_asl: str,
    out_csv: str = "0_rmsd_preview.csv",
    out_png: str = "0_rmsd_preview.png",
    rolling_window: int = 5,
    auto_trim: bool = True,
    rmsd_std_threshold: float = 0.25,
    rmsd_slope_threshold: float = 0.02,
    stable_windows_required: int = 3,
    explicit_end_frame: Optional[Union[int, str]] = None
) -> Dict[str, Any]:
    logger.info("Computing RMSD preview...")

    ref_aids, ref_sig = get_asl_indices_and_signature(ref_st, fit_asl)
    ref_xyz = extract_xyz(ref_st, ref_aids)

    rmsd_values: List[float] = []
    for fp in raw_files:
        st = read_first_structure(fp)
        aids, sig = get_asl_indices_and_signature(st, fit_asl)

        if sig != ref_sig:
            raise RuntimeError(
                "For RMSD preview, fit_asl atoms do not match the reference: %s" % fp
            )

        xyz = extract_xyz(st, aids)
        R, t = kabsch_align(xyz, ref_xyz)
        xyz_aligned = np.dot(xyz, R) + t
        r = rmsd_coords(xyz_aligned, ref_xyz)
        rmsd_values.append(r)

    rmsd_values_arr = np.array(rmsd_values, dtype=float)
    rolling_mean, rolling_std = rolling_mean_std(rmsd_values_arr, rolling_window)

    suggestion_idx = 0
    suggestion_frame = frame_numbers[0] if frame_numbers else 0
    auto_reason = "automatic trimming disabled"

    if auto_trim and len(frame_numbers) < rolling_window:
        auto_reason = (
            "insufficient frames for auto-trimming: "
            "n_frames=%d < rolling_window=%d"
            % (len(frame_numbers), rolling_window)
        )
        logger.warning(auto_reason)

    elif auto_trim and len(rolling_mean) > 0:
        if stable_windows_required > len(rolling_mean):
            auto_reason = (
                "stable_windows_required=%d exceeds the number of available rolling windows=%d; "
                "the start frame remains at %d"
                % (stable_windows_required, len(rolling_mean), suggestion_frame)
            )
            logger.warning(auto_reason)
        else:
            slope = np.abs(np.concatenate(([0.0], np.diff(rolling_mean))))
            stable = (rolling_std <= rmsd_std_threshold) & (slope <= rmsd_slope_threshold)

            count = 0
            found = False
            for i, ok in enumerate(stable):
                if ok:
                    count += 1
                else:
                    count = 0

                if count >= stable_windows_required:
                    start_roll_idx = i - stable_windows_required + 1
                    suggestion_idx = min(
                        len(frame_numbers) - 1,
                        start_roll_idx + rolling_window - 1
                    )
                    suggestion_frame = frame_numbers[suggestion_idx]
                    auto_reason = (
                        "rolling_std<=%s and rolling_slope<=%s satisfied for %d consecutive windows"
                        % (
                            rmsd_std_threshold,
                            rmsd_slope_threshold,
                            stable_windows_required
                        )
                    )
                    found = True
                    break

            if not found:
                suggestion_idx = 0
                suggestion_frame = frame_numbers[0]
                auto_reason = (
                    "no stable plateau was detected automatically; "
                    "the start frame remains at %d"
                    % suggestion_frame
                )

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_index_in_extracted_set",
            "trajectory_frame_number",
            "rmsd_to_experimental_A"
        ])
        for i, (fn, r) in enumerate(zip(frame_numbers, rmsd_values_arr)):
            writer.writerow([i, fn, "%.4f" % r])

    end_line_x = None
    end_line_label = None

    if explicit_end_frame == "auto":
        if len(frame_numbers) > 0:
            end_line_x = frame_numbers[-1]
            end_line_label = "Selected end = last extracted frame (%s)" % end_line_x
    elif explicit_end_frame == -1 or explicit_end_frame is None:
        end_line_x = None
        end_line_label = None
    else:
        end_line_x = explicit_end_frame
        end_line_label = "Selected exclusive end = %s" % explicit_end_frame

    plt.figure(figsize=(10, 5))
    plt.plot(frame_numbers, rmsd_values_arr, lw=1.2, label="Backbone RMSD")

    if len(rolling_mean) > 0:
        rm_x = frame_numbers[rolling_window - 1:]
        plt.plot(
            rm_x,
            rolling_mean,
            lw=2.0,
            label="Rolling mean (window=%d)" % rolling_window
        )

    if auto_trim:
        plt.axvline(
            suggestion_frame,
            linestyle="--",
            linewidth=1.5,
            label="Suggested start = %s" % suggestion_frame
        )

    if end_line_x is not None:
        plt.axvline(
            end_line_x,
            linestyle="--",
            linewidth=1.5,
            label=end_line_label
        )

    plt.xlabel("Trajectory frame number")
    plt.ylabel("RMSD to experimental structure (A)")
    plt.title("Equilibration preview")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    logger.info("RMSD preview CSV: %s", out_csv)
    logger.info("RMSD preview PNG: %s", out_png)
    logger.info(
        "Automatic trim suggestion: frame >= %s | %s",
        suggestion_frame,
        auto_reason
    )

    if explicit_end_frame == "auto":
        logger.info("Preview plot end marker: last extracted frame.")
    elif explicit_end_frame == -1:
        logger.info("Preview plot end marker: disabled (--end-frame -1).")
    elif explicit_end_frame is not None:
        logger.info(
            "Preview plot end marker: explicit exclusive cutoff at frame %s.",
            explicit_end_frame
        )

    return {
        "rmsd_values": rmsd_values_arr,
        "frame_numbers": frame_numbers,
        "suggestion_idx": suggestion_idx,
        "suggestion_frame": suggestion_frame,
        "auto_reason": auto_reason
    }


# =============================================================================
# 9. ALIGNMENT
# =============================================================================


def align_and_save_frames_to_reference(
    input_files: List[str],
    input_frame_numbers: List[int],
    aligned_folder: str,
    ref_st: Any,
    fit_asl: str
) -> Tuple[List[str], List[int]]:
    if os.path.exists(aligned_folder):
        shutil.rmtree(aligned_folder)
    os.makedirs(aligned_folder, exist_ok=True)

    logger.info(
        "Aligning frames to the reference structure and writing to %s...",
        aligned_folder
    )

    ref_aids, ref_sig = get_asl_indices_and_signature(ref_st, fit_asl)
    ref_xyz = extract_xyz(ref_st, ref_aids)

    aligned_files: List[str] = []
    aligned_frame_numbers: List[int] = []

    for fp, frnum in zip(input_files, input_frame_numbers):
        st = read_first_structure(fp)
        aids, sig = get_asl_indices_and_signature(st, fit_asl)

        if sig != ref_sig:
            raise RuntimeError(
                "For alignment, fit_asl atoms do not match the reference: %s" % fp
            )

        xyz = extract_xyz(st, aids)
        R, t = kabsch_align(xyz, ref_xyz)
        apply_transform(st, R, t)

        out_name = os.path.basename(fp)
        out_path = os.path.join(aligned_folder, out_name)
        st.write(out_path)

        aligned_files.append(out_path)
        aligned_frame_numbers.append(frnum)

    logger.info("%d frames were aligned.", len(aligned_files))
    return aligned_files, aligned_frame_numbers


# =============================================================================
# 10. FEATURE CONSTRUCTION
# =============================================================================


def build_ca_distance_features(xyz: np.ndarray) -> np.ndarray:
    n = xyz.shape[0]
    iu = np.triu_indices(n, k=1)
    diff = xyz[:, None, :] - xyz[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))
    return D[iu]


def build_features(
    aligned_files: List[str],
    ref_st: Any,
    feature_asl: str,
    feature_mode: str = "ca_distances"
) -> np.ndarray:
    logger.info(
        "Constructing the feature matrix | mode=%s | ASL=%s",
        feature_mode,
        feature_asl
    )

    ref_aids, ref_sig = get_asl_indices_and_signature(ref_st, feature_asl)
    n_ref = len(ref_aids)

    if feature_mode == "ca_distances" and n_ref < 2:
        raise RuntimeError(
            "For ca_distances, feature_asl must select at least two atoms."
        )

    if feature_mode == "ca_distances" and n_ref > 2000:
        raise RuntimeError(
            "feature_asl selected too many atoms for ca_distances (%d atoms). "
            "This mode is intended for compact atom sets such as C-alpha atoms. "
            "Use a smaller ASL or switch to --feature-mode ca_xyz."
            % n_ref
        )

    X = []
    for fp in aligned_files:
        st = read_first_structure(fp)
        aids, sig = get_asl_indices_and_signature(st, feature_asl)

        if sig != ref_sig:
            raise RuntimeError(
                "Feature ASL atoms do not match the reference: %s" % fp
            )

        xyz = extract_xyz(st, aids)

        if feature_mode == "ca_distances":
            feat = build_ca_distance_features(xyz)
        elif feature_mode == "ca_xyz":
            feat = xyz.reshape(-1)
        else:
            raise ValueError("Unknown feature_mode: %s" % feature_mode)

        X.append(feat)

    X_arr = np.array(X, dtype=float)
    logger.info(
        "Feature matrix dimensions: %d frames x %d features",
        X_arr.shape[0],
        X_arr.shape[1]
    )
    return X_arr


# =============================================================================
# 11. DIMENSIONALITY REDUCTION + CLUSTERING
# =============================================================================


def find_best_k_and_cluster(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    pca_variance_cutoff: Union[int, float] = 0.90,
    allow_singletons: bool = False,
    pca_random_state: int = 0
) -> Tuple[np.ndarray, np.ndarray, int, List[Dict[str, Any]], PCA]:
    """
    Determine the optimal number of clusters using a weighted consensus score
    derived from two complementary internal clustering metrics:

    1. Silhouette score
       Higher values indicate better separation between clusters.

    2. Davies-Bouldin index
       Lower values indicate tighter and better-separated clusters.

    To integrate these criteria into a unified decision framework, both metrics
    are normalized to the [0, 1] interval. The Davies-Bouldin index is inverted
    during normalization because smaller values are preferable. A weighted
    consensus score is then computed as:

        consensus_score = 0.70 * normalized_silhouette
                        + 0.30 * normalized_inverse_dbi

    The clustering solution with the highest consensus score is selected,
    subject to singleton-cluster constraints unless explicitly relaxed by the
    user.
    """
    if X.shape[0] < 3:
        raise RuntimeError("At least three frames are required for clustering.")

    safe_k_max = min(k_max, X.shape[0] - 1)
    if k_min > safe_k_max:
        raise RuntimeError(
            "No valid k range exists. n_frames=%d, k_min=%d, k_max=%d"
            % (X.shape[0], k_min, safe_k_max)
        )

    logger.info("Scaling features to zero mean and unit variance...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if isinstance(pca_variance_cutoff, float) and 0.0 < pca_variance_cutoff < 1.0:
        logger.info(
            "Applying PCA | explained_variance_cutoff=%s | svd_solver=full",
            pca_variance_cutoff
        )
        pca = PCA(
            n_components=pca_variance_cutoff,
            svd_solver="full"
        )
    else:
        n_components_int = int(pca_variance_cutoff)
        logger.info(
            "Applying PCA | n_components=%s | svd_solver=randomized | random_state=%s",
            n_components_int,
            pca_random_state
        )
        pca = PCA(
            n_components=n_components_int,
            svd_solver="randomized",
            random_state=pca_random_state
        )

    X_red = pca.fit_transform(X_scaled)

    ncomp = pca.n_components_
    evr_sum = np.sum(pca.explained_variance_ratio_)
    logger.info("Number of PCA components: %s", ncomp)
    logger.info("Cumulative explained variance: %.4f", evr_sum)

    if HAVE_NATIVE_DBI:
        logger.info(
            "Davies-Bouldin score: using the native sklearn implementation."
        )
    else:
        logger.info(
            "Davies-Bouldin score: older sklearn detected; using the fallback implementation."
        )

    k_metrics: List[Dict[str, Any]] = []

    logger.info(
        "Searching for the optimal number of clusters: k=%d..%d",
        k_min, safe_k_max
    )

    # -------------------------------------------------------------------------
    # Phase 1: Evaluate all candidate k values and collect clustering metrics
    # -------------------------------------------------------------------------
    for k in range(k_min, safe_k_max + 1):
        clusterer = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = clusterer.fit_predict(X_red)

        sil = silhouette_score(X_red, labels)
        dbi = compute_davies_bouldin_score(X_red, labels)
        singleton_count = count_singleton_clusters(labels)
        min_size = min_cluster_size(labels)
        valid_for_selection = allow_singletons or (singleton_count == 0)

        logger.info(
            "k=%d | silhouette=%.4f | davies_bouldin=%.4f | "
            "singleton_clusters=%d | min_cluster_size=%d",
            k, sil, dbi, singleton_count, min_size
        )

        row: Dict[str, Any] = {
            "k": int(k),
            "silhouette_score": float(sil),
            "davies_bouldin_score": float(dbi),
            "singleton_cluster_count": int(singleton_count),
            "min_cluster_size": int(min_size),
            "valid_for_selection": bool(valid_for_selection),
            "labels": labels
        }
        k_metrics.append(row)

    # -------------------------------------------------------------------------
    # Phase 2: Normalize metrics and compute the weighted consensus score
    # -------------------------------------------------------------------------
    sil_list = [m["silhouette_score"] for m in k_metrics]
    dbi_list = [m["davies_bouldin_score"] for m in k_metrics]

    sil_norm = normalize_array(sil_list, higher_is_better=True)
    dbi_norm = normalize_array(dbi_list, higher_is_better=False)

    silhouette_weight = 0.70
    dbi_weight = 0.30

    for i, row in enumerate(k_metrics):
        consensus_score = (
            silhouette_weight * sil_norm[i] +
            dbi_weight * dbi_norm[i]
        )

        row["normalized_silhouette_score"] = float(sil_norm[i])
        row["normalized_davies_bouldin_score"] = float(dbi_norm[i])
        row["consensus_score"] = float(consensus_score)

        logger.info(
            "k=%d | consensus_score=%.4f "
            "(norm_sil=%.4f, norm_dbi=%.4f, weights=%.2f/%.2f)",
            row["k"],
            row["consensus_score"],
            row["normalized_silhouette_score"],
            row["normalized_davies_bouldin_score"],
            silhouette_weight,
            dbi_weight
        )

    # -------------------------------------------------------------------------
    # Phase 3: Select the best clustering solution by consensus score
    # -------------------------------------------------------------------------
    best_valid: Optional[Dict[str, Any]] = None
    best_overall: Optional[Dict[str, Any]] = None

    for row in k_metrics:
        if best_overall is None or row["consensus_score"] > best_overall["consensus_score"]:
            best_overall = row

        if row["valid_for_selection"]:
            if best_valid is None or row["consensus_score"] > best_valid["consensus_score"]:
                best_valid = row

    if best_valid is not None:
        selected = best_valid
    else:
        selected = best_overall
        logger.warning(
            "No singleton-free clustering solution was found. "
            "The overall best consensus result was selected."
        )

    if selected is None:
        raise RuntimeError("No clustering solution could be selected.")

    logger.info(
        "Selected k=%d | consensus_score=%.4f | silhouette=%.4f | "
        "davies_bouldin=%.4f | singleton_clusters=%d",
        selected["k"],
        selected["consensus_score"],
        selected["silhouette_score"],
        selected["davies_bouldin_score"],
        selected["singleton_cluster_count"]
    )

    return selected["labels"], X_red, selected["k"], k_metrics, pca


# =============================================================================
# 12. CLUSTER / REPRESENTATIVE METRICS
# =============================================================================


def true_medoid_index(points: np.ndarray) -> int:
    n = points.shape[0]
    if n == 1:
        return 0

    dist_sums = np.zeros(n, dtype=float)
    for i in range(n):
        d = np.linalg.norm(points - points[i], axis=1)
        dist_sums[i] = np.sum(d)
    return int(np.argmin(dist_sums))


def compute_temporal_metrics_for_cluster(
    labels: np.ndarray,
    cluster_id: int
) -> Dict[str, Union[int, float]]:
    mask = (labels == cluster_id)
    segs = contiguous_segments(mask)
    if len(segs) == 0:
        return {
            "segment_count": 0,
            "longest_run": 0,
            "temporal_fraction": 0.0
        }

    lengths = [b - a + 1 for (a, b) in segs]
    longest_run = max(lengths)
    pop = int(np.sum(mask))
    temporal_fraction = float(longest_run) / float(pop) if pop > 0 else 0.0

    return {
        "segment_count": len(segs),
        "longest_run": int(longest_run),
        "temporal_fraction": float(temporal_fraction)
    }


def build_cluster_profiles(
    aligned_files: List[str],
    aligned_frame_numbers: List[int],
    X_original: np.ndarray,
    X_red: np.ndarray,
    labels: np.ndarray
) -> List[Dict[str, Any]]:
    profiles: List[Dict[str, Any]] = []
    n_total = len(labels)

    for cluster_id in sorted(np.unique(labels).tolist()):
        idx = np.where(labels == cluster_id)[0]
        cluster_points = X_original[idx]
        cluster_points_red = X_red[idx]

        medoid_local_idx = true_medoid_index(cluster_points)
        medoid_global_idx = idx[medoid_local_idx]

        medoid_point = X_original[medoid_global_idx]
        centroid = np.mean(cluster_points, axis=0)
        spread = float(np.mean(np.linalg.norm(cluster_points - centroid, axis=1)))

        medoid_dists = np.linalg.norm(cluster_points - medoid_point, axis=1)
        medoid_centrality = float(np.mean(medoid_dists))

        if len(cluster_points) > 1:
            sorted_d = np.sort(medoid_dists[medoid_dists > 0.0])
            k_nn = min(3, len(sorted_d))
            if k_nn > 0:
                nearest_neighbor_density = float(
                    1.0 / (np.mean(sorted_d[:k_nn]) + 1.0e-8)
                )
            else:
                nearest_neighbor_density = 0.0
        else:
            nearest_neighbor_density = 0.0

        temporal = compute_temporal_metrics_for_cluster(labels, cluster_id)

        profiles.append({
            "cluster_id": int(cluster_id),
            "population": int(len(idx)),
            "occupancy_percent": float((len(idx) / float(n_total)) * 100.0),
            "cluster_indices": idx,
            "medoid_global_idx": int(medoid_global_idx),
            "medoid_local_idx": int(medoid_local_idx),
            "medoid_traj_frame_number": int(aligned_frame_numbers[medoid_global_idx]),
            "medoid_file": aligned_files[medoid_global_idx],
            "cluster_mean_spread": float(spread),
            "medoid_centrality": float(medoid_centrality),
            "nearest_neighbor_density": float(nearest_neighbor_density),
            "segment_count": int(temporal["segment_count"]),
            "longest_run": int(temporal["longest_run"]),
            "temporal_fraction": float(temporal["temporal_fraction"]),
            "cluster_points": cluster_points,
            "cluster_points_red": cluster_points_red,
            "medoid_point": medoid_point
        })

    return profiles


def export_all_cluster_medoids(
    cluster_profiles: List[Dict[str, Any]],
    out_dir: str
) -> List[Dict[str, Any]]:
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    summaries: List[Dict[str, Any]] = []

    for prof in cluster_profiles:
        st = read_first_structure(prof["medoid_file"])
        out_name = "Cluster_%s_Medoid_TrajFrame_%s.mae" % (
            prof["cluster_id"],
            prof["medoid_traj_frame_number"]
        )
        out_path = os.path.join(out_dir, out_name)
        st.write(out_path)

        summaries.append({
            "cluster_id": prof["cluster_id"],
            "population": prof["population"],
            "occupancy_percent": prof["occupancy_percent"],
            "medoid_index_in_subset": prof["medoid_global_idx"],
            "medoid_traj_frame_number": prof["medoid_traj_frame_number"],
            "medoid_original_file": os.path.basename(prof["medoid_file"]),
            "medoid_output_file": out_name,
            "cluster_mean_spread": prof["cluster_mean_spread"],
            "medoid_centrality": prof["medoid_centrality"],
            "nearest_neighbor_density": prof["nearest_neighbor_density"],
            "segment_count": prof["segment_count"],
            "longest_run": prof["longest_run"],
            "temporal_fraction": prof["temporal_fraction"]
        })

    return summaries


# =============================================================================
# 13. REP1 / REP2 SELECTION WITH CONSENSUS
# =============================================================================


def choose_rep1(cluster_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Rep1: medoid of the largest cluster.
    Tie-breakers: lower medoid_centrality, then higher density.
    """
    sorted_profiles = sorted(
        cluster_profiles,
        key=lambda x: (
            -x["population"],
            x["medoid_centrality"],
            -x["nearest_neighbor_density"]
        )
    )
    return sorted_profiles[0]


def compute_rep_pair_metrics(
    rep1_profile: Dict[str, Any],
    rep2_profile: Dict[str, Any],
    X_original: np.ndarray,
    fit_asl: str,
    feature_asl: str,
    feature_mode: str
) -> Dict[str, Any]:
    rep1_file = rep1_profile["medoid_file"]
    rep2_file = rep2_profile["medoid_file"]

    st1 = read_first_structure(rep1_file)
    st2 = read_first_structure(rep2_file)

    fit1_aids, fit1_sig = get_asl_indices_and_signature(st1, fit_asl)
    fit2_aids, fit2_sig = get_asl_indices_and_signature(st2, fit_asl)
    if fit1_sig != fit2_sig:
        raise RuntimeError("Rep1/Rep2 fit_asl atoms do not match.")

    feat1_aids, feat1_sig = get_asl_indices_and_signature(st1, feature_asl)
    feat2_aids, feat2_sig = get_asl_indices_and_signature(st2, feature_asl)
    if feat1_sig != feat2_sig:
        raise RuntimeError("Rep1/Rep2 feature_asl atoms do not match.")

    fit1_xyz = extract_xyz(st1, fit1_aids)
    fit2_xyz = extract_xyz(st2, fit2_aids)
    feat1_xyz = extract_xyz(st1, feat1_aids)
    feat2_xyz = extract_xyz(st2, feat2_aids)

    backbone_rmsd = aligned_rmsd(fit2_xyz, fit1_xyz)
    feature_asl_rmsd = aligned_rmsd(feat2_xyz, feat1_xyz)

    rep1_idx = rep1_profile["medoid_global_idx"]
    rep2_idx = rep2_profile["medoid_global_idx"]

    f1 = X_original[rep1_idx]
    f2 = X_original[rep2_idx]
    feature_distance = float(np.linalg.norm(f1 - f2))
    mean_abs_feature_delta = float(np.mean(np.abs(f1 - f2)))
    max_abs_feature_delta = float(np.max(np.abs(f1 - f2)))

    return {
        "rep1_cluster_id": rep1_profile["cluster_id"],
        "rep2_cluster_id": rep2_profile["cluster_id"],
        "rep1_traj_frame": rep1_profile["medoid_traj_frame_number"],
        "rep2_traj_frame": rep2_profile["medoid_traj_frame_number"],
        "rep1_rep2_backbone_rmsd_A": float(backbone_rmsd),
        "rep1_rep2_feature_asl_rmsd_A": float(feature_asl_rmsd),
        "rep1_rep2_feature_distance": float(feature_distance),
        "rep1_rep2_mean_abs_feature_delta": float(mean_abs_feature_delta),
        "rep1_rep2_max_abs_feature_delta": float(max_abs_feature_delta),
        "feature_mode": feature_mode
    }


def rank_rep2_candidates(
    cluster_profiles: List[Dict[str, Any]],
    rep1_profile: Dict[str, Any],
    X_original: np.ndarray,
    fit_asl: str,
    feature_asl: str,
    feature_mode: str,
    rep2_min_occupancy_percent: float,
    rep2_min_longest_run: int,
    rep2_min_temporal_fraction: float,
    rep2_min_backbone_rmsd: float
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for prof in cluster_profiles:
        if prof["cluster_id"] == rep1_profile["cluster_id"]:
            continue

        pair = compute_rep_pair_metrics(
            rep1_profile,
            prof,
            X_original=X_original,
            fit_asl=fit_asl,
            feature_asl=feature_asl,
            feature_mode=feature_mode
        )

        criterion_clustering = bool(
            prof["occupancy_percent"] >= rep2_min_occupancy_percent
        )
        criterion_temporal = bool(
            (prof["longest_run"] >= rep2_min_longest_run) and
            (prof["temporal_fraction"] >= rep2_min_temporal_fraction)
        )
        criterion_difference = bool(
            pair["rep1_rep2_backbone_rmsd_A"] >= rep2_min_backbone_rmsd
        )

        consensus_pass_count = (
            int(criterion_clustering) +
            int(criterion_temporal) +
            int(criterion_difference)
        )

        row = dict(prof)
        row.update(pair)
        row["criterion_clustering"] = int(criterion_clustering)
        row["criterion_temporal"] = int(criterion_temporal)
        row["criterion_difference"] = int(criterion_difference)
        row["consensus_pass_count"] = int(consensus_pass_count)
        candidates.append(row)

    if len(candidates) == 0:
        return candidates

    occ_norm = normalize_array(
        [c["occupancy_percent"] for c in candidates],
        higher_is_better=True
    )
    cen_norm = normalize_array(
        [c["medoid_centrality"] for c in candidates],
        higher_is_better=False
    )
    diff_norm = normalize_array(
        [c["rep1_rep2_backbone_rmsd_A"] for c in candidates],
        higher_is_better=True
    )
    tmp_norm = normalize_array(
        [c["temporal_fraction"] for c in candidates],
        higher_is_better=True
    )

    for i, c in enumerate(candidates):
        score = (
            0.35 * occ_norm[i] +
            0.20 * cen_norm[i] +
            0.30 * diff_norm[i] +
            0.15 * tmp_norm[i]
        )
        c["selection_score"] = float(score)

    candidates = sorted(
        candidates,
        key=lambda x: (
            -x["consensus_pass_count"],
            -x["selection_score"],
            -x["occupancy_percent"]
        )
    )
    return candidates


def choose_top2_representatives(
    cluster_profiles: List[Dict[str, Any]],
    X_original: np.ndarray,
    fit_asl: str,
    feature_asl: str,
    feature_mode: str,
    rep2_min_occupancy_percent: float,
    rep2_min_longest_run: int,
    rep2_min_temporal_fraction: float,
    rep2_min_backbone_rmsd: float
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    rep1 = choose_rep1(cluster_profiles)

    ranked = rank_rep2_candidates(
        cluster_profiles=cluster_profiles,
        rep1_profile=rep1,
        X_original=X_original,
        fit_asl=fit_asl,
        feature_asl=feature_asl,
        feature_mode=feature_mode,
        rep2_min_occupancy_percent=rep2_min_occupancy_percent,
        rep2_min_longest_run=rep2_min_longest_run,
        rep2_min_temporal_fraction=rep2_min_temporal_fraction,
        rep2_min_backbone_rmsd=rep2_min_backbone_rmsd
    )

    if len(ranked) == 0:
        raise RuntimeError(
            "No Rep2 candidate could be identified. At least two clusters are required."
        )

    consensus_ok = [r for r in ranked if r["consensus_pass_count"] >= 2]
    if len(consensus_ok) > 0:
        rep2_cluster_id = consensus_ok[0]["cluster_id"]
    else:
        rep2_cluster_id = ranked[0]["cluster_id"]
        logger.warning(
            "No Rep2 candidate passed the 2/3 consensus threshold. "
            "The top-scoring candidate was selected."
        )

    rep2: Optional[Dict[str, Any]] = None
    for prof in cluster_profiles:
        if prof["cluster_id"] == rep2_cluster_id:
            rep2 = prof
            break

    if rep2 is None:
        raise RuntimeError("Rep2 cluster profile could not be found.")

    return rep1, rep2, ranked


def export_top2_representatives(
    rep1: Dict[str, Any],
    rep2: Dict[str, Any],
    out_dir: str
) -> Tuple[str, str]:
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    st1 = read_first_structure(rep1["medoid_file"])
    st2 = read_first_structure(rep2["medoid_file"])

    rep1_name = "REP1_cluster_%s_trajframe_%s.mae" % (
        rep1["cluster_id"],
        rep1["medoid_traj_frame_number"]
    )
    rep2_name = "REP2_cluster_%s_trajframe_%s.mae" % (
        rep2["cluster_id"],
        rep2["medoid_traj_frame_number"]
    )

    st1.write(os.path.join(out_dir, rep1_name))
    st2.write(os.path.join(out_dir, rep2_name))

    return rep1_name, rep2_name


# =============================================================================
# 14. REPORTING
# =============================================================================


def write_csv_reports(
    k_metrics: List[Dict[str, Any]],
    labels: np.ndarray,
    aligned_files: List[str],
    aligned_frame_numbers: List[int],
    cluster_summaries: List[Dict[str, Any]],
    pca: PCA,
    base_dir: str
) -> None:
    """
    Write clustering and dimensionality-reduction reports to CSV files.

    The k-metrics summary includes the weighted consensus score used for model
    selection, together with the normalized component metrics from which it was
    derived.
    """
    logger.info("Writing CSV reports...")

    with open(os.path.join(base_dir, "1_k_metrics_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "k_value",
            "silhouette_score",
            "davies_bouldin_score",
            "normalized_silhouette_score",
            "normalized_davies_bouldin_score",
            "consensus_score",
            "singleton_cluster_count",
            "min_cluster_size",
            "valid_for_selection"
        ])
        for m in k_metrics:
            writer.writerow([
                m["k"],
                "%.4f" % m["silhouette_score"],
                "%.4f" % m["davies_bouldin_score"],
                "%.4f" % m.get("normalized_silhouette_score", 0.0),
                "%.4f" % m.get("normalized_davies_bouldin_score", 0.0),
                "%.4f" % m.get("consensus_score", 0.0),
                m["singleton_cluster_count"],
                m["min_cluster_size"],
                int(m["valid_for_selection"])
            ])

    with open(os.path.join(base_dir, "1b_pca_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pc_index",
            "explained_variance_ratio",
            "cumulative_explained_variance"
        ])
        cum = 0.0
        for i, v in enumerate(pca.explained_variance_ratio_, start=1):
            cum += float(v)
            writer.writerow([i, "%.6f" % v, "%.6f" % cum])

    with open(os.path.join(base_dir, "2_cluster_assignments.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subset_index",
            "trajectory_frame_number",
            "aligned_file",
            "assigned_cluster"
        ])
        for i, (label, file_path, frnum) in enumerate(
            zip(labels, aligned_files, aligned_frame_numbers)
        ):
            writer.writerow([i, frnum, os.path.basename(file_path), int(label)])

    with open(os.path.join(base_dir, "3_cluster_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cluster_id",
            "population",
            "occupancy_percent",
            "medoid_index_in_subset",
            "medoid_traj_frame_number",
            "medoid_original_file",
            "medoid_output_file",
            "cluster_mean_spread",
            "medoid_centrality",
            "nearest_neighbor_density",
            "segment_count",
            "longest_run",
            "temporal_fraction"
        ])
        for c in sorted(cluster_summaries, key=lambda x: x["population"], reverse=True):
            writer.writerow([
                c["cluster_id"],
                c["population"],
                "%.2f" % c["occupancy_percent"],
                c["medoid_index_in_subset"],
                c["medoid_traj_frame_number"],
                c["medoid_original_file"],
                c["medoid_output_file"],
                "%.6f" % c["cluster_mean_spread"],
                "%.6f" % c["medoid_centrality"],
                "%.6f" % c["nearest_neighbor_density"],
                c["segment_count"],
                c["longest_run"],
                "%.6f" % c["temporal_fraction"]
            ])


def write_representative_ranking_csv(
    path: str,
    rep1: Dict[str, Any],
    ranked_rep2: List[Dict[str, Any]]
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cluster_id",
            "is_rep1",
            "candidate_for_rep2",
            "population",
            "occupancy_percent",
            "medoid_traj_frame_number",
            "medoid_centrality",
            "nearest_neighbor_density",
            "cluster_mean_spread",
            "segment_count",
            "longest_run",
            "temporal_fraction",
            "rep1_rep2_backbone_rmsd_A",
            "rep1_rep2_feature_asl_rmsd_A",
            "rep1_rep2_feature_distance",
            "criterion_clustering",
            "criterion_temporal",
            "criterion_difference",
            "consensus_pass_count",
            "selection_score"
        ])

        writer.writerow([
            rep1["cluster_id"],
            1,
            0,
            rep1["population"],
            "%.2f" % rep1["occupancy_percent"],
            rep1["medoid_traj_frame_number"],
            "%.6f" % rep1["medoid_centrality"],
            "%.6f" % rep1["nearest_neighbor_density"],
            "%.6f" % rep1["cluster_mean_spread"],
            rep1["segment_count"],
            rep1["longest_run"],
            "%.6f" % rep1["temporal_fraction"],
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
        ])

        for r in ranked_rep2:
            writer.writerow([
                r["cluster_id"],
                0,
                1,
                r["population"],
                "%.2f" % r["occupancy_percent"],
                r["medoid_traj_frame_number"],
                "%.6f" % r["medoid_centrality"],
                "%.6f" % r["nearest_neighbor_density"],
                "%.6f" % r["cluster_mean_spread"],
                r["segment_count"],
                r["longest_run"],
                "%.6f" % r["temporal_fraction"],
                "%.6f" % r["rep1_rep2_backbone_rmsd_A"],
                "%.6f" % r["rep1_rep2_feature_asl_rmsd_A"],
                "%.6f" % r["rep1_rep2_feature_distance"],
                r["criterion_clustering"],
                r["criterion_temporal"],
                r["criterion_difference"],
                r["consensus_pass_count"],
                "%.6f" % r["selection_score"]
            ])


def write_pair_metrics_csv(path: str, pair_metrics: Dict[str, Any]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rep1_cluster_id",
            "rep2_cluster_id",
            "rep1_traj_frame",
            "rep2_traj_frame",
            "rep1_rep2_backbone_rmsd_A",
            "rep1_rep2_feature_asl_rmsd_A",
            "rep1_rep2_feature_distance",
            "rep1_rep2_mean_abs_feature_delta",
            "rep1_rep2_max_abs_feature_delta",
            "feature_mode"
        ])
        writer.writerow([
            pair_metrics["rep1_cluster_id"],
            pair_metrics["rep2_cluster_id"],
            pair_metrics["rep1_traj_frame"],
            pair_metrics["rep2_traj_frame"],
            "%.6f" % pair_metrics["rep1_rep2_backbone_rmsd_A"],
            "%.6f" % pair_metrics["rep1_rep2_feature_asl_rmsd_A"],
            "%.6f" % pair_metrics["rep1_rep2_feature_distance"],
            "%.6f" % pair_metrics["rep1_rep2_mean_abs_feature_delta"],
            "%.6f" % pair_metrics["rep1_rep2_max_abs_feature_delta"],
            pair_metrics["feature_mode"]
        ])


def write_temporal_report_csv(path: str, cluster_profiles: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cluster_id",
            "population",
            "segment_count",
            "longest_run",
            "temporal_fraction"
        ])
        for p in sorted(cluster_profiles, key=lambda x: x["population"], reverse=True):
            writer.writerow([
                p["cluster_id"],
                p["population"],
                p["segment_count"],
                p["longest_run"],
                "%.6f" % p["temporal_fraction"]
            ])


# =============================================================================
# 15. START / END FRAME PARSERS + STRIDE RESOLUTION
# =============================================================================


def positive_int_arg(value: str) -> int:
    try:
        v = int(value)
        if v <= 0:
            raise ValueError
        return v
    except Exception:
        raise argparse.ArgumentTypeError("value must be a positive integer")


def non_negative_int_arg(value: str) -> int:
    try:
        v = int(value)
        if v < 0:
            raise ValueError
        return v
    except Exception:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")


def parse_start_frame_arg(value: str) -> Union[int, str]:
    if str(value).lower() == "auto":
        return "auto"
    try:
        v = int(value)
        if v < 0:
            raise ValueError
        return v
    except Exception:
        raise argparse.ArgumentTypeError(
            "--start-frame must be a non-negative integer or 'auto'"
        )


def parse_end_frame_arg(value: str) -> Union[int, str]:
    if str(value).lower() == "auto":
        return "auto"
    try:
        v = int(value)
        if v != -1 and v < 0:
            raise ValueError
        return v
    except Exception:
        raise argparse.ArgumentTypeError(
            "--end-frame must be -1, a non-negative integer, or 'auto'"
        )


def parse_pca_variance_cutoff_arg(value: str) -> Union[int, float]:
    try:
        v = float(value)
    except Exception:
        raise argparse.ArgumentTypeError(
            "--pca-variance-cutoff must be a positive float or integer"
        )

    if v <= 0.0:
        raise argparse.ArgumentTypeError(
            "--pca-variance-cutoff must be > 0"
        )

    if 0.0 < v < 1.0:
        return float(v)

    if float(v).is_integer():
        return int(v)

    raise argparse.ArgumentTypeError(
        "--pca-variance-cutoff values > 1 must be whole integers "
        "(e.g. 5, 10). Fractional values in (0, 1) are treated as "
        "explained-variance cutoffs."
    )


def resolve_strides(args: Any) -> Tuple[int, int]:
    if args.preview_stride is not None:
        preview_stride = args.preview_stride
    elif args.stride is not None:
        preview_stride = args.stride
    else:
        preview_stride = 20

    if args.cluster_stride is not None:
        cluster_stride = args.cluster_stride
    elif args.stride is not None:
        cluster_stride = args.stride
    else:
        cluster_stride = 50

    return preview_stride, cluster_stride


# =============================================================================
# 16. MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Representative-frame selection pipeline for Desmond trajectories"
        )
    )

    parser.add_argument("--cms", required=True, help="Desmond -out.cms file")
    parser.add_argument("--trj", required=True, help="Desmond _trj directory")

    parser.add_argument(
        "--stride",
        type=positive_int_arg,
        default=None,
        help=(
            "Legacy shared stride argument. If provided, it is used for both "
            "preview and clustering."
        )
    )
    parser.add_argument(
        "--preview-stride",
        type=positive_int_arg,
        default=None,
        help="Extraction stride used for the RMSD preview"
    )
    parser.add_argument(
        "--cluster-stride",
        type=positive_int_arg,
        default=None,
        help="Extraction stride used for clustering"
    )

    parser.add_argument(
        "--end-frame",
        type=parse_end_frame_arg,
        default="auto",
        help=(
            "Trajectory end frame number (exclusive), or 'auto'. "
            "'auto' uses the last available frame and marks the last extracted "
            "preview frame on the RMSD plot; -1 uses the last available frame "
            "without drawing an end marker."
        )
    )
    parser.add_argument(
        "--start-frame",
        type=parse_start_frame_arg,
        default="auto",
        help="Trajectory start frame number for clustering (inclusive), or 'auto'"
    )

    parser.add_argument(
        "--extract-asl",
        default="protein",
        help="ASL extracted from the trajectory"
    )
    parser.add_argument(
        "--fit-asl",
        default="protein and backbone and not atom.ele H",
        help="ASL used for structural alignment"
    )
    parser.add_argument(
        "--feature-asl",
        default="protein and atom.ptype CA",
        help="ASL used for feature generation"
    )
    parser.add_argument(
        "--feature-mode",
        choices=["ca_distances", "ca_xyz"],
        default="ca_distances",
        help="Feature representation mode (default: ca_distances)"
    )
    parser.add_argument(
        "--ref",
        default="experimental_structure.mae",
        help="Reference structure file (default: experimental_structure.mae)"
    )

    parser.add_argument("--k-min", type=positive_int_arg, default=2)
    parser.add_argument("--k-max", type=positive_int_arg, default=8)
    parser.add_argument(
        "--pca-variance-cutoff",
        type=parse_pca_variance_cutoff_arg,
        default=0.90,
        help=(
            "If 0<x<1, use cumulative explained variance cutoff; "
            "if x>=1, x must be an integer number of PCA components."
        )
    )
    parser.add_argument("--pca-random-state", type=int, default=0)

    parser.add_argument(
        "--rolling-window",
        type=positive_int_arg,
        default=5,
        help="Rolling window used for RMSD-based auto-trimming"
    )
    parser.add_argument(
        "--rmsd-std-threshold",
        type=float,
        default=0.25,
        help="Auto-trim RMSD standard-deviation threshold (A)"
    )
    parser.add_argument(
        "--rmsd-slope-threshold",
        type=float,
        default=0.02,
        help="Auto-trim rolling-slope threshold (A/window)"
    )
    parser.add_argument(
        "--stable-windows-required",
        type=positive_int_arg,
        default=3,
        help="Number of consecutive stable windows required"
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Generate only the RMSD preview and then exit"
    )

    parser.add_argument(
        "--max-ca-gap",
        type=float,
        default=8.0,
        help="Maximum consecutive C-alpha gap allowed in the PBC sanity check (A)"
    )
    parser.add_argument(
        "--pbc-error-fraction",
        type=float,
        default=0.05,
        help=(
            "Raise an error if the fraction of suspicious frames equals or exceeds "
            "this threshold"
        )
    )

    parser.add_argument(
        "--low-silhouette-threshold",
        type=float,
        default=0.20,
        help="Emit a quality warning if silhouette falls below this value"
    )
    parser.add_argument(
        "--allow-singletons",
        action="store_true",
        help=(
            "Allow clustering solutions containing singleton clusters to "
            "participate in model selection"
        )
    )

    parser.add_argument(
        "--rep2-min-occupancy-percent",
        type=float,
        default=10.0,
        help="Minimum occupancy percentage required for a Rep2 candidate"
    )
    parser.add_argument(
        "--rep2-min-longest-run",
        type=positive_int_arg,
        default=2,
        help="Minimum longest_run required for a Rep2 candidate"
    )
    parser.add_argument(
        "--rep2-min-temporal-fraction",
        type=float,
        default=0.25,
        help="Minimum temporal_fraction required for a Rep2 candidate"
    )
    parser.add_argument(
        "--rep2-min-backbone-rmsd",
        type=float,
        default=0.75,
        help="Minimum backbone RMSD from Rep1 required for a Rep2 candidate (A)"
    )

    args = parser.parse_args()

    if args.k_min > args.k_max:
        parser.error("--k-min cannot be greater than --k-max")
    if args.pca_variance_cutoff <= 0:
        parser.error("--pca-variance-cutoff must be > 0")
    if args.rmsd_std_threshold < 0.0:
        parser.error("--rmsd-std-threshold must be >= 0")
    if args.rmsd_slope_threshold < 0.0:
        parser.error("--rmsd-slope-threshold must be >= 0")
    if args.max_ca_gap <= 0.0:
        parser.error("--max-ca-gap must be > 0")
    if not (0.0 <= args.pbc_error_fraction <= 1.0):
        parser.error("--pbc-error-fraction must be in the interval [0, 1]")
    if args.low_silhouette_threshold < -1.0 or args.low_silhouette_threshold > 1.0:
        parser.error("--low-silhouette-threshold must be in the interval [-1, 1]")
    if args.rep2_min_occupancy_percent < 0.0 or args.rep2_min_occupancy_percent > 100.0:
        parser.error("--rep2-min-occupancy-percent must be in the interval [0, 100]")
    if args.rep2_min_temporal_fraction < 0.0 or args.rep2_min_temporal_fraction > 1.0:
        parser.error("--rep2-min-temporal-fraction must be in the interval [0, 1]")
    if args.rep2_min_backbone_rmsd < 0.0:
        parser.error("--rep2-min-backbone-rmsd must be >= 0")

    base_out_dir = "Clustering_Results"
    os.makedirs(base_out_dir, exist_ok=True)

    preview_stride, cluster_stride = resolve_strides(args)

    if args.end_frame == "auto":
        chosen_end_frame = -1
        preview_end_marker: Union[int, str] = "auto"
        logger.info(
            "Automatic end frame selected: using the last available frame "
            "and marking the last extracted frame on the preview plot."
        )
    elif args.end_frame == -1:
        chosen_end_frame = -1
        preview_end_marker = -1
        logger.info(
            "End frame set to -1: using the trajectory until the end "
            "without drawing an end marker on the preview plot."
        )
    else:
        chosen_end_frame = args.end_frame
        preview_end_marker = args.end_frame
        logger.info("User-defined end frame cutoff (exclusive): %s", chosen_end_frame)

    preview_frames_dir = os.path.join(base_out_dir, "0_preview_frames")
    cluster_raw_frames_dir = os.path.join(base_out_dir, "1_raw_frames")
    aligned_frames_dir = os.path.join(base_out_dir, "2_aligned_frames")

    ref_path = resolve_reference_structure(args.ref)
    validate_input_paths(args.cms, args.trj, ref_path)

    ref_st = read_first_structure(ref_path)
    logger.info("Reference structure: %s", ref_path)
    logger.info(
        "Preview stride: %d | Cluster stride: %d",
        preview_stride,
        cluster_stride
    )

    if SCHRODINGER_RUN_EXE is not None:
        logger.info("Schrödinger run executable: %s", SCHRODINGER_RUN_EXE)
    else:
        logger.info("Schrödinger run executable: resolved lazily when needed.")

    ref_fit_aids, ref_fit_sig = get_asl_indices_and_signature(ref_st, args.fit_asl)
    ref_feat_aids, ref_feat_sig = get_asl_indices_and_signature(ref_st, args.feature_asl)
    logger.info("fit_asl atom count: %d", len(ref_fit_aids))
    logger.info("feature_asl atom count: %d", len(ref_feat_aids))

    preview_files, preview_frame_numbers = extract_frames(
        cms_file=args.cms,
        trj_dir=args.trj,
        out_folder=preview_frames_dir,
        stride=preview_stride,
        extract_asl=args.extract_asl,
        start_frame=0,
        end_frame=chosen_end_frame,
        clean_folder=True
    )

    validate_asl_consistency_against_reference(
        preview_files,
        ref_fit_sig,
        args.fit_asl,
        "fit_asl (preview)"
    )
    validate_asl_consistency_against_reference(
        preview_files,
        ref_feat_sig,
        args.feature_asl,
        "feature_asl (preview)"
    )

    run_pbc_sanity_check(
        preview_files,
        ca_asl="protein and atom.ptype CA",
        max_ca_gap=args.max_ca_gap,
        suspicious_fraction_error=args.pbc_error_fraction
    )

    preview_csv_path = os.path.join(base_out_dir, "0_rmsd_preview.csv")
    preview_png_path = os.path.join(base_out_dir, "0_rmsd_preview.png")

    preview_info = compute_rmsd_preview(
        raw_files=preview_files,
        frame_numbers=preview_frame_numbers,
        ref_st=ref_st,
        fit_asl=args.fit_asl,
        out_csv=preview_csv_path,
        out_png=preview_png_path,
        rolling_window=args.rolling_window,
        auto_trim=(args.start_frame == "auto"),
        rmsd_std_threshold=args.rmsd_std_threshold,
        rmsd_slope_threshold=args.rmsd_slope_threshold,
        stable_windows_required=args.stable_windows_required,
        explicit_end_frame=preview_end_marker
    )

    if args.preview_only:
        logger.info("preview-only is active. The script is terminating here.")
        return

    if args.start_frame == "auto":
        chosen_start_frame = preview_info["suggestion_frame"]
        logger.info("Auto-trim selected start frame: %s", chosen_start_frame)
    else:
        chosen_start_frame = args.start_frame
        logger.info("User-defined start frame: %s", chosen_start_frame)

    if chosen_end_frame != -1 and chosen_start_frame >= chosen_end_frame:
        raise RuntimeError(
            "The chosen start frame (%d) must be smaller than the chosen end frame (%d). "
            "Note: end frame is exclusive."
            % (chosen_start_frame, chosen_end_frame)
        )

    raw_files, frame_numbers = extract_frames(
        cms_file=args.cms,
        trj_dir=args.trj,
        out_folder=cluster_raw_frames_dir,
        stride=cluster_stride,
        extract_asl=args.extract_asl,
        start_frame=chosen_start_frame,
        end_frame=chosen_end_frame,
        clean_folder=True
    )

    validate_asl_consistency_against_reference(
        raw_files,
        ref_fit_sig,
        args.fit_asl,
        "fit_asl (cluster)"
    )
    validate_asl_consistency_against_reference(
        raw_files,
        ref_feat_sig,
        args.feature_asl,
        "feature_asl (cluster)"
    )

    run_pbc_sanity_check(
        raw_files,
        ca_asl="protein and atom.ptype CA",
        max_ca_gap=args.max_ca_gap,
        suspicious_fraction_error=args.pbc_error_fraction
    )

    if len(raw_files) < 3:
        raise RuntimeError(
            "Too few frames remain for clustering: %d" % len(raw_files)
        )

    logger.info(
        "Clustering set: %d frames | initial trajectory frame=%s",
        len(raw_files),
        frame_numbers[0]
    )

    aligned_files, aligned_frame_numbers = align_and_save_frames_to_reference(
        input_files=raw_files,
        input_frame_numbers=frame_numbers,
        aligned_folder=aligned_frames_dir,
        ref_st=ref_st,
        fit_asl=args.fit_asl
    )

    X_features = build_features(
        aligned_files=aligned_files,
        ref_st=ref_st,
        feature_asl=args.feature_asl,
        feature_mode=args.feature_mode
    )

    labels, X_red, best_k, k_metrics, pca = find_best_k_and_cluster(
        X_features,
        k_min=args.k_min,
        k_max=args.k_max,
        pca_variance_cutoff=args.pca_variance_cutoff,
        allow_singletons=args.allow_singletons,
        pca_random_state=args.pca_random_state
    )

    final_silhouette = silhouette_score(X_red, labels)
    final_singletons = count_singleton_clusters(labels)

    selected_metric_row: Optional[Dict[str, Any]] = None
    for row in k_metrics:
        if int(row["k"]) == int(best_k):
            selected_metric_row = row
            break

    final_consensus_score = (
        float(selected_metric_row["consensus_score"])
        if selected_metric_row is not None and "consensus_score" in selected_metric_row
        else float("nan")
    )
    final_davies_bouldin = (
        float(selected_metric_row["davies_bouldin_score"])
        if selected_metric_row is not None and "davies_bouldin_score" in selected_metric_row
        else float("nan")
    )

    quality_lines: List[str] = []
    quality_lines.append("Representative frame selection quality report")
    quality_lines.append("")
    quality_lines.append("preview_stride = %d" % preview_stride)
    quality_lines.append("cluster_stride = %d" % cluster_stride)
    quality_lines.append("chosen_start_frame = %s" % chosen_start_frame)
    if args.end_frame == "auto":
        quality_lines.append(
            "chosen_end_frame = auto (marked at the last extracted frame on the preview plot)"
        )
    elif args.end_frame == -1:
        quality_lines.append(
            "chosen_end_frame = -1 (trajectory used until the end; no end marker on the preview plot)"
        )
    else:
        quality_lines.append(
            "chosen_end_frame = %s (explicit exclusive cutoff; marked on the preview plot)"
            % chosen_end_frame
        )
    quality_lines.append("n_cluster_frames = %d" % len(raw_files))
    quality_lines.append("best_k = %d" % best_k)
    quality_lines.append("final_consensus_score = %.4f" % final_consensus_score)
    quality_lines.append("final_silhouette = %.4f" % final_silhouette)
    quality_lines.append("final_davies_bouldin = %.4f" % final_davies_bouldin)
    quality_lines.append("final_singleton_clusters = %d" % final_singletons)

    if final_silhouette < args.low_silhouette_threshold:
        warn = (
            "WARNING: The silhouette score is low (%.4f < %.4f). "
            "Cluster separation may be weak. Consider using a smaller preview stride, "
            "a smaller cluster stride, more frames, or a different start frame."
            % (final_silhouette, args.low_silhouette_threshold)
        )
        logger.warning(warn)
        quality_lines.append(warn)

    if final_singletons > 0:
        warn = (
            "WARNING: The final clustering contains %d singleton clusters. "
            "This may indicate outliers, sparse sampling, or an overly large k."
            % final_singletons
        )
        logger.warning(warn)
        quality_lines.append(warn)

    cluster_profiles = build_cluster_profiles(
        aligned_files=aligned_files,
        aligned_frame_numbers=aligned_frame_numbers,
        X_original=X_features,
        X_red=X_red,
        labels=labels
    )

    rep_dir = os.path.join(base_out_dir, "3_representatives")
    cluster_summaries = export_all_cluster_medoids(
        cluster_profiles=cluster_profiles,
        out_dir=rep_dir
    )

    rep1, rep2, ranked_rep2 = choose_top2_representatives(
        cluster_profiles=cluster_profiles,
        X_original=X_features,
        fit_asl=args.fit_asl,
        feature_asl=args.feature_asl,
        feature_mode=args.feature_mode,
        rep2_min_occupancy_percent=args.rep2_min_occupancy_percent,
        rep2_min_longest_run=args.rep2_min_longest_run,
        rep2_min_temporal_fraction=args.rep2_min_temporal_fraction,
        rep2_min_backbone_rmsd=args.rep2_min_backbone_rmsd
    )

    top2_dir = os.path.join(base_out_dir, "3_top2_representatives")
    rep1_name, rep2_name = export_top2_representatives(
        rep1,
        rep2,
        out_dir=top2_dir
    )

    pair_metrics = compute_rep_pair_metrics(
        rep1_profile=rep1,
        rep2_profile=rep2,
        X_original=X_features,
        fit_asl=args.fit_asl,
        feature_asl=args.feature_asl,
        feature_mode=args.feature_mode
    )

    quality_lines.append("")
    quality_lines.append("Selected top-2 representatives")
    quality_lines.append("rep1_cluster_id = %d" % rep1["cluster_id"])
    quality_lines.append("rep1_traj_frame = %d" % rep1["medoid_traj_frame_number"])
    quality_lines.append("rep2_cluster_id = %d" % rep2["cluster_id"])
    quality_lines.append("rep2_traj_frame = %d" % rep2["medoid_traj_frame_number"])
    quality_lines.append(
        "rep1_rep2_backbone_rmsd_A = %.4f" %
        pair_metrics["rep1_rep2_backbone_rmsd_A"]
    )
    quality_lines.append(
        "rep1_rep2_feature_asl_rmsd_A = %.4f" %
        pair_metrics["rep1_rep2_feature_asl_rmsd_A"]
    )
    quality_lines.append(
        "rep1_rep2_feature_distance = %.4f" %
        pair_metrics["rep1_rep2_feature_distance"]
    )

    write_csv_reports(
        k_metrics=k_metrics,
        labels=labels,
        aligned_files=aligned_files,
        aligned_frame_numbers=aligned_frame_numbers,
        cluster_summaries=cluster_summaries,
        pca=pca,
        base_dir=base_out_dir
    )

    write_representative_ranking_csv(
        os.path.join(base_out_dir, "5_representative_ranking.csv"),
        rep1,
        ranked_rep2
    )
    write_pair_metrics_csv(
        os.path.join(base_out_dir, "6_representative_pair_metrics.csv"),
        pair_metrics
    )
    write_temporal_report_csv(
        os.path.join(base_out_dir, "7_temporal_continuity_report.csv"),
        cluster_profiles
    )
    write_quality_report(
        os.path.join(base_out_dir, "4_quality_report.txt"),
        quality_lines
    )

    logger.info("All tasks completed successfully.")
    logger.info("Selected representative frames:")
    logger.info(
        " - REP1: cluster=%d | traj_frame=%d | file=%s",
        rep1["cluster_id"],
        rep1["medoid_traj_frame_number"],
        rep1_name
    )
    logger.info(
        " - REP2: cluster=%d | traj_frame=%d | file=%s",
        rep2["cluster_id"],
        rep2["medoid_traj_frame_number"],
        rep2_name
    )
    logger.info("Pair metrics:")
    logger.info(
        " - Rep1/Rep2 backbone RMSD: %.4f A",
        pair_metrics["rep1_rep2_backbone_rmsd_A"]
    )
    logger.info(
        " - Rep1/Rep2 feature ASL RMSD: %.4f A",
        pair_metrics["rep1_rep2_feature_asl_rmsd_A"]
    )
    logger.info(
        " - Rep1/Rep2 feature distance: %.4f",
        pair_metrics["rep1_rep2_feature_distance"]
    )

    logger.info("Outputs saved in '%s/':", base_out_dir)
    logger.info(" - 0_preview_frames/")
    logger.info(" - 1_raw_frames/")
    logger.info(" - 2_aligned_frames/")
    logger.info(" - 3_representatives/")
    logger.info(" - 3_top2_representatives/")
    logger.info(" - 0_rmsd_preview.csv")
    logger.info(" - 0_rmsd_preview.png")
    logger.info(" - 1_k_metrics_summary.csv")
    logger.info(" - 1b_pca_summary.csv")
    logger.info(" - 2_cluster_assignments.csv")
    logger.info(" - 3_cluster_summary.csv")
    logger.info(" - 4_quality_report.txt")
    logger.info(" - 5_representative_ranking.csv")
    logger.info(" - 6_representative_pair_metrics.csv")
    logger.info(" - 7_temporal_continuity_report.csv")


if __name__ == "__main__":
    main()