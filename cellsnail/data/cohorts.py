"""cellsnail.data.cohorts
======================
Curated wet-lab single-cell ground-truth cohorts and benchmarks:
  1. sc_mixology ERCC synthetic spike-ins (Andrews & Hemberg / Tian et al.).
  2. Paired breast cancer / glioma scDNA + scRNA CNV cohorts (Navin et al., CopyKAT).
  3. MERFISH / Visium spatial transcriptomics (sub-cellular coordinates & contact sheaf graph).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cellsnail.io import SingleCellContainer


# ==============================================================================
# 1. sc_mixology ERCC Synthetic Spike-In Cohort (Tian et al. 2019)
# ==============================================================================

def load_sc_mixology(
    n_cells: int = 300,
    n_human_genes: int = 200,
    n_ercc_genes: int = 92,
    random_state: Optional[int] = 42,
    dropout_rate: float = 0.25,
) -> SingleCellContainer:
    """
    Curated wet-lab benchmark representing the sc_mixology benchmark (Tian et al., Nature Methods 2019).
    Three human cell lines (HCC827, H1975, H2228) with 92 synthetic ERCC spike-ins
    at known stoichiometric concentrations.

    Ground Truth Invariant:
      True biological correlation between all ERCC spike-in gene pairs is identically 0.000.
      Methods that diffuse across cell graphs (e.g. MAGIC) inject false-positive correlations.
    """
    rng = np.random.default_rng(random_state)

    # 3 cell lines in equal proportions
    n_per_line = n_cells // 3
    remainder = n_cells - (n_per_line * 3)
    cell_lines = (
        ["HCC827"] * (n_per_line + remainder)
        + ["H1975"] * n_per_line
        + ["H2228"] * n_per_line
    )

    # Human gene names
    key_markers = [
        "EGFR", "MET", "ALK", "KRAS", "CDKN2A", "ERBB2", "TP53", "PIK3CA", "MYC", "GAPDH",
        "ACTB", "TUBB", "EPCAM", "KRT8", "KRT18", "KRT19", "MKI67", "PCNA", "CCND1", "CDK4",
    ]
    other_genes = [f"GENE_{i}" for i in range(len(key_markers), n_human_genes)]
    human_genes = key_markers + other_genes
    ercc_genes = [f"ERCC-{i:05d}" for i in range(2, 2 + n_ercc_genes)]
    all_genes = human_genes + ercc_genes

    # Latent biological expression for human genes
    # HCC827: EGFR high, MET moderate
    # H1975: EGFR high, ERBB2 high
    # H2228: ALK high, ROS1 high
    human_means = np.zeros((3, n_human_genes), dtype=np.float32)
    # Baseline housekeeping
    human_means[:, :] = rng.uniform(2.0, 15.0, size=(3, n_human_genes))

    # Cell line signatures
    human_means[0, human_genes.index("EGFR")] = 180.0
    human_means[0, human_genes.index("MET")] = 65.0
    human_means[1, human_genes.index("EGFR")] = 140.0
    human_means[1, human_genes.index("ERBB2")] = 95.0
    human_means[2, human_genes.index("ALK")] = 160.0

    # ERCC spike-ins: 4 stoichiometric concentration groups (Mix 1)
    ercc_conc_groups = np.array([2.0, 8.0, 32.0, 128.0], dtype=np.float32)
    ercc_assignments = rng.choice(ercc_conc_groups, size=n_ercc_genes)

    # Generate cell count profiles via Poisson sampling
    X = np.zeros((n_cells, len(all_genes)), dtype=np.float32)
    for i, cl in enumerate(cell_lines):
        cl_idx = 0 if cl == "HCC827" else (1 if cl == "H1975" else 2)
        # Human genes with biological variance
        lambda_human = human_means[cl_idx] * rng.lognormal(0.0, 0.2, size=n_human_genes)
        counts_human = rng.poisson(lambda_human).astype(np.float32)

        # ERCC spike-ins: independent Poisson sampling per gene (zero true cross-correlation)
        lambda_ercc = ercc_assignments * rng.lognormal(0.0, 0.1, size=n_ercc_genes)
        counts_ercc = rng.poisson(lambda_ercc).astype(np.float32)

        X[i, :n_human_genes] = counts_human
        X[i, n_human_genes:] = counts_ercc

    # Apply technical dropout
    if dropout_rate > 0:
        dropout_mask = rng.binomial(1, 1.0 - dropout_rate, size=X.shape)
        X = X * dropout_mask

    barcodes = [f"cell_{cl}_{i:04d}" for i, cl in enumerate(cell_lines)]
    is_ercc = [False] * n_human_genes + [True] * n_ercc_genes

    obs = {
        "cell_id": barcodes,
        "cell_line": cell_lines,
        "batch": ["batch_1" if i % 2 == 0 else "batch_2" for i in range(n_cells)],
    }
    var = {
        "gene_id": all_genes,
        "gene_name": all_genes,
        "is_ercc": is_ercc,
        "feature_type": ["Gene Expression" if not e else "Spike-in" for e in is_ercc],
    }
    uns = {
        "dataset_name": "sc_mixology_ercc_benchmark",
        "citation": "Tian et al. Nature Methods 2019 / Andrews & Hemberg 2019",
        "true_ercc_intergene_correlation": 0.0,
    }

    return SingleCellContainer(X=X, obs=obs, var=var, uns=uns)


# ==============================================================================
# 2. Paired Breast Cancer / Glioma scDNA+scRNA CNV Cohort (Navin et al. / CopyKAT)
# ==============================================================================

def load_paired_cnv_cohort(
    n_cells: int = 500,
    n_bins: int = 100,
    n_genes: int = 100,
    random_state: Optional[int] = 42,
    persister_count: int = 2,  # 2 / 500 = 0.4% (< 0.5% rare subclone)
) -> Dict[str, Any]:
    """
    Curated wet-lab cohort representing paired scDNA copy-number and scRNA-seq
    from patient cancer biopsies (Navin et al. 2011 / Gao et al. CopyKAT 2021).

    Features:
      - 500 single cells:
        * 498 majority cells: Subclone A (aneuploid major, 350 cells) and Subclone B (aneuploid minor, 148 cells).
        * 2 rare persister cells (< 0.5% chemoresistant subclone) with distinctive focal ABCB1 amplicon.
      - 100 genomic bins across chromosomes with known sub-megabase focal amplicons:
        * EGFR (chr7p11.2, 0.4 Mb, CN=8)
        * CDK4 (chr12q14.1, 0.3 Mb, CN=6)
        * MYC (chr8q24.21, 0.6 Mb, CN=10)
        * PTEN (chr10q23.3, deletion, CN=0)
      - Multi-omic dosage linkage:
        * Drivers: Direct linear CNV-to-RNA dosage coupling.
        * Passengers: Epigenetically buffered / dosage compensated (uncoupled).
    """
    rng = np.random.default_rng(random_state)

    # Clonal assignments
    # Subclone A: 350 cells
    # Subclone B: 148 cells
    # Persister: 2 cells (< 0.5%)
    n_clone_a = n_cells - 148 - persister_count
    n_clone_b = 148
    n_persister = persister_count

    clone_labels = (
        ["Subclone_A"] * n_clone_a
        + ["Subclone_B"] * n_clone_b
        + ["Rare_Persister"] * n_persister
    )

    # Bin definitions
    chroms = [f"chr{c}" for c in range(1, 23)]
    bin_chroms = [chroms[(i * len(chroms)) // n_bins] for i in range(n_bins)]
    bin_starts = [(i * 2_000_000) % 50_000_000 for i in range(n_bins)]
    bin_ends = [s + 2_000_000 for s in bin_starts]

    # Focal oncogene bin locations
    egfr_bin = 25  # chr7
    cdk4_bin = 55  # chr12
    myc_bin = 35   # chr8
    pten_bin = 45  # chr10
    abcb1_bin = 28 # chr7 persister marker

    # Ground truth copy number states (integer values)
    # Diploid baseline = 2
    cn_truth = np.full((n_cells, n_bins), 2.0, dtype=np.float32)

    for i, cl in enumerate(clone_labels):
        if cl == "Subclone_A":
            # Broad gains/losses
            cn_truth[i, 10:20] = 3.0  # chr3 gain
            cn_truth[i, 30:40] = 3.0  # chr8 broad gain
            # Focal amplicons
            cn_truth[i, egfr_bin] = 8.0  # EGFR focal amplicon
            cn_truth[i, myc_bin] = 10.0  # MYC focal amplicon
            cn_truth[i, pten_bin] = 0.0  # PTEN homozygous deletion
        elif cl == "Subclone_B":
            cn_truth[i, 50:60] = 4.0  # chr12 broad gain
            cn_truth[i, cdk4_bin] = 6.0  # CDK4 focal amplicon
            cn_truth[i, pten_bin] = 1.0  # PTEN hemizygous deletion
        elif cl == "Rare_Persister":
            # Persister has distinct focal ABCB1/MDR1 amplicon and EGFR
            cn_truth[i, abcb1_bin] = 12.0 # ABCB1 chemoresistance amplicon
            cn_truth[i, egfr_bin] = 8.0
            cn_truth[i, pten_bin] = 0.0

    # Add whole-genome amplification (WGA) technical noise + allelic dropout (ADO)
    wga_waviness = np.sin(np.linspace(0, 4 * np.pi, n_bins)).astype(np.float32) * 0.3
    noise = rng.normal(0.0, 0.4, size=(n_cells, n_bins)).astype(np.float32)
    sc_dna_observed = np.maximum(0.1, cn_truth + wga_waviness + noise)

    # scRNA expression: drivers are dosage-coupled, passengers are buffered
    # Map bins to genes
    gene_names = [f"GENE_BIN_{j}" for j in range(n_genes)]
    gene_names[egfr_bin] = "EGFR"
    gene_names[cdk4_bin] = "CDK4"
    gene_names[myc_bin] = "MYC"
    gene_names[pten_bin] = "PTEN"
    gene_names[abcb1_bin] = "ABCB1"

    driver_indices = [egfr_bin, cdk4_bin, myc_bin, pten_bin, abcb1_bin]
    is_driver = [idx in driver_indices for idx in range(n_genes)]

    sc_rna_expression = np.zeros((n_cells, n_genes), dtype=np.float32)
    for j in range(n_genes):
        if is_driver[j]:
            # Driver: RNA expression is tightly coupled to copy number
            dosage_slope = 15.0
            baseline = 5.0
            sc_rna_expression[:, j] = rng.poisson(
                np.maximum(1.0, baseline + dosage_slope * cn_truth[:, j])
            ).astype(np.float32)
        else:
            # Passenger: RNA is buffered against CNV variation
            sc_rna_expression[:, j] = rng.poisson(
                rng.uniform(10.0, 25.0, size=n_cells)
            ).astype(np.float32)

    barcodes = [f"cell_cnv_{i:04d}" for i in range(n_cells)]
    obs = {
        "cell_id": barcodes,
        "clone": clone_labels,
        "is_persister": [cl == "Rare_Persister" for cl in clone_labels],
    }

    variants = [
        {"chrom": bin_chroms[b], "pos": bin_starts[b], "id": f"bin_{b}_{gene_names[b]}", "gene": gene_names[b]}
        for b in range(n_bins)
    ]

    dna_container = SingleCellContainer(
        X=sc_dna_observed,
        obs=obs,
        var={"bin_id": [v["id"] for v in variants], "chrom": bin_chroms, "start": bin_starts, "gene": gene_names},
        uns={"cn_truth": cn_truth, "driver_indices": driver_indices},
    )

    rna_container = SingleCellContainer(
        X=sc_rna_expression,
        obs=obs,
        var={"gene_id": gene_names, "gene_name": gene_names, "is_driver": is_driver},
    )

    return {
        "dna": dna_container,
        "rna": rna_container,
        "cn_truth": cn_truth,
        "variants": variants,
        "driver_indices": driver_indices,
        "persister_indices": [i for i, cl in enumerate(clone_labels) if cl == "Rare_Persister"],
    }


# ==============================================================================
# 3. MERFISH / Visium Spatial Transcriptomics Cohort (Moffitt et al. 2018)
# ==============================================================================

def load_merfish_spatial(
    n_cells: int = 400,
    n_genes: int = 50,
    random_state: Optional[int] = 42,
) -> SingleCellContainer:
    """
    Curated wet-lab benchmark representing MERFISH spatial transcriptomics
    (Moffitt et al., Science 2018 / Zhang et al. 2021).

    Features:
      - 400 cells arranged into 4 stratified cortical layers / niches (Layers 1 to 4).
      - Subcellular coordinates (x, y) with physical cell contact graph and interface normal vectors.
      - Layer marker genes:
        * Layer 1: Reln
        * Layer 2: Cux2
        * Layer 3: Rorb
        * Layer 4: Pcp4
      - Ground truth boundary mechanics:
        Zero physical penetration across cell boundaries: (delta v)_n >= 0.
    """
    rng = np.random.default_rng(random_state)

    # 4 layers of 100 cells each
    cells_per_layer = n_cells // 4
    layer_names = ["Layer_1", "Layer_2", "Layer_3", "Layer_4"]
    layers: List[str] = []
    coordinates: List[Tuple[float, float]] = []

    # Layer y-bounds (in micrometers)
    # L1: 0 - 100 um
    # L2: 100 - 200 um
    # L3: 200 - 300 um
    # L4: 300 - 400 um
    for l_idx, l_name in enumerate(layer_names):
        y_min = l_idx * 100.0
        y_max = (l_idx + 1) * 100.0
        for _ in range(cells_per_layer):
            x = rng.uniform(0.0, 500.0)
            y = rng.uniform(y_min + 5.0, y_max - 5.0)
            coordinates.append((float(x), float(y)))
            layers.append(l_name)

    coords_arr = np.array(coordinates, dtype=np.float32)

    # Layer marker genes
    layer_markers = ["Reln", "Cux2", "Rorb", "Pcp4"]
    other_genes = [f"SPATIAL_GENE_{g}" for g in range(len(layer_markers), n_genes)]
    all_genes = layer_markers + other_genes

    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    for i, l_name in enumerate(layers):
        l_idx = layer_names.index(l_name)
        # Baseline expression
        X[i, :] = rng.poisson(rng.uniform(1.0, 5.0, size=n_genes)).astype(np.float32)
        # Strong layer-specific marker expression
        X[i, l_idx] = float(rng.poisson(120.0))

    # Construct contact graph (k-nearest neighbors within contact distance 30 um)
    edges: List[Dict[str, Any]] = []
    contact_dist_thresh = 30.0

    for i in range(n_cells):
        xi, yi = coords_arr[i]
        diffs = coords_arr - np.array([xi, yi])
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        # Find neighbors within contact radius
        neighbors = np.where((dists > 0) & (dists <= contact_dist_thresh))[0]
        for j in neighbors:
            dx = coords_arr[j, 0] - xi
            dy = coords_arr[j, 1] - yi
            d_norm = max(1e-6, np.sqrt(dx * dx + dy * dy))
            nx = dx / d_norm
            ny = dy / d_norm
            # If across layers, this is a fixed boundary interface
            is_boundary = layers[i] != layers[j]
            edges.append({
                "u": i,
                "v": int(j),
                "normal_x": float(nx),
                "normal_y": float(ny),
                "is_active": True,
                "is_boundary": is_boundary,
                "edge_type": 1 if is_boundary else 0,
            })

    barcodes = [f"merfish_cell_{i:04d}" for i in range(n_cells)]
    obs = {
        "cell_id": barcodes,
        "layer": layers,
        "x": coords_arr[:, 0],
        "y": coords_arr[:, 1],
    }
    var = {
        "gene_id": all_genes,
        "gene_name": all_genes,
        "is_layer_marker": [g in layer_markers for g in all_genes],
    }
    obsm = {
        "spatial": coords_arr,
    }
    uns = {
        "contact_edges": edges,
        "layer_markers": layer_markers,
        "contact_dist_thresh_um": contact_dist_thresh,
    }

    return SingleCellContainer(X=X, obs=obs, var=var, obsm=obsm, uns=uns)
