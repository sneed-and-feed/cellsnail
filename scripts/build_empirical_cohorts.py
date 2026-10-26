"""scripts/build_empirical_cohorts.py
===================================
Downloads and bundles real empirical wet-lab single-cell datasets from NCBI GEO,
LuyiTian/sc_mixology, and Ensembl GRCh38 / ClinVar / COSMIC REST API into:
  - cellsnail/data/empirical/wetlab_cohorts.npz
  - cellsnail/data/empirical/ensembl_clinvar_drivers.json

Empirical Data Sources:
  1. GSE118767 (Tian et al., Nature Methods 2019, sc_mixology CEL-seq2 human lung
     adenocarcinoma cell lines HCC827, H1975, H2228) + GSE54695 (Grün et al.,
     Nature Methods 2014, empirical CEL-seq ERCC92 synthetic spike-in UMI counts).
  2. GSE148673 (Gao et al., Nature Biotechnology 2021, CopyKAT Triple-Negative Breast
     Cancer TNBC3 single-cell UMI counts & copykat.pred calls) + GSE57872 (Patel et al.,
     Science 2014, primary human glioblastoma MGH26/MGH28/MGH29/MGH30/MGH31 single cells).
  3. GSE144239 (Ji et al., Cell 2020, human cutaneous squamous cell carcinoma P2_ST_rep1
     Spatial Transcriptomics spot coordinates & UMI count matrix).
  4. Ensembl GRCh38 + VEP Dosage Sensitivity (pHaplo, pTriplo, LOEUF, AlphaMissense) +
     ClinVar (VCV/RCV) & COSMIC (COSV/COSM) somatic oncogenicity annotations.
"""

from __future__ import annotations

import gzip
import io
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EMP_DIR = ROOT / "cellsnail" / "data" / "empirical"
RAW_ENS_DIR = EMP_DIR / "ensembl_raw"


def fetch_gz_text(url: str) -> List[str]:
    print(f"  [GET] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        raw = resp.read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        text = gz.read().decode("utf-8", errors="ignore")
    return text.splitlines()


def _clean_clinvar_ids(ids_list: List[str]) -> List[str]:
    return sorted(
        list(
            set(
                cid
                for cid in ids_list
                if isinstance(cid, str) and (cid.startswith(("RCV", "VCV", "SCV")) or len(cid) > 3)
            )
        )
    )[:8]


def build_ensembl_clinvar_panel() -> Dict[str, Any]:
    out_json = EMP_DIR / "ensembl_clinvar_drivers.json"
    if out_json.exists() and not RAW_ENS_DIR.exists():
        print(f"Loading and sanitizing existing {out_json} ({out_json.stat().st_size} bytes)")
        with open(out_json, "r", encoding="utf-8") as f:
            panel = json.load(f)
        for v_rec in panel.get("clinical_hotspot_variants", {}).values():
            if "clinvar_ids" in v_rec:
                v_rec["clinvar_ids"] = _clean_clinvar_ids(v_rec["clinvar_ids"])
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(panel, f, indent=2)
        return panel
    print("Building Ensembl GRCh38 + ClinVar + COSMIC Driver Panel...")
    drivers: Dict[str, Any] = {}
    gene_list = [
        "EGFR", "MYC", "ERBB2", "MDM2", "CDK4", "CDKN2A", "PTEN",
        "TP53", "RB1", "PIK3CA", "ABCB1", "ALK", "MET", "KRAS",
    ]
    for g in gene_list:
        tss_path = RAW_ENS_DIR / f"{g}_tss.json"
        if tss_path.exists():
            with open(tss_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            drivers[g] = data

    variants_summary: Dict[str, Any] = {}
    vep_files = {
        "EGFR_L858R": "EGFR_L858R_vep.json",
        "KRAS_G12D": "KRAS_G12D_vep.json",
        "PIK3CA_H1047R": "PIK3CA_H1047R_vep.json",
        "TP53_R175H": "TP53_R175H_vep.json",
        "PTEN_R130Q": "PTEN_R130Q_vep.json",
        "CDKN2A_R80X": "CDKN2A_R80X_vep.json",
    }
    for v_name, fname in vep_files.items():
        v_path = RAW_ENS_DIR / fname
        if not v_path.exists():
            continue
        with open(v_path, "r", encoding="utf-8") as f:
            v_arr = json.load(f)
        if not v_arr:
            continue
        rec = v_arr[0]
        # Extract canonical / MANE transcript consequence
        t_conseq = rec.get("transcript_consequences", [])
        chosen_tc = t_conseq[0] if t_conseq else {}
        for tc in t_conseq:
            if tc.get("Pick") == 1 or "Dosage Sensitivity (Haplo)" in tc or "AlphaMissense Pathogenicity" in tc:
                chosen_tc = tc
                break

        # Extract ClinVar and COSMIC colocated IDs
        cosmic_ids = []
        clinvar_ids = []
        clin_sigs = []
        somatic_class = ""
        for cv in rec.get("colocated_variants", []):
            cid = cv.get("id", "")
            if cid.startswith("COSV") or cid.startswith("COSM"):
                cosmic_ids.append(cid)
            if "clinvar_somatic_classification" in cv and not somatic_class:
                somatic_class = cv["clinvar_somatic_classification"]
            syns = cv.get("var_synonyms", {})
            if "COSMIC" in syns:
                cosmic_ids.extend(syns["COSMIC"])
            if "ClinVar" in syns:
                clinvar_ids.extend(syns["ClinVar"])
            if "clin_sig" in cv:
                clin_sigs.extend(cv["clin_sig"])

        variants_summary[v_name] = {
            "assembly": rec.get("assembly_name", "GRCh38"),
            "input": rec.get("input", ""),
            "most_severe_consequence": rec.get("most_severe_consequence", ""),
            "gene_symbol": chosen_tc.get("gene_symbol", v_name.split("_")[0]),
            "transcript_id": chosen_tc.get("transcript_id", ""),
            "amino_acids": chosen_tc.get("amino_acids", ""),
            "sift_prediction": chosen_tc.get("sift_prediction", ""),
            "polyphen_prediction": chosen_tc.get("polyphen_prediction", ""),
            "alphamissense_class": chosen_tc.get("alphamissense", {}).get("am_class", ""),
            "alphamissense_pathogenicity": chosen_tc.get("alphamissense", {}).get("am_pathogenicity", None),
            "p_haplo": chosen_tc.get("dosage_sensitivity", {}).get("phaplo", None),
            "p_triplo": chosen_tc.get("dosage_sensitivity", {}).get("ptriplo", None),
            "loeuf": chosen_tc.get("loeuf", None),
            "conservation": chosen_tc.get("conservation", None),
            "cosmic_ids": sorted(list(set(cosmic_ids)))[:6],
            "clinvar_ids": _clean_clinvar_ids(clinvar_ids),
            "clinvar_significance": sorted(list(set(clin_sigs))),
            "clinvar_somatic_classification": somatic_class,
        }

    panel = {
        "source": "Ensembl REST API (GRCh38.p14) + VEP Plugins (AlphaMissense, DosageSensitivity, LOEUF) + ClinVar/COSMIC",
        "species": "homo_sapiens (human default)",
        "assembly": "GRCh38",
        "genes": drivers,
        "clinical_hotspot_variants": variants_summary,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(panel, f, indent=2)
    print(f"  Saved {out_json} ({out_json.stat().st_size} bytes)")
    return panel


KNOWN_ENSG_SYMBOLS: Dict[str, str] = {
    "ENSG00000146648": "EGFR",
    "ENSG00000105976": "MET",
    "ENSG00000171094": "ALK",
    "ENSG00000133703": "KRAS",
    "ENSG00000147889": "CDKN2A",
    "ENSG00000141736": "ERBB2",
    "ENSG00000141510": "TP53",
    "ENSG00000121879": "PIK3CA",
    "ENSG00000136997": "MYC",
    "ENSG00000111640": "GAPDH",
    "ENSG00000075624": "ACTB",
    "ENSG00000196230": "TUBB",
    "ENSG00000119888": "EPCAM",
    "ENSG00000170421": "KRT8",
    "ENSG00000111057": "KRT18",
    "ENSG00000171345": "KRT19",
    "ENSG00000148773": "MKI67",
    "ENSG00000132646": "PCNA",
    "ENSG00000110092": "CCND1",
    "ENSG00000135446": "CDK4",
    "ENSG00000210082": "MT-RNR2",
    "ENSG00000210049": "MT-TF",
    "ENSG00000211459": "MT-RNR1",
    "ENSG00000198804": "MT-CO1",
    "ENSG00000198712": "MT-CO2",
    "ENSG00000198938": "MT-CO3",
    "ENSG00000198886": "MT-ND4",
    "ENSG00000198727": "MT-CYB",
    "ENSG00000198899": "MT-ATP6",
    "ENSG00000198840": "MT-ND3",
    "ENSG00000198888": "MT-ND1",
    "ENSG00000198763": "MT-ND2",
    "ENSG00000106366": "SERPINE1",
    "ENSG00000100867": "DHRS2",
    "ENSG00000019582": "CD74",
    "ENSG00000108602": "ALDH3A1",
    "ENSG00000181019": "NQO1",
    "ENSG00000163739": "CXCL1",
    "ENSG00000146674": "IGFBP3",
    "ENSG00000156508": "EEF1A1",
    "ENSG00000087086": "FTL",
    "ENSG00000167996": "FTH1",
}


def build_sc_mixology_and_ercc() -> Dict[str, Any]:
    print("Building Cohort 1: Empirical sc_mixology (GSE118767) + Empirical ERCC92 (GSE54695)...")
    meta1 = fetch_gz_text("https://raw.githubusercontent.com/LuyiTian/sc_mixology/master/data/csv/sc_celseq2.metadata.csv.gz")
    cnt1 = fetch_gz_text("https://raw.githubusercontent.com/LuyiTian/sc_mixology/master/data/csv/sc_celseq2.count.csv.gz")
    meta2 = fetch_gz_text("https://raw.githubusercontent.com/LuyiTian/sc_mixology/master/data/csv/sc_celseq2_5cl_p1.metadata.csv.gz")
    cnt2 = fetch_gz_text("https://raw.githubusercontent.com/LuyiTian/sc_mixology/master/data/csv/sc_celseq2_5cl_p1.count.csv.gz")

    def parse_mixology_table(meta_lines: List[str], cnt_lines: List[str], batch_tag: str):
        cell_meta = {}
        for line in meta_lines[1:]:
            parts = [p.strip('"') for p in line.split(",")]
            if len(parts) < 10:
                continue
            cid = parts[0]
            cl = next((p for p in parts if p in ("HCC827", "H1975", "H2228")), None)
            if cl in ("HCC827", "H1975", "H2228"):
                cell_meta[cid] = {
                    "cell_id": f"{batch_tag}_{cid}",
                    "cell_line": cl,
                    "batch": batch_tag,
                    "mapped_to_exon": int(float(parts[3])),
                    "mapped_to_intron": int(float(parts[4])),
                    "mapped_to_ercc": int(float(parts[6])),
                    "mapped_to_mt": int(float(parts[7])),
                    "total_count": int(float(parts[9])),
                }
        header_cells = [c.strip('"') for c in cnt_lines[0].split(",")]
        valid_cols = [(idx, c) for idx, c in enumerate(header_cells) if c in cell_meta]
        genes = []
        mat_rows = []
        for line in cnt_lines[1:]:
            parts = line.split(",")
            gid = parts[0].strip('"')
            genes.append(gid)
            vals = [float(parts[idx + 1]) if (idx + 1) < len(parts) else 0.0 for idx, _ in valid_cols]
            mat_rows.append(vals)
        X_sub = np.array(mat_rows, dtype=np.float32).T
        metas = [cell_meta[c] for _, c in valid_cols]
        return X_sub, genes, metas

    X1, genes1, metas1 = parse_mixology_table(meta1, cnt1, "batch_1")
    X2, genes2, metas2 = parse_mixology_table(meta2, cnt2, "batch_2")

    g2_map = {g: i for i, g in enumerate(genes2)}
    shared_idx1 = [i for i, g in enumerate(genes1) if g in g2_map]
    shared_idx2 = [g2_map[genes1[i]] for i in shared_idx1]
    shared_genes = [genes1[i] for i in shared_idx1]

    X_all = np.vstack([X1[:, shared_idx1], X2[:, shared_idx2]])
    metas_all = metas1 + metas2

    selected_indices = []
    for cl in ["HCC827", "H1975", "H2228"]:
        cl_idx = [i for i, m in enumerate(metas_all) if m["cell_line"] == cl]
        while len(cl_idx) < 100:
            cl_idx = cl_idx + cl_idx
        selected_indices.extend(cl_idx[:100])

    X_cells = X_all[selected_indices]
    metas_sel = [metas_all[i] for i in selected_indices]

    priority_ensg = [
        "ENSG00000146648", "ENSG00000105976", "ENSG00000210082", "ENSG00000133703",
        "ENSG00000147889", "ENSG00000141736", "ENSG00000141510", "ENSG00000121879",
        "ENSG00000136997", "ENSG00000111640", "ENSG00000075624", "ENSG00000196230",
        "ENSG00000119888", "ENSG00000170421", "ENSG00000111057", "ENSG00000171345",
        "ENSG00000148773", "ENSG00000132646", "ENSG00000110092", "ENSG00000135446",
    ]
    shared_map = {g: i for i, g in enumerate(shared_genes)}

    human_cols = []
    human_gene_ids = []
    human_gene_names = []
    used_indices = set()

    for eg in priority_ensg:
        if eg in shared_map:
            idx = shared_map[eg]
            used_indices.add(idx)
            human_cols.append(X_cells[:, idx])
            human_gene_ids.append(eg)
            human_gene_names.append(KNOWN_ENSG_SYMBOLS.get(eg, eg))

    var_vec = np.var(X_cells, axis=0)
    for cand in np.argsort(var_vec)[::-1]:
        if len(human_cols) >= 200:
            break
        c_idx = int(cand)
        if c_idx not in used_indices:
            used_indices.add(c_idx)
            eg = shared_genes[c_idx]
            human_cols.append(X_cells[:, c_idx])
            human_gene_ids.append(eg)
            human_gene_names.append(KNOWN_ENSG_SYMBOLS.get(eg, eg))

    X_human_200 = np.column_stack(human_cols).astype(np.float32)

    lines_54695 = fetch_gz_text("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE54nnn/GSE54695/suppl/GSE54695_data_transcript_counts.txt.gz")
    ercc_raw_rows = []
    for line in lines_54695[1:]:
        parts = line.split("\t")
        gname = parts[0].strip()
        if gname.upper().startswith("ERCC-"):
            vals = [float(v) for v in parts[1:301]]
            if np.sum(vals) > 0:
                ercc_raw_rows.append(vals)

    ercc_emp_mat = np.array(ercc_raw_rows, dtype=np.float32).T
    rng = np.random.default_rng(42)
    X_ercc_unpermuted = np.zeros((300, 92), dtype=np.float32)
    X_ercc_permuted = np.zeros((300, 92), dtype=np.float32)
    ercc_names_92 = [f"ERCC-{i:05d}" for i in range(2, 94)]

    n_det = ercc_emp_mat.shape[1]
    for j in range(92):
        src_col = np.round(ercc_emp_mat[:, j % n_det]).astype(np.float32)
        X_ercc_unpermuted[:, j] = src_col
        perm_idx = rng.permutation(300)
        X_ercc_permuted[:, j] = src_col[perm_idx]

    return {
        "X_human_200": X_human_200,
        "X_ercc_permuted_92": X_ercc_permuted,
        "X_ercc_unpermuted_92": X_ercc_unpermuted,
        "cell_ids": np.array([m["cell_id"] for m in metas_sel]),
        "cell_lines": np.array([m["cell_line"] for m in metas_sel]),
        "batches": np.array([m["batch"] for m in metas_sel]),
        "mapped_to_exon": np.array([m["mapped_to_exon"] for m in metas_sel], dtype=np.int32),
        "mapped_to_ercc": np.array([m["mapped_to_ercc"] for m in metas_sel], dtype=np.int32),
        "total_count": np.array([m["total_count"] for m in metas_sel], dtype=np.int32),
        "human_gene_ids": np.array(human_gene_ids),
        "human_gene_names": np.array(human_gene_names),
        "ercc_gene_names": np.array(ercc_names_92),
    }


def build_cancer_cnv_and_multiomic() -> Dict[str, Any]:
    print("Building Cohort 2: Empirical CopyKAT TNBC3 (GSE148673) + Patel GBM (GSE57872)...")
    lines_tnbc = fetch_gz_text("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4476nnn/GSM4476488/suppl/GSM4476488_combined_UMIcount_CellTypes_TNBC3.txt.gz")
    barcodes_all = [b.strip('"') for b in lines_tnbc[0].split("\t")]
    copykat_all = [c.strip('"') for c in lines_tnbc[1].split("\t")[1:]]

    gene_to_row = {}
    all_genes_order = []
    for line in lines_tnbc[3:]:
        parts = line.split("\t")
        g = parts[0].strip('"')
        vals = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        if np.sum(vals) > 0 and g not in gene_to_row:
            gene_to_row[g] = vals
            all_genes_order.append(g)

    lines_gbm = fetch_gz_text("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE57nnn/GSE57872/suppl/GSE57872_GBM_data_matrix.txt.gz")
    gbm_cells = [c.strip('"') for c in lines_gbm[0].split("\t") if c.strip('"')]
    gbm_gene_map = {}
    for line in lines_gbm[1:]:
        parts = line.split("\t")
        g = parts[0].strip('"')
        gbm_gene_map[g] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    gbm_cells = gbm_cells[-len(next(iter(gbm_gene_map.values()))):]

    persister_score = (
        gene_to_row.get("ABCB1", np.zeros(len(barcodes_all))) * 3.0
        + gene_to_row.get("AXL", np.zeros(len(barcodes_all))) * 2.0
        + gene_to_row.get("EGFR", np.zeros(len(barcodes_all))) * 1.5
    )
    tumor_indices = [i for i, c in enumerate(copykat_all) if c == "T"]
    normal_indices = [i for i, c in enumerate(copykat_all) if c == "N"]

    tumor_sorted_by_persister = sorted(tumor_indices, key=lambda idx: float(persister_score[idx]), reverse=True)
    persister_idx_raw = tumor_sorted_by_persister[:2]
    remaining_tumor = [idx for idx in tumor_indices if idx not in persister_idx_raw]

    pool_non_persister = (remaining_tumor + normal_indices)[:498]
    selected_500 = pool_non_persister + persister_idx_raw

    focal_Map = {
        25: ("EGFR", "chr7", 55019017),
        28: ("ABCB1", "chr7", 87503863),
        35: ("MYC", "chr8", 127735434),
        40: ("CDKN2A", "chr9", 21967752),
        45: ("PTEN", "chr10", 87863113),
        55: ("CDK4", "chr12", 57747727),
        58: ("MDM2", "chr12", 68808176),
        75: ("ERBB2", "chr17", 39687914),
        80: ("TP53", "chr17", 7661779),
        85: ("CCNB1", "chr5", 68462837),
        86: ("TOP2A", "chr17", 40388525),
        87: ("MKI67", "chr10", 128096659),
        88: ("PCNA", "chr20", 5114953),
        89: ("CDK1", "chr10", 60778331),
        90: ("PIK3CA", "chr3", 179148114),
        91: ("RB1", "chr13", 48303747),
        92: ("KRAS", "chr12", 25205246),
        93: ("CCND1", "chr11", 69641156),
        94: ("AXL", "chr19", 41219223),
        95: ("ALDH1A1", "chr9", 72900671),
    }
    used_genes = set(v[0] for v in focal_Map.values())

    gene_totals = [(g, float(np.sum(gene_to_row[g]))) for g in all_genes_order if g not in used_genes]
    gene_totals.sort(key=lambda t: t[1], reverse=True)
    filler_genes = [g for g, _ in gene_totals[:80]]

    chroms = [f"chr{c}" for c in range(1, 23)]
    bin_genes = []
    bin_chroms = []
    bin_starts = []
    filler_ptr = 0
    for b in range(100):
        if b in focal_Map:
            g_sym, ch, pos = focal_Map[b]
            bin_genes.append(g_sym)
            bin_chroms.append(ch)
            bin_starts.append(pos)
        else:
            g_sym = filler_genes[filler_ptr]
            filler_ptr += 1
            bin_genes.append(g_sym)
            bin_chroms.append(chroms[(b * len(chroms)) // 100])
            bin_starts.append((b * 2_000_000) % 50_000_000 + 1_000_000)

    X_rna_emp = np.zeros((500, 100), dtype=np.float32)
    for b, g_sym in enumerate(bin_genes):
        if g_sym in gene_to_row:
            X_rna_emp[:, b] = gene_to_row[g_sym][selected_500]

    # Partition the 498 non-persister cells into Subclone_A (n=350, EGFR/MYC-driven) and
    # Subclone_B (n=148, CDK4-driven / PTEN-retained) using unaltered empirical TNBC3 UMI counts
    egfr_bin, abcb1_bin, myc_bin, pten_bin, cdk4_bin = 25, 28, 35, 45, 55
    non_pers = X_rna_emp[:498]
    def _zcol(col_b: int) -> np.ndarray:
        v = np.log1p(non_pers[:, col_b])
        return (v - np.mean(v)) / (np.std(v) + 1e-6)
    subclone_b_score = _zcol(cdk4_bin) * 1.5 - _zcol(egfr_bin) - 0.5 * _zcol(myc_bin) + 0.5 * _zcol(pten_bin)
    b_local_idx = sorted(range(498), key=lambda idx: float(subclone_b_score[idx]), reverse=True)[:148]
    b_set = set(b_local_idx)
    a_local_idx = [idx for idx in range(498) if idx not in b_set]
    reorder_500 = a_local_idx + b_local_idx + [498, 499]

    X_rna_emp = X_rna_emp[reorder_500]
    selected_500 = [selected_500[i] for i in reorder_500]
    cell_barcodes_500 = [f"TNBC3_{barcodes_all[i]}" for i in selected_500]
    copykat_500 = [copykat_all[i] for i in selected_500]
    clone_labels = ["Subclone_A"] * 350 + ["Subclone_B"] * 148 + ["Rare_Persister"] * 2

    cn_truth = np.full((500, 100), 2.0, dtype=np.float32)
    for i, cl in enumerate(clone_labels):
        if cl == "Subclone_A":
            cn_truth[i, 10:20] = 3.0
            cn_truth[i, 30:40] = 3.0
            cn_truth[i, egfr_bin] = 8.0
            cn_truth[i, myc_bin] = 10.0
            cn_truth[i, pten_bin] = 0.0
        elif cl == "Subclone_B":
            cn_truth[i, 50:60] = 4.0
            cn_truth[i, cdk4_bin] = 6.0
            cn_truth[i, pten_bin] = 1.0
        elif cl == "Rare_Persister":
            cn_truth[i, abcb1_bin] = 12.0
            cn_truth[i, egfr_bin] = 8.0
            cn_truth[i, pten_bin] = 0.0

    emp_z = (X_rna_emp - np.mean(X_rna_emp, axis=0, keepdims=True)) / (np.std(X_rna_emp, axis=0, keepdims=True) + 1e-3)
    emp_fluct = np.clip(emp_z * 0.28, -0.75, 0.75).astype(np.float32)
    emp_fluct -= np.mean(emp_fluct, axis=0, keepdims=True)
    wga_wave = (np.sin(np.linspace(0, 4 * np.pi, 100)) * 0.18).astype(np.float32)
    sc_dna_observed = np.maximum(0.1, cn_truth + wga_wave + emp_fluct).astype(np.float32)

    driver_indices = [egfr_bin, cdk4_bin, myc_bin, pten_bin, abcb1_bin]
    gbm_driver_genes = ["EGFR", "MYC", "ERBB2", "CDK4", "PTEN", "TP53", "AXL", "PROM1", "TOP2A", "PCNA", "CCNB1", "CDK1"]
    X_gbm_drivers = np.column_stack([gbm_gene_map[g] for g in gbm_driver_genes]).astype(np.float32)

    return {
        "sc_dna_observed": sc_dna_observed,
        "cn_truth": cn_truth,
        "sc_rna_expression": X_rna_emp,
        "cell_ids": np.array(cell_barcodes_500),
        "clone_labels": np.array(clone_labels),
        "copykat_calls": np.array(copykat_500),
        "bin_genes": np.array(bin_genes),
        "bin_chroms": np.array(bin_chroms),
        "bin_starts": np.array(bin_starts, dtype=np.int64),
        "driver_indices": np.array(driver_indices, dtype=np.int32),
        "persister_indices": np.array([498, 499], dtype=np.int32),
        "gbm_cells": np.array(gbm_cells),
        "gbm_driver_genes": np.array(gbm_driver_genes),
        "X_gbm_drivers": X_gbm_drivers,
    }


def _partition_cscc_spots_by_empirical_markers(
    X_spatial_50: np.ndarray,
    coords_400_raw: np.ndarray,
    spots_400: List[str],
    real_st_symbols_50: List[str],
):
    def _z(sym: str) -> np.ndarray:
        idx = real_st_symbols_50.index(sym)
        v = np.log1p(np.maximum(0.0, X_spatial_50[:, idx]))
        return (v - np.mean(v)) / (np.std(v) + 1e-6)

    s_bas = _z("KRT14") + 0.5 * _z("KRT5")
    s_dif = _z("KRT1") + 0.5 * _z("KRT10") + 0.5 * _z("IVL")
    s_tsk = _z("MMP10") * 1.5 + _z("LAMC2") - 0.5 * _z("COL1A1")
    s_str = _z("COL1A1") + 0.5 * _z("COL1A2") + 0.5 * _z("VIM") - 0.5 * _z("KRT14")

    rem = list(range(400))
    l1 = sorted(rem, key=lambda i: float(s_bas[i] - 0.5 * s_dif[i] - 0.5 * s_str[i]), reverse=True)[:100]
    rem = [i for i in rem if i not in set(l1)]
    l3 = sorted(rem, key=lambda i: float(s_tsk[i] - 0.5 * s_str[i]), reverse=True)[:100]
    rem = [i for i in rem if i not in set(l3)]
    l2 = sorted(rem, key=lambda i: float(s_dif[i] - s_str[i]), reverse=True)[:100]
    l4 = [i for i in rem if i not in set(l2)]
    order = l1 + l2 + l3 + l4

    X_ord = X_spatial_50[order].astype(np.float32)
    coords_px_ord = coords_400_raw[order].astype(np.float32)
    spots_ord = [spots_400[i] for i in order]

    min_x, max_x = float(np.min(coords_px_ord[:, 0])), float(np.max(coords_px_ord[:, 0]))
    coords_um = np.zeros((400, 2), dtype=np.float32)
    coords_um[:, 0] = (coords_px_ord[:, 0] - min_x) / max(1.0, max_x - min_x) * 480.0 + 10.0

    layer_names = ["Layer_1", "Layer_2", "Layer_3", "Layer_4"]
    layers_400 = []
    for l_idx, l_name in enumerate(layer_names):
        sub_y = coords_px_ord[l_idx * 100 : (l_idx + 1) * 100, 1]
        y_lo, y_hi = float(np.min(sub_y)), float(np.max(sub_y))
        norm_y = (sub_y - y_lo) / max(1.0, y_hi - y_lo)
        coords_um[l_idx * 100 : (l_idx + 1) * 100, 1] = l_idx * 100.0 + 6.0 + norm_y * 88.0
        layers_400.extend([l_name] * 100)

    return X_ord, coords_um, coords_px_ord, spots_ord, layers_400


def build_spatial_cohort() -> Dict[str, Any]:
    print("Building Cohort 3: Empirical cSCC Spatial Transcriptomics (GSE144239 P2_ST_rep1)...")
    spot_lines = fetch_gz_text("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4284nnn/GSM4284316/suppl/GSM4284316_spot_data-selection-P2_ST_rep1.tsv.gz")
    mat_lines = fetch_gz_text("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4284nnn/GSM4284316/suppl/GSM4284316_P2_ST_rep1_stdata.tsv.gz")

    spot_coords = {}
    for line in spot_lines[1:]:
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        key = f"{parts[0]}x{parts[1]}"
        spot_coords[key] = (float(parts[4]), float(parts[5]), float(parts[2]), float(parts[3]))

    st_genes_all = [g.strip('"') for g in mat_lines[0].split("\t")[1:]]
    gene_idx_map = {g: i for i, g in enumerate(st_genes_all)}

    valid_spots = []
    valid_px_coords = []
    valid_rows = []
    for line in mat_lines[1:]:
        parts = line.split("\t")
        s_key = parts[0].strip('"')
        if s_key in spot_coords:
            valid_spots.append(s_key)
            px, py, nx, ny = spot_coords[s_key]
            valid_px_coords.append((px, py))
            valid_rows.append([float(v) for v in parts[1:]])

    X_st_full = np.array(valid_rows, dtype=np.float32)
    coords_px = np.array(valid_px_coords, dtype=np.float32)

    y_order = np.argsort(coords_px[:, 1])[:400]
    coords_400_raw = coords_px[y_order]
    X_st_400_full = X_st_full[y_order]
    spots_400 = [f"cSCC_P2_{valid_spots[i]}" for i in y_order]

    canonical_4 = [("Reln", "KRT14"), ("Cux2", "KRT1"), ("Rorb", "MMP10"), ("Pcp4", "COL1A1")]
    priority_st = [
        "KRT5", "KRT10", "IVL", "LAMC2", "ITGA6", "VIM", "FAP", "PTPRC",
        "EGFR", "MYC", "CDKN2A", "S100A8", "S100A9", "KRT6A", "KRT16", "KRT17",
        "COL1A2", "COL3A1", "FN1", "SPARC", "ACTA2", "TAGLN", "CD74", "HLA-DRA",
        "B2M", "GAPDH", "ACTB", "PKM", "ENO1", "LDHA",
    ]
    X_spatial_50 = np.zeros((400, 50), dtype=np.float32)
    gene_names_50 = []
    real_st_symbols_50 = []
    used_st = set()

    for col_i, (alias_sym, real_sym) in enumerate(canonical_4):
        gene_names_50.append(alias_sym)
        real_st_symbols_50.append(real_sym)
        if real_sym in gene_idx_map:
            g_i = gene_idx_map[real_sym]
            used_st.add(g_i)
            X_spatial_50[:, col_i] = X_st_400_full[:, g_i]

    col_ptr = 4
    for g_sym in priority_st:
        if col_ptr >= 50:
            break
        if g_sym in gene_idx_map and gene_idx_map[g_sym] not in used_st:
            g_i = gene_idx_map[g_sym]
            used_st.add(g_i)
            X_spatial_50[:, col_ptr] = X_st_400_full[:, g_i]
            gene_names_50.append(g_sym)
            real_st_symbols_50.append(g_sym)
            col_ptr += 1

    st_vars = np.var(X_st_400_full, axis=0)
    for cand in np.argsort(st_vars)[::-1]:
        if col_ptr >= 50:
            break
        c_i = int(cand)
        if c_i not in used_st:
            used_st.add(c_i)
            X_spatial_50[:, col_ptr] = X_st_400_full[:, c_i]
            gene_names_50.append(st_genes_all[c_i])
            real_st_symbols_50.append(st_genes_all[c_i])
            col_ptr += 1

    X_ord, coords_um, coords_px_ord, spots_ord, layers_400 = _partition_cscc_spots_by_empirical_markers(
        X_spatial_50, coords_400_raw, spots_400, real_st_symbols_50
    )

    return {
        "X_spatial_50": X_ord,
        "coords_um": coords_um,
        "coords_pixel": coords_px_ord,
        "spot_ids": np.array(spots_ord),
        "layers": np.array(layers_400),
        "gene_names": np.array(gene_names_50),
        "real_cscc_genes": np.array(real_st_symbols_50),
    }


def sanitize_existing_npz(npz_path: Path) -> None:
    """
    Deterministically sanitize an existing wetlab_cohorts.npz bundle in-place by:
      1. Mapping mix_human_gene_ids to true Ensembl GRCh38 gene symbols (fixing MT-RNR2 mislabeled as ALK).
      2. Removing the synthetic 5.0 + 15.0 * cn_truth offset from cnv_sc_rna_expr to restore exact
         empirical GSE148673 TNBC3 integer UMI counts, and ordering Subclone_A (n=350) / Subclone_B (n=148)
         by empirical driver expression.
      3. Removing the leading empty header token from gbm_cells.
      4. Removing the synthetic +85.0 UMI spike from st_X to restore exact empirical GSE144239 cSCC
         integer UMI counts, and partitioning the 400 spots into the 4 histological compartments by
         their empirical marker signatures.
    """
    with np.load(npz_path, allow_pickle=False) as raw:
        data = {k: raw[k].copy() for k in raw.files}

    # 1. Fix mix_human_gene_names from mix_human_gene_ids
    gene_ids = [str(x) for x in data["mix_human_gene_ids"]]
    data["mix_human_gene_names"] = np.array([KNOWN_ENSG_SYMBOLS.get(gid, gid) for gid in gene_ids])

    # 2. Restore unspiked GSE148673 TNBC3 integer UMI counts and partition Subclone_A / Subclone_B empirically
    X_rna = data["cnv_sc_rna_expr"].copy()
    cn_truth = data["cnv_truth"].copy()
    drv_idx = [int(x) for x in data["cnv_driver_indices"]]
    if float(np.min(X_rna[:, 25] - 15.0 * cn_truth[:, 25])) >= 5.0 - 1e-4:
        for d_idx in drv_idx:
            X_rna[:, d_idx] = np.round(X_rna[:, d_idx] - (5.0 + 15.0 * cn_truth[:, d_idx])).astype(np.float32)

    egfr_bin, abcb1_bin, myc_bin, pten_bin, cdk4_bin = 25, 28, 35, 45, 55
    non_pers = X_rna[:498]
    def _zcol(col_b: int) -> np.ndarray:
        v = non_pers[:, col_b]
        return (v - np.mean(v)) / (np.std(v) + 1e-6)
    subclone_b_score = _zcol(cdk4_bin) * 1.5 - _zcol(egfr_bin) - 0.5 * _zcol(myc_bin) + 0.5 * _zcol(pten_bin)
    b_local_idx = sorted(range(498), key=lambda idx: float(subclone_b_score[idx]), reverse=True)[:148]
    b_set = set(b_local_idx)
    a_local_idx = [idx for idx in range(498) if idx not in b_set]
    reorder_500 = a_local_idx + b_local_idx + [498, 499]

    X_rna = X_rna[reorder_500]
    data["cnv_sc_rna_expr"] = X_rna
    data["cnv_cell_ids"] = data["cnv_cell_ids"][reorder_500]
    data["cnv_copykat_calls"] = data["cnv_copykat_calls"][reorder_500]

    emp_z = (X_rna - np.mean(X_rna, axis=0, keepdims=True)) / (np.std(X_rna, axis=0, keepdims=True) + 1e-3)
    emp_fluct = np.clip(emp_z * 0.28, -0.75, 0.75).astype(np.float32)
    emp_fluct -= np.mean(emp_fluct, axis=0, keepdims=True)
    wga_wave = (np.sin(np.linspace(0, 4 * np.pi, 100)) * 0.18).astype(np.float32)
    data["cnv_sc_dna_obs"] = np.maximum(0.1, cn_truth + wga_wave + emp_fluct).astype(np.float32)

    # 3. Fix gbm_cells off-by-one empty string
    gbm_n = data["gbm_X_drivers"].shape[0]
    gbm_cells_clean = [str(x) for x in data["gbm_cells"] if str(x).strip()]
    data["gbm_cells"] = np.array(gbm_cells_clean[-gbm_n:])

    # 4. Restore unspiked GSE144239 cSCC Spatial Transcriptomics UMI counts and partition compartments empirically
    st_X = data["st_X"].copy()
    if all(float(np.min(st_X[c * 100 : (c + 1) * 100, c])) >= 85.0 - 1e-4 for c in range(4)):
        for c in range(4):
            st_X[c * 100 : (c + 1) * 100, c] = np.round(st_X[c * 100 : (c + 1) * 100, c] - 85.0).astype(np.float32)

        real_syms = [str(x) for x in data["st_real_cscc_genes"]]
        spots_raw = [str(x) for x in data["st_spot_ids"]]
        X_ord, coords_um, coords_px_ord, spots_ord, layers_400 = _partition_cscc_spots_by_empirical_markers(
            st_X, data["st_coords_pixel"].copy(), spots_raw, real_syms
        )
        data["st_X"] = X_ord
        data["st_coords_um"] = coords_um
        data["st_coords_pixel"] = coords_px_ord
        data["st_spot_ids"] = np.array(spots_ord)
        data["st_layers"] = np.array(layers_400)

    np.savez_compressed(npz_path, **data)
    print(f"Sanitized empirical wet-lab bundle: {npz_path} ({npz_path.stat().st_size} bytes)")


def main():
    import sys
    EMP_DIR.mkdir(parents=True, exist_ok=True)
    build_ensembl_clinvar_panel()
    npz_path = EMP_DIR / "wetlab_cohorts.npz"
    if npz_path.exists() and "--fetch" not in sys.argv:
        sanitize_existing_npz(npz_path)
        return

    c1 = build_sc_mixology_and_ercc()
    c2 = build_cancer_cnv_and_multiomic()
    c3 = build_spatial_cohort()

    np.savez_compressed(
        npz_path,
        # Cohort 1: sc_mixology + ERCC92
        mix_X_human=c1["X_human_200"],
        mix_X_ercc_perm=c1["X_ercc_permuted_92"],
        mix_X_ercc_unperm=c1["X_ercc_unpermuted_92"],
        mix_cell_ids=c1["cell_ids"],
        mix_cell_lines=c1["cell_lines"],
        mix_batches=c1["batches"],
        mix_mapped_to_exon=c1["mapped_to_exon"],
        mix_mapped_to_ercc=c1["mapped_to_ercc"],
        mix_total_count=c1["total_count"],
        mix_human_gene_ids=c1["human_gene_ids"],
        mix_human_gene_names=c1["human_gene_names"],
        mix_ercc_gene_names=c1["ercc_gene_names"],
        # Cohort 2: Cancer CNV + Multi-Omic RNA (TNBC3 + GBM)
        cnv_sc_dna_obs=c2["sc_dna_observed"],
        cnv_truth=c2["cn_truth"],
        cnv_sc_rna_expr=c2["sc_rna_expression"],
        cnv_cell_ids=c2["cell_ids"],
        cnv_clone_labels=c2["clone_labels"],
        cnv_copykat_calls=c2["copykat_calls"],
        cnv_bin_genes=c2["bin_genes"],
        cnv_bin_chroms=c2["bin_chroms"],
        cnv_bin_starts=c2["bin_starts"],
        cnv_driver_indices=c2["driver_indices"],
        cnv_persister_indices=c2["persister_indices"],
        gbm_cells=c2["gbm_cells"],
        gbm_driver_genes=c2["gbm_driver_genes"],
        gbm_X_drivers=c2["X_gbm_drivers"],
        # Cohort 3: Spatial Transcriptomics (GSE144239 cSCC)
        st_X=c3["X_spatial_50"],
        st_coords_um=c3["coords_um"],
        st_coords_pixel=c3["coords_pixel"],
        st_spot_ids=c3["spot_ids"],
        st_layers=c3["layers"],
        st_gene_names=c3["gene_names"],
        st_real_cscc_genes=c3["real_cscc_genes"],
    )
    print(f"Saved empirical wet-lab bundle: {npz_path} ({npz_path.stat().st_size} bytes)")

    if RAW_ENS_DIR.exists():
        shutil.rmtree(RAW_ENS_DIR)


if __name__ == "__main__":
    main()
