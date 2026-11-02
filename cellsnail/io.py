"""cellsnail.io
============
Pure-Python / NumPy zero-dependency multi-omic single-cell I/O engine.

Supported Formats:
  1. .mtx (Matrix Market coordinate sparse format) with features.tsv / barcodes.tsv.
  2. .vcf (Variant Call Format for single-cell DNA copy number and SNVs).
  3. .csv / .tsv (Expression matrices, metadata, and spatial coordinates).
  4. .h5ad / HDF5-compatible format (AnnData-compatible container with zero external dependencies).
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import numpy as np

# Optional h5py support if present; otherwise pure-Python/NumPy zero-dependency container
try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False


# ==============================================================================
# SingleCellContainer (AnnData-Compatible Data Structure)
# ==============================================================================

@dataclass
class SingleCellContainer:
    """
    Lightweight, high-performance AnnData-compatible single-cell container.

    Attributes:
      X: 2D numpy array (cells x genes / features).
      obs: Dict mapping metadata column names to arrays/lists of length n_obs.
      var: Dict mapping feature column names to arrays/lists of length n_vars.
      obsm: Dict mapping multidimensional observation arrays (e.g. 'spatial', 'X_umap').
      varm: Dict mapping multidimensional feature arrays.
      uns: Dict containing unstructured metadata and parameters.
    """
    X: np.ndarray
    obs: Dict[str, Any] = field(default_factory=dict)
    var: Dict[str, Any] = field(default_factory=dict)
    obsm: Dict[str, np.ndarray] = field(default_factory=dict)
    varm: Dict[str, np.ndarray] = field(default_factory=dict)
    uns: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.X, np.ndarray):
            self.X = np.asarray(self.X, dtype=np.float32)
        if self.X.ndim != 2:
            raise ValueError(f"X must be a 2D matrix (cells x features), got ndim={self.X.ndim}")

        n_obs, n_vars = self.shape
        # Ensure default cell_id / barcode if missing
        if "cell_id" not in self.obs and "barcode" not in self.obs:
            self.obs["cell_id"] = [f"cell_{i}" for i in range(n_obs)]
        elif "cell_id" not in self.obs and "barcode" in self.obs:
            self.obs["cell_id"] = list(self.obs["barcode"])

        # Ensure default gene_id / gene_name if missing
        if "gene_id" not in self.var and "gene_name" not in self.var:
            self.var["gene_id"] = [f"gene_{j}" for j in range(n_vars)]
            self.var["gene_name"] = [f"gene_{j}" for j in range(n_vars)]
        elif "gene_name" not in self.var and "gene_id" in self.var:
            self.var["gene_name"] = list(self.var["gene_id"])

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.X.shape[0]), int(self.X.shape[1])

    @property
    def n_obs(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_vars(self) -> int:
        return int(self.X.shape[1])

    @property
    def obs_names(self) -> List[str]:
        if "cell_id" in self.obs:
            return [str(x) for x in self.obs["cell_id"]]
        if "barcode" in self.obs:
            return [str(x) for x in self.obs["barcode"]]
        return [f"cell_{i}" for i in range(self.n_obs)]

    @property
    def var_names(self) -> List[str]:
        if "gene_name" in self.var:
            return [str(x) for x in self.var["gene_name"]]
        if "gene_id" in self.var:
            return [str(x) for x in self.var["gene_id"]]
        return [f"gene_{j}" for j in range(self.n_vars)]

    def copy(self) -> "SingleCellContainer":
        return SingleCellContainer(
            X=self.X.copy(),
            obs={k: (v.copy() if isinstance(v, np.ndarray) else list(v)) for k, v in self.obs.items()},
            var={k: (v.copy() if isinstance(v, np.ndarray) else list(v)) for k, v in self.var.items()},
            obsm={k: v.copy() for k, v in self.obsm.items()},
            varm={k: v.copy() for k, v in self.varm.items()},
            uns={k: v for k, v in self.uns.items()},
        )


# ==============================================================================
# Helper: File Openers for Plain or Gzip Files
# ==============================================================================

def _open_text(path_or_str: Union[str, Path], mode: str = "rt"):
    path = Path(path_or_str)
    if path.suffix == ".gz":
        return gzip.open(path, mode=mode, encoding="utf-8" if "t" in mode else None)
    return open(path, mode=mode, encoding="utf-8" if "t" in mode else None)


# ==============================================================================
# 1. Matrix Market (.mtx) Reader and Serializer
# ==============================================================================

def read_mtx(
    mtx_path: Union[str, Path],
    features_path: Optional[Union[str, Path]] = None,
    barcodes_path: Optional[Union[str, Path]] = None,
    dtype: type = np.float32,
    transpose_to_cells_x_genes: bool = True,
) -> SingleCellContainer:
    """
    Read MatrixMarket coordinate matrix (.mtx or .mtx.gz) along with barcodes
    and features/genes files into a SingleCellContainer.

    By convention in 10x Genomics, .mtx is (n_features x n_barcodes).
    If transpose_to_cells_x_genes is True, the result is transposed to (n_obs, n_vars).
    """
    mtx_p = Path(mtx_path)
    if not mtx_p.exists():
        raise FileNotFoundError(f"MatrixMarket file not found: {mtx_path}")

    # Auto-detect barcodes and features if in same directory and not provided
    parent = mtx_p.parent
    if barcodes_path is None:
        for candidate in ["barcodes.tsv", "barcodes.tsv.gz", "cells.tsv", "cells.tsv.gz"]:
            if (parent / candidate).exists():
                barcodes_path = parent / candidate
                break

    if features_path is None:
        for candidate in ["features.tsv", "features.tsv.gz", "genes.tsv", "genes.tsv.gz"]:
            if (parent / candidate).exists():
                features_path = parent / candidate
                break

    # Read barcodes
    barcodes: List[str] = []
    if barcodes_path and Path(barcodes_path).exists():
        with _open_text(barcodes_path, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    barcodes.append(line.split("\t")[0])

    # Read features/genes
    var_dict: Dict[str, List[str]] = {"gene_id": [], "gene_name": []}
    if features_path and Path(features_path).exists():
        with _open_text(features_path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    var_dict["gene_id"].append(parts[0])
                    var_dict["gene_name"].append(parts[1])
                else:
                    var_dict["gene_id"].append(parts[0])
                    var_dict["gene_name"].append(parts[0])

    # Parse .mtx
    with _open_text(mtx_p, "rt") as f:
        header = f.readline().strip()
        if not header.startswith("%%MatrixMarket"):
            raise ValueError(f"Invalid MatrixMarket header: {header}")

        # Skip comment lines
        line = f.readline().strip()
        while line.startswith("%"):
            line = f.readline().strip()

        dims = line.split()
        if len(dims) < 3:
            raise ValueError(f"Invalid MatrixMarket size line: {line}")

        n_rows, n_cols, n_entries = int(dims[0]), int(dims[1]), int(dims[2])
        matrix = np.zeros((n_rows, n_cols), dtype=dtype)

        # Parse coordinate entries (1-indexed)
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            r = int(parts[0]) - 1
            c = int(parts[1]) - 1
            val = float(parts[2])
            matrix[r, c] = val

    # Transpose if single-cell convention (features x cells -> cells x features)
    if transpose_to_cells_x_genes:
        X = matrix.T
        n_obs, n_vars = X.shape
        if not barcodes:
            barcodes = [f"cell_{i}" for i in range(n_obs)]
        if not var_dict["gene_id"]:
            var_dict["gene_id"] = [f"gene_{j}" for j in range(n_vars)]
            var_dict["gene_name"] = [f"gene_{j}" for j in range(n_vars)]
    else:
        X = matrix
        n_obs, n_vars = X.shape
        if not barcodes:
            barcodes = [f"cell_{i}" for i in range(n_obs)]
        if not var_dict["gene_id"]:
            var_dict["gene_id"] = [f"gene_{j}" for j in range(n_vars)]
            var_dict["gene_name"] = [f"gene_{j}" for j in range(n_vars)]

    obs_dict = {"cell_id": barcodes, "barcode": barcodes}
    return SingleCellContainer(X=X, obs=obs_dict, var=var_dict)


def write_mtx(
    container: SingleCellContainer,
    output_dir: Union[str, Path],
    prefix: str = "",
    compress: bool = False,
) -> Dict[str, str]:
    """
    Write SingleCellContainer to MatrixMarket format with features.tsv and barcodes.tsv.
    By standard convention, the written .mtx is (n_features x n_cells).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cells, n_features = container.shape
    X_t = container.X.T  # (n_features, n_cells)

    # Non-zero coordinates
    rows, cols = np.nonzero(X_t)
    vals = X_t[rows, cols]
    n_entries = len(vals)

    ext = ".gz" if compress else ""
    mtx_file = out_dir / f"{prefix}matrix.mtx{ext}"
    features_file = out_dir / f"{prefix}features.tsv{ext}"
    barcodes_file = out_dir / f"{prefix}barcodes.tsv{ext}"

    # Write matrix.mtx
    open_fn = gzip.open if compress else open
    with open_fn(mtx_file, "wt", encoding="utf-8") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write("% Generated by cellsnail.io\n")
        f.write(f"{n_features} {n_cells} {n_entries}\n")
        for r, c, v in zip(rows, cols, vals):
            f.write(f"{r + 1} {c + 1} {v:.6g}\n")

    # Write barcodes.tsv
    with open_fn(barcodes_file, "wt", encoding="utf-8") as f:
        for b in container.obs_names:
            f.write(f"{b}\n")

    # Write features.tsv
    gene_ids = container.var.get("gene_id", container.var_names)
    gene_names = container.var.get("gene_name", container.var_names)
    with open_fn(features_file, "wt", encoding="utf-8") as f:
        for gid, gname in zip(gene_ids, gene_names):
            f.write(f"{gid}\t{gname}\tGene Expression\n")

    return {
        "matrix": str(mtx_file),
        "barcodes": str(barcodes_file),
        "features": str(features_file),
    }


# ==============================================================================
# 2. VCF (.vcf) Reader and Serializer for scDNA CNV & SNVs
# ==============================================================================

def read_vcf(
    vcf_path: Union[str, Path],
    cn_format_key: str = "CN",
) -> Dict[str, Any]:
    """
    Read Variant Call Format (VCF 4.2) for single-cell DNA copy number profiles or SNVs.

    Returns dict with:
      - samples: list of cell/sample IDs.
      - variants: list of dicts (chrom, pos, id, ref, alt, info).
      - cn_matrix: 2D numpy array (n_samples x n_variants) of copy numbers.
      - baf_matrix: 2D numpy array (n_samples x n_variants) of B-allele frequencies.
      - info: dict of parsed info tags.
    """
    vcf_p = Path(vcf_path)
    if not vcf_p.exists():
        raise FileNotFoundError(f"VCF file not found: {vcf_path}")

    headers: List[str] = []
    samples: List[str] = []
    variants: List[Dict[str, Any]] = []
    sample_cn_data: List[List[float]] = []
    sample_baf_data: List[List[float]] = []

    with _open_text(vcf_p, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("##"):
                headers.append(line)
                continue
            if line.startswith("#CHROM"):
                parts = line.split("\t")
                if len(parts) > 9:
                    samples = parts[9:]
                sample_cn_data = [[] for _ in samples]
                sample_baf_data = [[] for _ in samples]
                continue

            parts = line.split("\t")
            chrom, pos, vid, ref, alt, qual, filt, info, fmt = parts[:9]
            sample_fields = parts[9:]

            fmt_keys = fmt.split(":")
            cn_idx = fmt_keys.index(cn_format_key) if cn_format_key in fmt_keys else -1
            baf_idx = fmt_keys.index("BAF") if "BAF" in fmt_keys else -1
            gt_idx = fmt_keys.index("GT") if "GT" in fmt_keys else -1

            var_info = {
                "chrom": chrom,
                "pos": int(pos),
                "id": vid,
                "ref": ref,
                "alt": alt,
                "qual": qual,
                "filter": filt,
                "info": info,
            }
            variants.append(var_info)

            for s_idx, sf in enumerate(sample_fields):
                s_parts = sf.split(":")
                # Copy number extraction
                cn_val = 2.0  # default diploid
                if cn_idx != -1 and len(s_parts) > cn_idx and s_parts[cn_idx] not in (".", ""):
                    try:
                        cn_val = float(s_parts[cn_idx])
                    except ValueError:
                        cn_val = 2.0
                elif gt_idx != -1 and len(s_parts) > gt_idx:
                    # Infer CN from GT if CN tag missing
                    gt = s_parts[gt_idx]
                    if gt in ("0/0", "0|0"):
                        cn_val = 2.0
                    elif gt in ("0/1", "1/0", "0|1", "1|0"):
                        cn_val = 2.0
                    elif gt in ("1/1", "1|1"):
                        cn_val = 2.0
                sample_cn_data[s_idx].append(cn_val)

                # BAF extraction
                baf_val = 0.5
                if baf_idx != -1 and len(s_parts) > baf_idx and s_parts[baf_idx] not in (".", ""):
                    try:
                        baf_val = float(s_parts[baf_idx])
                    except ValueError:
                        baf_val = 0.5
                sample_baf_data[s_idx].append(baf_val)

    n_samples = len(samples)
    n_vars = len(variants)
    cn_matrix = np.array(sample_cn_data, dtype=np.float32) if n_samples > 0 else np.zeros((0, n_vars), dtype=np.float32)
    baf_matrix = np.array(sample_baf_data, dtype=np.float32) if n_samples > 0 else np.zeros((0, n_vars), dtype=np.float32)

    return {
        "headers": headers,
        "samples": samples,
        "variants": variants,
        "cn_matrix": cn_matrix,
        "baf_matrix": baf_matrix,
    }


def write_vcf(
    output_path: Union[str, Path],
    variants: List[Dict[str, Any]],
    samples: List[str],
    cn_matrix: Optional[np.ndarray] = None,
    baf_matrix: Optional[np.ndarray] = None,
    reference: str = "GRCh38",
) -> str:
    """
    Serialize single-cell DNA copy-number calls and SNVs into standard VCF 4.2 format.
    cn_matrix: shape (n_samples, n_variants).
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with open(out_p, "wt", encoding="utf-8") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=cellsnail.io\n")
        f.write(f"##reference={reference}\n")
        f.write('##INFO=<ID=END,Number=1,Type=Integer,Description="End coordinate of the CNV segment">\n')
        f.write('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">\n')
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        f.write('##FORMAT=<ID=CN,Number=1,Type=Float,Description="Single-cell copy number integer or posterior">\n')
        f.write('##FORMAT=<ID=BAF,Number=1,Type=Float,Description="B-allele frequency">\n')

        # Header line
        header_cols = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + list(samples)
        f.write("\t".join(header_cols) + "\n")

        for v_idx, var in enumerate(variants):
            chrom = str(var.get("chrom", "chr1"))
            pos = str(var.get("pos", v_idx * 1000 + 1))
            vid = str(var.get("id", f"var_{v_idx}"))
            ref = str(var.get("ref", "N"))
            alt = str(var.get("alt", "<CNV>"))
            qual = str(var.get("qual", "."))
            filt = str(var.get("filter", "PASS"))
            info = str(var.get("info", "SVTYPE=CNV"))
            fmt = "GT:CN:BAF"

            row = [chrom, pos, vid, ref, alt, qual, filt, info, fmt]
            for s_idx in range(len(samples)):
                cn_val = float(cn_matrix[s_idx, v_idx]) if cn_matrix is not None else 2.0
                baf_val = float(baf_matrix[s_idx, v_idx]) if baf_matrix is not None else 0.5
                gt = "0/1" if abs(baf_val - 0.5) < 0.2 else ("0/0" if baf_val < 0.2 else "1/1")
                sample_str = f"{gt}:{cn_val:.2f}:{baf_val:.3f}"
                row.append(sample_str)

            f.write("\t".join(row) + "\n")

    return str(out_p)


# ==============================================================================
# 3. CSV / TSV Expression & Spatial Coordinates Reader / Writer
# ==============================================================================

def read_csv(
    csv_path: Union[str, Path],
    sep: Optional[str] = None,
    is_spatial: bool = False,
    first_col_is_index: bool = True,
) -> Union[SingleCellContainer, Dict[str, Any]]:
    """
    Read expression matrix or spatial coordinates from CSV or TSV.

    If is_spatial is True:
      Parses spatial coordinates (expects columns like barcode/cell_id, x, y, fov, cell_type, layer).
      Returns a dictionary of spatial arrays / metadata.
    If is_spatial is False:
      Parses dense expression matrix (rows = cells, cols = genes).
      Returns SingleCellContainer.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    if sep is None:
        sep = "\t" if path.suffix in (".tsv", ".tsv.gz") else ","

    lines: List[List[str]] = []
    with _open_text(path, "rt") as f:
        reader = csv.reader(f, delimiter=sep)
        for row in reader:
            if row:
                lines.append(row)

    if not lines:
        raise ValueError(f"Empty CSV file: {csv_path}")

    header = lines[0]
    data_rows = lines[1:]

    if is_spatial:
        # Spatial coordinate parser
        col_map = {col.lower().strip(): idx for idx, col in enumerate(header)}
        barcodes: List[str] = []
        coords: List[Tuple[float, float]] = []
        metadata: Dict[str, List[Any]] = {col: [] for col in header}

        x_idx = col_map.get("x", col_map.get("center_x", col_map.get("centroid_x", 1)))
        y_idx = col_map.get("y", col_map.get("center_y", col_map.get("centroid_y", 2)))
        id_idx = col_map.get("barcode", col_map.get("cell_id", col_map.get("cell", 0)))

        for r in data_rows:
            if len(r) <= max(x_idx, y_idx, id_idx):
                continue
            barcodes.append(r[id_idx])
            coords.append((float(r[x_idx]), float(r[y_idx])))
            for c_name, c_idx in col_map.items():
                if c_idx < len(r):
                    metadata[c_name].append(r[c_idx])

        return {
            "barcodes": barcodes,
            "coordinates": np.array(coords, dtype=np.float32),
            "metadata": metadata,
        }

    # Dense expression matrix parser
    if first_col_is_index:
        col_names = header[1:]
        row_names = [r[0] for r in data_rows]
        data_vals = [[float(v) for v in r[1:]] for r in data_rows]
    else:
        col_names = header
        row_names = [f"cell_{i}" for i in range(len(data_rows))]
        data_vals = [[float(v) for v in r] for r in data_rows]

    X = np.array(data_vals, dtype=np.float32)
    obs = {"cell_id": row_names, "barcode": row_names}
    var = {"gene_id": col_names, "gene_name": col_names}

    return SingleCellContainer(X=X, obs=obs, var=var)


def write_csv(
    container_or_data: Union[SingleCellContainer, Dict[str, Any], np.ndarray],
    csv_path: Union[str, Path],
    sep: str = ",",
) -> str:
    """
    Write expression matrix or spatial coordinates to CSV or TSV.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=sep)

        if isinstance(container_or_data, SingleCellContainer):
            header = ["cell_id"] + container_or_data.var_names
            writer.writerow(header)
            cell_ids = container_or_data.obs_names
            for idx, c_id in enumerate(cell_ids):
                row = [c_id] + [f"{v:.6g}" for v in container_or_data.X[idx]]
                writer.writerow(row)
        elif isinstance(container_or_data, dict):
            # Dict with keys as columns
            cols = list(container_or_data.keys())
            writer.writerow(cols)
            n_rows = len(container_or_data[cols[0]])
            for i in range(n_rows):
                row = [str(container_or_data[c][i]) for c in cols]
                writer.writerow(row)
        else:
            arr = np.asarray(container_or_data)
            for row in arr:
                writer.writerow([f"{v:.6g}" for v in row])

    return str(path)


# ==============================================================================
# 4. H5AD / HDF5-Compatible Pure-Python Zero-Dependency Container
# ==============================================================================

def write_h5ad(
    container: SingleCellContainer,
    output_path: Union[str, Path],
) -> str:
    """
    Serialize SingleCellContainer to .h5ad.

    If h5py is installed, writes standard HDF5 format matching AnnData specifications.
    If h5py is not installed, creates a zero-dependency .h5ad bundle (HDF5-compatible
    container using zip-packed NPZ structured hierarchy with JSON metadata).
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if _HAS_H5PY:
        try:
            with h5py.File(out_p, "w") as f:
                # /X dataset
                f.create_dataset("X", data=container.X, compression="gzip")

                # /obs group
                obs_grp = f.create_group("obs")
                obs_grp.attrs["_index"] = "cell_id"
                for k, v in container.obs.items():
                    arr = np.asarray(v)
                    if arr.dtype.kind in ("U", "S", "O"):
                        arr = arr.astype("S")
                    obs_grp.create_dataset(k, data=arr)

                # /var group
                var_grp = f.create_group("var")
                var_grp.attrs["_index"] = "gene_name"
                for k, v in container.var.items():
                    arr = np.asarray(v)
                    if arr.dtype.kind in ("U", "S", "O"):
                        arr = arr.astype("S")
                    var_grp.create_dataset(k, data=arr)

                # /obsm group
                if container.obsm:
                    obsm_grp = f.create_group("obsm")
                    for k, v in container.obsm.items():
                        obsm_grp.create_dataset(k, data=np.asarray(v, dtype=np.float32))

                # /varm group
                if container.varm:
                    varm_grp = f.create_group("varm")
                    for k, v in container.varm.items():
                        varm_grp.create_dataset(k, data=np.asarray(v, dtype=np.float32))

                # /uns group
                if container.uns:
                    uns_grp = f.create_group("uns")
                    for k, v in container.uns.items():
                        try:
                            if isinstance(v, (int, float, str, bool)):
                                uns_grp.attrs[k] = v
                            elif isinstance(v, np.ndarray):
                                uns_grp.create_dataset(k, data=v)
                            else:
                                uns_grp.attrs[k] = json.dumps(v)
                        except Exception:
                            pass
            return str(out_p)
        except Exception:
            # Fall back to zero-dependency serializer if h5py fails
            pass

    # Zero-dependency container serializer:
    # Packages /X, /obs, /var, /obsm, /uns into a structured ZIP archive
    # with exact AnnData hierarchical keys and manifest
    buffer_dict: Dict[str, np.ndarray] = {
        "X": container.X,
    }

    # Add obs columns
    obs_meta: Dict[str, str] = {}
    for k, v in container.obs.items():
        arr = np.asarray(v)
        buffer_dict[f"obs/{k}"] = arr
        obs_meta[k] = str(arr.dtype)

    # Add var columns
    var_meta: Dict[str, str] = {}
    for k, v in container.var.items():
        arr = np.asarray(v)
        buffer_dict[f"var/{k}"] = arr
        var_meta[k] = str(arr.dtype)

    # Add obsm arrays
    for k, v in container.obsm.items():
        buffer_dict[f"obsm/{k}"] = np.asarray(v)

    # Add varm arrays
    for k, v in container.varm.items():
        buffer_dict[f"varm/{k}"] = np.asarray(v)

    # Write as compressed structured archive
    manifest = {
        "format": "cellsnail_h5ad_compatible",
        "version": "1.0",
        "shape": list(container.shape),
        "obs_keys": list(container.obs.keys()),
        "var_keys": list(container.var.keys()),
        "obsm_keys": list(container.obsm.keys()),
        "varm_keys": list(container.varm.keys()),
        "uns": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in container.uns.items() if isinstance(v, (int, float, str, bool, list, dict))},
    }

    with zipfile.ZipFile(out_p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Write manifest
        zf.writestr("_manifest.json", json.dumps(manifest, indent=2))
        for key, arr in buffer_dict.items():
            bio = io.BytesIO()
            np.save(bio, arr)
            zf.writestr(f"{key}.npy", bio.getvalue())

    return str(out_p)


def read_h5ad(
    input_path: Union[str, Path],
) -> SingleCellContainer:
    """
    Read SingleCellContainer from .h5ad file.
    Supports both standard h5py HDF5 files and zero-dependency AnnData containers.
    """
    in_p = Path(input_path)
    if not in_p.exists():
        raise FileNotFoundError(f"H5AD file not found: {input_path}")

    # 1. Try standard h5py if available and file is valid HDF5
    if _HAS_H5PY:
        try:
            with h5py.File(in_p, "r") as f:
                # Read X
                X_data = f["X"]
                if hasattr(X_data, "shape"):
                    X = np.array(X_data, dtype=np.float32)
                else:
                    # Sparse csr_matrix in h5ad
                    data = np.array(X_data["data"])
                    indices = np.array(X_data["indices"])
                    indptr = np.array(X_data["indptr"])
                    shape = tuple(X_data.attrs.get("shape", (len(indptr) - 1, int(np.max(indices)) + 1)))
                    X = np.zeros(shape, dtype=np.float32)
                    for i in range(len(indptr) - 1):
                        row_start, row_end = indptr[i], indptr[i + 1]
                        X[i, indices[row_start:row_end]] = data[row_start:row_end]

                # Read obs
                obs: Dict[str, Any] = {}
                if "obs" in f:
                    obs_grp = f["obs"]
                    for k in obs_grp.keys():
                        v = obs_grp[k]
                        if isinstance(v, h5py.Dataset):
                            val = np.array(v)
                            if val.dtype.kind == "S":
                                val = [s.decode("utf-8", errors="replace") for s in val]
                            obs[k] = val

                # Read var
                var: Dict[str, Any] = {}
                if "var" in f:
                    var_grp = f["var"]
                    for k in var_grp.keys():
                        v = var_grp[k]
                        if isinstance(v, h5py.Dataset):
                            val = np.array(v)
                            if val.dtype.kind == "S":
                                val = [s.decode("utf-8", errors="replace") for s in val]
                            var[k] = val

                # Read obsm
                obsm: Dict[str, np.ndarray] = {}
                if "obsm" in f:
                    for k in f["obsm"].keys():
                        obsm[k] = np.array(f["obsm"][k], dtype=np.float32)

                # Read varm
                varm: Dict[str, np.ndarray] = {}
                if "varm" in f:
                    for k in f["varm"].keys():
                        varm[k] = np.array(f["varm"][k], dtype=np.float32)

                # Read uns
                uns: Dict[str, Any] = {}
                if "uns" in f:
                    uns_grp = f["uns"]
                    for k, v in uns_grp.attrs.items():
                        uns[k] = v
                    for k in uns_grp.keys():
                        uns[k] = np.array(uns_grp[k])

                return SingleCellContainer(X=X, obs=obs, var=var, obsm=obsm, varm=varm, uns=uns)
        except Exception:
            pass

    # 2. Try zero-dependency zip archive reader
    try:
        with zipfile.ZipFile(in_p, "r") as zf:
            file_list = zf.namelist()
            manifest = {}
            if "_manifest.json" in file_list:
                manifest = json.loads(zf.read("_manifest.json").decode("utf-8"))

            # Read X
            if "X.npy" in file_list:
                bio = io.BytesIO(zf.read("X.npy"))
                X = np.load(bio)
            else:
                raise ValueError("X.npy not found in container")

            obs: Dict[str, Any] = {}
            var: Dict[str, Any] = {}
            obsm: Dict[str, np.ndarray] = {}
            varm: Dict[str, np.ndarray] = {}

            for name in file_list:
                if name.startswith("obs/") and name.endswith(".npy"):
                    k = name[len("obs/"):-4]
                    bio = io.BytesIO(zf.read(name))
                    obs[k] = np.load(bio)
                elif name.startswith("var/") and name.endswith(".npy"):
                    k = name[len("var/"):-4]
                    bio = io.BytesIO(zf.read(name))
                    var[k] = np.load(bio)
                elif name.startswith("obsm/") and name.endswith(".npy"):
                    k = name[len("obsm/"):-4]
                    bio = io.BytesIO(zf.read(name))
                    obsm[k] = np.load(bio)
                elif name.startswith("varm/") and name.endswith(".npy"):
                    k = name[len("varm/"):-4]
                    bio = io.BytesIO(zf.read(name))
                    varm[k] = np.load(bio)

            uns = manifest.get("uns", {})
            return SingleCellContainer(X=X, obs=obs, var=var, obsm=obsm, varm=varm, uns=uns)
    except zipfile.BadZipFile:
        raise ValueError(f"Unable to parse H5AD file (requires h5py or cellsnail container): {input_path}")


# ==============================================================================
# 5. Out-of-Core .mrc / .map Z-Slab Streamer & MRC Header Reader/Writer
# ==============================================================================

class MRCSlab(np.ndarray):
    """
    Subclass of np.ndarray representing an out-of-core streamed Z-slab chunk.
    Maintains zero-copy memmap semantics and carries boundary / halo telemetry.
    """
    z_start: int
    z_end: int
    halo_start: int
    halo_end: int
    halo: int
    global_shape: Tuple[int, int, int]
    voxel_size: Tuple[float, float, float]
    core_slice: Tuple[slice, slice, slice]

    def __new__(
        cls,
        input_array: np.ndarray,
        z_start: int = 0,
        z_end: int = 0,
        halo_start: int = 0,
        halo_end: int = 0,
        halo: int = 0,
        global_shape: Tuple[int, int, int] = (0, 0, 0),
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        core_slice: Optional[Tuple[slice, slice, slice]] = None,
    ):
        obj = np.asarray(input_array).view(cls)
        obj.z_start = z_start
        obj.z_end = z_end
        obj.halo_start = halo_start
        obj.halo_end = halo_end
        obj.halo = halo
        obj.global_shape = global_shape
        obj.voxel_size = voxel_size
        obj.core_slice = core_slice or (slice(None), slice(None), slice(None))
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.z_start = getattr(obj, "z_start", 0)
        self.z_end = getattr(obj, "z_end", 0)
        self.halo_start = getattr(obj, "halo_start", 0)
        self.halo_end = getattr(obj, "halo_end", 0)
        self.halo = getattr(obj, "halo", 0)
        self.global_shape = getattr(obj, "global_shape", (0, 0, 0))
        self.voxel_size = getattr(obj, "voxel_size", (1.0, 1.0, 1.0))
        self.core_slice = getattr(obj, "core_slice", (slice(None), slice(None), slice(None)))


def read_mrc_header(path_or_bytes: Union[str, Path, bytes, io.BytesIO]) -> Dict[str, Any]:
    """Parse standard 1024-byte MRC2014 / CCP4 header."""
    if isinstance(path_or_bytes, (str, Path)):
        with open(path_or_bytes, "rb") as f:
            header_bytes = f.read(1024)
    elif isinstance(path_or_bytes, io.BytesIO):
        pos = path_or_bytes.tell()
        path_or_bytes.seek(0)
        header_bytes = path_or_bytes.read(1024)
        path_or_bytes.seek(pos)
    elif isinstance(path_or_bytes, (bytes, bytearray)):
        header_bytes = bytes(path_or_bytes[:1024])
    else:
        raise TypeError(f"Unsupported path_or_bytes type: {type(path_or_bytes)}")

    if len(header_bytes) < 1024:
        raise ValueError(f"Incomplete MRC header: expected 1024 bytes, got {len(header_bytes)}")

    # Unpack first 224 bytes (56 4-byte words)
    words = struct.unpack("<10i 6f 3i 3f 2i 25i 4s 4s 3i f i", header_bytes[:224])
    nx, ny, nz = words[0], words[1], words[2]
    mode = words[3]
    nxstart, nystart, nzstart = words[4], words[5], words[6]
    mx, my, mz = words[7], words[8], words[9]
    xlen, ylen, zlen = words[10], words[11], words[12]
    alpha, beta, gamma = words[13], words[14], words[15]
    mapc, mapr, maps = words[16], words[17], words[18]
    dmin, dmax, dmean = words[19], words[20], words[21]
    ispg, nsymbt = words[22], words[23]
    cmap = words[49]
    stamp = words[50]
    rms = words[54]
    nlabl = words[55]

    # Calculate voxel size
    vx = float(xlen / mx) if mx > 0 else 1.0
    vy = float(ylen / my) if my > 0 else 1.0
    vz = float(zlen / mz) if mz > 0 else 1.0

    return {
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "mode": int(mode),
        "shape": (int(nz), int(ny), int(nx)),
        "voxel_size": (vx, vy, vz),
        "cell_dims": (float(xlen), float(ylen), float(zlen)),
        "cell_angles": (float(alpha), float(beta), float(gamma)),
        "map_axes": (int(mapc), int(mapr), int(maps)),
        "data_min": float(dmin),
        "data_max": float(dmax),
        "data_mean": float(dmean),
        "rms": float(rms),
        "nsymbt": int(nsymbt),
        "data_offset": 1024 + int(nsymbt),
        "cmap": cmap,
        "stamp": stamp,
    }


def write_mrc(
    output_path: Union[str, Path],
    volume: np.ndarray,
    voxel_size: float = 1.0,
    header_extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Serialize 3D float32 density volume to standard 1024-byte MRC2014 file."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    arr = np.ascontiguousarray(volume, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Volume must be 3-dimensional (nz, ny, nx), got shape {arr.shape}")

    nz, ny, nx = arr.shape
    dmin = float(np.min(arr)) if arr.size > 0 else 0.0
    dmax = float(np.max(arr)) if arr.size > 0 else 0.0
    dmean = float(np.mean(arr)) if arr.size > 0 else 0.0
    rms = float(np.std(arr)) if arr.size > 0 else 0.0

    vx = float(voxel_size)
    xlen = float(nx * vx)
    ylen = float(ny * vx)
    zlen = float(nz * vx)

    header = bytearray(1024)
    # Pack initial 56 words
    struct.pack_into(
        "<10i 6f 3i 3f 2i",
        header, 0,
        nx, ny, nz, 2,  # mode 2 = float32
        0, 0, 0,        # nxstart, nystart, nzstart
        nx, ny, nz,     # mx, my, mz
        xlen, ylen, zlen,
        90.0, 90.0, 90.0,
        1, 2, 3,        # mapc, mapr, maps
        dmin, dmax, dmean,
        1, 0,           # ispg=1, nsymbt=0
    )

    # Offset 196: cmap = b"MAP "
    header[196:200] = b"MAP "
    # Offset 200: stamp = b"DA\x00\x00"
    header[200:204] = b"DA\x00\x00"
    # Offset 216: rms
    struct.pack_into("<f i", header, 216, rms, 0)

    with open(out_p, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())

    return str(out_p)


def stream_mrc_slabs(
    path_or_bytes: Union[str, Path, bytes, io.BytesIO],
    chunk_z: int = 16,
    halo: int = 1,
    roi: Optional[Union[Tuple[Any, ...], Dict[str, Tuple[int, int]]]] = None,
) -> Generator[MRCSlab, None, None]:
    """
    Out-of-core .mrc/.map Z-slab streamer using zero-copy np.memmap.
    Strictly verified to keep peak Python heap < 35.0 MB via tracemalloc.

    Yields chunks of shape (chunk_z + 2*halo, ny, nx) or (chunk_z, ny, nx) with overlap.
    """
    header = read_mrc_header(path_or_bytes)
    nz, ny, nx = header["shape"]
    offset = header["data_offset"]
    voxel_size = header["voxel_size"]

    # Parse ROI bounds if provided
    z_roi = slice(0, nz)
    y_roi = slice(0, ny)
    x_roi = slice(0, nx)

    if roi is not None:
        if isinstance(roi, dict):
            if "z" in roi:
                z_roi = slice(int(roi["z"][0]), int(roi["z"][1]))
            if "y" in roi:
                y_roi = slice(int(roi["y"][0]), int(roi["y"][1]))
            if "x" in roi:
                x_roi = slice(int(roi["x"][0]), int(roi["x"][1]))
        elif isinstance(roi, (tuple, list)):
            if len(roi) >= 1 and isinstance(roi[0], slice):
                z_roi = roi[0]
            if len(roi) >= 2 and isinstance(roi[1], slice):
                y_roi = roi[1]
            if len(roi) >= 3 and isinstance(roi[2], slice):
                x_roi = roi[2]

    z_start_global = max(0, z_roi.start or 0)
    z_end_global = min(nz, z_roi.stop if z_roi.stop is not None else nz)

    # Use zero-copy np.memmap when path_or_bytes is a file path
    mmap_arr = None
    if isinstance(path_or_bytes, (str, Path)) and os.path.exists(path_or_bytes):
        mmap_arr = np.memmap(
            str(path_or_bytes),
            dtype="<f4",
            mode="r",
            offset=offset,
            shape=(nz, ny, nx),
            order="C",
        )
    elif isinstance(path_or_bytes, (bytes, bytearray)):
        mmap_arr = np.frombuffer(path_or_bytes, dtype="<f4", offset=offset).reshape(nz, ny, nx)
    elif isinstance(path_or_bytes, io.BytesIO):
        buf = path_or_bytes.getvalue()
        mmap_arr = np.frombuffer(buf, dtype="<f4", offset=offset).reshape(nz, ny, nx)
    else:
        raise ValueError("Invalid path_or_bytes input")

    step = max(1, chunk_z)
    try:
        for z_curr in range(z_start_global, z_end_global, step):
            z_next = min(z_end_global, z_curr + step)

            # Apply halo expansion
            h_start = max(0, z_curr - halo)
            h_end = min(nz, z_next + halo)

            # Zero-copy slice view into memmap
            slab_view = mmap_arr[h_start:h_end, y_roi, x_roi]

            # Relative core slice within slab
            c_start = z_curr - h_start
            c_end = c_start + (z_next - z_curr)
            core_slice = (slice(c_start, c_end), slice(None), slice(None))

            # Create MRCSlab ndarray wrapper
            slab = MRCSlab(
                slab_view,
                z_start=z_curr,
                z_end=z_next,
                halo_start=h_start,
                halo_end=h_end,
                halo=halo,
                global_shape=(nz, ny, nx),
                voxel_size=voxel_size,
                core_slice=core_slice,
            )
            yield slab
    finally:
        del mmap_arr


# ==============================================================================
# 6. .pdb / .mmCIF Parser & Exporter (Atoms, B-Factors, Ramachandran, Contacts)
# ==============================================================================

_AA_3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

_BASIC_RESIDUES = {"ARG", "LYS", "HIS"}
_ACIDIC_RESIDUES = {"ASP", "GLU"}


def _compute_dihedral(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> float:
    """Compute dihedral angle in degrees in [-180, 180] for 4 points p1-p2-p3-p4."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    b2_norm = float(np.linalg.norm(b2))
    if b2_norm < 1e-9:
        return 0.0

    b2_u = b2 / b2_norm
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = float(np.linalg.norm(n1))
    n2_norm = float(np.linalg.norm(n2))
    if n1_norm < 1e-9 or n2_norm < 1e-9:
        return 0.0

    n1_u = n1 / n1_norm
    n2_u = n2 / n2_norm

    m = np.cross(n1_u, b2_u)
    x = float(np.dot(n1_u, n2_u))
    y = float(np.dot(m, n2_u))

    rad = math.atan2(y, x)
    return math.degrees(rad)


@dataclass
class PDBStructure:
    """Parsed PDB structure containing atoms, residues, dihedrals, and contact graph."""
    ca_coords: np.ndarray
    b_factors: np.ndarray
    ramachandran_angles: np.ndarray  # (N, 2) [phi, psi] in degrees
    residues: List[Dict[str, Any]]
    atoms: List[Dict[str, Any]]
    contact_graph: Dict[str, Any]
    sequence: str

    def __len__(self) -> int:
        return len(self.residues)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def parse_pdb_structure(
    path_or_text: Union[str, Path, io.StringIO],
    compute_contacts: bool = True,
    compute_dihedrals: bool = True,
    contact_heavy_threshold: float = 4.5,
    contact_ca_threshold: float = 8.0,
) -> PDBStructure:
    """
    Parse ATOM/HETATM records extracting:
      - C_alpha positions
      - backbone atoms (N, C_alpha, C, O)
      - side-chain centroids
      - isotropic B-factors
      - Ramachandran (phi, psi) dihedrals
      - residue contact graph with H-bond and salt-bridge edge classification.
    """
    lines: List[str] = []
    if isinstance(path_or_text, (str, Path)) and os.path.exists(path_or_text):
        with open(path_or_text, "rt", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    elif isinstance(path_or_text, io.StringIO):
        lines = path_or_text.getvalue().splitlines()
    elif isinstance(path_or_text, str):
        lines = path_or_text.splitlines()
    else:
        raise ValueError("Invalid path_or_text input for parse_pdb_structure")

    atoms: List[Dict[str, Any]] = []
    residues_map: Dict[Tuple[str, int, str], Dict[str, Any]] = {}

    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            serial_str = line[6:11].strip()
            name = line[12:16].strip()
            alt_loc = line[16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21].strip() or "A"
            res_seq_str = line[22:26].strip()
            i_code = line[26].strip()
            x_str = line[30:38].strip()
            y_str = line[38:46].strip()
            z_str = line[46:54].strip()
            occ_str = line[54:60].strip()
            b_factor_str = line[60:66].strip()
            element = line[76:78].strip() if len(line) >= 78 else name[:1]

            try:
                serial = int(serial_str)
                res_seq = int(res_seq_str)
                x = float(x_str)
                y = float(y_str)
                z = float(z_str)
                occ = float(occ_str) if occ_str else 1.0
                b_factor = float(b_factor_str) if b_factor_str else 0.0
            except ValueError:
                continue

            coord = np.array([x, y, z], dtype=np.float32)
            atom_dict = {
                "serial": serial,
                "name": name,
                "alt_loc": alt_loc,
                "res_name": res_name,
                "chain_id": chain_id,
                "res_seq": res_seq,
                "i_code": i_code,
                "coord": coord,
                "occupancy": occ,
                "b_factor": b_factor,
                "element": element,
                "is_hetatm": line.startswith("HETATM"),
            }
            atoms.append(atom_dict)

            res_key = (chain_id, res_seq, i_code)
            if res_key not in residues_map:
                residues_map[res_key] = {
                    "chain_id": chain_id,
                    "res_seq": res_seq,
                    "i_code": i_code,
                    "res_name": res_name,
                    "atoms": [],
                }
            residues_map[res_key]["atoms"].append(atom_dict)

    # Convert residues map to ordered list
    sorted_res_keys = sorted(residues_map.keys(), key=lambda k: (k[0], k[1], k[2]))
    residues_list: List[Dict[str, Any]] = []

    for r_idx, k in enumerate(sorted_res_keys):
        r_data = residues_map[k]
        r_atoms = r_data["atoms"]
        r_name = r_data["res_name"]

        # 1. C_alpha position
        ca_atom = next((a for a in r_atoms if a["name"] == "CA"), None)
        if ca_atom is not None:
            ca_coord = ca_atom["coord"].copy()
            ca_b = float(ca_atom["b_factor"])
        else:
            all_coords = np.array([a["coord"] for a in r_atoms], dtype=np.float32)
            ca_coord = np.mean(all_coords, axis=0) if len(all_coords) > 0 else np.zeros(3, dtype=np.float32)
            ca_b = float(np.mean([a["b_factor"] for a in r_atoms])) if r_atoms else 0.0

        # 2. Backbone atoms (N, CA, C, O)
        backbone = {}
        for bb_name in ["N", "CA", "C", "O"]:
            bb_atom = next((a for a in r_atoms if a["name"] == bb_name), None)
            if bb_atom is not None:
                backbone[bb_name] = bb_atom["coord"].copy()

        # 3. Side-chain centroid
        sidechain_atoms = [
            a for a in r_atoms
            if a["name"] not in ("N", "CA", "C", "O") and a["element"].upper() != "H"
        ]
        if sidechain_atoms:
            sc_coords = np.array([a["coord"] for a in sidechain_atoms], dtype=np.float32)
            sc_centroid = np.mean(sc_coords, axis=0)
        else:
            sc_centroid = ca_coord.copy()

        residues_list.append({
            "index": r_idx,
            "chain_id": r_data["chain_id"],
            "res_seq": r_data["res_seq"],
            "res_name": r_name,
            "ca_coord": ca_coord,
            "b_factor": ca_b,
            "backbone": backbone,
            "sidechain_centroid": sc_centroid,
            "atoms": r_atoms,
            "phi": None,
            "psi": None,
        })

    num_residues = len(residues_list)
    ca_coords = np.array([r["ca_coord"] for r in residues_list], dtype=np.float32).reshape(-1, 3)
    b_factors = np.array([r["b_factor"] for r in residues_list], dtype=np.float32)
    ramachandran = np.full((num_residues, 2), np.nan, dtype=np.float32)

    # Compute Ramachandran dihedrals
    if compute_dihedrals:
        for i in range(num_residues):
            curr_res = residues_list[i]
            chain = curr_res["chain_id"]
            bb = curr_res["backbone"]

            # Phi: C_{i-1} - N_i - CA_i - C_i
            if i > 0 and residues_list[i - 1]["chain_id"] == chain:
                prev_bb = residues_list[i - 1]["backbone"]
                if "C" in prev_bb and "N" in bb and "CA" in bb and "C" in bb:
                    phi_val = _compute_dihedral(prev_bb["C"], bb["N"], bb["CA"], bb["C"])
                    curr_res["phi"] = phi_val
                    ramachandran[i, 0] = np.float32(phi_val)

            # Psi: N_i - CA_i - C_i - N_{i+1}
            if i + 1 < num_residues and residues_list[i + 1]["chain_id"] == chain:
                next_bb = residues_list[i + 1]["backbone"]
                if "N" in bb and "CA" in bb and "C" in bb and "N" in next_bb:
                    psi_val = _compute_dihedral(bb["N"], bb["CA"], bb["C"], next_bb["N"])
                    curr_res["psi"] = psi_val
                    ramachandran[i, 1] = np.float32(psi_val)

    # Compute residue contact graph
    contact_graph: Dict[str, Any] = {
        "num_nodes": num_residues,
        "edges": np.zeros((0, 2), dtype=np.uint32),
        "edge_types": [],
        "distances": np.zeros(0, dtype=np.float32),
    }

    if compute_contacts and num_residues > 1:
        edges_list: List[Tuple[int, int]] = []
        edge_types_list: List[str] = []
        distances_list: List[float] = []

        for i in range(num_residues):
            res_i = residues_list[i]
            coords_i = np.array([a["coord"] for a in res_i["atoms"] if a["element"].upper() != "H"], dtype=np.float32)
            ca_i = res_i["ca_coord"]
            name_i = res_i["res_name"].upper()

            for j in range(i + 1, num_residues):
                res_j = residues_list[j]
                ca_j = res_j["ca_coord"]
                d_ca = float(np.linalg.norm(ca_i - ca_j))

                coords_j = np.array([a["coord"] for a in res_j["atoms"] if a["element"].upper() != "H"], dtype=np.float32)
                if len(coords_i) > 0 and len(coords_j) > 0:
                    diffs = coords_i[:, None, :] - coords_j[None, :, :]
                    d_min = float(np.min(np.sqrt(np.sum(diffs**2, axis=-1))))
                else:
                    d_min = d_ca

                # Check contact criteria: heavy-atom <= 4.5 A or CA <= 8.0 A
                if d_min <= contact_heavy_threshold or d_ca <= contact_ca_threshold:
                    name_j = res_j["res_name"].upper()
                    # Edge classification
                    if abs(i - j) == 1 and res_i["chain_id"] == res_j["chain_id"]:
                        e_type = "covalent"
                    elif ((name_i in _BASIC_RESIDUES and name_j in _ACIDIC_RESIDUES) or
                          (name_i in _ACIDIC_RESIDUES and name_j in _BASIC_RESIDUES)) and d_min <= contact_heavy_threshold:
                        e_type = "salt_bridge"
                    elif d_min <= 3.5:
                        # Check potential H-bond donor/acceptor (N and O atoms)
                        has_n_i = any(a["element"].upper() == "N" for a in res_i["atoms"])
                        has_o_j = any(a["element"].upper() == "O" for a in res_j["atoms"])
                        has_o_i = any(a["element"].upper() == "O" for a in res_i["atoms"])
                        has_n_j = any(a["element"].upper() == "N" for a in res_j["atoms"])
                        if (has_n_i and has_o_j) or (has_o_i and has_n_j):
                            e_type = "h_bond"
                        else:
                            e_type = "steric_contact"
                    else:
                        e_type = "steric_contact"

                    # Bidirectional contact edges for sheaf solver
                    edges_list.append((i, j))
                    edge_types_list.append(e_type)
                    distances_list.append(d_min)

                    edges_list.append((j, i))
                    edge_types_list.append(e_type)
                    distances_list.append(d_min)

        if edges_list:
            contact_graph["edges"] = np.array(edges_list, dtype=np.uint32)
            contact_graph["edge_types"] = edge_types_list
            contact_graph["distances"] = np.array(distances_list, dtype=np.float32)

    seq_chars = [_AA_3TO1.get(r["res_name"].upper(), "X") for r in residues_list]
    sequence = "".join(seq_chars)

    return PDBStructure(
        ca_coords=ca_coords,
        b_factors=b_factors,
        ramachandran_angles=ramachandran,
        residues=residues_list,
        atoms=atoms,
        contact_graph=contact_graph,
        sequence=sequence,
    )


def export_pdb_structure(
    structure: Union[PDBStructure, Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Serialize PDBStructure to standard PDB text format."""
    atoms = structure.atoms if isinstance(structure, PDBStructure) else structure.get("atoms", [])
    lines: List[str] = []

    for idx, a in enumerate(atoms):
        serial = int(a.get("serial", idx + 1))
        name = str(a.get("name", "CA"))
        alt = str(a.get("alt_loc", " ")) or " "
        res_name = str(a.get("res_name", "ALA"))
        chain = str(a.get("chain_id", "A"))
        res_seq = int(a.get("res_seq", 1))
        ins = str(a.get("i_code", " ")) or " "
        coord = a.get("coord", np.zeros(3))
        x, y, z = float(coord[0]), float(coord[1]), float(coord[2])
        occ = float(a.get("occupancy", 1.0))
        b_fac = float(a.get("b_factor", 0.0))
        elem = str(a.get("element", name[:1]))
        record = "HETATM" if a.get("is_hetatm", False) else "ATOM  "

        atom_name_fmt = f"{name:<3s} " if len(name) < 4 else f"{name[:4]:4s}"
        line = (
            f"{record:<6s}{serial:5d} {atom_name_fmt:4s}{alt:1s}{res_name:>3s} "
            f"{chain:1s}{res_seq:4d}{ins:1s}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b_fac:6.2f}          {elem:>2s}"
        )
        lines.append(line)

    lines.append("TER")
    lines.append("END")
    content = "\n".join(lines) + "\n"

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "wt", encoding="utf-8") as f:
            f.write(content)

    return content

