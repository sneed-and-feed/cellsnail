"""tests/test_io.py
================
Unit tests for cellsnail.io zero-dependency single-cell readers and serializers.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from cellsnail.io import (
    SingleCellContainer,
    read_csv,
    read_h5ad,
    read_mtx,
    read_vcf,
    write_csv,
    write_h5ad,
    write_mtx,
    write_vcf,
)


def test_single_cell_container_basics():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    container = SingleCellContainer(X=X)
    assert container.shape == (2, 3)
    assert container.n_obs == 2
    assert container.n_vars == 3
    assert len(container.obs_names) == 2
    assert len(container.var_names) == 3


def test_mtx_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        X = np.array([[10.0, 0.0, 5.0], [0.0, 20.0, 0.0]], dtype=np.float32)
        obs = {"cell_id": ["cell_A", "cell_B"]}
        var = {"gene_id": ["g1", "g2", "g3"], "gene_name": ["Gene1", "Gene2", "Gene3"]}
        c_orig = SingleCellContainer(X=X, obs=obs, var=var)

        # Write
        out_files = write_mtx(c_orig, tmp_path)
        assert Path(out_files["matrix"]).exists()
        assert Path(out_files["features"]).exists()
        assert Path(out_files["barcodes"]).exists()

        # Read back
        c_read = read_mtx(out_files["matrix"], out_files["features"], out_files["barcodes"])
        assert c_read.shape == c_orig.shape
        np.testing.assert_allclose(c_read.X, c_orig.X, atol=1e-5)
        assert c_read.obs_names == c_orig.obs_names
        assert c_read.var_names == c_orig.var_names


def test_vcf_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        vcf_file = Path(tmpdir) / "test_sc_cnv.vcf"
        samples = ["cell_001", "cell_002", "cell_003"]
        variants = [
            {"chrom": "chr7", "pos": 55000000, "id": "var_egfr", "gene": "EGFR"},
            {"chrom": "chr12", "pos": 58000000, "id": "var_cdk4", "gene": "CDK4"},
        ]
        cn_matrix = np.array([[8.0, 2.0], [2.0, 6.0], [2.0, 2.0]], dtype=np.float32)
        baf_matrix = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float32)

        # Write
        write_vcf(vcf_file, variants, samples, cn_matrix=cn_matrix, baf_matrix=baf_matrix)
        assert vcf_file.exists()

        # Read back
        res = read_vcf(vcf_file)
        assert res["samples"] == samples
        assert len(res["variants"]) == 2
        np.testing.assert_allclose(res["cn_matrix"], cn_matrix, atol=1e-2)


def test_csv_expression_and_spatial_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Expression CSV
        expr_file = Path(tmpdir) / "expr.csv"
        X = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        c_orig = SingleCellContainer(
            X=X,
            obs={"cell_id": ["c1", "c2"]},
            var={"gene_name": ["G1", "G2"]},
        )
        write_csv(c_orig, expr_file)
        c_read = read_csv(expr_file, is_spatial=False)
        np.testing.assert_allclose(c_read.X, c_orig.X, atol=1e-5)

        # Spatial CSV
        spatial_file = Path(tmpdir) / "spatial.csv"
        spatial_dict = {
            "barcode": ["c1", "c2", "c3"],
            "x": [10.5, 20.5, 30.5],
            "y": [100.2, 100.8, 101.4],
            "layer": ["L1", "L1", "L2"],
        }
        write_csv(spatial_dict, spatial_file)
        s_read = read_csv(spatial_file, is_spatial=True)
        assert len(s_read["barcodes"]) == 3
        np.testing.assert_allclose(s_read["coordinates"][:, 0], [10.5, 20.5, 30.5])


def test_h5ad_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        h5ad_file = Path(tmpdir) / "test_data.h5ad"
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        spatial = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        c_orig = SingleCellContainer(
            X=X,
            obs={"cell_id": ["c1", "c2"], "batch": ["b1", "b2"]},
            var={"gene_id": ["g1", "g2"], "gene_name": ["GeneA", "GeneB"]},
            obsm={"spatial": spatial},
            uns={"experiment": "test"},
        )

        write_h5ad(c_orig, h5ad_file)
        assert h5ad_file.exists()

        c_read = read_h5ad(h5ad_file)
        assert c_read.shape == c_orig.shape
        np.testing.assert_allclose(c_read.X, c_orig.X, atol=1e-5)
        np.testing.assert_allclose(c_read.obsm["spatial"], spatial, atol=1e-5)
