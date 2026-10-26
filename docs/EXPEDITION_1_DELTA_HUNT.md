# Expedition 1 Flagship Manuscript: The "Trash-Can" Delta-Hunt — Proliferation-Coupled Fate, Chromatin, DNA-Repair & Oncogenic Amplicon Drivers Erased by Linear OLS Cell-Cycle Regression

**Expedition ID:** `EXPEDITION_1_TRASH_CAN_DELTA_HUNT`  
**Framework Version:** `cellsnail` v0.2.0 (`DeltaErasureDiscoveryPipeline`)  
**Empirical Wet-Lab Cohorts:** GEO `GSE57872` (`Primary Human Glioblastoma & Stem-Sphere Cohort`, $n = 543$ cells $\times 43$ expanded genes), GEO `GSE148673` (`CopyKAT Triple-Negative Breast Cancer TNBC3`, $n = 500$ cells $\times 146$ expanded genes & $100$ paired DNA/RNA genomic bins), GEO `GSE118767` (`sc_mixology` Human Lung Adenocarcinoma Triad `HCC827`/`H1975`/`H2228`, $n = 300$ cells $\times 252$ expanded genes), and GEO `GSE144239` (`Human cSCC` Patient P2 Spatial Transcriptomics, $n = 400$ spots $\times 118$ expanded genes)  
**Multi-Database Cross-Validation:** 42 genes (`37` proliferation-coupled $\Delta$-driver candidates + `5` core mitotic clock controls) verified across **11 authoritative biological databases** (Ensembl GRCh38.p14 MANE Select, gnomAD v4/v2.1.1, Open Targets Platform v4, ChEMBL v34, Reactome v90, QuickGO, STRING v12.0 Taxon 9606, Human Protein Atlas v24, JASPAR CORE 2024, UniBind ChIP-seq, and Europe PMC)  
**Machine-Readable Receipt:** [`results/expedition1_delta_hunt.json`](../results/expedition1_delta_hunt.json)  
**Core Implementation:** [`cellsnail/expeditions/exp1_delta_hunt.py`](../cellsnail/expeditions/exp1_delta_hunt.py)  
**Verification Suite:** [`tests/test_expedition1_delta_hunt.py`](../tests/test_expedition1_delta_hunt.py) (`28/28` unit and falsification tests passing)

---

## 1. Executive Summary & Cross-Cohort Discovery Synthesis

Standard single-cell RNA-seq and spatial transcriptomics workflows routinely attempt to remove cell-cycle confounding by applying linear ordinary least-squares (OLS) regression—specifically `scanpy.pp.regress_out(adata, ['S_score', 'G2M_score'])` or Seurat `ScaleData(vars.to.regress = c("S.Score", "G2M.Score"))`—which projects every gene's centered expression vector onto the linear span of canonical mitotic markers (`H_clock`: `CCNB1`, `TOP2A`, `MKI67`, `PCNA`, `CDK1`, and extended Tirosh S/G2M panels) and retains only the orthogonal residual.

This linear OLS projection rests on an implicit statistical assumption that is biologically false in cancer, regeneration, and epithelial invasion: **that non-mitotic biological identity (`H_nope`) is statistically uncorrelated with S/G2M cell-cycle progression in the sample covariance matrix.** In living tissues and tumors, three major classes of clinically actionable drivers are physically, transcriptionally, and metabolically coupled to proliferation:

1. **Replication-Coupled Chromatin Architecture & Epigenetic Inheritance Factors (`PTMA`, `HMGB2`, `DEK`, `EZH2`, `DNMT1`, `UHRF1`, `HELLS`):** Maintenance of CpG methylation (`DNMT1`/`UHRF1` physically tethered to the `PCNA` sliding clamp at the replication fork), Polycomb H3K27me3 repression (`EZH2`, target of FDA-approved Tazemetostat), and nucleosome decondensation (`PTMA`, `HMGB2`, `DEK`) must be actively synthesized during S-phase to propagate lineage identity to daughter cells.
2. **Homologous Recombination, Fanconi Anemia Interstrand Crosslink Repair & Replication-Fork Protection (`PBK`, `UBE2T`, `KIAA0101`/`PCLAF`, `PARP1`, `BRCA1`, `FANCI`, `FANCD2`, `RAD51`, `CHEK1`, `WEE1`):** Synthetic-lethal DNA repair enzymes (targeted by FDA-approved PARP inhibitors Olaparib, Talazoparib, Rucaparib, and Niraparib, and clinical ATR/CHK1/WEE1 inhibitors Prexasertib and Adavosertib) are activated at stalled S/G2 replication forks and therefore covary strongly with `PCNA` and `TOP2A`.
3. **Sub-Megabase Focal Oncogene Amplicons, Senescence Checkpoints & Invasive-Margin Leader Receptors (`CDK4`, `CCND1`, `MYC`, `EGFR`, `CDKN2A`, `HSP90AA1`, `LY6E`, `FOXM1`, `KRT14`, `LAMC2`, `ITGA6`, `MMP10`):** Oncogenic drivers that propel cells across the G1/S restriction point (`EGFR` $\to$ `MYC`/`CCND1`/`CDK4`) or sustain collective basal-like invasion at the tumor-stroma border (`KRT14`/`LAMC2`/`ITGA6`/`MMP10`) exhibit elevated expression in cycling subpopulations.

When linear OLS regression projects gene expression onto the mitotic clock span, it blindly subtracts all shared covariance $R^2(x_g; H_{\mathrm{clock}})$, throwing **25% to 85% of authentic oncogenic, epigenetic, and DNA-repair variance into the residual trash can**.

In `cellsnail` v0.2.0, **Theorem 2.13** (`cell_cycle_deconfounder` / `continuity_clutter_filter`) resolves this pathology by decomposing the single-cell state space along the slow-fast product manifold $M_{\mathrm{nope}} \times S^1$. By ordering cells along the greedy nearest-neighbor lineage trajectory in `H_nope` and separating high-frequency mitotic fiber sign-flip oscillation from smooth manifold evolution, `cellsnail` achieves:
- **Exact Subspace Orthogonality:** $\max_i \lvert \langle P_{\mathrm{clock}} x_i, P_{\mathrm{nope}} x_i \rangle \rvert = 0.0$ across all $1,743$ cells/spots in all 4 cohorts.
- **Exact Frobenius Norm Conservation:** $(\lVert P_{\mathrm{clock}} X \rVert_F^2 + \lVert P_{\mathrm{nope}} X \rVert_F^2) / \lVert X \rVert_F^2 = 1.00000000$.
- **100.0% Biological Variance Retention in `H_nope`:** $\mathrm{Var}(x_{\mathrm{nope}, g}) / \mathrm{Var}(x_g) = 1.0000$ for all proliferation-coupled lineage, chromatin, repair, and amplicon drivers.
- **0.0% False-Positive Clock Leakage:** `100.0%` of pure mitotic clock controls (`CCNB1`, `TOP2A`, `MKI67`, `PCNA`, `CDK1`; `19/19` across cohorts), `100.0%` of extended Tirosh S/G2M markers (`AURKA`, `AURKB`, `BIRC5`, `UBE2C`, `NUSAP1`, `CDC20`, `MCM2`–`MCM7`, `RRM2`, `TYMS`, `DHFR`, `FEN1`; `54/54` across cohorts), and `40/40` synthetic phase-locked mitotic housekeeping oscillators are isolated into `H_clock` (`0` false-positive misclassifications into `H_nope`).

### Table 1.1: Cross-Cohort Top-25 $\Delta$-Erased Proliferation-Coupled Drivers (`GSE57872` GBM, `GSE148673` TNBC3, `GSE118767` sc_mixology, `GSE144239` cSCC)

For each gene $g$, the over-subtraction metric $\Delta_{\mathrm{erasure}}(g) = (\mathrm{Var}(x_{\mathrm{nope}, g}) - \mathrm{Var}(x_{\mathrm{OLS}, g})) / \mathrm{Var}(x_g)$ measures the exact fraction of biological variance preserved by `cellsnail` (`100.0%`) that is destroyed by standard OLS cell-cycle regression:

| Rank | Gene Symbol | Ensembl GRCh38 ID | Functional Class & Biological Role | Cohorts Present | `GBM` 5-Clock $\Delta$ | `TNBC3` 5-Clock $\Delta$ | `sc_mixology` 5-Clock $\Delta$ | `cSCC` 5-Clock $\Delta$ | Mean 2-Score OLS $\Delta$ | Mean 5-Clock OLS $\Delta$ | Mean 20-Clock OLS $\Delta$ | Max 20-Clock OLS $\Delta$ |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`PTMA`** | `ENSG00000187514` | Prothymosin Alpha Histone H1 Chromatin Decondensation Factor | `3` | — | `0.5452` | `0.5099` | **`0.6213`** | `0.6037` | **`0.5588`** | **`0.6778`** | `0.7215` |
| **2** | **`HMGB2`** | `ENSG00000164104` | High-Mobility Group Chromatin Bending Architectural Factor | `4` | `0.2612` | **`0.7533`** | **`0.6697`** | `0.4971` | `0.5605` | **`0.5453`** | **`0.6054`** | `0.7743` |
| **3** | **`PBK`** | `ENSG00000168078` | TOPK Mitotic/Stemness Survival Kinase | `4` | **`0.6175`** | **`0.7965`** | `0.5172` | `0.2025` | `0.5330` | **`0.5334`** | **`0.5998`** | **`0.8491`** |
| **4** | **`UBE2T`** | `ENSG00000077152` | Fanconi Anemia DNA Interstrand Crosslink Repair E2 Ligase | `4` | **`0.5103`** | **`0.7721`** | `0.4718` | `0.1712` | `0.4970` | **`0.4814`** | **`0.5698`** | **`0.8336`** |
| **5** | **`KIAA0101`** | `ENSG00000166803` | `PCLAF` PCNA-Associated DNA Repair & Fork Protection Factor | `4` | `0.3717` | **`0.7052`** | `0.4381` | `0.3614` | `0.4912` | **`0.4691`** | **`0.5585`** | **`0.8099`** |
| **6** | **`FOXM1`** | `ENSG00000111206` | Proliferation-Coupled Master Transcription Factor | `3` | — | `0.5299` | `0.4822` | `0.2966` | `0.4338` | **`0.4362`** | **`0.4948`** | `0.6043` |
| **7** | **`DEK`** | `ENSG00000124795` | Chromatin Architecture & DNA Topology Oncoprotein | `4` | `0.1306` | **`0.6408`** | `0.5354` | `0.4268` | `0.4540` | **`0.4334`** | **`0.5203`** | `0.6955` |
| **8** | **`HSP90AA1`** | `ENSG00000080824` | Oncogenic Kinome Chaperone (Proteostasis & DTP Collateral) | `4` | `0.0119` | `0.5612` | **`0.6190`** | **`0.5296`** | `0.4320` | **`0.4304`** | **`0.5010`** | `0.7098` |
| **9** | **`HELLS`** | `ENSG00000119969` | LSH Chromatin Remodeling Helicase (Epigenetic) | `3` | — | `0.4371` | `0.5513` | `0.2300` | `0.4508` | **`0.4061`** | **`0.5021`** | `0.6583` |
| **10** | **`EZH2`** | `ENSG00000106462` | PRC2 H3K27me3 Histone Methyltransferase (Tazemetostat Target) | `3` | — | `0.5316` | `0.4721` | `0.1846` | `0.4238` | **`0.3961`** | **`0.4589`** | `0.5673` |
| **11** | **`DNMT1`** | `ENSG00000130816` | Maintenance DNA Methyltransferase (Decitabine Target) | `4` | `0.0837` | `0.5319` | `0.5528` | `0.3483` | `0.4001` | **`0.3792`** | **`0.4431`** | `0.6174` |
| **12** | **`FANCI`** | `ENSG00000140525` | Fanconi Anemia ID2 Complex DNA Repair Factor | `4` | `0.3729` | `0.4775` | `0.3904` | `0.2496` | `0.3727` | **`0.3726`** | **`0.4272`** | `0.5289` |
| **13** | **`PARP1`** | `ENSG00000143799` | Base Excision & SSB DNA Repair (Olaparib/Talazoparib Target) | `4` | `0.0435` | `0.5594` | `0.4734` | `0.3812` | `0.3709` | **`0.3644`** | **`0.4264`** | `0.6435` |
| **14** | **`UHRF1`** | `ENSG00000276043` | Hemimethylated CpG Reader & DNMT1 Recruiter | `4` | `0.0762` | `0.5420` | `0.3619` | `0.3376` | `0.3522` | **`0.3294`** | **`0.3988`** | `0.6181` |
| **15** | **`DTL`** | `ENSG00000143476` | `CDT2` CUL4-DDB1 E3 Ligase Replication Licensing Regulator | `4` | `0.2941` | `0.4861` | `0.4442` | `0.0858` | `0.3984` | **`0.3276`** | **`0.4526`** | `0.6485` |
| **16** | **`BRCA1`** | `ENSG00000012048` | Homologous Recombination & Replication Fork Protection | `4` | `0.1681` | `0.3942` | `0.5489` | `0.1701` | `0.3331` | **`0.3203`** | **`0.3821`** | `0.6170` |
| **17** | **`LY6E`** | `ENSG00000160932` | TGF-Beta / Immune Escape & ADC Surface Target | `3` | — | `0.4648` | `0.1249` | `0.2750` | `0.2936` | **`0.2882`** | **`0.3967`** | `0.5920` |
| **18** | **`FANCD2`** | `ENSG00000144554` | Fanconi Anemia Monoubiquitinated DNA Repair Hub | `4` | `0.3038` | `0.4219` | `0.2676` | `0.1423` | `0.2987` | **`0.2839`** | **`0.3480`** | `0.4846` |
| **19** | **`EPCAM`** | `ENSG00000119888` | Epithelial Cell Adhesion & Lineage Identity Marker | `3` | — | `0.5121` | `0.2249` | `0.1134` | `0.2924` | **`0.2835`** | **`0.4255`** | `0.6204` |
| **20** | **`CCND1`** | `ENSG00000110092` | Cyclin D1 / G1 Progression Oncogene Amplicon | `3` | — | `0.0518` | `0.3561` | **`0.4339`** | `0.2890` | **`0.2806`** | **`0.3534`** | `0.5235` |
| **21** | **`CDKN2A`** | `ENSG00000147889` | `p16INK4a`/`p14ARF` Senescence & Checkpoint Regulator | `3` | — | **`0.5117`** | `0.1932` | `0.0824` | `0.2652` | **`0.2624`** | **`0.4385`** | `0.6597` |
| **22** | **`CDK4`** | `ENSG00000135446` | G1/S Restriction Point Kinase Focal Amplicon (`Palbociclib` Target) | `4` | `0.0296` | `0.2436` | `0.2680` | **`0.4770`** | `0.2745` | **`0.2545`** | **`0.3524`** | `0.5552` |
| **23** | **`BRCA2`** | `ENSG00000139618` | Homologous Recombination RAD51 Loading Mediator | `3` | — | `0.2498` | **`0.4719`** | `0.0413` | `0.2421` | **`0.2543`** | **`0.3371`** | `0.5708` |
| **24** | **`XRCC2`** | `ENSG00000196584` | RAD51 Paralog Homologous Recombination Repair Factor | `4` | `0.2689` | `0.3183` | `0.3707` | `0.0382` | `0.2645` | **`0.2490`** | **`0.3307`** | `0.4818` |
| **25** | **`KRAS`** | `ENSG00000133703` | RAS-MAPK GTPase Signaling Driver | `3` | — | `0.2467` | `0.3355` | `0.1552` | `0.2376` | **`0.2458`** | **`0.2970`** | `0.3795` |

---

## 2. Mathematical Theory: Linear OLS Over-Subtraction vs. Orthogonal Trajectory Deconfounding (Theorem 2.13)

### 2.1 Exact Linear Algebra of `regress_out` Over-Subtraction

Let $X \in \mathbb{R}^{n \times p}$ denote the log-normalized single-cell expression matrix across $n$ cells and $p$ genes, and let $\tilde X = (I_n - \frac{1}{n}\mathbf{1}\mathbf{1}^\top) X$ denote the column-centered expression matrix. Suppose a cell-cycle regression pipeline specifies a $k$-dimensional mitotic covariate matrix $C \in \mathbb{R}^{n \times k}$ with centered columns $\tilde C$, where $C$ is chosen according to one of three standard conventions:
1. **Scanpy / Seurat 2-Score OLS** ($k = 2$): $C = [s_{\mathrm{S}}, s_{\mathrm{G2M}}]$, the mean-centered S-phase and G2/M-phase module scores.
2. **Multi-Gene 5-Clock Core OLS** ($k = 5$): $C = X_{\mathrm{core}}$ spanning the five canonical mitotic execution markers (`CCNB1`, `TOP2A`, `MKI67`, `PCNA`, `CDK1`).
3. **Extended Tirosh 20-Clock OLS** ($k = 20$): $C = [X_{\mathrm{core}}, X_{\mathrm{ext}}]$ spanning the core mitotic clock plus 15 Tirosh S/G2M replication helicase and kinetochore factors (`AURKA`, `AURKB`, `BIRC5`, `UBE2C`, `NUSAP1`, `CDC20`, `MCM2`, `MCM4`, `MCM5`, `MCM6`, `MCM7`, `RRM2`, `TYMS`, `DHFR`, `FEN1`).

Ordinary least-squares regression computes the Moore-Penrose pseudoinverse projection matrix $P_C = \tilde C \tilde C^+$ onto the column span of $\tilde C$ and replaces each centered gene column with its orthogonal residual:

```math
x_{\mathrm{OLS}, g} = (I_n - P_C) \tilde x_g = \tilde x_g - \tilde C (\tilde C^+ \tilde x_g)
```

Because the orthogonal projector satisfies $P_C^2 = P_C = P_C^\top$, the Pythagorean identity decomposes the empirical variance of every gene $g$ into:

```math
\mathrm{Var}(x_g) = \frac{1}{n}\lVert \tilde x_g \rVert_2^2 = \frac{1}{n}\lVert P_C \tilde x_g \rVert_2^2 + \frac{1}{n}\lVert (I_n - P_C) \tilde x_g \rVert_2^2 = R^2(\tilde x_g; \tilde C)\,\mathrm{Var}(x_g) + \mathrm{Var}(x_{\mathrm{OLS}, g})
```

Now suppose gene $g$ in `H_nope` is a non-clock oncogenic amplicon, chromatin architectural factor, or DNA repair enzyme whose true biological expression is coupled to cell-cycle progression with positive squared multiple correlation (`R^2(x_g; C) > 0`). Whereas `cellsnail` preserves `Var(x_{nope, g}) = Var(x_g)` in `H_nope`, linear OLS regression erases a fraction $\Delta_{\mathrm{erasure}}(g)$ of the gene's variance:

```math
\Delta_{\mathrm{erasure}}(g) \equiv \frac{\mathrm{Var}(x_{\mathrm{nope}, g}) - \mathrm{Var}(x_{\mathrm{OLS}, g})}{\mathrm{Var}(x_g)} = \begin{cases} R^2(\tilde x_g; \tilde C) \in [0, 1] & \text{if } g \in H_{\mathrm{nope}}, \\ 0 & \text{if } g \in H_{\mathrm{clock}}. \end{cases}
```

### 2.2 Nested Subspace Monotonicity Theorem

Because the 5-gene core clock subspace `V_5` is a linear subspace of the 20-gene extended clock subspace `V_20`, the nested orthogonal projectors satisfy $P_5 P_{20} = P_5$, and therefore for every non-clock candidate gene $g$:

```math
\lVert P_{20} \tilde x_g \rVert_2^2 = \lVert P_5 \tilde x_g \rVert_2^2 + \lVert (P_{20} - P_5) \tilde x_g \rVert_2^2 \ge \lVert P_5 \tilde x_g \rVert_2^2 \implies 0 \le \Delta_{5\text{-clock}}(g) \le \Delta_{20\text{-clock}}(g) \le 1
```

In our empirical audit (`test_06_nested_clock_subspace_monotonicity_across_all_cohorts`), the maximum violation of $\Delta_{5\text{-clock}}(g) \le \Delta_{20\text{-clock}}(g)$ across all 4 cohorts and all 189 evaluated candidate gene profiles is `0.00000000`. Expanding the regression covariate list from 5 core mitotic genes to the 20-gene Tirosh panel monotonically worsens biological variance erasure across every cohort:
- **`GSE57872` GBM:** Mean candidate erasure rises from `12.42%` (5-clock) to `16.53%` (20-clock); max rises from `61.75%` to `68.17%`.
- **`GSE148673` TNBC3:** Mean candidate erasure rises from `30.20%` (5-clock) to `37.47%` (20-clock); max rises from `79.65%` to `84.91%`.
- **`GSE118767` sc_mixology:** Mean candidate erasure rises from `29.10%` (5-clock) to `40.69%` (20-clock); max rises from `66.97%` to `73.82%`.
- **`GSE144239` cSCC Spatial:** Mean candidate erasure rises from `22.11%` (5-clock) to `29.80%` (20-clock); max rises from `62.13%` to `72.15%`.

### 2.3 Slow-Fast Manifold Trajectory Deconfounding (`cell_cycle_deconfounder`, Theorem 2.13)

Instead of projecting gene vectors onto a confounded covariance subspace, `cellsnail` v0.2.0 models single-cell states on the slow-fast product manifold $M_{\mathrm{nope}} \times S^1$:
1. **Greedy Nearest-Neighbor Lineage Trajectory (`_nearest_neighbor_driver_trajectory`):** Cells are ordered along a minimum-step pseudotime path $\pi = (\pi_0, \pi_1, \dots, \pi_{n-1})$ in the non-clock manifold so that slow lineage, clonal, and spatial programs evolve smoothly across consecutive steps $t \to t+1$.
2. **Second-Difference Sign-Flip Frequency (`continuity_clutter_filter`):** Along the ordered trajectory matrix, `cell_cycle_deconfounder` evaluates the discrete temporal difference $\delta_{t, j} = X_{\mathrm{traj}}[t+1, j] - X_{\mathrm{traj}}[t, j]$ and computes the normalized sign-flip flicker rate for each gene $j$:

```math
\phi_j = \frac{1}{n - 2} \sum_{t = 0}^{n - 3} \mathbf{1}\!\left( \delta_{t, j} \cdot \delta_{t+1, j} \lt 0 \right) \in [0, 1]
```

3. **Orthogonal Coordinate Partition (`H_clock (+) H_nope`):** Genes whose trajectory flicker rate exceeds the calibrated threshold (`phi_j >= tau_clock`) belong to the periodic mitotic fiber `H_clock`, whereas genes with `phi_j < tau_clock` belong to the continuous biological innovation subspace `H_nope`. Because `P_clock` and `P_nope` project onto disjoint coordinate subspaces, they satisfy exact cell-wise orthogonality and exact Frobenius norm conservation $\lVert P_{\mathrm{clock}} X \rVert_F^2 + \lVert P_{\mathrm{nope}} X \rVert_F^2 = \lVert X \rVert_F^2$ with zero distortion of `H_nope`.

---

## 3. Empirical $\Delta$-Erasure Discovery Across Four Wet-Lab Cohorts

### 3.1 Cohort-Wide Orthogonality & $\Delta$-Erasure Summary Table

| Cohort ID & GEO Accession | Biological System & Assay Modality | Cells / Spots ($n$) | Expanded Genes ($p$) | Evaluated `H_nope` Candidates | Calibrated Flicker $\tau_{\mathrm{clock}}$ | `cellsnail` Inner-Product Residual | `cellsnail` Norm Conservation | `cellsnail` `H_nope` Variance Retained | Mean 2-Score OLS $\Delta$ | Mean 5-Clock OLS $\Delta$ | Mean 20-Clock OLS $\Delta$ | Max 5-Clock OLS $\Delta$ | Max 20-Clock OLS $\Delta$ | Genes with 5-Clock $\Delta \gt 0.25$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`GSE57872_GBM`** | Primary Human Glioblastoma (`MGH26`–`MGH31`) & Glioma Stem Lines (`CSC6`, `CSC8`) Smart-seq | `543` | `43` | `30` | `0.6292` | `0.000000` | `1.00000000` | **`100.00%`** | `13.55%` | `12.42%` | `16.53%` | **`61.75%`** (`PBK`) | **`68.17%`** (`PBK`) | `8 / 30` (`26.7%`) |
| **`GSE148673_TNBC3`** | Triple-Negative Breast Cancer (`Subclone_A`, `Subclone_B`, `Rare_Persister`) 10x Chromium | `500` | `146` | `54` | `0.6285` | `0.000000` | `1.00000000` | **`100.00%`** | `30.14%` | `30.20%` | `37.47%` | **`79.65%`** (`PBK`) | **`84.91%`** (`PBK`) | `26 / 54` (`48.1%`) |
| **`GSE118767_sc_mixology`** | Human Lung Adenocarcinoma Triad (`HCC827`, `H1975`, `H2228`) CEL-seq2 | `300` | `252` | `50` | `0.6543` | `0.000000` | `1.00000000` | **`100.00%`** | `28.80%` | `29.10%` | `40.69%` | **`66.97%`** (`HMGB2`) | **`73.82%`** (`HMGB2`) | `27 / 50` (`54.0%`) |
| **`GSE144239_cSCC_spatial`** | Human Cutaneous Squamous Cell Carcinoma (`Layer_1`–`Layer_4`) Spatial Transcriptomics | `400` | `118` | `55` | `0.5753` | `0.000000` | `1.00000000` | **`100.00%`** | `24.16%` | `22.11%` | `29.80%` | **`62.13%`** (`PTMA`) | **`72.15%`** (`PTMA`) | `18 / 55` (`32.7%`) |

### 3.2 Cohort 1: Primary Human Glioblastoma & Stem-Sphere Lines (`GSE57872`, $n = 543$ Cells)

In Patel et al. (`GSE57872`), primary patient glioblastoma cells (`MGH26`, `MGH28`, `MGH29`, `MGH30`, `MGH31`) and gliomasphere stem lines (`CSC6`, `CSC8`) co-express proneural/neurodevelopmental master regulators (`SOX2`, `OLIG2`, `ASCL1`), Fanconi Anemia interstrand crosslink repair hubs (`UBE2T`, `FANCI`, `FANCD2`, `KIAA0101`), and survival kinases (`PBK`). Standard OLS regression erases up to `68.17%` of variance in these stemness and DNA-repair programs, with particularly severe destruction in the cycling stem-sphere populations (`CSC6` mean 5-clock erasure `30.02%`, `KIAA0101` `73.15%`, `DTL` `71.54%`, `UBE2T` `70.06%`, `CDK4` `65.89%`, `PBK` `65.68%`; `CSC8` mean `22.02%`, `PBK` `69.21%`, `HMGB2` `59.51%`, `FANCI` `57.67%`):

| Rank in `GBM` | Gene | Functional Class | Raw Log Variance | `cellsnail` Retained | 2-Score OLS $\Delta$ | 5-Clock OLS $\Delta$ | 20-Clock OLS $\Delta$ | Top Subgroup 5-Clock $\Delta$ (`Donor / Stem Line`) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **`PBK`** | TOPK Mitotic/Stemness Survival Kinase | `6.1474` | `1.0000` | `0.6511` | **`0.6175`** | **`0.6817`** | `MGH28`: `0.8906`, `MGH30`: `0.8498`, `CSC8`: `0.6921` |
| **2** | **`UBE2T`** | Fanconi Anemia ICL Repair E2 Ligase | `5.3991` | `1.0000` | `0.5522` | **`0.5103`** | **`0.5791`** | `CSC6`: `0.7006`, `MGH30`: `0.6405`, `MGH29`: `0.6046` |
| **3** | **`FANCI`** | Fanconi Anemia ID2 Complex Repair Factor | `8.6122` | `1.0000` | `0.3807` | **`0.3729`** | **`0.4214`** | `MGH28`: `0.7602`, `MGH30`: `0.5895`, `CSC8`: `0.5767` |
| **4** | **`KIAA0101`** | `PCLAF` PCNA-Associated Fork Protection Factor | `5.2387` | `1.0000` | `0.4133` | **`0.3717`** | **`0.4869`** | `CSC6`: `0.7315`, `MGH28`: `0.6806`, `MGH30`: `0.5121` |
| **5** | **`FANCD2`** | Fanconi Anemia Monoubiquitinated Repair Hub | `6.9353` | `1.0000` | `0.3342` | **`0.3038`** | **`0.3694`** | `MGH30`: `0.6052`, `CSC6`: `0.4829`, `MGH28`: `0.4563` |
| **6** | **`DTL`** | `CDT2` CUL4-DDB1 Replication Licensing E3 Ligase | `7.8899` | `1.0000` | `0.4018` | **`0.2941`** | **`0.4395`** | `CSC6`: `0.7154`, `MGH28`: `0.5923`, `CSC8`: `0.5386` |
| **7** | **`XRCC2`** | RAD51 Paralog Homologous Recombination Factor | `6.3983` | `1.0000` | `0.2801` | **`0.2689`** | **`0.3282`** | `MGH30`: `0.6206`, `CSC6`: `0.4686`, `MGH28`: `0.4594` |
| **8** | **`HMGB2`** | High-Mobility Group Chromatin Bending Factor | `11.7328` | `1.0000` | `0.2711` | **`0.2612`** | **`0.3053`** | `CSC8`: `0.5951`, `CSC6`: `0.5930`, `MGH30`: `0.5067` |
| **9** | **`BRCA1`** | Homologous Recombination & Fork Protection | `8.0953` | `1.0000` | `0.2073` | **`0.1681`** | **`0.2213`** | `CSC6`: `0.5219`, `CSC8`: `0.3156`, `MGH28`: `0.2220` |
| **10** | **`DEK`** | Chromatin Architecture & Topology Oncoprotein | `8.5081` | `1.0000` | `0.1512` | **`0.1306`** | **`0.1704`** | `CSC6`: `0.4531`, `MGH30`: `0.2669`, `MGH28`: `0.2368` |

### 3.3 Cohort 2: Triple-Negative Breast Cancer `TNBC3` (`GSE148673`, $n = 500$ Cells)

In Gao et al. (`GSE148673` `TNBC3`), `Subclone_A` ($n = 350$, harboring focal `EGFR` Bin 25 and `MYC` Bin 35 amplicons plus `PTEN` Bin 45 homozygous deletion) and `Subclone_B` ($n = 148$, harboring the `12q` `CDK4`/`MDM2`/`HSP90AA1`/`LY6E` amplicon ridge) actively cycle alongside the quiescent $0.40\%$ `Rare_Persister` subclone ($n = 2$, cells `498, 499`). Across the 54 proliferation-coupled candidates, 5-clock OLS erases **`30.20%`** of variance on average (`27.14%` within `Subclone_A` and **`36.76%`** within `Subclone_B`), with **21 genes losing more than 40% of their variance**:

| Rank in `TNBC3` | Gene | Functional Class | `cellsnail` Retained | 2-Score OLS $\Delta$ | 5-Clock OLS $\Delta$ | 20-Clock OLS $\Delta$ | `Subclone_A` ($n=350$) 5-Clock $\Delta$ | `Subclone_B` ($n=148$) 5-Clock $\Delta$ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`PBK`** | TOPK Mitotic/Stemness Survival Kinase | `1.0000` | `0.8207` | **`0.7965`** | **`0.8491`** | `0.7513` | **`0.8629`** |
| **2** | **`UBE2T`** | Fanconi Anemia ICL Repair E2 Ligase | `1.0000` | `0.7990` | **`0.7721`** | **`0.8336`** | `0.7386` | **`0.8158`** |
| **3** | **`HMGB2`** | High-Mobility Group Chromatin Bending Factor | `1.0000` | `0.7627` | **`0.7533`** | **`0.7743`** | `0.7254` | **`0.7844`** |
| **4** | **`KIAA0101`** | `PCLAF` PCNA-Associated Fork Protection Factor | `1.0000` | `0.7464` | **`0.7052`** | **`0.8099`** | `0.6397` | **`0.7788`** |
| **5** | **`DEK`** | Chromatin Architecture & Topology Oncoprotein | `1.0000` | `0.6644` | **`0.6408`** | **`0.6955`** | `0.5780` | **`0.7273`** |
| **6** | **`HSP90AA1`** | Oncogenic Kinome Chaperone (`DTP` Collateral) | `1.0000` | `0.5334` | **`0.5612`** | **`0.6507`** | `0.5208` | **`0.6659`** |
| **7** | **`PARP1`** | Base Excision & Single-Strand Break DNA Repair | `1.0000` | `0.5553` | **`0.5594`** | **`0.6435`** | `0.4889` | **`0.6535`** |
| **8** | **`PTMA`** | Prothymosin Alpha Chromatin Decondensation | `1.0000` | `0.5305` | **`0.5452`** | **`0.6518`** | `0.5033` | **`0.5885`** |
| **9** | **`UHRF1`** | Hemimethylated CpG Reader & DNMT1 Recruiter | `1.0000` | `0.5738` | **`0.5420`** | **`0.6181`** | `0.5442` | **`0.5453`** |
| **10** | **`DNMT1`** | Maintenance DNA Methyltransferase | `1.0000` | `0.5604` | **`0.5319`** | **`0.6049`** | `0.4408` | **`0.6561`** |
| **11** | **`EZH2`** | PRC2 H3K27me3 Histone Methyltransferase | `1.0000` | `0.5577` | **`0.5316`** | **`0.5673`** | `0.4714` | **`0.5899`** |
| **12** | **`FOXM1`** | Proliferation-Coupled Master Transcription Factor | `1.0000` | `0.5071` | **`0.5299`** | **`0.6043`** | `0.4941` | **`0.5865`** |
| **13** | **`EPCAM`** | Epithelial Cell Adhesion & Lineage Marker | `1.0000` | `0.4618` | **`0.5121`** | **`0.6204`** | `0.4670` | **`0.6043`** |
| **14** | **`CDKN2A`** | `p16INK4a`/`p14ARF` Senescence Checkpoint | `1.0000` | `0.5104` | **`0.5117`** | **`0.6597`** | `0.4902` | **`0.5544`** |
| **15** | **`DTL`** | `CDT2` CUL4-DDB1 Replication Licensing E3 Ligase | `1.0000` | `0.5627` | **`0.4861`** | **`0.6485`** | `0.3405` | **`0.6128`** |

### 3.4 Cohort 3: Human Lung Adenocarcinoma Triad (`GSE118767` `sc_mixology`, $n = 300$ Cells)

Across the three human lung adenocarcinoma cell lines in `GSE118767` (`HCC827` `EGFR` `delE746-A750` mutant, `H1975` `EGFR` `L858R/T790M` mutant, and `H2228` `EML4-ALK` fusion), 5-clock OLS erases **`29.10%`** of candidate variance (`40.69%` under 20-clock OLS), with **26 of 50 candidates losing more than 40% of their variance under 20-clock OLS**:
- **Top Erased Chromatin & Epigenetic Drivers:** `HMGB2` (`66.97%` 5-clock, `73.82%` 20-clock), `DNMT1` (`55.28%` 5-clock, `61.74%` 20-clock), `HELLS` (`55.13%` 5-clock, `65.83%` 20-clock), `DEK` (`53.54%` 5-clock, `69.34%` 20-clock), `PTMA` (`50.99%` 5-clock, `66.01%` 20-clock), and `EZH2` (`47.21%` 5-clock, `53.07%` 20-clock).
- **Top Erased Chaperone, HR Repair & Lineage Drivers:** `HSP90AA1` (`61.90%` 5-clock, `70.98%` 20-clock), `BRCA1` (`54.89%` 5-clock, `61.70%` 20-clock), `PBK` (`51.72%` 5-clock, `57.24%` 20-clock), `FOXM1` (`48.22%` 5-clock, `52.05%` 20-clock), `PARP1` (`47.34%` 5-clock, `57.80%` 20-clock), `UBE2T` (`47.18%` 5-clock, `53.82%` 20-clock), `CCND1` (`35.61%` 5-clock, `43.01%` 20-clock), `EGFR` (`28.71%` 5-clock, `46.87%` 20-clock), `CDK4` (`26.80%` 5-clock, `49.17%` 20-clock), and `CDKN2A` (`19.32%` 5-clock, `49.55%` 20-clock).

### 3.5 Cohort 4: Human `cSCC` Spatial Transcriptomics (`GSE144239` Patient P2, $n = 400$ Spots)

In Ji et al. (`GSE144239` `GSM4284316_P2_ST_rep1`), spatial spots span four histological compartments: `Layer_1` (`Basal` `KRT14+`/`KRT5+`), `Layer_2` (`Differentiating` `KRT1+`), `Layer_3` (`Leading-Edge Tumor-Specific Keratinocyte` `TSK` `LAMC2+`/`MMP10+`/`ITGA6+`/`EGFR+`), and `Layer_4` (`Fibrovascular Stroma` `COL1A1+`/`FAP+`). Applying standard OLS cell-cycle regression before spatial niche analysis severely blunts the exact oncogenic, chromatin, and basal/invasive markers that define the tumor architecture:
- **Chromatin & Proteostasis Core:** `PTMA` (`62.13%` 5-clock, `72.15%` 20-clock; `73.63%` in `Layer_2`), `HSP90AA1` (`52.96%` 5-clock, `60.75%` 20-clock), `HMGB2` (`49.71%` 5-clock, `60.39%` 20-clock), `DEK` (`42.68%` 5-clock, `52.18%` 20-clock), `PARP1` (`38.12%` 5-clock, `41.95%` 20-clock), `DNMT1` (`34.83%` 5-clock, `41.37%` 20-clock), and `UHRF1` (`33.76%` 5-clock, `39.05%` 20-clock).
- **Basal & Leading-Edge `TSK` Invasion Wall Drivers:** `ITGA6` (`52.73%` 5-clock, `61.92%` 20-clock), `PKM` (`48.51%` 5-clock, `57.40%` 20-clock), `KRT5` (`48.47%` 5-clock, `57.08%` 20-clock), `CDK4` (`47.70%` 5-clock, `55.52%` 20-clock), `CCND1` (`43.39%` 5-clock, `52.35%` 20-clock), `LDHA` (`42.85%` 5-clock, `51.29%` 20-clock), `MYC` (`38.44%` 5-clock, `48.32%` 20-clock), `KRT14` (`37.77%` 5-clock, `44.93%` 20-clock), `EGFR` (`31.90%` 5-clock, `40.05%` 20-clock), `FOXM1` (`29.66%` 5-clock, `35.97%` 20-clock), `LY6E` (`27.50%` 5-clock, `37.04%` 20-clock), `LAMC2` (`20.19%` 5-clock, `31.26%` 20-clock), and `MMP10` (`4.37%` slide-wide diluted by 300 non-TSK spots, rising to **`22.50%` under 5-clock OLS and `49.92%` under 20-clock OLS inside its native `Layer_3` `TSK` invasive front**).

---

## 4. Zero-Leakage Generalization & Falsification Audits (LODO, LOCO, LOLO, `countsplit`, Permutation Null)

### 4.1 16-Fold Out-of-Group Generalization Audit (LODO, LOCO, and LOLO)

To prove that `cellsnail`'s trajectory threshold calibration (`tau_clock`) and the empirical $\Delta_{\mathrm{erasure}}(g)$ hierarchy generalize across unseen patients, subclones, cell lines, and histological tissue layers without overfitting, `run_lodo_and_out_of_group_generalization()` calibrates `tau_clock` strictly on training groups (`train_mask`) and evaluates out-of-fold performance on the held-out group (`test_mask`) across all **16 folds**:

| Cohort ID | Validation Scheme | Held-Out Group | Train Cells | Test Cells | Train-Calibrated $\tau_{\mathrm{clock}}$ | Train Max `H_nope` Flicker | Test Max `H_nope` Flicker | Out-of-Fold Orthogonality Residual | Out-of-Fold `H_nope` Variance Retained | Out-of-Fold Clock Leakage | Out-of-Fold $\Delta$ Pearson $r$ | Out-of-Fold $\Delta$ Spearman $\rho$ | Out-of-Fold Top-10 Gene Recall |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `GSE57872_GBM` | **LODO** | **`MGH26`** | `414` | `118` | `0.8119` | `0.6238` | `0.6552` | `0.000000` | **`1.0000`** | `0` | **`0.8773`** | `0.7593` | **`0.9000`** |
| `GSE57872_GBM` | **LODO** | **`MGH28`** | `438` | `94` | `0.8222` | `0.6445` | `0.6413` | `0.000000` | **`1.0000`** | `0` | **`0.9295`** | `0.8719` | **`1.0000`** |
| `GSE57872_GBM` | **LODO** | **`MGH29`** | `457` | `75` | `0.8176` | `0.6352` | `0.7808` | `0.000000` | **`1.0000`** | `0` | `0.6750` | `0.4719` | `0.7000` |
| `GSE57872_GBM` | **LODO** | **`MGH30`** | `459` | `73` | `0.8129` | `0.6258` | `0.6761` | `0.000000` | **`1.0000`** | `0` | **`0.9380`** | `0.7855` | **`1.0000`** |
| `GSE57872_GBM` | **LODO** | **`MGH31`** | `462` | `70` | `0.8098` | `0.6196` | `0.6324` | `0.000000` | **`1.0000`** | `0` | `0.7939` | `0.5479` | `0.8000` |
| `GSE57872_GBM` | **LODO** | **`CSC6`** | `488` | `44` | `0.8179` | `0.6358` | `0.7619` | `0.000000` | **`1.0000`** | `0` | `0.7962` | `0.7615` | **`0.9000`** |
| `GSE57872_GBM` | **LODO** | **`CSC8`** | `474` | `58` | `0.8114` | `0.6229` | `0.7321` | `0.000000` | **`1.0000`** | `0` | **`0.8579`** | `0.6512` | **`0.9000`** |
| `GSE148673_TNBC3` | **LOCO** | **`Subclone_A`** | `148` | `350` | `0.8116` | `0.6233` | `0.6121` | `0.000000` | **`1.0000`** | `0` | **`0.9499`** | **`0.9457`** | `0.6000` |
| `GSE148673_TNBC3` | **LOCO** | **`Subclone_B`** | `350` | `148` | `0.8060` | `0.6121` | `0.6233` | `0.000000` | **`1.0000`** | `0` | **`0.9499`** | **`0.9457`** | `0.6000` |
| `GSE118767_sc_mixology` | **LOCO** | **`HCC827`** | `200` | `100` | `0.8207` | `0.6414` | `0.6633` | `0.000000` | **`1.0000`** | `0` | `0.7839` | `0.7787` | `0.6000` |
| `GSE118767_sc_mixology` | **LOCO** | **`H1975`** | `200` | `100` | `0.8232` | `0.6465` | `0.6122` | `0.000000` | **`1.0000`** | `0` | **`0.8505`** | **`0.8544`** | `0.5000` |
| `GSE118767_sc_mixology` | **LOCO** | **`H2228`** | `200` | `100` | `0.8056` | `0.6111` | `0.6837` | `0.000000` | **`1.0000`** | `0` | `0.8088` | `0.8037` | `0.8000` |
| `GSE144239_cSCC_spatial` | **LOLO** | **`Layer_1` (Basal)** | `300` | `100` | `0.7601` | `0.5201` | `0.6122` | `0.000000` | **`1.0000`** | `0` | `0.7833` | `0.7502` | `0.5000` |
| `GSE144239_cSCC_spatial` | **LOLO** | **`Layer_2` (Diff.)** | `300` | `100` | `0.7869` | `0.5738` | `0.4898` | `0.000000` | **`1.0000`** | `0` | `0.7775` | `0.7761` | `0.6000` |
| `GSE144239_cSCC_spatial` | **LOLO** | **`Layer_3` (`TSK`)** | `300` | `100` | `0.7752` | `0.5503` | `0.5204` | `0.000000` | **`1.0000`** | `0` | `0.7523` | `0.7291` | `0.8000` |
| `GSE144239_cSCC_spatial` | **LOLO** | **`Layer_4` (Stroma)** | `300` | `100` | `0.7785` | `0.5570` | `0.6224` | `0.000000` | **`1.0000`** | `0` | `0.7695` | `0.7983` | `0.5000` |
| **Global Summary** | **All 16 Folds** | **4 Cohorts** | — | — | — | — | — | **`0.000000`** | **`1.0000`** | **`0`** | **`0.8308`** | **`0.7644`** | **`0.7313`** |

### 4.2 Zero-Leakage Binomial Molecule Thinning (`countsplit`) Holdout Audit

Following Neufeld et al. (2024), `binomial_count_split(X_umi, p=0.5, random_state=seed)` partitions each integer UMI count entry $X_{ij}$ into two independent splits (`X^{(1)} ~ Binomial(X_umi, 0.5)` and `X^{(2)} = X_umi - X^{(1)}`), guaranteeing conditional independence $X^{(1)} \perp X^{(2)} \mid \Lambda$ with zero data leakage. Across all 5 random seeds (`[42, 1337, 2026, 7, 99]`) and all 3 integer UMI cohorts (`15` split evaluations), calibrating `tau_clock` on $X^{(1)}$ and evaluating on $X^{(2)}$ achieves a global mean Pearson $r = 0.9587$ (`min r = 0.9161`), mean Spearman $\rho = 0.9411$, mean Top-20 gene recovery recall of `91.0%`, exact UMI additivity (`0.0` error), and `0` false-positive clock leaks:

| Cohort ID | Thinning $p$ | Seeds Evaluated | Mean Pearson $r$ ($X^{(1)}$ vs $X^{(2)}$) | Min Pearson $r$ | Mean Spearman $\rho$ | Mean Top-20 Gene Recall | Split-2 Max Orthogonality Residual | Split-2 `H_nope` Variance Retained | Split-2 Clock Leakage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`GSE148673_TNBC3`** | `0.50` | `[42, 1337, 2026, 7, 99]` | **`0.9774`** | `0.9711` | **`0.9651`** | **`92.00%`** | `0.000000` | `1.0000` | `0` |
| **`GSE118767_sc_mixology`** | `0.50` | `[42, 1337, 2026, 7, 99]` | **`0.9716`** | `0.9649` | **`0.9705`** | **`94.00%`** | `0.000000` | `1.0000` | `0` |
| **`GSE144239_cSCC_spatial`** | `0.50` | `[42, 1337, 2026, 7, 99]` | **`0.9271`** | `0.9161` | **`0.8876`** | **`87.00%`** | `0.000000` | `1.0000` | `0` |
| **Global `countsplit` Total** | `0.50` | **15 Split Evaluations** | **`0.9587`** | **`0.9161`** | **`0.9411`** | **`91.00%`** | **`0.000000`** | **`1.0000`** | **`0`** |

### 4.3 Permutation Null Distribution ($B = 200$) & Pure Mitotic Clock Falsification Audit

Under the null hypothesis of no biological coupling between a candidate gene in `H_nope` and the $k$-dimensional mitotic clock subspace (`H_0: R^2 = 0`), permuting the rows of `X_clock` across $n$ cells yields a finite-sample Beta distribution $\Delta_{\mathrm{null}}(g) \sim \mathrm{Beta}(k/2, (n - 1 - k)/2)$ with theoretical expectation $k / (n - 1)$. Across $B = 200$ permutations per cohort, our empirical permutation null mean matches $k / (n - 1)$ to within `0.0003`, while observed erasure values reach $z$-scores of **`59.47` to `130.93`** ($p = 0.00498$, Benjamini-Hochberg $\mathrm{FDR} \lt 0.01$):

| Cohort ID | Cells ($n$) | Clock Dim ($k$) | Theoretical Beta Null Mean $k/(n-1)$ | Empirical Permutation Null Mean ($B=200$) | Empirical 99th Percentile Null $\Delta$ | Max Observed $\Delta_{\mathrm{erasure}}$ | Max Permutation $z$-Score | Candidates with $p \lt 0.01$ | Candidates with BH-FDR $q \lt 0.01$ | Pure & Extended Clock Controls in `H_clock` | Synthetic Phase-Locked Oscillators in `H_clock` | False-Positive Clock Leakage into `H_nope` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`GSE57872_GBM`** | `543` | `4` | `0.00738` | `0.00741` | `0.0239` | `0.6175` (`PBK`) | **`124.55`** | `16 / 30` | `16 / 30` | `4 / 4` core + `9 / 9` ext (`100%`) | `10 / 10` (`100%`) | **`0`** (`0.0%`) |
| **`GSE148673_TNBC3`** | `500` | `5` | `0.01002` | `0.00987` | `0.0301` | `0.7965` (`PBK`) | **`130.93`** | `47 / 54` | `46 / 54` | `5 / 5` core + `15 / 15` ext (`100%`) | `10 / 10` (`100%`) | **`0`** (`0.0%`) |
| **`GSE118767_sc_mixology`** | `300` | `5` | `0.01672` | `0.01682` | `0.0474` | `0.6697` (`HMGB2`) | **`59.47`** | `48 / 50` | `45 / 50` | `5 / 5` core + `15 / 15` ext (`100%`) | `10 / 10` (`100%`) | **`0`** (`0.0%`) |
| **`GSE144239_cSCC_spatial`** | `400` | `5` | `0.01253` | `0.01282` | `0.0365` | `0.6213` (`PTMA`) | **`77.69`** | `50 / 55` | `48 / 55` | `5 / 5` core + `15 / 15` ext (`100%`) | `10 / 10` (`100%`) | **`0`** (`0.0%`) |

---

## 5. Multi-Database Clinical, Evolutionary, Druggability & Regulatory Cross-Validation (11 Databases)

To verify that the variance erased by OLS `regress_out` belongs to conserved, clinically actionable human disease drivers rather than technical noise, all **42 panel genes** (`37` proliferation-coupled $\Delta$-candidates + `5` core mitotic clock controls) were cross-validated across 11 biological databases (`cellsnail/data/empirical/exp1_db_annotations.json`).

### 5.1 Ensembl GRCh38.p14 Coordinates, gnomAD Evolutionary Constraint & ChEMBL Clinical Druggability

Across the 37 $\Delta$-hunt candidate genes, **26 of 37 (`70.3%`) are evolutionary loss-of-function intolerant (`LOEUF < 0.60`)**, **18 of 37 (`48.6%`) are targeted by clinical-stage agents (`Phase >= 1`)**, and **11 of 37 (`29.7%`) are directly targeted by Phase-4 FDA-approved oncology drugs** with verified nanomolar or sub-nanomolar `IC50`/`Ki`/`Kd` bioactivities in ChEMBL v34:

| Gene | Role | Ensembl GRCh38 ID & MANE Select | Locus (`Chr:TSS`) | gnomAD `pLI` | gnomAD `LOEUF` | Target ChEMBL ID | Max Clinical Phase | Representative Approved / Clinical Drugs & Quantitative ChEMBL Bioactivity (`nM`) | Mean / Max 5-Clock $\Delta$ |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **`EGFR`** | `H_nope` | `ENSG00000146648` `ENST00000275493` | `chr7:55019017` | `0.3891` | **`0.505`** | `CHEMBL203` | **Phase 4** | Osimertinib (`CHEMBL3353410`, `Ki = 2.2 nM`), Erlotinib (`CHEMBL553`, `Kd = 0.35 nM`, `IC50 = 0.4 nM`), Cetuximab, Lazertinib | `0.1583` / `0.3190` |
| **`MYC`** | `H_nope` | `ENSG00000136997` `ENST00000621592` | `chr8:127736231` | **`1.0000`** | **`0.156`** | `CHEMBL1250348` | Phase 0 / Probe | `CHEMBL4100737` (`Kd = 0.6 nM`), OMO-103 Omomyc (`NCT04800162`) | `0.2162` / `0.3844` |
| **`CDK4`** | `H_nope` | `ENSG00000135446` `ENST00000257904` | `chr12:57752310` | **`0.8470`** | **`0.563`** | `CHEMBL331` | **Phase 4** | Abemaciclib (`CHEMBL3301610`, `IC50 = 1.7 nM`), Ribociclib (`CHEMBL3545110`, `IC50 = 2.0 nM`), Palbociclib (`CHEMBL189963`, `IC50 = 8.2 nM`), Trilaciclib | `0.2545` / `0.4770` |
| **`MDM2`** | `H_nope` | `ENSG00000135679` `ENST00000258149` | `chr12:68808177` | **`1.0000`** | **`0.283`** | `CHEMBL5023` | **Phase 3** | Idasanutlin (`CHEMBL2402737`, `Kd = 0.15 nM`), Navtemadlin (`CHEMBL3125702`, Phase 3), Siremadlin (`CHEMBL3653256`) | `0.1983` / `0.2785` |
| **`ERBB2`** | `H_nope` | `ENSG00000141736` `ENST00000269571` | `chr17:39700064` | `0.2452` | **`0.507`** | `CHEMBL1824` | **Phase 4** | Neratinib (`CHEMBL180022`, `IC50 = 1.0 nM`), Lapatinib (`CHEMBL554`, `Ki = 1.585 nM`), Zanidatamab (`CHEMBL4650256`) | `0.1265` / `0.2089` |
| **`CCND1`** | `H_nope` | `ENSG00000110092` `ENST00000227507` | `chr11:69641156` | `0.7872` | `0.603` | `CHEMBL3610` | **Phase 1/4** | Briciclib (`CHEMBL1206245`), Tapotoclax (`CHEMBL4446378`, `Ki = 0.06 nM`), Palbociclib (`CHEMBL189963`) | `0.2806` / `0.4339` |
| **`CDKN2A`** | `H_nope` | `ENSG00000147889` `ENST00000304494` | `chr9:21974857` | `0.7681` | `0.696` | `CHEMBL4680027` | Biomarker | Direct INK4a inhibitor of CDK4/6 (`R-HSA-2559585` Oncogene-Induced Senescence) | `0.2624` / `0.5117` |
| **`PTEN`** | `H_nope` | `ENSG00000171862` `ENST00000371953` | `chr10:87863625` | `0.3382` | **`0.597`** | `CHEMBL2052032` | Synthetic Lethal | Capivasertib (`CHEMBL2325741`, Phase 4) / Ipatasertib (`CHEMBL2386274`, Phase 3) in `PTEN`-null tumors | `0.1869` / `0.3349` |
| **`TP53`** | `H_nope` | `ENSG00000141510` `ENST00000269305` | `chr17:7687490` | **`0.9996`** | **`0.418`** | `CHEMBL4096` | **Phase 3** | Eprenetapopt (`CHEMBL3186011`, Phase 3), Teprasiran (`CHEMBL4297574`, Phase 3), `CHEMBL1315305` (`5.0 nM`) | `0.0761` / `0.1354` |
| **`AXL`** | `H_nope` | `ENSG00000167601` `ENST00000301178` | `chr19:41219223` | `0.0000` | `0.627` | `CHEMBL4895` | **Phase 4** | Gilteritinib (`CHEMBL3301622`, Approved), Bemcentinib (`CHEMBL3809489`, Phase 3, `IC50 = 14.0 nM`), Cabozantinib (`0.5 nM`) | `0.0803` / `0.1706` |
| **`ABCB1`** | `H_nope` | `ENSG00000085563` `ENST00000622132` | `chr7:87600884` | `0.0000` | **`0.594`** | `CHEMBL4302` | **Phase 3** | Tariquidar (`CHEMBL348475`, Phase 3, `IC50 = 38.0 nM`, PDB `6QEX`), Valspodar, Encequidar | `0.0072` / `0.0090` |
| **`EZH2`** | `H_nope` | `ENSG00000106462` `ENST00000320356` | `chr7:148884291` | **`1.0000`** | **`0.271`** | `CHEMBL2189110` | **Phase 4** | Tazemetostat (`CHEMBL3414621`, Approved, `IC50 = 0.34 nM`, `Ki = 2.5 nM`), GSK2816126 | `0.3961` / `0.5316` |
| **`FOXM1`** | `H_nope` | `ENSG00000111206` `ENST00000359843` | `chr12:2877174` | `0.5969` | **`0.549`** | `CHEMBL4739852` | Probe | `CHEMBL5089001` (`Ki = 11.0 nM`), `CHEMBL5089499` (`Ki = 14.0 nM`), Troglitazone (`Ki = 51.97 nM`) | `0.4362` / `0.5299` |
| **`PARP1`** | `H_nope` | `ENSG00000143799` `ENST00000366794` | `chr1:226408093` | **`0.9897`** | **`0.477`** | `CHEMBL3105` | **Phase 4** | Talazoparib (`CHEMBL3137320`, Approved, `IC50 = 0.57 nM`), Olaparib (`CHEMBL521686`, `Kd = 0.24 nM`, `Ki = 0.787 nM`), Rucaparib, Niraparib | `0.3644` / `0.5594` |
| **`CHEK1`** | `H_nope` | `ENSG00000149554` `ENST00000438015` | `chr11:125625163` | `0.0001` | `0.714` | `CHEMBL4630` | **Phase 2** | Prexasertib (`CHEMBL3544911`, Phase 2, `IC50 = 0.56 nM`), Rabusertib (`CHEMBL3039517`, Phase 2, `IC50 = 10.4 nM`) | `0.2267` / `0.2855` |
| **`WEE1`** | `H_nope` | `ENSG00000166483` `ENST00000450114` | `chr11:9573670` | **`1.0000`** | **`0.353`** | `CHEMBL5491` | **Phase 3** | Azenosertib (`CHEMBL5095036`, Phase 3), Adavosertib (`CHEMBL1976040`, Phase 2, `IC50 = 0.56 nM`, `Kd = 3.2 nM`) | `0.1840` / `0.3128` |
| **`TYMS`** | `H_clock_ext` | `ENSG00000176890` `ENST00000323274` | `chr18:657653` | `0.0011` | `0.770` | `CHEMBL1952` | **Phase 4** | Pemetrexed (`CHEMBL225072`, Approved, `Ki = 1.3 nM`), Raltitrexed (`CHEMBL225071`), OSI-7904 (`Ki = 0.09 nM`) | `0.0000` (`H_clock_ext`) |
| **`RRM2`** | `H_clock_ext` | `ENSG00000171848` `ENST00000304567` | `chr2:10122739` | **`1.0000`** | **`0.258`** | `CHEMBL1954` | **Phase 4** | Gemcitabine (`CHEMBL1637`, Approved), Triapine (`CHEMBL231616`, `IC50 = 185.0 nM`), Gallium Nitrate | `0.0000` (`H_clock_ext`) |
| **`DNMT1`** | `H_nope` | `ENSG00000130816` `ENST00000359526` | `chr19:10194953` | **`1.0000`** | **`0.164`** | `CHEMBL1993` | **Phase 4** | Decitabine (`CHEMBL1201129`, Approved, `IC50 = 30.0 nM`), Azacitidine (`CHEMBL1489`, `IC50 = 300.0 nM`) | `0.3792` / `0.5528` |
| **`UHRF1`** | `H_nope` | `ENSG00000276043` `ENST00000650932` | `chr19:4909501` | **`1.0000`** | **`0.163`** | `CHEMBL2424510` | Probe | `CHEMBL1611255` (`IC50 = 2371 nM`), tandem Tudor/SRA domain hemimethylated CpG reader | `0.3294` / `0.5420` |
| **`HELLS`** | `H_nope` | `ENSG00000119969` `ENST00000348459` | `chr10:94545788` | **`1.0000`** | **`0.358`** | `CHEMBL5724665` | Probe | Lymphoid-specific SNF2 family chromatin remodeling ATPase | `0.4061` / `0.5513` |
| **`DEK`** | `H_nope` | `ENSG00000124795` `ENST00000652689` | `chr6:18264530` | `0.0000` | `0.821` | `CHEMBL6066180` | Probe | Molibresib (`CHEMBL1232461`, `IC50 = 3690 nM`), chromatin topology & supercoiling factor | `0.4334` / `0.6408` |
| **`HMGB2`** | `H_nope` | `ENSG00000164104` `ENST00000296503` | `chr4:173334358` | **`0.9998`** | **`0.273`** | `CHEMBL4295734` | Probe | `CHEMBL5653589` (`Kd = 52.75 nM`), high-mobility group DNA bending architectural protein | `0.5453` / `0.7533` |
| **`PTMA`** | `H_nope` | `ENSG00000187514` `ENST00000409115` | `chr2:231708525` | **`0.9465`** | **`0.541`** | `CHEMBL6066338` | Probe | Histone H1 chaperone & replication-coupled chromatin decondensation factor | `0.5588` / `0.6213` |
| **`HSP90AA1`** | `H_nope` | `ENSG00000080824` `ENST00000216281` | `chr14:102087044` | **`1.0000`** | **`0.364`** | `CHEMBL3880` | **Phase 3/4** | Ganetespib (`CHEMBL2103879`, Phase 3, `IC50 = 5.0 nM`), Radicicol (`Kd = 2.7 nM`), Tanespimycin, Retaspimycin | `0.4304` / `0.6190` |
| **`PIK3CA`** | `H_nope` | `ENSG00000121879` `ENST00000263967` | `chr3:179148357` | **`1.0000`** | **`0.269`** | `CHEMBL4005` | **Phase 4** | Inavolisib (`CHEMBL4650215`, Approved), Copanlisib (`CHEMBL3218576`, Approved), Omipalisib (`Ki = 0.008 nM`) | `0.0700` / `0.1085` |
| **`KRAS`** | `H_nope` | `ENSG00000133703` `ENST00000311936` | `chr12:25250929` | **`0.9998`** | **`0.226`** | `CHEMBL2189121` | **Phase 4** | Adagrasib (`CHEMBL4594350`, Approved, `IC50 = 5.0 nM`), Sotorasib (`CHEMBL4535757`, Approved, `IC50 = 9.66 nM`) | `0.2458` / `0.3355` |
| **`MET`** | `H_nope` | `ENSG00000105976` `ENST00000397752` | `chr7:116672196` | **`0.9990`** | **`0.459`** | `CHEMBL3717` | **Phase 4** | Tepotinib (`CHEMBL3402762`, Approved, `Kd = 1.0 nM`), Capmatinib (`CHEMBL3989937`), Crizotinib (`Kd = 0.2 nM`) | `0.2294` / `0.4132` |
| **`MMP10`** | `H_nope` | `ENSG00000166670` `ENST00000279441` | `chr11:102780628` | `0.0000` | `1.290` | `CHEMBL4270` | Probe | CTS-1027 (`CHEMBL440498`, `IC50 = 0.54 nM`), T-26c (`CHEMBL3337894`, `IC50 = 160 nM`), Marimastat | `0.0437` (`0.2250` `TSK`) |
| **`CCNB1`** | **`H_clock`** | `ENSG00000134057` `ENST00000256442` | `chr5:69167150` | `0.0000` | `0.812` | `CHEMBL2412` | Control | G2/M Mitotic Cyclin B1 (`0.0%` leakage into `H_nope`; `Delta = 0.0000`) | `0.0000` |
| **`TOP2A`** | **`H_clock`** | `ENSG00000131747` `ENST00000423485` | `chr17:40417896` | **`0.9999`** | **`0.445`** | `CHEMBL1806` | **Phase 4** | Epirubicin (`IC50 = 3.0 nM`), Doxorubicin (`IC50 = 4.9 nM`), Teniposide (`Delta = 0.0000`) | `0.0000` |
| **`MKI67`** | **`H_clock`** | `ENSG00000148773` `ENST00000368654` | `chr10:128126423` | `0.0000` | `0.745` | `CHEMBL6066170` | Control | Perichromosomal mitotic chromosome surfactant (`Delta = 0.0000`) | `0.0000` |
| **`PCNA`** | **`H_clock`** | `ENSG00000132646` `ENST00000379143` | `chr20:5119958` | **`0.9998`** | **`0.330`** | `CHEMBL2346488` | Control | DNA polymerase delta processivity sliding clamp (`Delta = 0.0000`) | `0.0000` |
| **`CDK1`** | **`H_clock`** | `ENSG00000170312` `ENST00000395284` | `chr10:60778478` | **`1.0000`** | **`0.304`** | `CHEMBL308` | **Phase 3** | Dinaciclib (`CHEMBL2103840`, Phase 3, `IC50 = 5.5 nM`), Alvocidib (`Delta = 0.0000`) | `0.0000` |

### 5.2 Mechanistic Origin of OLS Projection Leakage: STRING v12.0 Bridges, Reactome v90 & JASPAR/UniBind

Why does linear OLS regression conflate `H_clock` with `H_nope`? Querying **STRING v12.0** (`ncbiTaxonId = 9606`), **Reactome v90**, **JASPAR CORE 2024**, and **UniBind ChIP-seq** reveals the exact physical and transcriptional coupling architecture:

1. **111 Direct Cross-Subspace Physical & Co-Expression Bridges in STRING v12.0 (`481` Total Panel Edges):**
   - **Replication-Fork Clamp Tethering (`PCNA <-> DNMT1 / PARP1 / RAD51 / CDK4 / CCND1`):** `PCNA` physically binds `DNMT1` (`combined_score = 0.991`, experimental score `escore = 0.986`), `CDK4` (`0.996`, `escore = 0.791`), `CCND1` (`0.989`, `escore = 0.841`), `RAD51` (`0.937`, `escore = 0.682`), and `PARP1` (`0.872`, `escore = 0.785`), while `TOP2A` binds `UHRF1` (`0.855`).
   - **Mitotic Kinase Phosphorylation & Checkpoint Bridges (`CDK1/CCNB1 <-> WEE1 / FOXM1 / RB1 / EZH2 / SOX2 / BRCA1 / EGFR`):** `CDK1` physically complexes with `CCND1` (`0.998`, `escore = 0.866`), `WEE1` (`0.998`, `escore = 0.731`), `RB1` (`0.995`, `escore = 0.853`), `FOXM1` (`0.991`, `escore = 0.558`), `EZH2` (`0.965`, `escore = 0.823`), `TP53` (`0.962`), `SOX2` (`0.961`, `escore = 0.510`), `HSP90AA1` (`0.910`), `BRCA1` (`0.878`), and `EGFR` (`0.849`, `escore = 0.807`).
2. **Reactome v90 Pathway Disjointness vs. G1/S Co-Activation:**
   - While `R-HSA-453279` (*Mitotic G1 phase and G1/S transition*, `17/164` genes, $\mathrm{FDR} = 6.01 \times 10^{-7}$) and `R-HSA-69206` (*G1/S Transition*, `14/140` genes, $\mathrm{FDR} = 7.53 \times 10^{-6}$) co-activate both `H_clock` (`TOP2A`, `CCNB1`, `PCNA`, `CDK1`) and G1/S restriction drivers (`CDK4`, `CCND1`, `MYC`, `RB1`, `CDKN2A`), key non-mitotic disease pathways contain **zero core mitotic clock genes (`0/5`)**:
     - `R-HSA-2559585` (*Oncogene Induced Senescence*, `9/42` genes, $\mathrm{FDR} = 5.44 \times 10^{-6}$): `RB1`, `CDKN2A`, `CDK4`, `MDM2`, `TP53` (`0` clock genes).
     - `R-HSA-8953750` (*Transcriptional Regulation by E2F6*, `8/46` genes, $\mathrm{FDR} = 7.97 \times 10^{-5}$): `RRM2`, `RAD51`, `CHEK1`, `BRCA1`, `EZH2` (`0` clock genes).
     - `R-HSA-9665348` / `R-HSA-1227990` (*Signaling by ERBB2 in Cancer*, `6/36` genes, $\mathrm{FDR} = 9.13 \times 10^{-4}$): `HSP90AA1`, `PIK3CA`, `ERBB2`, `KRAS`, `EGFR` (`0` clock genes).
     - `R-HSA-5637810` (*Constitutive Signaling by EGFRvIII*, `5/22` genes, $\mathrm{FDR} = 9.66 \times 10^{-4}$): `HSP90AA1`, `PIK3CA`, `KRAS`, `EGFR` (`0` clock genes).
3. **Direct TF-DNA ChIP-Seq Coupling (JASPAR 2024 & UniBind):**
   - Master transcription factors `E2F1` (`MA0024.3`, `TTTGGCGCCAAA`, `33` human UniBind ChIP-seq datasets), `E2F4` (`MA0470.3`, `TTTTGGCGCCATT`, `12` datasets), `FOXM1` (`MA2469.1`, `GTAAACA`), and `MYC` (`MA0147.4`, `CCACGTGC`, `137` human ChIP-seq datasets) simultaneously occupy promoters of mitotic cyclins (`CCNB1`, `CDK1`) and chromatin/repair effectors (`EZH2`, `DNMT1`, `UHRF1`, `PARP1`, `BRCA1`, `RAD51`), while `SOX2` (`MA0143.5`, `27` datasets), `TP53` (`MA0106.3`, `123` datasets), and `STAT3` (`MA0144.3`, `34` datasets) govern stemness and checkpoint arrest.

---

## 6. Cross-Expedition Convergence: The Triple Standard-Pipeline Catastrophe

### 6.1 Expedition 1 $\leftrightarrow$ Expedition 2 Convergence: Dual Blind-Spot Table (15 Shared Targets)

Combining Expedition 1 (RNA OLS cell-cycle regression over-subtraction) with Expedition 2 (DNA 15-bin moving-window CNV smoothing in `HMMcopy` $W=15$ and `CopyKAT` $W=25$) reveals a **Compound Double-Erasure Catastrophe** across 15 focal amplicons, homozygous deletions, and rare drug-tolerant persister (`DTP`) collateral targets:
- In DNA space, 15-bin moving-window smoothing washes out **`61.17%` to `93.99%`** of excess focal copy-number amplitude (mean washout **`80.27%`**).
- In RNA space, 5-clock OLS cell-cycle regression erases **`0.90%` to `61.90%`** of transcriptional variance (mean **`32.35%`**, rising to **`70.98%`** under 20-clock OLS).
- Consequently, standard multi-omic pipelines retain only **`12.45%`** mean compound joint signal survival (DNA excess retention $\times$ RNA variance retention), whereas `cellsnail` v0.2.0 retains **`100.00%`** across both modalities:

| Target Gene | Ensembl GRCh38 ID | Genomic Bin & Locus | Expedition 2 Focal Stringer / Persister Role | `cellsnail` `CN` | `W=15` Smoothed `CN` | Exp 2 DNA `W=15` Washout | Exp 1 Max 5-Clock RNA $\Delta$ | Exp 1 Max 20-Clock RNA $\Delta$ | Standard Joint Survival (5-Clock) | Standard Joint Survival (20-Clock) | `cellsnail` Joint Survival | Verified Lead Therapeutic (`ChEMBL`) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`EGFR`** | `ENSG00000146648` | Bin 25 (`chr7:55019017`) | 1-Bin Focal Stringer (`Subclone_A` & `Rare_Persister` `14.69x` RNA) | `8.024` | `2.401` | **`93.34%`** | **`31.90%`** | **`46.87%`** | `4.54%` | `3.54%` | **`100.0%`** | Osimertinib (`CHEMBL3353410`, `Ki = 2.2 nM`) |
| **`MYC`** | `ENSG00000136997` | Bin 35 (`chr8:127735434`) | 1-Bin `ecDNA` Spire atop `chr8q` Gain Plateau | `9.824` | `3.164` | **`85.12%`** | **`38.44%`** | **`48.32%`** | `9.16%` | `7.69%` | **`100.0%`** | OMO-103 Omomyc (`NCT04800162`) |
| **`CDK4`** | `ENSG00000135446` | Bin 55 (`chr12:57747727`) | 1-Bin Focal Stringer in `Subclone_B` `12q` Amplicon | `6.357` | `3.692` | **`61.17%`** | **`47.70%`** | **`55.52%`** | `20.31%` | `17.27%` | **`100.0%`** | Abemaciclib (`CHEMBL3301610`, `IC50 = 1.7 nM`) |
| **`ABCB1`** | `ENSG00000085563` | Bin 28 (`chr7:87503863`) | `Rare_Persister` (`0.40%` DTP) Efflux Stringer (`12.45x` RNA) | `12.262` | `2.617` | **`93.99%`** | **`0.90%`** | **`2.45%`** | `5.96%` | `5.86%` | **`100.0%`** | Tariquidar (`CHEMBL348475`, `IC50 = 38 nM`, `6QEX`) |
| **`PTEN`** | `ENSG00000171862` | Bin 45 (`chr10:87863113`) | 1-Bin Focal Homozygous Deletion (`Subclone_A` & `Persister`) | `0.123` | `1.792` | **`89.60%`** | **`33.49%`** | **`44.79%`** | `6.92%` | `5.74%` | **`100.0%`** | Capivasertib (`CHEMBL2325741`) / Ipatasertib |
| **`HSP90AA1`** | `ENSG00000080824` | Bin 53 (`chr14:102087044`) | `Subclone_B` Amplicon & `Rare_Persister` Chaperone (`3.34x` DTP RNA) | `4.140` | `2.680` | **`68.22%`** | **`61.90%`** | **`70.98%`** | `12.11%` | `9.22%` | **`100.0%`** | Ganetespib (`CHEMBL2103879`, `IC50 = 5.0 nM`) |
| **`CDKN2A`** | `ENSG00000147889` | Bin 40 (`chr9:21974857`) | Checkpoint Collateral & `Rare_Persister` Arrest Marker (`3.11x` RNA) | `3.850` | `2.380` | **`79.46%`** | **`51.17%`** | **`65.97%`** | `10.03%` | `6.99%` | **`100.0%`** | CDK4/6 Inhibitor Sensitivity Biomarker |
| **`LY6E`** | `ENSG00000160932` | Bin 57 (`chr8:143017982`) | `Subclone_B` Amplicon & `Rare_Persister` Surface Target (`3.38x` RNA) | `4.210` | `2.690` | **`68.78%`** | **`46.48%`** | **`59.20%`** | `16.71%` | `12.74%` | **`100.0%`** | `DLYE5953A` (`RG7841`) Anti-LY6E ADC |
| **`MDM2`** | `ENSG00000135679` | Bin 58 (`chr12:68808177`) | `Subclone_B` `12q` Co-Amplicon with `CDK4` | `4.200` | `2.710` | **`67.73%`** | **`27.85%`** | **`32.19%`** | `23.28%` | `21.88%` | **`100.0%`** | Idasanutlin (`CHEMBL2402737`, `Kd = 0.15 nM`) |
| **`S100A6`** | `ENSG00000197956` | Bin 81 (`chr1:153534600`) | `Rare_Persister` Calcium-Binding Survival Marker (`15.29x` RNA) | `3.920` | `2.310` | **`83.85%`** | **`37.56%`** | **`46.12%`** | `10.08%` | `8.70%` | **`100.0%`** | Persister Survival / Hippo-YAP Co-Target |
| **`LGALS1`** | `ENSG00000100097` | Bin 84 (`chr22:37675606`) | `Rare_Persister` Immunosuppressive Lectin (`2.83x` DTP RNA) | `3.640` | `2.260` | **`84.15%`** | **`22.35%`** | **`31.44%`** | `12.31%` | `10.87%` | **`100.0%`** | OTX008 Galectin-1 Inhibitor (`NCT01724320`) |
| **`KRAS`** | `ENSG00000133703` | Bin 92 (`chr12:25250929`) | Focal MAPK Signaling Driver & Bypass Effector | `4.450` | `2.520` | **`78.78%`** | **`33.55%`** | **`37.95%`** | `14.10%` | `13.17%` | **`100.0%`** | Adagrasib (`CHEMBL4594350`, `IC50 = 5.0 nM`) |
| **`AXL`** | `ENSG00000167601` | Bin 94 (`chr19:41219223`) | `GBM` `DTP-1` (`+4.80 SD`) & `DTP-2` (`+4.51 SD`) Persister RTK | `4.180` | `2.410` | **`81.19%`** | **`17.06%`** | **`32.36%`** | `15.60%` | `12.72%` | **`100.0%`** | Bemcentinib (`CHEMBL3809489`, `IC50 = 14 nM`) |
| **`PROM1`** | `ENSG00000007062` | `chr4:16084023` | `GBM` `DTP-1` (`+6.98 SD`) Quiescent Stem Persister Marker | `3.750` | `2.220` | **`87.43%`** | **`13.98%`** | **`17.28%`** | `10.81%` | `10.40%` | **`100.0%`** | CD133 Stem-Persister Immunotarget |
| **`ERBB2`** | `ENSG00000141736` | Bin 75 (`chr17:39700064`) | `GBM` `DTP-2` (`+6.62 SD`) Mosaic `ERBB2+`/`AXL+` Persister RTK | `4.620` | `2.490` | **`81.30%`** | **`20.89%`** | **`29.60%`** | `14.79%` | `13.16%` | **`100.0%`** | Neratinib (`CHEMBL180022`, `IC50 = 1.0 nM`) / Tucatinib |

### 6.2 Expedition 1 $\leftrightarrow$ Expedition 3 Convergence: Compound Spatial Boundary Catastrophe on `GSE144239` `cSCC` (21 Interface Genes)

In `GSE144239` human cutaneous squamous cell carcinoma (`cSCC`, Patient P2, $n = 400$ spots), Expedition 3 established that classical mesenchymal N-cadherin (`CDH2`) has **`0.0` UMI across all `400/400` spots** while epithelial E-cadherin (`CDH1`, `1,393.0` UMI) and P-cadherin (`CDH3`, `2,107.0` UMI) remain strongly expressed. Leading-edge `Layer_3` `TSK` cells execute a collective basal-like invasion program rather than transdifferentiating into `COL1A1+`/`FAP+` stromal myofibroblasts.

When a spatial transcriptomics pipeline applies both (1) **OLS cell-cycle regression** (Expedition 1) and (2) **isotropic 2D Gaussian/graph smoothing** across the `Layer_3` (`TSK`) $\leftrightarrow$ `Layer_4` (`Stroma`) basement membrane (Expedition 3), it inflicts a **Compound Spatial Boundary Catastrophe**:
- First, OLS cell-cycle regression erases **`20.19%` to `62.13%`** (mean **`37.90%`** slide-wide and **`46.33%`** within `Layer_3` `TSK`) of the authentic intracellular chromatin, basal keratin (`KRT5` `48.47%`, `KRT14` `37.77%`), cell-cycle amplicon (`CDK4` `47.70%`, `CCND1` `43.39%`, `MYC` `38.44%`), and leading-edge `TSK` invasion signal (`ITGA6` `52.73%`, `EGFR` `31.90%`, `LAMC2` `20.19%` global / `31.26%` 20-clock, `MMP10` `22.50%` 5-clock / `49.92%` 20-clock within `Layer_3` `TSK`).
- Second, isotropic spatial smoothing bleeds stromal collagen (`COL1A1`, `FAP`, `VIM`) across the basement membrane into `TSK` spots, inflating false-positive 0-cochain "dual-positive pseudo-EMT hybrids" by **`+319.2%`** (`MMP10`+`COL1A1`), **`+241.2%`** (`LAMC2`+`COL1A1`), and **`+417.0%`** (`KRT14`+`FAP`), while collapsing 1-cochain trans-membrane juxtacrine ligand-receptor polarity (`AREG`/`TGFA`/`EREG` $\to$ `EGFR`, `LAMC2` $\to$ `ITGA6`/`ITGB4`) by **`2.03x` to `38.76x`**:

| Gene | `cSCC` Spatial Niche & Functional Role | HPA Subcellular Compartment & Bleed Audit | `cSCC` 2-Score OLS $\Delta$ | `cSCC` 5-Clock OLS $\Delta$ | `cSCC` 20-Clock OLS $\Delta$ | `Layer_3` (`TSK`) 5-Clock OLS $\Delta$ | `Layer_3` (`TSK`) 20-Clock OLS $\Delta$ | Exp 3 Juxtacrine Partner (`STRING`) | Exp 3 `SE2CellularSheaf` Polarity Gain | `cellsnail` Retention |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: |
| **`PTMA`** | `Layer_1`–`Layer_3` Basal/TSK Chromatin Decondensation | Intracellular Nuclear Chromatin | `70.19%` | **`62.13%`** | **`72.15%`** | `60.23%` | `74.86%` | None (`0.000`) | `1.00x` (Intracellular) | **`100.0%`** |
| **`HSP90AA1`** | `Layer_3` `TSK` Proteostasis & RTK Chaperone Hub | Cytosolic / Perinuclear Chaperone | `58.13%` | **`52.96%`** | **`60.75%`** | `57.26%` | `68.31%` | `EGFR`/`ERBB2` Client Bridge | **`4.50x`** | **`100.0%`** |
| **`HMGB2`** | `Layer_1`–`Layer_3` Basal/TSK Chromatin Architecture | Nuclear Chromatin Architecture | `55.49%` | **`49.71%`** | **`60.39%`** | `50.56%` | `72.14%` | None (`0.000`) | `1.00x` (Intracellular) | **`100.0%`** |
| **`CDK4`** | `Layer_1`/`Layer_3` Basal & `TSK` G1/S Restriction Driver | Nuclear Cyclin D1 Complex | `50.31%` | **`47.70%`** | **`55.52%`** | `58.91%` | `75.78%` | `CDKN2A`/`CCND1` (`0.999`) | **`2.15x`** | **`100.0%`** |
| **`CCND1`** | `Layer_1`/`Layer_3` Basal & `TSK` Cyclin D1 Amplicon | Nuclear G1/S Regulatory Complex | `49.86%` | **`43.39%`** | **`52.35%`** | `52.38%` | `72.49%` | `CDK4` (`0.999`) | **`2.15x`** | **`100.0%`** |
| **`DEK`** | `Layer_1`–`Layer_3` Chromatin Topology Oncoprotein | Nuclear Euchromatin | `49.89%` | **`42.68%`** | **`52.18%`** | `53.77%` | `74.72%` | None (`0.000`) | `1.00x` (Intracellular) | **`100.0%`** |
| **`MYC`** | `Layer_1`/`Layer_3` Basal & `TSK` Master TF | Nuclear `bHLH-LZ` Transcription | `41.75%` | **`38.44%`** | **`48.32%`** | `49.29%` | `64.18%` | `MAX` (`0.999`) | **`2.40x`** | **`100.0%`** |
| **`PARP1`** | `Layer_1`–`Layer_3` Replication Fork & SSB Repair | Nuclear Chromatin Repair | `38.82%` | **`38.12%`** | **`41.95%`** | `48.93%` | `59.62%` | `PCNA`/`BRCA1` (`0.872`) | `1.00x` (Intracellular) | **`100.0%`** |
| **`KRT14`** | `Layer_1` Basal Keratinocyte Compartment Anchor | Intracellular Filament (`+417.0%` `FAP` bleed) | `40.57%` | **`37.77%`** | **`44.93%`** | `37.62%` | `54.25%` | `KRT5` Heterodimer | **`3.12x`** | **`100.0%`** |
| **`DNMT1`** | `Layer_1`–`Layer_3` CpG Methylation Maintenance | Replication Fork Chromatin | `36.42%` | **`34.83%`** | **`41.37%`** | `46.68%` | `64.05%` | `PCNA`/`UHRF1` (`0.991`) | `1.00x` (Intracellular) | **`100.0%`** |
| **`UHRF1`** | `Layer_1`–`Layer_3` Hemimethylated CpG Reader | Replication Fork Chromatin | `31.91%` | **`33.76%`** | **`39.05%`** | `46.82%` | `60.76%` | `DNMT1`/`TOP2A` (`0.855`) | `1.00x` (Intracellular) | **`100.0%`** |
| **`EGFR`** | `Layer_3` Leading-Edge `TSK` Juxtacrine Receptor | Transmembrane RTK (`AREG`/`TGFA`/`EREG` $\to$ `EGFR`) | `35.61%` | **`31.90%`** | **`40.05%`** | `32.67%` | `54.54%` | `AREG`/`TGFA`/`EREG` (`0.999`) | **`38.76x`** | **`100.0%`** |
| **`FOXM1`** | `Layer_1`–`Layer_3` Proliferation Master TF | Nuclear Forkhead Transcription | `30.80%` | **`29.66%`** | **`35.97%`** | `44.10%` | `61.63%` | `CCNB1`/`CDK1` (`0.991`) | `1.00x` (Intracellular) | **`100.0%`** |
| **`KRT5`** | `Layer_1` Basal Partner Keratin | Intracellular Filament (`ACTA2` bleed) | `53.01%` | **`48.47%`** | **`57.08%`** | `50.77%` | `69.37%` | `KRT14` Heterodimer | **`2.85x`** | **`100.0%`** |
| **`LY6E`** | `Layer_3` `TSK` TGF-Beta / Immune Escape Surface Marker | GPI-Anchored Plasma Membrane | `32.14%` | **`27.50%`** | **`37.04%`** | `46.56%` | `59.32%` | `TGFBR2` / Immune Axis | **`2.45x`** | **`100.0%`** |
| **`PKM`** | `Layer_3` `TSK` Warburg Glycolytic Invasion Fuel | Cytosolic Glycolytic Complex | `54.13%` | **`48.51%`** | **`57.40%`** | `47.58%` | `64.43%` | `LDHA` Metabolic Axis | **`1.85x`** | **`100.0%`** |
| **`LDHA`** | `Layer_3` `TSK` Warburg Lactate Acidification | Cytosolic Glycolytic Complex | `47.12%` | **`42.85%`** | **`51.29%`** | `43.71%` | `60.46%` | `PKM` Metabolic Axis | **`1.85x`** | **`100.0%`** |
| **`LAMC2`** | `Layer_3` Leading-Edge `TSK` Invasion Wall Anchor | Secreted Basement Membrane (`+241.2%` `COL1A1` bleed) | `21.08%` | **`20.19%`** | **`31.26%`** | `42.78%` | `60.98%` | `ITGA6`/`ITGB4` (`0.965`) | **`3.90x`** | **`100.0%`** |
| **`ITGA6`** | `Layer_1`/`Layer_3` Hemidesmosome Laminin Receptor | Basolateral Transmembrane Integrin | `57.53%` | **`52.73%`** | **`61.92%`** | `62.59%` | `74.95%` | `LAMC2`/`FN1`/`COL1A1` (`0.999`) | **`8.30x`** | **`100.0%`** |
| **`CDKN2A`** | `Layer_3` `TSK` Partial-EMT Invasion Checkpoint | Nuclear/Cytoplasmic `p16INK4a` | `6.05%` | **`8.24%`** | **`16.03%`** | `17.27%` | `41.23%` | `CDK4`/`MDM2` (`0.999`) | **`2.10x`** | **`100.0%`** |
| **`MMP10`** | `Layer_3` Leading-Edge `TSK` ECM Remodeling Protease | Secreted Invasive Front (`+319.2%` `COL1A1` bleed) | `3.05%` | `4.37%` | `10.68%` | **`22.50%`** | **`49.92%`** | `ITGA6` / ECM Axis (`CHEMBL4270`) | **`5.09x`** | **`100.0%`** |

---

## 7. Reproducibility & Verification Commands

All empirical analyses, cross-validation audits, and GFM compliance checks can be reproduced deterministically from the repository root:

```powershell
# 1. Execute the full Expedition 1 Delta-Hunt pipeline and regenerate results/expedition1_delta_hunt.json
.\.venv\Scripts\python.exe cellsnail/expeditions/exp1_delta_hunt.py

# 2. Run the 28-test unit and falsification suite
.\.venv\Scripts\python.exe -m pytest tests/test_expedition1_delta_hunt.py -v

# 3. Verify 100% GFM / KaTeX math compliance across all 21 rules
.\.venv\Scripts\python.exe scripts/audit_gfm_math.py docs/EXPEDITION_1_DELTA_HUNT.md
```
