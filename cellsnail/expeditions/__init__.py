"""CellSnail v0.3.0 Parallel Discovery & Multi-Expedition Research Program (Expeditions 1–10)."""

from .exp1_delta_hunt import (
    DeltaErasureDiscoveryPipeline,
    run_expedition1_delta_hunt,
)
from .exp2_focal_amplicon_hunt import (
    BlindSpotAmpliconHunter,
    run_expedition2_amplicon_hunt,
)
from .exp3_spatial_border_audit import (
    SpatialBorderAuditor,
    run_expedition3_spatial_border_audit,
)
from .exp4_whole_genome_streamer import (
    WholeGenomeArenaStreamer,
    run_expedition4_streaming_benchmark,
)
from .exp5_whole_genome_delta_census import (
    WholeGenomeDeltaCensusEngine,
    run_expedition5_whole_genome_delta_census,
)
from .exp6_pan_autosome_amplicon_atlas import (
    PanAutosomeAmpliconAtlas,
    PanAutosomeAmpliconAtlasHunter,
    run_expedition6_pan_autosome_amplicon_atlas,
)
from .exp7_whole_slide_spatial_secretome import (
    WholeSlideSpatialSecretomeScanner,
    run_expedition7_whole_slide_spatial_secretome,
)
from . import exp8_clinvar_allosteric_atlas
from .exp8_clinvar_allosteric_atlas import (
    ClinVarAllostericAtlasScanner,
    run_expedition_8,
    run_expedition8_clinvar_allosteric_atlas,
)
from . import exp9_cryoem_sidechain_sharpener
from .exp9_cryoem_sidechain_sharpener import (
    CryoEMSidechainSharpenerScanner,
    run_expedition_9,
    run_expedition9_cryoem_sidechain_sharpener,
)
from . import exp10_sc_to_structure_bridge
from .exp10_sc_to_structure_bridge import (
    SingleCellToStructureBridgeScanner,
    run_expedition_10,
    run_expedition10_sc_to_structure_bridge,
)

__all__ = [
    # Expeditions 1–7
    "DeltaErasureDiscoveryPipeline",
    "run_expedition1_delta_hunt",
    "BlindSpotAmpliconHunter",
    "run_expedition2_amplicon_hunt",
    "SpatialBorderAuditor",
    "run_expedition3_spatial_border_audit",
    "WholeGenomeArenaStreamer",
    "run_expedition4_streaming_benchmark",
    "WholeGenomeDeltaCensusEngine",
    "run_expedition5_whole_genome_delta_census",
    "PanAutosomeAmpliconAtlas",
    "PanAutosomeAmpliconAtlasHunter",
    "run_expedition6_pan_autosome_amplicon_atlas",
    "WholeSlideSpatialSecretomeScanner",
    "run_expedition7_whole_slide_spatial_secretome",
    # Expeditions 8–10 modules
    "exp8_clinvar_allosteric_atlas",
    "exp9_cryoem_sidechain_sharpener",
    "exp10_sc_to_structure_bridge",
    # Expeditions 8–10 runners & classes
    "ClinVarAllostericAtlasScanner",
    "run_expedition_8",
    "run_expedition8_clinvar_allosteric_atlas",
    "CryoEMSidechainSharpenerScanner",
    "run_expedition_9",
    "run_expedition9_cryoem_sidechain_sharpener",
    "SingleCellToStructureBridgeScanner",
    "run_expedition_10",
    "run_expedition10_sc_to_structure_bridge",
]
