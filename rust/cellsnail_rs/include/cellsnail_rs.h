/**
 * @file cellsnail_rs.h
 * @brief C-ABI header for Cellsnail #![no_std] Rust SIMD library.
 *
 * Exposes 1D chromosomal coordinate filtering C-ABI symbols,
 * genomic coordinate validation, and static arena geometry.
 *
 * License: CC0-1.0
 */

#ifndef CELLSNAIL_RS_H
#define CELLSNAIL_RS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
  #define CELLSNAIL_ALIGN64 __declspec(align(64))
#elif defined(__GNUC__) || defined(__clang__)
  #define CELLSNAIL_ALIGN64 __attribute__((aligned(64)))
#else
  #define CELLSNAIL_ALIGN64
#endif

#define CELLSNAIL_MAX_PIXELS 16384
#define CELLSNAIL_RING_BUFFER_SIZE 131072
#define CELLSNAIL_GENOTYPE_TENSOR_SIZE 160
#define CELLSNAIL_MULTIOMIC_COV_SIZE 36
#define CELLSNAIL_CRYOEM_SLAB_SIZE 32768
#define CELLSNAIL_SE3_TWISTS_SIZE 6144
#define CELLSNAIL_CONTACT_EDGES_SIZE 8192

#define CELLSNAIL_GENOMIC_VALID 0
#define CELLSNAIL_ERR_GENOMIC_NULL_OR_EMPTY -1
#define CELLSNAIL_ERR_GENOMIC_INVERTED_COORDINATES -2
#define CELLSNAIL_ERR_GENOMIC_UNSORTED_CHROMOSOMES -3
#define CELLSNAIL_ERR_GENOMIC_UNSORTED_POSITIONS -4

typedef struct {
    uint32_t chrom_id;
    uint64_t start;
    uint64_t end;
} GenomicInterval;

typedef struct CELLSNAIL_ALIGN64 {
    float spatial_input[16384];
    float spatial_output[16384];
    float salience_map[16384];
    float curl_buffer[16384];
    float ring_buffer[131072];
    float cell_cycle_mask[16384];
    float genotype_tensor[160];
    float trajectory_buffer[16384];
    float sheaf_coboundary[16384];
    float multiomic_cov[36];
    uint8_t _pad[48];
    float cryoem_slab_in[32768];
    float cryoem_slab_out[32768];
    float se3_twists[6144];
    uint32_t contact_edges[8192];
} CellSnailArena;

/* Arena Lifecycle */
void cellsnail_arena_init(CellSnailArena* arena);
void cellsnail_arena_reset(CellSnailArena* arena);
size_t cellsnail_arena_size_bytes(void);
CellSnailArena* cellsnail_arena_global(void);

/* Genomic Coordinate Validation */
int32_t validate_genomic_coordinates_c(const uint32_t* chrom_ids, const uint64_t* starts, const uint64_t* ends, size_t n);
int32_t validate_genomic_intervals_c(const GenomicInterval* intervals, size_t n);

/* 1D Morse Gate & CNV Filtering Kernels */
void morse_gate_1d_f32(const float* in, float* out, float* salience, size_t n, float mad_floor, float tau);
void morse_dropout_filter_f32(const float* in, float* out, float* salience, size_t n, float mad_floor, float tau);

int32_t morse_gate_genomic_f32(const uint32_t* chrom_ids, const uint64_t* starts, const uint64_t* ends, const float* in, float* out, float* salience, size_t n, float mad_floor, float tau);
int32_t morse_gate_genomic_intervals_f32(const GenomicInterval* intervals, const float* in, float* out, float* salience, size_t n, float mad_floor, float tau);

void morse_dropout_filter_adaptive_f32(const float* in, const float* prior, float* out, float* salience, size_t n, float mad_floor, float tau, float lambda_blend);
void morse_plateau_filter_f32(const float* in, float* out, size_t n, size_t window_size);

/* Continuity Clutter Filter / Cell-Cycle Deconfounder */
void continuity_clutter_filter_f32(const float* in, float* out, uint8_t* clock_mask, size_t n, size_t t_steps);
void cell_cycle_deconfounder_f32(const float* in, float* out, uint8_t* clock_mask, size_t n, size_t t_steps);

/* Skew Triad Step / Genotype Tiebreaker SO(10) */
void skew_triad_step_f32(const float* u, const float* v, const float* W, float* out, size_t dim);
void genotype_tiebreaker_so10_f32(const float* u, const float* v, const float* W, float* out, size_t dim);

/* Slow-Fast Cascade K16 / Pseudotime Spherical Cascade */
void slow_fast_cascade_k16_f32(const float* v0, const float* u_seq, float gamma, float* out, size_t dim);
void pseudotime_spherical_cascade_f32(const float* v0, const float* u_seq, float gamma, float* out, size_t dim);

/* Contact Sheaf SE(2) Step / Spatial Tissue Sheaf Solver */
void contact_sheaf_se2_step_f32(float dx, float dy, float nx, float ny, float* out_dx, float* out_dy, uint8_t* hit_wall);
void spatial_tissue_sheaf_solver_f32(float dx, float dy, float nx, float ny, float* out_dx, float* out_dy, uint8_t* hit_wall);
void contact_sheaf_se2_step_adaptive_f32(float dx, float dy, float nx, float ny, float delta_jitter, float huber_delta, float* out_dx, float* out_dy, uint8_t* hit_wall);
void spatial_tissue_sheaf_solver_huber_f32(float dx, float dy, float nx, float ny, float delta_jitter, float huber_delta, float* out_dx, float* out_dy, uint8_t* hit_wall);

/* Kelvin-Mandel SO(6) Rotor / Multi-Omic Covariance Rotor */
void kelvin_mandel_so6_rotor_f32(const float* rot3, const float* cov6, float* rotor6, float* cov6_out);
void multiomic_covariance_rotor_f32(const float* rot3, const float* cov6, float* rotor6, float* cov6_out);
void multiomic_covariance_rotor_shrinkage_f32(const float* rot3, const float* cov6, float alpha_shrink, float* rotor6, float* cov6_out);

/* Cryo-EM 3D Morse Gate / Side-Chain Morse Filter */
void cellsnail_cryoem_morse_3d_f32(const float* in, float* out, float* salience, size_t nx, size_t ny, size_t nz, float mad_floor, float tau);
void cryoem_sidechain_morse_filter(const float* in, float* out, float* salience, size_t nx, size_t ny, size_t nz, float mad_floor, float tau);
void cryoem_sidechain_morse_filter_f32(const float* in, float* out, float* salience, size_t nx, size_t ny, size_t nz, float mad_floor, float tau);

/* Contact Sheaf SE(3) Step / Allosteric Pocket Sheaf Solver */
void cellsnail_contact_sheaf_se3_f32(const float* coords, const uint32_t* edges, size_t num_residues, size_t num_edges, size_t mut_residue, const float* mut_twist, size_t num_iters, float* twists_out, float* eta_out, float* penetration_res);
void allosteric_pocket_sheaf_solver(const float* coords, const uint32_t* edges, size_t num_residues, size_t num_edges, size_t mut_residue, const float* mut_twist, size_t num_iters, float* twists_out, float* eta_out, float* penetration_res);
void allosteric_pocket_sheaf_solver_f32(const float* coords, const uint32_t* edges, size_t num_residues, size_t num_edges, size_t mut_residue, const float* mut_twist, size_t num_iters, float* twists_out, float* eta_out, float* penetration_res);

/* Cayley Ramachandran Integrator / SO(3) Dihedral Cascade */
void cellsnail_cayley_ramachandran_f32(const float* coords_in, const float* u_seq, const float* v_seq, const float* alpha_seq, float* coords_out, size_t num_coords, size_t k_steps, float* rot_out);
void ramachandran_cayley_integrator(const float* coords_in, const float* u_seq, const float* v_seq, const float* alpha_seq, float* coords_out, size_t num_coords, size_t k_steps, float* rot_out);
void ramachandran_cayley_integrator_f32(const float* coords_in, const float* u_seq, const float* v_seq, const float* alpha_seq, float* coords_out, size_t num_coords, size_t k_steps, float* rot_out);

#ifdef __cplusplus
}
#endif

#endif /* CELLSNAIL_RS_H */
