/**
 * @file cellsnail_rs.h
 * @brief C-ABI header for Cellsnail #![no_std] Rust SIMD library.
 *
 * Exposes canonical DarkSnail symbols alongside domain-native single-cell
 * bioinformatics aliases and static arena geometry.
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
} CellSnailArena;

/* Arena Lifecycle */
void cellsnail_arena_init(CellSnailArena* arena);
void cellsnail_arena_reset(CellSnailArena* arena);
size_t cellsnail_arena_size_bytes(void);
CellSnailArena* cellsnail_arena_global(void);

/* Pair 1: Morse Gate / Dropout Filter */
void beltrami_morse_gate_f32(const float* in, float* out, float* salience, size_t n, float mad_floor, float tau);
void morse_dropout_filter_f32(const float* in, float* out, float* salience, size_t n, float mad_floor, float tau);

/* Pair 2: Continuity Clutter Filter / Cell-Cycle Deconfounder */
void continuity_clutter_filter_f32(const float* in, float* out, uint8_t* clock_mask, size_t n, size_t t_steps);
void cell_cycle_deconfounder_f32(const float* in, float* out, uint8_t* clock_mask, size_t n, size_t t_steps);

/* Pair 3: Skew Triad Step / Genotype Tiebreaker SO(10) */
void skew_triad_step_f32(const float* u, const float* v, const float* W, float* out, size_t dim);
void genotype_tiebreaker_so10_f32(const float* u, const float* v, const float* W, float* out, size_t dim);

/* Pair 4: Slow-Fast Cascade K16 / Pseudotime Spherical Cascade */
void slow_fast_cascade_k16_f32(const float* v0, const float* u_seq, float gamma, float* out, size_t dim);
void pseudotime_spherical_cascade_f32(const float* v0, const float* u_seq, float gamma, float* out, size_t dim);

/* Pair 5: Contact Sheaf SE(2) Step / Spatial Tissue Sheaf Solver */
void contact_sheaf_se2_step_f32(float dx, float dy, float nx, float ny, float* out_dx, float* out_dy, uint8_t* hit_wall);
void spatial_tissue_sheaf_solver_f32(float dx, float dy, float nx, float ny, float* out_dx, float* out_dy, uint8_t* hit_wall);

/* Pair 6: Kelvin-Mandel SO(6) Rotor / Multi-Omic Covariance Rotor */
void kelvin_mandel_so6_rotor_f32(const float* rot3, const float* cov6, float* rotor6, float* cov6_out);
void multiomic_covariance_rotor_f32(const float* rot3, const float* cov6, float* rotor6, float* cov6_out);

#ifdef __cplusplus
}
#endif

#endif /* CELLSNAIL_RS_H */
