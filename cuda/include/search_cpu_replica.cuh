#pragma once
#include "secp256k1.cuh"
#include "bloom.cuh"
#include "batch_inversion.cuh"

namespace secp256k1 {
namespace cuda {

// EXACT CPU ALGORITHM REPLICA FOR GPU
// CPU does: Q[i+1] = Q[i] + G, KQ[i+1] = KQ[i] + KG (sequential)
// GPU must parallelize this with proper algorithm

struct SearchResult {
    uint64_t x[4];
    int64_t index;
};

// Step 1: Each thread computes Q_base + tid*G using double-and-add
__global__ void init_Q_points_kernel(
    JacobianPoint Q_base,
    JacobianPoint G,
    JacobianPoint* Q_out,
    int count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    // Compute Q_base + tid*G using binary method
    JacobianPoint result = Q_base;
    JacobianPoint G_acc = G;
    
    uint64_t scalar = tid;
    while (scalar > 0) {
        if (scalar & 1) {
            jacobian_add(&result, &G_acc, &result);
        }
        jacobian_add(&G_acc, &G_acc, &G_acc);  // Double
        scalar >>= 1;
    }
    
    Q_out[tid] = result;
}

// Step 2: Each thread computes KQ_base + tid*KG
__global__ void init_KQ_points_kernel(
    JacobianPoint KQ_base,
    JacobianPoint KG,
    JacobianPoint* KQ_out,
    int count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    JacobianPoint result = KQ_base;
    JacobianPoint KG_acc = KG;
    
    uint64_t scalar = tid;
    while (scalar > 0) {
        if (scalar & 1) {
            jacobian_add(&result, &KG_acc, &result);
        }
        jacobian_add(&KG_acc, &KG_acc, &KG_acc);
        scalar >>= 1;
    }
    
    KQ_out[tid] = result;
}

// Step 3: Extract Z coordinates for batch inversion
__global__ void extract_z_kernel(
    JacobianPoint* KQ_points,
    FieldElement* z_out,
    int count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    z_out[tid] = KQ_points[tid].z;
}

// Step 4: Convert to affine and check Bloom filter
__global__ void affine_and_bloom_kernel(
    JacobianPoint* KQ_points,
    FieldElement* z_inv,
    DeviceBloom bloom,
    SearchResult* results,
    uint32_t* result_count,
    uint64_t batch_offset,
    int count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    // Convert to affine X
    FieldElement z2_inv;
    field_sqr(&z_inv[tid], &z2_inv);
    FieldElement x_affine;
    field_mul(&KQ_points[tid].x, &z2_inv, &x_affine);
    
    // Convert to Little-Endian bytes for Bloom
    uint8_t x_bytes[32];
    for (int i = 0; i < 4; i++) {
        uint64_t limb = x_affine.limbs[i];
        for (int j = 0; j < 8; j++) {
            x_bytes[i*8 + j] = (limb >> (j*8)) & 0xFF;
        }
    }
    
    if (bloom.test(x_bytes, 32)) {
        uint32_t idx = atomicAdd(result_count, 1);
        if (idx < 1024) {
            results[idx].x[0] = x_affine.limbs[0];
            results[idx].x[1] = x_affine.limbs[1];
            results[idx].x[2] = x_affine.limbs[2];
            results[idx].x[3] = x_affine.limbs[3];
            results[idx].index = batch_offset + tid;
        }
    }
}

// Main wrapper function
void run_search_batch(
    JacobianPoint Q_base,
    JacobianPoint KQ_base,
    JacobianPoint G,
    JacobianPoint KG,
    DeviceBloom bloom,
    SearchResult* d_results,
    uint32_t* d_result_count,
    uint64_t batch_offset,
    int batch_size,
    JacobianPoint* d_Q_temp,
    JacobianPoint* d_KQ_temp,
    FieldElement* d_z_temp,
    FieldElement* d_z_inv_temp
) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    
    // Step 1: Initialize Q points
    init_Q_points_kernel<<<blocks, threads>>>(Q_base, G, d_Q_temp, batch_size);
    
    // Step 2: Initialize KQ points
    init_KQ_points_kernel<<<blocks, threads>>>(KQ_base, KG, d_KQ_temp, batch_size);
    
    // Step 3: Extract Z coordinates
    extract_z_kernel<<<blocks, threads>>>(d_KQ_temp, d_z_temp, batch_size);
    
    // Step 4: Batch inversion
    size_t shared_mem = 2 * threads * sizeof(FieldElement);
    batch_inverse_kernel<<<blocks, threads, shared_mem>>>(d_z_temp, d_z_inv_temp, batch_size);
    
    // Step 5: Affine conversion and Bloom check
    affine_and_bloom_kernel<<<blocks, threads>>>(
        d_KQ_temp, d_z_inv_temp, bloom, d_results, d_result_count, batch_offset, batch_size
    );
    
    cudaDeviceSynchronize();
}

} // namespace cuda
} // namespace secp256k1
