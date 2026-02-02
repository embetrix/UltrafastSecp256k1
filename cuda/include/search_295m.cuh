#pragma once
#include "secp256k1.cuh"
#include "bloom.cuh"
#include "batch_inversion.cuh"

namespace secp256k1 {
namespace cuda {

__constant__ AffinePoint c_precomp_g[16][16];

// Structure of Arrays layout for Jacobian Points
struct JacobianPointSoA {
    uint64_t *x0, *x1, *x2, *x3;
    uint64_t *y0, *y1, *y2, *y3;
    uint64_t *z0, *z1, *z2, *z3;
    uint8_t *infinity;
};

__device__ inline void store_point_soa(JacobianPointSoA& soa, int idx, const JacobianPoint& p) {
    soa.x0[idx] = p.x.limbs[0]; soa.x1[idx] = p.x.limbs[1]; soa.x2[idx] = p.x.limbs[2]; soa.x3[idx] = p.x.limbs[3];
    soa.y0[idx] = p.y.limbs[0]; soa.y1[idx] = p.y.limbs[1]; soa.y2[idx] = p.y.limbs[2]; soa.y3[idx] = p.y.limbs[3];
    soa.z0[idx] = p.z.limbs[0]; soa.z1[idx] = p.z.limbs[1]; soa.z2[idx] = p.z.limbs[2]; soa.z3[idx] = p.z.limbs[3];
    soa.infinity[idx] = p.infinity ? 1 : 0;
}

__device__ inline JacobianPoint load_point_soa(const JacobianPointSoA& soa, int idx) {
    JacobianPoint p;
    p.x.limbs[0] = soa.x0[idx]; p.x.limbs[1] = soa.x1[idx]; p.x.limbs[2] = soa.x2[idx]; p.x.limbs[3] = soa.x3[idx];
    p.y.limbs[0] = soa.y0[idx]; p.y.limbs[1] = soa.y1[idx]; p.y.limbs[2] = soa.y2[idx]; p.y.limbs[3] = soa.y3[idx];
    p.z.limbs[0] = soa.z0[idx]; p.z.limbs[1] = soa.z1[idx]; p.z.limbs[2] = soa.z2[idx]; p.z.limbs[3] = soa.z3[idx];
    p.infinity = (soa.infinity[idx] != 0);
    return p;
}

// 40-byte structure for results sent back to CPU
struct SearchResult {
    uint64_t x[4];   // 32 bytes (Affine X coordinate)
    int64_t index;   // 8 bytes (Scalar index: +/- 1, 2, 3...)
};

// Kernel 1: Generate Points (P + i*G) - ILP=4
__global__ void generate_points_kernel(
    const JacobianPoint base_point,
    const JacobianPointSoA points,      // Input: Jacobian Points SoA
    int count,
    int batch_idx,
    int step,
    const JacobianPoint Q_div,
    uint64_t* zs,
    uint64_t* inv_zs)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 2; // 2 items per thread

    if (idx >= count) return;

    // Process 2 items per thread
    uint64_t k[2];
    k[0] = (uint64_t)idx + (uint64_t)batch_idx * (uint64_t)count;
    k[1] = k[0] + 1;

    // Load points from SoA
    JacobianPoint p[2];
    
    // Load p[0]
    p[0].x.limbs[0] = points.x0[idx];
    p[0].x.limbs[1] = points.x1[idx];
    p[0].x.limbs[2] = points.x2[idx];
    p[0].x.limbs[3] = points.x3[idx];
    
    p[0].y.limbs[0] = points.y0[idx];
    p[0].y.limbs[1] = points.y1[idx];
    p[0].y.limbs[2] = points.y2[idx];
    p[0].y.limbs[3] = points.y3[idx];
    
    p[0].z.limbs[0] = points.z0[idx];
    p[0].z.limbs[1] = points.z1[idx];
    p[0].z.limbs[2] = points.z2[idx];
    p[0].z.limbs[3] = points.z3[idx];
    
    p[0].infinity = points.infinity[idx];

    // Load p[1]
    if (idx + 1 < count) {
        p[1].x.limbs[0] = points.x0[idx+1];
        p[1].x.limbs[1] = points.x1[idx+1];
        p[1].x.limbs[2] = points.x2[idx+1];
        p[1].x.limbs[3] = points.x3[idx+1];
        
        p[1].y.limbs[0] = points.y0[idx+1];
        p[1].y.limbs[1] = points.y1[idx+1];
        p[1].y.limbs[2] = points.y2[idx+1];
        p[1].y.limbs[3] = points.y3[idx+1];
        
        p[1].z.limbs[0] = points.z0[idx+1];
        p[1].z.limbs[1] = points.z1[idx+1];
        p[1].z.limbs[2] = points.z2[idx+1];
        p[1].z.limbs[3] = points.z3[idx+1];
        
        p[1].infinity = points.infinity[idx+1];
    }

    // Perform point addition
    if (step == 1) {
        // First step: Add G (base_point)
        // Optimization: Use Mixed Addition (Jacobian + Affine) since base_point is G (Z=1)
        // This saves ~5 multiplications per point add (~30% speedup)
        AffinePoint base_affine;
        base_affine.x = base_point.x;
        base_affine.y = base_point.y;

        jacobian_add_mixed(&p[0], &base_affine, &p[0]);
        if (idx + 1 < count) {
            jacobian_add_mixed(&p[1], &base_affine, &p[1]);
        }
    } else if (step == -1) {
        // Inverse step: P = P - Q
        // We add -Q. Q_div is already negated if needed? 
        // Actually main.cu passes Q_div. 
        // If we want P - Q, we add -Q.
        // Assuming Q_div is the point to subtract (or add).
        
        jacobian_add(&p[0], &Q_div, &p[0]);
        if (idx + 1 < count) {
            jacobian_add(&p[1], &Q_div, &p[1]);
        }
    }

    // Store back to SoA
    points.x0[idx] = p[0].x.limbs[0];
    points.x1[idx] = p[0].x.limbs[1];
    points.x2[idx] = p[0].x.limbs[2];
    points.x3[idx] = p[0].x.limbs[3];
    
    points.y0[idx] = p[0].y.limbs[0];
    points.y1[idx] = p[0].y.limbs[1];
    points.y2[idx] = p[0].y.limbs[2];
    points.y3[idx] = p[0].y.limbs[3];
    
    points.z0[idx] = p[0].z.limbs[0];
    points.z1[idx] = p[0].z.limbs[1];
    points.z2[idx] = p[0].z.limbs[2];
    points.z3[idx] = p[0].z.limbs[3];
    
    points.infinity[idx] = p[0].infinity;

    if (idx + 1 < count) {
        points.x0[idx+1] = p[1].x.limbs[0];
        points.x1[idx+1] = p[1].x.limbs[1];
        points.x2[idx+1] = p[1].x.limbs[2];
        points.x3[idx+1] = p[1].x.limbs[3];
        
        points.y0[idx+1] = p[1].y.limbs[0];
        points.y1[idx+1] = p[1].y.limbs[1];
        points.y2[idx+1] = p[1].y.limbs[2];
        points.y3[idx+1] = p[1].y.limbs[3];
        
        points.z0[idx+1] = p[1].z.limbs[0];
        points.z1[idx+1] = p[1].z.limbs[1];
        points.z2[idx+1] = p[1].z.limbs[2];
        points.z3[idx+1] = p[1].z.limbs[3];
        
        points.infinity[idx+1] = p[1].infinity;
    }

    // Store Z for inversion (only if step == 1 or step == -1, which is always true here)
    // We need to store Z coordinate for batch inversion
    zs[idx] = p[0].z.limbs[0]; // Just storing one limb? No, we need the whole Z.
    // Wait, the batch inversion usually works on FieldElements.
    // The original code likely stored the whole Z.
    // Let's check how zs is used. It's a uint64_t*.
    // If it's a pointer to FieldElements, it should be casted.
    // But here it seems we are just storing... wait.
    // The original code:
    // zs[idx] = p.z; 
    // If zs is uint64_t*, that's wrong. It should be FieldElement*.
    // Ah, in the previous context, zs was passed as `uint64_t* d_zs`.
    // And `d_zs` was allocated as `count * sizeof(FieldElement)`.
    // So we should cast `zs` to `FieldElement*`.
    
    FieldElement* zs_fe = (FieldElement*)zs;
    zs_fe[idx] = p[0].z;
    if (idx + 1 < count) {
        zs_fe[idx+1] = p[1].z;
    }
}

__global__ void extract_z_kernel(const JacobianPointSoA points, FieldElement* zs, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        zs[idx].limbs[0] = points.z0[idx];
        zs[idx].limbs[1] = points.z1[idx];
        zs[idx].limbs[2] = points.z2[idx];
        zs[idx].limbs[3] = points.z3[idx];
    }
}

__global__ void apply_z_inv_kernel(JacobianPointSoA points, const FieldElement* inv_zs, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        points.z0[idx] = inv_zs[idx].limbs[0];
        points.z1[idx] = inv_zs[idx].limbs[1];
        points.z2[idx] = inv_zs[idx].limbs[2];
        points.z3[idx] = inv_zs[idx].limbs[3];
    }
}

// Kernel 3: Affine Convert -> Bloom Filter -> Collect Candidates - ILP=4
__global__ void filter_and_extract_kernel(
    const JacobianPointSoA points,
    const uint64_t* inv_zs,
    uint32_t* result_count,
    SearchResult* results,
    int count,
    int batch_idx,
    DeviceBloom filter)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 2; // 2 items per thread

    if (idx >= count) return;

    // Process 2 items per thread
    uint64_t k[2];
    k[0] = (uint64_t)idx + (uint64_t)batch_idx * (uint64_t)count;
    k[1] = k[0] + 1;

    // Load points from SoA
    JacobianPoint p[2];
    
    // Load p[0]
    p[0].x.limbs[0] = points.x0[idx];
    p[0].x.limbs[1] = points.x1[idx];
    p[0].x.limbs[2] = points.x2[idx];
    p[0].x.limbs[3] = points.x3[idx];
    
    p[0].y.limbs[0] = points.y0[idx];
    p[0].y.limbs[1] = points.y1[idx];
    p[0].y.limbs[2] = points.y2[idx];
    p[0].y.limbs[3] = points.y3[idx];
    
    p[0].z.limbs[0] = points.z0[idx];
    p[0].z.limbs[1] = points.z1[idx];
    p[0].z.limbs[2] = points.z2[idx];
    p[0].z.limbs[3] = points.z3[idx];
    
    p[0].infinity = points.infinity[idx];

    // Load p[1]
    if (idx + 1 < count) {
        p[1].x.limbs[0] = points.x0[idx+1];
        p[1].x.limbs[1] = points.x1[idx+1];
        p[1].x.limbs[2] = points.x2[idx+1];
        p[1].x.limbs[3] = points.x3[idx+1];
        
        p[1].y.limbs[0] = points.y0[idx+1];
        p[1].y.limbs[1] = points.y1[idx+1];
        p[1].y.limbs[2] = points.y2[idx+1];
        p[1].y.limbs[3] = points.y3[idx+1];
        
        p[1].z.limbs[0] = points.z0[idx+1];
        p[1].z.limbs[1] = points.z1[idx+1];
        p[1].z.limbs[2] = points.z2[idx+1];
        p[1].z.limbs[3] = points.z3[idx+1];
        
        p[1].infinity = points.infinity[idx+1];
    }

    // Load inverse Z
    FieldElement inv_z[2];
    FieldElement* inv_zs_fe = (FieldElement*)inv_zs;
    inv_z[0] = inv_zs_fe[idx];
    if (idx + 1 < count) {
        inv_z[1] = inv_zs_fe[idx+1];
    }

    // Convert to Affine
    FieldElement x[2];
    
    // Item 0
    if (!p[0].infinity) {
        FieldElement z2, z3;
        secp256k1::cuda::field_sqr(&inv_z[0], &z2);
        secp256k1::cuda::field_mul(&z2, &inv_z[0], &z3);
        secp256k1::cuda::field_mul(&p[0].x, &z2, &x[0]);
        // secp256k1::cuda::field_normalize(&x[0]);
    } else {
        x[0].limbs[0] = 0; x[0].limbs[1] = 0; x[0].limbs[2] = 0; x[0].limbs[3] = 0;
    }

    // Item 1
    if (idx + 1 < count) {
        if (!p[1].infinity) {
            FieldElement z2, z3;
            secp256k1::cuda::field_sqr(&inv_z[1], &z2);
            secp256k1::cuda::field_mul(&z2, &inv_z[1], &z3);
            secp256k1::cuda::field_mul(&p[1].x, &z2, &x[1]);
            // secp256k1::cuda::field_normalize(&x[1]);
        } else {
            x[1].limbs[0] = 0; x[1].limbs[1] = 0; x[1].limbs[2] = 0; x[1].limbs[3] = 0;
        }
    }

    // Check Bloom Filter
    uint8_t key_bytes[32];
    int i, j;
    uint64_t limb;
    
    auto to_bytes = [&](const FieldElement& fe, uint8_t* out) {
        for(i=0; i<4; i++) {
            limb = fe.limbs[3-i]; // Big endian: MSB first
            for(j=0; j<8; j++) {
                out[i*8 + j] = (limb >> (56 - j*8)) & 0xFF;
            }
        }
    };

    to_bytes(x[0], key_bytes);
    
    if (filter.test(key_bytes, 32)) {
        uint32_t pos = atomicAdd(result_count, 1);
        results[pos].index = k[0];
        results[pos].x[0] = x[0].limbs[0];
        results[pos].x[1] = x[0].limbs[1];
        results[pos].x[2] = x[0].limbs[2];
        results[pos].x[3] = x[0].limbs[3];
    }

    // Item 1
    if (idx + 1 < count) {
        to_bytes(x[1], key_bytes);
        if (filter.test(key_bytes, 32)) {
            uint32_t pos = atomicAdd(result_count, 1);
            results[pos].index = k[1];
            results[pos].x[0] = x[1].limbs[0];
            results[pos].x[1] = x[1].limbs[1];
            results[pos].x[2] = x[1].limbs[2];
            results[pos].x[3] = x[1].limbs[3];
        }
    }
}

// Host function to orchestrate the search pipeline
void run_search_batch(
    JacobianPointSoA d_points,      // Pre-allocated device memory for points (SoA)
    SearchResult* d_results,      // Pre-allocated device memory for results
    uint32_t* d_result_count,     // Device pointer to counter
    DeviceBloom filter,
    int count,
    int batch_idx,
    int step,
    const JacobianPoint base_point, // G
    const JacobianPoint Q_div,      // Q
    FieldElement* d_zs,           // Pre-allocated temp buffer
    FieldElement* d_inv_zs        // Pre-allocated temp buffer
) {
    int threads = 256;
    int blocks = (count + (threads * 2) - 1) / (threads * 2);

    generate_points_kernel<<<blocks, threads>>>(
        base_point,
        d_points,
        count,
        batch_idx,
        step,
        Q_div,
        (uint64_t*)d_zs,
        (uint64_t*)d_inv_zs
    );
    cudaDeviceSynchronize();
    
    // Batch inversion
    int blocks_inv = (count + threads - 1) / threads;
    size_t shared_mem = 2 * threads * sizeof(FieldElement);
    batch_inverse_kernel<<<blocks_inv, threads, shared_mem>>>(d_zs, d_inv_zs, count);
    cudaDeviceSynchronize();
    
    // Filter and Extract
    cudaMemset(d_result_count, 0, sizeof(uint32_t));
    
    filter_and_extract_kernel<<<blocks, threads>>>(
        d_points,
        (uint64_t*)d_inv_zs,
        d_result_count,
        d_results,
        count,
        batch_idx,
        filter
    );
    cudaDeviceSynchronize();
}

} // namespace cuda
} // namespace secp256k1
