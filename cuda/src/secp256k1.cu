#include "secp256k1.cuh"

namespace secp256k1 {
namespace cuda {

__global__ void field_mul_kernel(const FieldElement* a, const FieldElement* b, FieldElement* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        field_mul(&a[idx], &b[idx], &r[idx]);
    }
}

__global__ void field_add_kernel(const FieldElement* a, const FieldElement* b, FieldElement* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        field_add(&a[idx], &b[idx], &r[idx]);
    }
}

__global__ void field_sub_kernel(const FieldElement* a, const FieldElement* b, FieldElement* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        field_sub(&a[idx], &b[idx], &r[idx]);
    }
}

__global__ void field_inv_kernel(const FieldElement* a, FieldElement* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        field_inv(&a[idx], &r[idx]);
    }
}

// MEGA BATCH: Scalar multiplication P * k
__global__ void scalar_mul_batch_kernel(const JacobianPoint* points, const Scalar* scalars, 
                                         JacobianPoint* results, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        scalar_mul(&points[idx], &scalars[idx], &results[idx]);
    }
}

// Generator multiplication G * k (optimized)
__global__ void generator_mul_batch_kernel(const Scalar* scalars, JacobianPoint* results, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // Compute G * scalar using pre-loaded constant generator (reduces per-thread setup)
        scalar_mul(&GENERATOR_JACOBIAN, &scalars[idx], &results[idx]);
    }
}

__global__ void point_add_kernel(const JacobianPoint* a, const JacobianPoint* b, JacobianPoint* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        jacobian_add(&a[idx], &b[idx], &r[idx]);
    }
}

__global__ void point_dbl_kernel(const JacobianPoint* a, JacobianPoint* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        jacobian_double(&a[idx], &r[idx]);
    }
}

__global__ void hash160_pubkey_kernel(const uint8_t* pubkeys, int pubkey_len, uint8_t* out_hashes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const uint8_t* pk = pubkeys + static_cast<size_t>(idx) * static_cast<size_t>(pubkey_len);
        uint8_t* out = out_hashes + static_cast<size_t>(idx) * 20U;
        hash160_pubkey(pk, static_cast<size_t>(pubkey_len), out);
    }
}

} // namespace cuda
} // namespace secp256k1
