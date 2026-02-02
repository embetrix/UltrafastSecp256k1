#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace secp256k1 {
namespace cuda {

__device__ __forceinline__ uint64_t fast_reduce64_device(uint64_t x, uint64_t range) {
    // Equivalent to: ((unsigned __int128)x * range) >> 64
    return __umul64hi(x, range);
}

// FNV-1a 64-bit hash (Device version)
__device__ inline uint64_t fnv1a64_device(const uint8_t* data, int len) {
    uint64_t h = 1469598103934665603ULL;
    int i;
    for(i=0; i<len; ++i) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// SplitMix64 (Device version)
__device__ inline uint64_t splitmix64_device(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

struct DeviceBloom {
    uint64_t* bitwords; // Pointer to device memory
    uint64_t m_bits;
    uint32_t k;
    uint64_t salt;

    // Check if data is in the filter
    __device__ bool test(const uint8_t* data, int len) const {
        const uint64_t h1 = fnv1a64_device(data, len);
        const uint64_t h2 = splitmix64_device(h1 ^ salt) | 1ULL;
        
        uint64_t idx, w, mask;
        uint32_t i;

        for(i=0; i<k; ++i) {
            idx = fast_reduce64_device(h1 + (uint64_t)i * h2, m_bits);
            w = idx >> 6;
            mask = 1ULL << (idx & 63ULL);
            
            if((bitwords[w] & mask) == 0ULL) return false;
        }
        return true;
    }
    
    // Add data to the filter (Atomic)
    __device__ void add(const uint8_t* data, int len) {
        const uint64_t h1 = fnv1a64_device(data, len);
        const uint64_t h2 = splitmix64_device(h1 ^ salt) | 1ULL;
        
        uint64_t idx, w, mask;
        uint32_t i;

        for(i=0; i<k; ++i) {
            idx = fast_reduce64_device(h1 + (uint64_t)i * h2, m_bits);
            w = idx >> 6;
            mask = 1ULL << (idx & 63ULL);
            
            atomicOr((unsigned long long*)&bitwords[w], (unsigned long long)mask);
        }
    }
};

// Kernel to batch add items to bloom filter
__global__ void bloom_add_kernel(DeviceBloom filter, const uint8_t* data, int item_len, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        filter.add(data + idx * item_len, item_len);
    }
}

// Kernel to batch check items against bloom filter
__global__ void bloom_check_kernel(DeviceBloom filter, const uint8_t* data, int item_len, int count, uint8_t* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        results[idx] = filter.test(data + idx * item_len, item_len) ? 1 : 0;
    }
}

} // namespace cuda
} // namespace secp256k1
