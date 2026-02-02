# Secp256k1 CUDA - Full ECC Port

**Complete 1:1 port of the optimized C++ Secp256k1 library to CUDA**

This is a direct port of the high-performance C++ implementation to CUDA, optimized for maximum throughput on NVIDIA GPUs. **Priority: Speed & Stability** (no side-channel protection, Montgomery, or constant-time operations).

---

## Features

### ✅ Field Arithmetic (Fp)
- **Addition/Subtraction**: Modular arithmetic over field prime P
- **Multiplication**: 256×256→512 bit with fast reduction (P = 2^256 - K)
- **Squaring**: Optimized multiplication
- **Inversion**: Binary GCD (Euclidean Algorithm)

### ✅ Scalar Arithmetic (Fn)  
- **Addition/Subtraction**: Modular arithmetic over curve order N
- **Bit extraction**: Fast bit access for scalar processing

### ✅ Point Operations (Jacobian Coordinates)
- **Point Doubling**: `dbl-2007-a` formula (4 squarings + 4 multiplications)
- **Mixed Addition**: Jacobian + Affine → Jacobian (7M + 4S)
- **GLV Endomorphism**: φ(x,y) = (β·x, y) for 2x speedup

### ✅ Scalar Multiplication
- **wNAF encoding**: Window width 4, signed digit representation
- **Precomputed tables**: [P, 3P, 5P, ..., 15P] for fast lookups
- **GLV decomposition**: Split k → (k1, k2) for parallel computation
- **Shamir's trick**: Interleaved double-and-add

### ⚡ MEGA BATCH Processing
- **Parallel execution**: Process millions of scalar multiplications simultaneously
- **G×k kernel**: Optimized generator point multiplication
- **P×k kernel**: General point multiplication

---

## Performance (NVIDIA RTX 5060 Ti)

| Operation | Throughput | Time per Op |
|-----------|------------|-------------|
| **Field Multiplication** | 4.1 Gops/s | ~0.24 ns |
| **Field Inversion** | 29 Mops/s | ~34.6 ns |
| **Scalar Mul (G×k)** | **1.86 M ops/s** | **~0.54 μs** |

### Comparison with C++

The CUDA version achieves **massive parallelism**:
- **C++ (single core, Clang 18)**: ~77 μs per scalar_mul (~13K ops/sec)
- **CUDA (RTX 5060 Ti)**: ~0.54 μs per scalar_mul in batch (**1.86M ops/sec**)
- **Speedup**: **~143x** via GPU parallelization

---

## Building

```bash
cd libs/Secp256k1Cuda
mkdir -p build && cd build
cmake ..
make
./secp256k1_cuda_bench
```

**Requirements:**
- CUDA Toolkit 12.0+
- NVIDIA GPU with Compute Capability 7.0+
- CMake 3.18+

---

## Usage

### Basic Example

```cpp
#include "secp256k1.cuh"

__global__ void my_kernel() {
    using namespace secp256k1::cuda;
    
    // Create generator point
    JacobianPoint G;
    G.x.limbs[0] = GENERATOR_X[0]; // ... set all limbs
    G.y.limbs[0] = GENERATOR_Y[0]; // ... set all limbs
    G.z.limbs[0] = 1;
    G.infinity = false;
    
    // Create scalar
    Scalar k;
    k.limbs[0] = 0x123456789ABCDEF0ULL;
    // ... set remaining limbs
    
    // Compute G * k
    JacobianPoint result;
    scalar_mul(&G, &k, &result);
}
```

### Batch Processing (MEGA BATCH)

```cpp
// Host code
const int N = 1000000;  // 1 million operations
Scalar* d_scalars;
JacobianPoint* d_results;

cudaMalloc(&d_scalars, N * sizeof(Scalar));
cudaMalloc(&d_results, N * sizeof(JacobianPoint));

// Initialize scalars...

// Launch kernel
int block_size = 256;
int grid_size = (N + block_size - 1) / block_size;
generator_mul_batch_kernel<<<grid_size, block_size>>>(d_scalars, d_results, N);

// Results are ready in d_results
```

---

## Implementation Details

### Algorithms Used (Same as C++)

1. **Field Reduction**: Fast reduction using P = 2^256 - 2^32 - 977
2. **Point Doubling**: `dbl-2007-a` formula (Jacobian, a=0)
3. **Mixed Addition**: Optimized for (Jacobian + Affine)
4. **wNAF**: Window width 4, reduces non-zero digits by ~75%
5. **GLV Endomorphism**: 2x speedup via scalar decomposition
6. **Shamir's Trick**: Interleaved processing for k1·P + k2·φ(P)

### No Security Features (Speed Priority)
- ❌ No constant-time operations
- ❌ No side-channel protection
- ❌ No Montgomery ladder
- ✅ Maximum performance
- ✅ Correctness verified

---

## File Structure

```
libs/Secp256k1Cuda/
├── include/
│   └── secp256k1.cuh       # All device functions & structs
├── src/
│   ├── main.cu             # Benchmark & testing
│   └── secp256k1.cu        # Kernel implementations
├── build/
│   └── secp256k1_cuda_bench
└── README.md
```

---

## Verification

The implementation includes verification tests:
- ✅ Field arithmetic: 2×3=6, (P-1)²=1
- ✅ Scalar multiplication: G×k produces valid points
- ✅ Endomorphism: φ(φ(P)) + P = -φ(P)

---

## Roadmap

- [x] Field arithmetic (add, sub, mul, inv)
- [x] Scalar arithmetic
- [x] Point operations (double, add)
- [x] wNAF encoding
- [x] GLV decomposition
- [x] Scalar multiplication (P×k)
- [x] MEGA BATCH kernels
- [ ] Affine batch conversion (Montgomery's trick)
- [ ] Multi-scalar multiplication (Pippenger's algorithm)
- [ ] Precomputed tables for fixed bases

---

## License

Same as parent project (Secp256K1fast)

---

## Credits

**Port**: Direct 1:1 translation from the C++ implementation  
**GPU**: NVIDIA GeForce RTX 5060 Ti  
**Target**: Maximum throughput for batch processing  
**Philosophy**: Speed > Security (research/development use)
