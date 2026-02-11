# UltrafastSecp256k1 API Reference

Complete API documentation for both CPU and CUDA implementations.

---

## Table of Contents

1. [CPU API](#cpu-api)
   - [FieldElement](#fieldelement)
   - [Scalar](#scalar)
   - [Point](#point)
   - [Utility Functions](#utility-functions)
2. [CUDA API](#cuda-api)
   - [Data Structures](#cuda-data-structures)
   - [Field Operations](#cuda-field-operations)
   - [Point Operations](#cuda-point-operations)
   - [Batch Operations](#cuda-batch-operations)
3. [Performance Tips](#performance-tips)
4. [Examples](#examples)

---

## CPU API

**Namespace:** `secp256k1::fast`

**Headers:**
```cpp
#include <secp256k1/field.hpp>
#include <secp256k1/scalar.hpp>
#include <secp256k1/point.hpp>
```

---

### FieldElement

256-bit field element for secp256k1 curve (mod p where p = 2^256 - 2^32 - 977).

#### Construction

```cpp
// Zero element
FieldElement a = FieldElement::zero();

// One element
FieldElement b = FieldElement::one();

// From 64-bit integer
FieldElement c = FieldElement::from_uint64(12345);

// From 4 x 64-bit limbs (little-endian, RECOMMENDED for binary I/O)
std::array<uint64_t, 4> limbs = {0x123, 0x456, 0x789, 0xABC};
FieldElement d = FieldElement::from_limbs(limbs);

// From 32 bytes (big-endian, for hex/test vectors only)
std::array<uint8_t, 32> bytes = {...};
FieldElement e = FieldElement::from_bytes(bytes);

// From hex string (developer-friendly)
FieldElement f = FieldElement::from_hex(
    "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
);
```

#### Arithmetic Operations

```cpp
FieldElement a, b;

// Basic arithmetic (immutable, returns new object)
FieldElement sum = a + b;
FieldElement diff = a - b;
FieldElement prod = a * b;
FieldElement sq = a.square();
FieldElement inv = a.inverse();

// In-place arithmetic (mutable, ~10-15% faster)
a += b;
a -= b;
a *= b;
a.square_inplace();    // a = a²
a.inverse_inplace();   // a = a⁻¹
```

#### Serialization

```cpp
FieldElement a;

// To bytes (big-endian)
std::array<uint8_t, 32> bytes = a.to_bytes();

// To bytes into existing buffer (no allocation)
uint8_t buffer[32];
a.to_bytes_into(buffer);

// To hex string
std::string hex = a.to_hex();

// Access raw limbs (little-endian)
const auto& limbs = a.limbs();  // std::array<uint64_t, 4>
```

#### Comparison

```cpp
FieldElement a, b;
if (a == b) { ... }
if (a != b) { ... }
```

---

### Scalar

256-bit scalar for secp256k1 curve (mod n where n is the group order).

#### Construction

```cpp
// Zero
Scalar a = Scalar::zero();

// One
Scalar b = Scalar::one();

// From 64-bit integer
Scalar c = Scalar::from_uint64(12345);

// From limbs (little-endian)
std::array<uint64_t, 4> limbs = {...};
Scalar d = Scalar::from_limbs(limbs);

// From bytes (big-endian)
std::array<uint8_t, 32> bytes = {...};
Scalar e = Scalar::from_bytes(bytes);

// From hex string
Scalar f = Scalar::from_hex(
    "E9873D79C6D87DC0FB6A5778633389F4453213303DA61F20BD67FC233AA33262"
);
```

#### Arithmetic Operations

```cpp
Scalar a, b;

// Basic arithmetic
Scalar sum = a + b;
Scalar diff = a - b;
Scalar prod = a * b;

// In-place
a += b;
a -= b;
a *= b;
```

#### Utility Methods

```cpp
Scalar s;

// Check if zero
bool isZero = s.is_zero();

// Get specific bit
uint8_t bit = s.bit(index);  // 0 or 1

// NAF encoding (Non-Adjacent Form)
std::vector<int8_t> naf = s.to_naf();

// wNAF encoding (width-w NAF)
std::vector<int8_t> wnaf = s.to_wnaf(4);  // width = 4
```

---

### Point

Elliptic curve point on secp256k1 (internally Jacobian coordinates).

#### Construction

```cpp
// Generator point G
Point G = Point::generator();

// Point at infinity (identity)
Point inf = Point::infinity();

// From affine coordinates
FieldElement x = FieldElement::from_hex("...");
FieldElement y = FieldElement::from_hex("...");
Point p = Point::from_affine(x, y);

// From hex strings (developer-friendly)
Point p2 = Point::from_hex(
    "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
    "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"
);
```

#### Point Operations

```cpp
Point p, q;
Scalar k;

// Addition and doubling
Point sum = p.add(q);       // p + q
Point doubled = p.dbl();    // 2p

// Scalar multiplication
Point result = p.scalar_mul(k);  // k * p

// Negation
Point neg = p.negate();  // -p
```

#### Optimized Scalar Multiplication

```cpp
// For fixed K × variable Q pattern (same K, different Q points):
Scalar K = Scalar::from_hex("...");
KPlan plan = KPlan::from_scalar(K);  // Precompute once

// Then for each Q:
Point Q1 = Point::from_hex("...", "...");
Point Q2 = Point::from_hex("...", "...");

Point R1 = Q1.scalar_mul_with_plan(plan);  // Fastest!
Point R2 = Q2.scalar_mul_with_plan(plan);
```

#### In-Place Operations (Fastest)

```cpp
Point p;

// Increment/decrement by generator
p.next_inplace();    // p += G
p.prev_inplace();    // p -= G

// In-place arithmetic
p.add_inplace(q);    // p += q
p.sub_inplace(q);    // p -= q
p.dbl_inplace();     // p = 2p
p.negate_inplace();  // p = -p

// Mixed addition (when q is affine, z=1)
FieldElement qx, qy;
p.add_mixed_inplace(qx, qy);  // Branchless, ~12% faster
```

#### Serialization

```cpp
Point p;

// Get affine coordinates
FieldElement x = p.x();
FieldElement y = p.y();

// Compressed format (33 bytes: 0x02/0x03 + x)
std::array<uint8_t, 33> compressed = p.to_compressed();

// Uncompressed format (65 bytes: 0x04 + x + y)
std::array<uint8_t, 65> uncompressed = p.to_uncompressed();

// Split x-coordinate for database lookups
std::array<uint8_t, 16> first_half = p.x_first_half();
std::array<uint8_t, 16> second_half = p.x_second_half();
```

#### Properties

```cpp
Point p;

// Check if point at infinity
bool isInf = p.is_infinity();

// Direct Jacobian coordinate access
const FieldElement& X = p.X();  // Jacobian X
const FieldElement& Y = p.Y();  // Jacobian Y
const FieldElement& Z = p.z();  // Jacobian Z
```

---

### Utility Functions

#### Self-Test

```cpp
#include <secp256k1/point.hpp>

// Run correctness tests
bool passed = secp256k1::fast::Selftest(true);  // verbose=true
if (!passed) {
    std::cerr << "Self-test failed!" << std::endl;
}
```

---

## CUDA API

**Namespace:** `secp256k1::cuda`

**Header:**
```cpp
#include <secp256k1.cuh>
```

---

### CUDA Data Structures

```cpp
// Field element (4 × 64-bit limbs, little-endian)
struct FieldElement {
    uint64_t limbs[4];
};

// Scalar (4 × 64-bit limbs)
struct Scalar {
    uint64_t limbs[4];
};

// Jacobian point (X, Y, Z)
struct JacobianPoint {
    FieldElement x;
    FieldElement y;
    FieldElement z;
    bool infinity;
};

// Affine point (x, y)
struct AffinePoint {
    FieldElement x;
    FieldElement y;
};

// 32-bit view for optimized operations (zero-cost conversion)
struct MidFieldElement {
    uint32_t limbs[8];
};
```

---

### CUDA Field Operations

All functions are `__device__` and can only be called from GPU kernels.

```cpp
// Initialization
__device__ void field_set_zero(FieldElement* r);
__device__ void field_set_one(FieldElement* r);

// Comparison
__device__ bool field_is_zero(const FieldElement* a);
__device__ bool field_eq(const FieldElement* a, const FieldElement* b);

// Arithmetic
__device__ void field_add(const FieldElement* a, const FieldElement* b, FieldElement* r);
__device__ void field_sub(const FieldElement* a, const FieldElement* b, FieldElement* r);
__device__ void field_mul(const FieldElement* a, const FieldElement* b, FieldElement* r);
__device__ void field_sqr(const FieldElement* a, FieldElement* r);
__device__ void field_inv(const FieldElement* a, FieldElement* r);
__device__ void field_neg(const FieldElement* a, FieldElement* r);

// Domain conversion (Montgomery mode only)
__device__ void field_to_mont(const FieldElement* a, FieldElement* r);
__device__ void field_from_mont(const FieldElement* a, FieldElement* r);
```

---

### CUDA Point Operations

```cpp
// Initialization
__device__ void jacobian_set_infinity(JacobianPoint* p);
__device__ void jacobian_set_generator(JacobianPoint* p);
__device__ bool jacobian_is_infinity(const JacobianPoint* p);

// Point arithmetic
__device__ void jacobian_double(const JacobianPoint* p, JacobianPoint* r);
__device__ void jacobian_add(const JacobianPoint* p, const JacobianPoint* q, JacobianPoint* r);
__device__ void jacobian_add_mixed(const JacobianPoint* p, const AffinePoint* q, JacobianPoint* r);

// Scalar multiplication
__device__ void scalar_mul(const JacobianPoint* p, const Scalar* k, JacobianPoint* r);
__device__ void scalar_mul_generator(const Scalar* k, JacobianPoint* r);

// Conversion
__device__ void jacobian_to_affine(const JacobianPoint* p, AffinePoint* r);
```

---

### CUDA Batch Operations

```cpp
#include <batch_inversion.cuh>

// Batch field inversion (Montgomery's trick)
// Inverts n field elements using only 1 modular inversion + 3(n-1) multiplications
__device__ void batch_invert(FieldElement* elements, int n, FieldElement* scratch);
```

---

### CUDA Hash Operations

```cpp
#include <hash160.cuh>

// Compute HASH160 = RIPEMD160(SHA256(pubkey))
__device__ void hash160_compressed(const uint8_t pubkey[33], uint8_t hash[20]);
__device__ void hash160_uncompressed(const uint8_t pubkey[65], uint8_t hash[20]);
```

---

## Performance Tips

### CPU

1. **Use in-place operations** when possible:
   ```cpp
   // Slower: creates temporary
   point = point.add(other);
   
   // Faster: no allocation
   point.add_inplace(other);
   ```

2. **Use KPlan for fixed-K multiplication**:
   ```cpp
   // If K is constant and Q varies, precompute K once
   KPlan plan = KPlan::from_scalar(K);
   for (auto& Q : points) {
       result = Q.scalar_mul_with_plan(plan);
   }
   ```

3. **Use `from_limbs` for binary I/O** (not `from_bytes`):
   ```cpp
   // Database/binary files: use from_limbs (native little-endian)
   FieldElement::from_limbs(limbs);
   
   // Hex strings/test vectors: use from_bytes (big-endian)
   FieldElement::from_bytes(bytes);
   ```

### CUDA

1. **Batch operations**: Process thousands of points in parallel
2. **Avoid divergence**: Use branchless algorithms where possible
3. **Memory coalescing**: Align data structures to 32/64 bytes
4. **Use hybrid 32-bit multiplication**: Enabled by default (`SECP256K1_CUDA_USE_HYBRID_MUL=1`)

---

## Examples

### Generate Bitcoin Address (CPU)

```cpp
#include <secp256k1/point.hpp>
#include <secp256k1/scalar.hpp>

using namespace secp256k1::fast;

int main() {
    // Private key (256-bit)
    Scalar private_key = Scalar::from_hex(
        "E9873D79C6D87DC0FB6A5778633389F4453213303DA61F20BD67FC233AA33262"
    );
    
    // Public key = private_key × G
    Point G = Point::generator();
    Point public_key = G.scalar_mul(private_key);
    
    // Get compressed public key (33 bytes)
    auto compressed = public_key.to_compressed();
    
    // Print as hex
    for (auto byte : compressed) {
        printf("%02x", byte);
    }
    printf("\n");
    
    return 0;
}
```

### Batch Point Generation (CUDA)

```cpp
#include <secp256k1.cuh>

using namespace secp256k1::cuda;

__global__ void generate_points_kernel(
    const Scalar* private_keys,
    AffinePoint* public_keys,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    JacobianPoint result;
    scalar_mul_generator(&private_keys[idx], &result);
    jacobian_to_affine(&result, &public_keys[idx]);
}

void generate_points(
    const Scalar* d_private_keys,
    AffinePoint* d_public_keys,
    int count
) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    generate_points_kernel<<<blocks, threads>>>(
        d_private_keys, d_public_keys, count
    );
}
```

### Verify Self-Test (CPU)

```cpp
#include <secp256k1/point.hpp>
#include <iostream>

int main() {
    bool ok = secp256k1::fast::Selftest(true);
    if (ok) {
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } else {
        std::cerr << "TESTS FAILED!" << std::endl;
        return 1;
    }
}
```

---

## Build Configuration Macros

### CPU

| Macro | Default | Description |
|-------|---------|-------------|
| `SECP256K1_USE_ASM` | ON | Enable x64/RISC-V assembly |
| `SECP256K1_RISCV_FAST_REDUCTION` | ON | Fast modular reduction (RISC-V) |
| `SECP256K1_RISCV_USE_VECTOR` | ON | RVV vector extension |

### CUDA

| Macro | Default | Description |
|-------|---------|-------------|
| `SECP256K1_CUDA_USE_HYBRID_MUL` | 1 | 32-bit hybrid multiplication (~10% faster) |
| `SECP256K1_CUDA_USE_MONTGOMERY` | 0 | Montgomery domain arithmetic |
| `SECP256K1_CUDA_LIMBS_32` | 0 | Use 8×32-bit limbs (experimental) |

---

## Platform Support

| Platform | Assembly | SIMD | Status |
|----------|----------|------|--------|
| x86-64 Linux/Windows | BMI2/ADX | AVX2 | ✅ Production |
| RISC-V 64 | RV64GC | RVV 1.0 | ✅ Production |
| CUDA (sm_75+) | PTX | - | ✅ Production |
| ARM64 | - | NEON | 🚧 Planned |
| OpenCL | - | - | 🚧 Planned |

---

## Version

UltrafastSecp256k1 v1.0.0

For more information, see the [README](../README.md) or [GitHub repository](https://github.com/shrec/UltrafastSecp256k1).

