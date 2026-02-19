# UltrafastSecp256k1

The **world's fastest open-source secp256k1** elliptic curve cryptography library — GPU-accelerated ECDSA & Schnorr signatures, multi-platform, zero dependencies.

> **4.88M ECDSA signs/s** · **2.44M ECDSA verifies/s** · **3.66M Schnorr signs/s** · **2.82M Schnorr verifies/s** on a single GPU

[![GitHub stars](https://img.shields.io/github/stars/shrec/UltrafastSecp256k1?style=social)](https://github.com/shrec/UltrafastSecp256k1/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/shrec/UltrafastSecp256k1?style=social)](https://github.com/shrec/UltrafastSecp256k1/network/members)

[![CI](https://img.shields.io/github/actions/workflow/status/shrec/UltrafastSecp256k1/ci.yml?branch=main&label=CI)](https://github.com/shrec/UltrafastSecp256k1/actions/workflows/ci.yml)
[![Benchmark](https://img.shields.io/github/actions/workflow/status/shrec/UltrafastSecp256k1/benchmark.yml?branch=main&label=Bench)](https://shrec.github.io/UltrafastSecp256k1/dev/bench/)
[![Release](https://img.shields.io/github/v/release/shrec/UltrafastSecp256k1?label=Release)](https://github.com/shrec/UltrafastSecp256k1/releases/latest)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)

**Supported Blockchains (secp256k1-based):**

[![Bitcoin](https://img.shields.io/badge/Bitcoin-BTC-F7931A.svg?logo=bitcoin&logoColor=white)](https://bitcoin.org)
[![Ethereum](https://img.shields.io/badge/Ethereum-ETH-3C3C3D.svg?logo=ethereum&logoColor=white)](https://ethereum.org)
[![Litecoin](https://img.shields.io/badge/Litecoin-LTC-A6A9AA.svg?logo=litecoin&logoColor=white)](https://litecoin.org)
[![Dogecoin](https://img.shields.io/badge/Dogecoin-DOGE-C2A633.svg?logo=dogecoin&logoColor=white)](https://dogecoin.com)
[![Bitcoin Cash](https://img.shields.io/badge/Bitcoin%20Cash-BCH-8DC351.svg?logo=bitcoincash&logoColor=white)](https://bitcoincash.org)
[![Zcash](https://img.shields.io/badge/Zcash-ZEC-F4B728.svg)](https://z.cash)
[![Dash](https://img.shields.io/badge/Dash-DASH-008CE7.svg?logo=dash&logoColor=white)](https://dash.org)
[![BNB Chain](https://img.shields.io/badge/BNB%20Chain-BNB-F0B90B.svg?logo=binance&logoColor=white)](https://www.bnbchain.org)
[![Polygon](https://img.shields.io/badge/Polygon-MATIC-8247E5.svg?logo=polygon&logoColor=white)](https://polygon.technology)
[![Avalanche](https://img.shields.io/badge/Avalanche-AVAX-E84142.svg?logo=avalanche&logoColor=white)](https://avax.network)
[![Arbitrum](https://img.shields.io/badge/Arbitrum-ARB-28A0F0.svg)](https://arbitrum.io)
[![Optimism](https://img.shields.io/badge/Optimism-OP-FF0420.svg)](https://optimism.io)
[![+15 more](https://img.shields.io/badge/+15%20more-secp256k1%20coins-grey.svg)](#supported-coins)

**GPU & Platform Support:**

[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCL](https://img.shields.io/badge/OpenCL-3.0-green.svg)](https://www.khronos.org/opencl/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3%2FM4-black.svg?logo=apple)](metal/)
[![Metal](https://img.shields.io/badge/Metal-GPU%20Compute-silver.svg?logo=apple)](metal/)
[![ROCm](https://img.shields.io/badge/ROCm-6.3%20HIP-red.svg)](cuda/README.md)
[![WebAssembly](https://img.shields.io/badge/WebAssembly-Emscripten-purple.svg)](wasm/)
[![ARM64](https://img.shields.io/badge/ARM64-Cortex--A55%2FA76-orange.svg)](https://developer.android.com/ndk)
[![RISC-V](https://img.shields.io/badge/RISC--V-RV64GC-orange.svg)](https://riscv.org/)
[![Android](https://img.shields.io/badge/Android-NDK%20r27-brightgreen.svg)](android/)
[![iOS](https://img.shields.io/badge/iOS-17%2B%20XCFramework-lightgrey.svg)](cmake/ios.toolchain.cmake)
[![ESP32-S3](https://img.shields.io/badge/ESP32--S3-Xtensa%20LX7-orange.svg)](https://www.espressif.com/en/products/socs/esp32-s3)
[![ESP32](https://img.shields.io/badge/ESP32-Xtensa%20LX6-orange.svg)](https://www.espressif.com/en/products/socs/esp32)
[![STM32](https://img.shields.io/badge/STM32-Cortex--M3-orange.svg)](https://www.st.com/en/microcontrollers-microprocessors/stm32f103ze.html)

## ⚠️ Security Notice

**Research & Development Project - Not Audited**

This library has **not undergone independent security audits**. It is provided for research, educational, and experimental purposes.

**Production Use:**
- ❌ Not recommended without independent cryptographic audit
- ❌ No formal security guarantees
- ✅ All self-tests pass (76/76 including all backends)
- ✅ Dual-layer constant-time architecture (FAST + CT always active)
- ✅ Stable C ABI (`ufsecp`) with 45 exported functions

**Reporting Security Issues:**
- Email: [payysoon@gmail.com](mailto:payysoon@gmail.com)
- GitHub Issues: [UltrafastSecp256k1/issues](https://github.com/shrec/UltrafastSecp256k1/issues)

**Disclaimer:**
Users assume all risks. For production cryptographic systems, prefer audited libraries like [libsecp256k1](https://github.com/bitcoin-core/secp256k1).

---

## 🚀 Features

- **Multi-Platform Architecture**
  - CPU: Optimized for x86-64 (BMI2/ADX), RISC-V (RV64GC), and ARM64 (MUL/UMULH)
  - Mobile: Android ARM64 (NDK r27, Clang 18) + iOS 17+ (XCFramework, SPM, CocoaPods)
  - WebAssembly: Emscripten ES6 module with TypeScript declarations
  - Embedded: ESP32-S3 (Xtensa LX7) + ESP32-PICO-D4 (Xtensa LX6) + STM32F103 (ARM Cortex-M3)
  - GPU/CUDA: Batch ECDSA sign 4.88M/s, verify 2.44M/s, Schnorr sign 3.66M/s, verify 2.82M/s
  - GPU/Metal: Apple Silicon (M1/M2/M3/M4) with Comba-accelerated field arithmetic
  - GPU/ROCm (HIP): Portable PTX→__int128 fallbacks for AMD GPUs
  - GPU/OpenCL: PTX inline asm, 3.39M kG/s

- **Performance**
  - x86-64: 3-5× speedup with BMI2/ADX assembly
  - ARM64: ~5× speedup with MUL/UMULH inline assembly
  - RISC-V: 2-3× speedup with native assembly
  - CUDA: Batch ECDSA & Schnorr signatures at millions/second

- **Features**
  - Complete secp256k1 field and scalar arithmetic
  - Point addition, doubling, and multiplication
  - GLV endomorphism optimization
  - Efficient batch operations
  - ECDSA sign/verify (RFC 6979 deterministic nonce, low-S)
  - Schnorr BIP-340 sign/verify
  - SHA-256 hashing
  - Constant-time (CT) layer for side-channel resistance
  - Public key derivation

### Feature Coverage (v3.4.0)

| Category | Component | Status |
|----------|-----------|--------|
| **Core** | Field, Scalar, Point, GLV, Precompute | ✅ |
| **Assembly** | x64 MASM/GAS, BMI2/ADX, RISC-V | ✅ |
| **SIMD** | AVX2/AVX-512 batch ops, Montgomery batch inverse | ✅ |
| **CT** | Constant-time field/scalar/point | ✅ |
| **ECDSA** | Sign/Verify, RFC 6979, DER/Compact, low-S | ✅ |
| **Schnorr** | BIP-340 sign/verify | ✅ |
| **Recovery** | ECDSA pubkey recovery (recid) | ✅ |
| **ECDH** | Key exchange (raw, xonly, SHA-256) | ✅ |
| **Multi-scalar** | Strauss/Shamir | ✅ |
| **Batch verify** | ECDSA + Schnorr batch | ✅ |
| **BIP-32** | HD derivation, path parsing, xprv/xpub | ✅ |
| **MuSig2** | BIP-327, key aggregation, 2-round | ✅ |
| **Taproot** | BIP-341/342, tweak, Merkle | ✅ |
| **Pedersen** | Commitments, homomorphic, switch | ✅ |
| **FROST** | Threshold signatures, t-of-n | ✅ |
| **Adaptor** | Schnorr + ECDSA adaptor sigs | ✅ |
| **Address** | P2PKH, P2WPKH, P2TR, Base58, Bech32/m | ✅ |
| **Silent Pay** | BIP-352 | ✅ |
| **Hashing** | SHA-256 (SHA-NI), SHA-512, HMAC, Keccak-256 | ✅ |
| **Coins** | 27 coins, auto-dispatch, EIP-55 | ✅ |
| **Custom G** | CurveContext, custom generator/curve | ✅ |
| **BIP-44** | Coin-type HD, auto-purpose | ✅ |
| **C ABI** | `ufsecp` stable FFI (45 exports, C/C#/Python/Go/…) | ✅ |
| **Self-test** | Known vector verification | ✅ |
| **GPU** | CUDA, Metal, OpenCL, ROCm kernels | ✅ |
| **Platforms** | x64, ARM64, RISC-V, ESP32, WASM, iOS, Android, Metal, ROCm | ✅ |

## � Batch Modular Inverse (Montgomery Trick)

All backends include **batch modular inversion** — a critical building block for Jacobian→Affine conversion and high-throughput point operations:

| Backend | File | Function(s) |
|---------|------|-------------|
| **CPU** | `cpu/src/field.cpp` | `fe_batch_inverse(FieldElement*, size_t)` — Montgomery trick with scratch buffer |
| **CPU** | `cpu/src/precompute.cpp` | `batch_inverse(std::vector<FieldElement>&)` — vector variant |
| **CUDA** | `cuda/include/batch_inversion.cuh` | `batch_inverse_montgomery` — GPU Montgomery trick kernel |
| **CUDA** | `cuda/include/batch_inversion.cuh` | `batch_inverse_fermat` — Fermat's little theorem variant |
| **CUDA** | `cuda/include/batch_inversion.cuh` | `batch_inverse_kernel` — production kernel (`__launch_bounds__(256, 4)`) |
| **CUDA** | `cuda/src/test_suite.cu` | `fe_batch_inverse()` — host wrapper + unit tests |
| **Metal** | `metal/shaders/secp256k1_kernels.metal` | `batch_inverse` — chunked Montgomery inverse (parallel threadgroups) |

**Algorithm**: Montgomery batch inverse computes N field inversions using only **1 modular inversion + 3(N−1) multiplications**, amortizing the expensive inversion across the entire batch.
## ⚡ Mixed Addition (Jacobian + Affine)

The library provides **branchless mixed addition** (`add_mixed_inplace`) — the fastest way to add a point with known affine coordinates (Z=1) to a Jacobian point. Uses the **madd-2007-bl** formula (7M + 4S, vs 11M + 5S for full Jacobian add).

| Backend | File | Function |
|---------|------|----------|
| **CPU** | `cpu/src/point.cpp` | `jacobian_add_mixed(JacobianPoint&, AffinePoint&)` |
| **CPU** | `cpu/src/point.cpp` | `Point::add_mixed_inplace(FieldElement&, FieldElement&)` |
| **CPU** | `cpu/src/point.cpp` | `Point::sub_mixed_inplace(FieldElement&, FieldElement&)` |
| **CPU** | `cpu/src/precompute.cpp` | `jacobian_add_mixed_local(JacobianPoint&, AffinePointPacked&)` |
| **OpenCL** | `opencl/kernels/secp256k1_point.cl` | `point_add_mixed_impl(JacobianPoint*, AffinePoint*)` |
| **Metal** | `metal/shaders/secp256k1_point.h` | `jacobian_add_mixed(JacobianPoint&, AffinePoint&)` |

### Usage Example (CPU)

```cpp
#include <secp256k1/point.hpp>

using namespace secp256k1::fast;

// Start with generator point G
Point P = Point::generator();

// Get affine coordinates of G for mixed addition
FieldElement gx = P.x();
FieldElement gy = P.y();

// Compute 2G using mixed add (Jacobian + Affine, 7M + 4S)
Point Q = Point::generator();
Q.add_mixed_inplace(gx, gy);  // Q = G + G = 2G

// Subtraction variant: Q = Q - G
Q.sub_mixed_inplace(gx, gy);  // Q = 2G - G = G

// Batch walk: P, P+G, P+2G, ... using repeated mixed add
Point walker = P;
for (int i = 0; i < 1000; ++i) {
    walker.add_mixed_inplace(gx, gy);  // walker += G each step
    // ... process walker ...
}
```
### Mixed Add + Batch Inverse: Collecting Z Values for Cheap Jacobian→Affine

During serial mixed additions, each point accumulates a growing Z coordinate.
To extract affine X for comparison, you need Z⁻² — which requires an expensive modular inversion.
**Solution**: Collect Z values in a batch, then invert them all at once with Montgomery trick (1 inversion + 3N multiplications instead of N inversions).

```cpp
#include <secp256k1/point.hpp>
#include <secp256k1/field.hpp>

using namespace secp256k1::fast;

constexpr size_t BATCH_SIZE = 1024;

// Buffers (allocate once, reuse)
Point batch_points[BATCH_SIZE];
FieldElement batch_z[BATCH_SIZE];

// Start from some point P
Point walker = Point::generator();
FieldElement gx = walker.x();
FieldElement gy = walker.y();

size_t idx = 0;

for (uint64_t j = 0; j < total_count; ++j) {
    // Save point and its Z coordinate
    batch_points[idx] = walker;
    batch_z[idx] = walker.z();
    idx++;

    // Advance walker using mixed add (7M + 4S)
    walker.add_mixed_inplace(gx, gy);

    // When batch is full — do batch inversion
    if (idx == BATCH_SIZE) {
        // ONE modular inversion for 1024 points!
        fe_batch_inverse(batch_z.data(), idx);

        // Now batch_z[i] contains Z_i^(-1)
        for (size_t i = 0; i < idx; ++i) {
            FieldElement z_inv_sq = batch_z[i].square();         // Z^(-2)
            FieldElement x_affine = batch_points[i].X() * z_inv_sq;  // X_affine = X_jac * Z^(-2)
            // Use x_affine as needed
        }
        idx = 0;  // Reset batch
    }
}
```

**Performance**: For N=1024 batch, this is **~500× cheaper** than individual inversions. A single field inversion costs ~3.5μs (Fermat), while batch amortizes to ~7ns per element.

### GPU Pattern: H-Product Serial Inversion (`jacobian_add_mixed_h`)

Production GPU apps use a more memory-efficient variant: instead of storing full Z coordinates,
`jacobian_add_mixed_h` returns **H = U2 − X1** separately from each addition. Since Z_{k} = Z_0 · H_0 · H_1 · … · H_{k-1},
we can reconstruct and invert the entire Z chain from just the H values + initial Z_0.

**Step 1 — Collect H values during serial additions** (CUDA kernel):
```cuda
// jacobian_add_mixed_h: madd-2004-hmv (8M+3S), outputs H separately
// H = U2 - X1, and internally computes Z3 = Z1 * H
__device__ void jacobian_add_mixed_h(
    const JacobianPoint* p, const AffinePoint* q,
    JacobianPoint* r, FieldElement& h_out);

// --- Step kernel: add G repeatedly, save X and H at each slot ---
FieldElement h;
win_z0[tid] = P.z;                    // Save initial Z_0

for (int slot = 0; slot < batch_interval; ++slot) {
    win_x[tid + slot * stride] = P.x; // Save Jacobian X
    jacobian_add_mixed_h(&P, &G, &P, h);
    win_h[tid + slot * stride] = h;   // Save H (not Z!)
}
```

**Step 2 — Serial Z chain inversion** (1 Fermat inversion per thread):
```cuda
// Forward: reconstruct Z_final = Z_0 * H_0 * H_1 * ... * H_{N-1}
FieldElement z_current = z0_values[tid];
for (int slot = 0; slot < batch_interval; ++slot) {
    z_current = z_current * h_array[tid + slot * stride];
}

// ONE inversion of Z_final (Fermat: 255 sqr + 16 mul)
FieldElement z_inv = field_inverse(z_current);

// Backward: unwind to get Z_slot^{-2} at each position
for (int slot = batch_interval - 1; slot >= 0; --slot) {
    int idx = tid + slot * stride;
    z_inv = z_inv * h_array[idx];     // Z_{slot}^{-1}
    h_array[idx] = z_inv * z_inv;     // Z_{slot}^{-2} (overwrite H in-place!)
}
```

**Step 3 — Affine X extraction**:
```cuda
// h_array now contains Z^{-2} at each slot
for (int slot = 0; slot < batch_interval; ++slot) {
    int idx = tid + slot * stride;
    FieldElement x_affine = win_x[idx] * h_array[idx];  // X_jac * Z^{-2}
    // Use x_affine as needed
}
```

**Why H instead of Z?**
- **Memory**: H is a single field element; Z would also be a field element, but H is computed "for free" inside the addition — no extra multiply needed
- **Serial inversion**: Z_k = Z_0 · ∏H_i, so the backward sweep naturally yields Z_k^{-1} at each step using just the stored H values
- **In-place**: H array is overwritten with Z^{-2} — zero extra memory allocation
- **Cost**: 1 Fermat inversion + 2N multiplications per thread (vs N Fermat inversions naively)

> See production usage: `apps/secp256k1_search_gpu_only/gpu_only.cu` (step kernel) + `unified_split.cuh` (batch inversion kernel)

### Other Batch Inverse Use Cases

#### 1. Full Point Conversion: Jacobian → Affine (X + Y)

When you need both X and Y (precompute table, serialization, debugging):

```cpp
// N Jacobian points → N Affine points (1 inversion)
FieldElement z_values[N];
for (size_t i = 0; i < N; ++i)
    z_values[i] = points[i].z();

fe_batch_inverse(z_values.data(), N);  // z_values[i] = Z_i^(-1)

for (size_t i = 0; i < N; ++i) {
    FieldElement z_inv = z_values[i];
    FieldElement z2 = z_inv.square();          // Z^(-2)
    FieldElement z3 = z2 * z_inv;              // Z^(-3)
    affine_x[i] = points[i].X() * z2;         // X_affine = X_jac · Z^(-2)
    affine_y[i] = points[i].Y() * z3;         // Y_affine = Y_jac · Z^(-3)
}
```

#### 2. X-Only Coordinate Extraction

In most cases you don't need Y — only the affine X coordinate is required:

```cpp
// CPU pattern
constexpr size_t BATCH_SIZE = 1024;
Point batch_points[BATCH_SIZE];
FieldElement batch_z[BATCH_SIZE];
size_t batch_idx = 0;

for (uint64_t j = start; j < end; ++j) {
    batch_points[batch_idx] = p;
    batch_z[batch_idx] = p.z();
    batch_idx++;
    p.next_inplace();

    if (batch_idx == BATCH_SIZE || j == end - 1) {
        fe_batch_inverse(batch_z.data(), batch_idx);  // 1 inversion!

        for (size_t i = 0; i < batch_idx; ++i) {
            FieldElement z_inv_sq = batch_z[i].square();           // Z^(-2)
            FieldElement x_affine = batch_points[i].X() * z_inv_sq;  // X only!
            // Use x_affine as needed
        }
        batch_idx = 0;
    }
}
```

#### 3. CUDA: Z Extraction → batch_inverse_kernel → Affine X

On GPU where you have an array of `JacobianPoint` — Z coordinates are extracted separately, inversion uses shared memory:

```cuda
// Step 1: Extract Z coordinates (1 kernel)
__global__ void extract_z_kernel(const JacobianPoint* points,
                                 FieldElement* zs, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) zs[idx] = points[idx].z;
}

// Step 2: Montgomery batch inverse (shared memory prefix/suffix scan)
//         1 inversion per block, inner elements use multiplications only
batch_inverse_kernel<<<blocks, 256, shared_mem>>>(d_zs, d_inv_zs, N);

// Step 3: Affine X = X_jac * Z_inv²
__global__ void affine_extraction_kernel(const JacobianPoint* points,
                                         const FieldElement* inv_zs, ...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    FieldElement z_inv = inv_zs[idx];
    FieldElement z2;
    field_sqr(&z_inv, &z2);           // Z^(-2)
    FieldElement x_aff;
    field_mul(&points[idx].x, &z2, &x_aff);  // X_affine
    // Use x_aff as needed
}
```

#### 4. Batch Modular Division: a[i] / b[i]

Arbitrary batch division for field elements:

```cpp
FieldElement denominators[] = {b0, b1, b2, b3};
fe_batch_inverse(denominators, 4);
// denominators[i] = b_i^(-1)
FieldElement r0 = a0 * denominators[0];  // a0 / b0
FieldElement r1 = a1 * denominators[1];  // a1 / b1
FieldElement r2 = a2 * denominators[2];  // a2 / b2
FieldElement r3 = a3 * denominators[3];  // a3 / b3
```

#### 5. Scratch Buffer Reuse

When processing multiple rounds, a single pre-allocated scratch buffer is reused across all rounds:

```cpp
std::vector<FieldElement> scratch;
scratch.reserve(BATCH_SIZE);  // Allocate once

for (int round = 0; round < total_rounds; ++round) {
    // ... fill batch_z[] ...
    fe_batch_inverse(batch_z.data(), N, scratch);  // Reuses scratch buffer
    // ... affine conversion ...
}
```

### Montgomery Trick — Full Algorithm Explanation

```
Input: [a₀, a₁, a₂, ..., aₙ₋₁]

1) Forward pass — cumulative products:
   prod[0] = a₀
   prod[1] = a₀ · a₁
   prod[2] = a₀ · a₁ · a₂
   ...
   prod[N-1] = a₀ · a₁ · ... · aₙ₋₁

2) Single inversion:
   inv = prod[N-1]⁻¹ = (a₀ · a₁ · ... · aₙ₋₁)⁻¹

3) Backward pass — extract individual inverses:
   aₙ₋₁⁻¹ = inv · prod[N-2]
   inv ← inv · aₙ₋₁(original)
   aₙ₋₂⁻¹ = inv · prod[N-3]
   inv ← inv · aₙ₋₂(original)
   ...
   a₀⁻¹ = inv

Cost: 1 inversion + 3(N-1) multiplications
N=1024: 1×3.5μs + 3069×5ns ≈ 18.8μs (vs 1024×3.5μs = 3584μs → 190× faster!)
```

## �📦 Use Cases

> ### ⚠️ Testers Wanted
> We need community testers for platforms we cannot fully validate in CI:
> - **iOS** — Build & run on real iPhone/iPad hardware with Xcode
> - **AMD GPU (ROCm/HIP)** — Test on AMD Radeon RX / Instinct GPUs
>
> If you can help, please [open an issue](https://github.com/shrec/UltrafastSecp256k1/issues) with your results!

- **Cryptocurrency Applications**
  - Bitcoin/Ethereum address generation
  - Transaction signing and verification
  - Hardware wallet integration
  - Bulk address validation

- **Cryptographic Research**
  - ECC algorithm testing
  - Performance benchmarking
  - Custom curve implementations

- **General Purpose**
  - Any application requiring secp256k1 operations
  - High-throughput cryptographic services
  - Embedded systems (RISC-V support)

## 🔐 Security Model

UltrafastSecp256k1 is a performance-focused secp256k1 engine with two security profiles.
See [THREAT_MODEL.md](THREAT_MODEL.md) for a full layer-by-layer risk assessment.

⚠️ **Constant-time behavior is NOT guaranteed unless you use the `ct::` namespace.**

### FAST Profile (Default)

* Optimized for maximum throughput
* Variable-time algorithms (timing side-channels possible)
* Intended for:
  * Public-key operations and verification
  * Batch processing and GPU workloads
  * Research and benchmarking

### CT / HARDENED Profile (Implemented)

* Constant-time arithmetic — no secret-dependent branches or memory access
* ~5–7× performance penalty vs FAST
* Provides: `ct::field_mul`, `ct::field_inv`, `ct::scalar_mul`, `ct::point_add_complete`, `ct::point_dbl`
* Use for: private key handling, signing, nonce operations

**Choose the appropriate profile for your use case.** Using FAST with secret data is a security vulnerability.

## � Stable C ABI (`ufsecp`)

Starting with **v3.4.0**, UltrafastSecp256k1 ships a stable C ABI — `ufsecp` — designed for FFI bindings (C#, Python, Rust, Go, Java, etc.) and embedding.

### Architecture

```
┌──────────────────────────────────────────────────┐
│                  Your Application                │
│          (C, C#, Python, Go, Rust, …)            │
└──────────────────┬───────────────────────────────┘
                   │  ufsecp C ABI (45 functions)
┌──────────────────▼───────────────────────────────┐
│           ufsecp.dll / libufsecp.so              │
│  Opaque ctx  │  Error model  │  ABI versioning   │
├──────────────┴───────────────┴───────────────────┤
│                FAST layer                        │
│  Variable-time point/field/scalar operations     │
├──────────────────────────────────────────────────┤
│                CT layer (always active)           │
│  Constant-time signing, nonce gen, secret ops    │
│  Complete addition (12M+2S), Valgrind markers    │
└──────────────────────────────────────────────────┘
```

Both layers are **always active** — no flag-based selection. Public operations use the FAST layer; secret-key operations (sign, derive, ECDH) use the CT layer internally.

### Quick Start (C)

```c
#include "ufsecp.h"

ufsecp_ctx* ctx = NULL;
ufsecp_ctx_create(&ctx);

// Generate keypair
unsigned char seckey[32], pubkey[33];
ufsecp_keygen(ctx, seckey, pubkey);

// ECDSA sign
unsigned char msg[32] = { /* SHA-256 hash */ };
unsigned char sig[64];
ufsecp_ecdsa_sign(ctx, seckey, msg, sig);

// Verify
int valid = 0;
ufsecp_ecdsa_verify(ctx, pubkey, 33, msg, sig, &valid);

ufsecp_ctx_destroy(ctx);
```

### API Coverage

| Category | Functions |
|----------|-----------|
| **Context** | `ctx_create`, `ctx_destroy`, `selftest`, `last_error` |
| **Keys** | `keygen`, `seckey_verify`, `pubkey_create`, `pubkey_parse`, `pubkey_serialize` |
| **ECDSA** | `ecdsa_sign`, `ecdsa_verify`, `ecdsa_sign_der`, `ecdsa_verify_der`, `ecdsa_recover` |
| **Schnorr** | `schnorr_sign`, `schnorr_verify` |
| **SHA-256** | `sha256` (SHA-NI accelerated) |
| **ECDH** | `ecdh_compressed`, `ecdh_xonly`, `ecdh_raw` |
| **BIP-32** | `bip32_from_seed`, `bip32_derive_child`, `bip32_serialize` |
| **Address** | `address_p2pkh`, `address_p2wpkh`, `address_p2tr` |
| **WIF** | `wif_encode`, `wif_decode` |
| **Tweak** | `pubkey_tweak_add`, `pubkey_tweak_mul` |
| **Version** | `version`, `abi_version`, `version_string` |

### Building ufsecp

```bash
# Sub-project (from UltrafastSecp256k1 root — preferred)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Standalone
cmake -S include/ufsecp -B build-ufsecp -DCMAKE_BUILD_TYPE=Release
cmake --build build-ufsecp -j
```

Output: `ufsecp.dll` (shared) + `ufsecp_s.lib` (static).

See [SUPPORTED_GUARANTEES.md](include/ufsecp/SUPPORTED_GUARANTEES.md) for Tier 1/2/3 stability guarantees.

## �🛠️ Building

### Prerequisites

- CMake 3.18+
- C++20 compiler (GCC 11+, Clang/LLVM 15+)
  - MSVC 2022+ (optional, disabled by default - use `-DSECP256K1_ALLOW_MSVC=ON`)
- CUDA Toolkit 12.0+ (optional, for GPU support)
- Ninja (recommended)

### CPU-Only Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### With CUDA Support

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_CUDA=ON
cmake --build build -j
```

### WebAssembly (Emscripten)

```bash
# Requires Emscripten SDK (emsdk)
./scripts/build_wasm.sh        # → build-wasm/dist/
```

Output: `secp256k1_wasm.wasm` + `secp256k1.mjs` (ES6 module with TypeScript types). See [wasm/README.md](wasm/README.md) for JS/TS usage.

### iOS (XCFramework)

```bash
./scripts/build_xcframework.sh  # → build-xcframework/output/
```

Produces a universal XCFramework (arm64 device + arm64 simulator). Also available via **Swift Package Manager** and **CocoaPods**.

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `SECP256K1_USE_ASM` | ON | Enable assembly optimizations (x64/RISC-V) |
| `SECP256K1_BUILD_CUDA` | OFF | Build CUDA GPU support |
| `SECP256K1_BUILD_OPENCL` | OFF | Build OpenCL GPU support |
| `SECP256K1_BUILD_ROCM` | OFF | Build ROCm/HIP GPU support (AMD) |
| `SECP256K1_BUILD_TESTS` | ON | Build test suite |
| `SECP256K1_BUILD_BENCH` | ON | Build benchmarks |
| `SECP256K1_RISCV_FAST_REDUCTION` | ON | Fast modular reduction (RISC-V) |
| `SECP256K1_RISCV_USE_VECTOR` | ON | RVV vector extension (RISC-V) |

### Build Profiles

UltrafastSecp256k1 is designed with two conceptual build targets:

#### 1️⃣ FAST (Performance Research Mode)

* Maximum throughput
* Aggressive compiler optimizations allowed
* Suitable for:
  * Benchmarking
  * Public key generation
  * Batch verification
  * High-performance research environments

#### 2️⃣ CT (Constant-Time Hardened Mode)

* Secret-dependent branches avoided
* Deterministic execution paths
* Safer for:
  * Private key operations
  * Signing workflows
  * External-facing cryptographic services

CT mode is under continuous development and will be expanded with:

* Montgomery ladder options
* Constant-time table selection
* Optional blinding techniques
* Timing regression testing integration

## 🎯 Quick Start

### Basic CPU Usage

```cpp
#include <secp256k1/field.hpp>
#include <secp256k1/point.hpp>
#include <secp256k1/scalar.hpp>
#include <iostream>

using namespace secp256k1::fast;

int main() {
    // 1. Field arithmetic
    auto a = FieldElement::from_hex(
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141"
    );
    auto b = FieldElement::from_hex(
        "1234567890ABCDEF1234567890ABCDEF1234567890ABCDEF1234567890ABCDEF"
    );
    
    auto sum = a + b;
    auto product = a * b;
    auto inverse = a.inverse();
    
    std::cout << "Sum: " << sum.to_hex() << "\n";
    std::cout << "Product: " << product.to_hex() << "\n";
    
    // 2. Point operations (public key derivation)
    auto generator = Point::generator();
    auto private_key = Scalar::from_hex(
        "E9873D79C6D87DC0FB6A5778633389F4453213303DA61F20BD67FC233AA33262"
    );
    
    // Multiply generator by private key
    auto public_key = generator * private_key;
    
    std::cout << "Public Key X: " << public_key.x().to_hex() << "\n";
    std::cout << "Public Key Y: " << public_key.y().to_hex() << "\n";
    
    // 3. Point addition
    auto point1 = Point::from_coordinates(
        FieldElement::from_hex("..."),
        FieldElement::from_hex("...")
    );
    auto point2 = Point::from_coordinates(
        FieldElement::from_hex("..."),
        FieldElement::from_hex("...")
    );
    
    auto result = point1 + point2;
    
    return 0;
}
```

**Compile & Run:**
```bash
# Link with the library
g++ -std=c++20 example.cpp -lsecp256k1-fast-cpu -o example
./example
```

### Advanced: Batch Signature Verification

```cpp
#include <secp256k1/point.hpp>
#include <secp256k1/scalar.hpp>
#include <vector>

using namespace secp256k1::fast;

bool verify_signatures_batch(
    const std::vector<Point>& public_keys,
    const std::vector<std::array<uint8_t, 32>>& messages,
    const std::vector<Scalar>& r_values,
    const std::vector<Scalar>& s_values
) {
    auto generator = Point::generator();
    
    for (size_t i = 0; i < public_keys.size(); ++i) {
        // Hash message
        auto msg_hash = Scalar::from_bytes(messages[i]);
        
        // Verify: s*G = R + hash*PubKey
        auto s_inv = s_values[i].inverse();
        auto u1 = msg_hash * s_inv;
        auto u2 = r_values[i] * s_inv;
        
        auto point = generator * u1 + public_keys[i] * u2;
        
        if (point.x().to_scalar() != r_values[i]) {
            return false;
        }
    }
    
    return true;
}
```

### CUDA GPU Acceleration

```cpp
#include <secp256k1_cuda/batch_operations.hpp>
#include <secp256k1/point.hpp>
#include <vector>

using namespace secp256k1::fast;

int main() {
    // Prepare batch data (1 million operations)
    std::vector<Point> base_points(1'000'000);
    std::vector<Scalar> scalars(1'000'000);
    
    // Fill with data...
    for (size_t i = 0; i < base_points.size(); ++i) {
        base_points[i] = Point::generator();
        scalars[i] = Scalar::random();
    }
    
    // GPU batch multiplication
    cuda::BatchConfig config{
        .device_id = 0,
        .threads_per_block = 256,
        .streams = 4
    };
    
    auto results = cuda::batch_multiply(
        base_points, 
        scalars, 
        config
    );
    
    std::cout << "Processed " << results.size() 
              << " point multiplications on GPU\n";
    
    // Results are already on host memory
    for (const auto& result : results) {
        std::cout << "Result: " << result.x().to_hex() << "\n";
    }
    
    return 0;
}
```

**Compile with CUDA:**
```bash
nvcc -std=c++20 cuda_example.cpp \
     -lsecp256k1-fast-cpu -lsecp256k1-fast-cuda \
     -o cuda_example
./cuda_example
```

### CUDA: Batch Address Generation

```cpp
#include <secp256k1_cuda/batch_operations.hpp>
#include <secp256k1_cuda/address_generator.hpp>

int main() {
    // Generate 10 million Bitcoin addresses on GPU
    std::vector<Scalar> private_keys(10'000'000);
    
    // Fill with sequential or random keys
    for (size_t i = 0; i < private_keys.size(); ++i) {
        private_keys[i] = Scalar::from_int(i + 1);
    }
    
    // GPU batch generation
    auto addresses = cuda::generate_addresses(
        private_keys,
        cuda::AddressType::P2PKH // Bitcoin P2PKH format
    );
    
    std::cout << "Generated " << addresses.size() << " addresses\n";
    
    // First few addresses
    for (size_t i = 0; i < 10; ++i) {
        std::cout << "Address " << i << ": " 
                  << addresses[i] << "\n";
    }
    
    return 0;
}
```

### Performance Tuning Example

```cpp
#include <secp256k1/field.hpp>
#include <secp256k1/field_asm.hpp>
#include <chrono>

using namespace secp256k1::fast;

void benchmark_field_multiply() {
    auto a = FieldElement::random();
    auto b = FieldElement::random();
    
    const int iterations = 1'000'000;
    
    // Warm-up
    for (int i = 0; i < 1000; ++i) {
        volatile auto result = a * b;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        volatile auto result = a * b;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - start
    ).count();
    
    std::cout << "Field multiply: " 
              << (duration / iterations) << " ns/op\n";
    
    // Check if using assembly
    if (has_bmi2_support()) {
        std::cout << "Using BMI2 intrinsics: YES\n";
    }
    
#ifdef SECP256K1_HAS_ASM
    std::cout << "Using assembly: YES\n";
#else
    std::cout << "Using portable C++\n";
#endif
}
```

## 📊 Performance

All CPU benchmarks use median of 3 passes after warm-up. Windows results from Clang 21.1.0, Release, AVX2.
RISC-V results collected on **Milk-V Mars** (RV64 + RVV).

### x86_64 / Windows (Clang 21.1.0, AVX2, BMI2/ADX, Release)

| Operation | Time |
|-----------|------:|
| Field Mul (5×52) | 17 ns |
| Field Square (5×52) | 13 ns |
| Field Add | 1 ns |
| Field Negate | <1 ns |
| Field Inverse | 1 μs |
| Point Add | 172 ns |
| Point Double | 83 ns |
| Point Scalar Mul (k×P) | 24 μs |
| Generator Mul (k×G) | 7 μs |
| **ECDSA Sign** | **33 μs** |
| **ECDSA Verify** | **57 μs** |
| **Schnorr Sign (BIP-340)** | **23 μs** |
| **Schnorr Verify (BIP-340)** | **58 μs** |
| Batch Inverse (n=100) | 118 ns/elem |
| Batch Inverse (n=1000) | 105 ns/elem |

#### Signature Performance Summary

| Operation | Time | Notes |
|-----------|------:|-------|
| ECDSA Sign (RFC 6979) | 33 μs | Deterministic nonce, low-S normalized |
| ECDSA Verify | 57 μs | Accepts both low-S and high-S |
| Schnorr Sign (BIP-340) | 23 μs | Tagged hashing, x-only pubkeys |
| Schnorr Verify (BIP-340) | 58 μs | Standard BIP-340 verification |

*Schnorr sign is ~30% faster than ECDSA sign due to simpler nonce derivation (no modular inverse). Verification speed is comparable — both require two scalar multiplications (k₁×G + k₂×Q).*

#### Scalar Multiplication Breakdown

| Method | Time |
|--------|------:|
| k×G (Generator, precomputed) | 7 μs |
| k×P (Arbitrary point) | 24 μs |

#### Field Representation Comparison (5×52 vs 4×64)

| Operation | 4×64 | 5×52 | Speedup |
|-----------|------:|------:|--------:|
| Multiplication | 42 ns | 15 ns | **2.76×** |
| Squaring | 31 ns | 13 ns | **2.44×** |
| Addition | 4.3 ns | 1.6 ns | **2.69×** |
| Add chain (32 ops) | 286 ns | 57 ns | **5.01×** |

*5×52 uses `__int128` lazy reduction — ideal for 64-bit platforms. 4×64 is the default portable representation.*

#### Constant-Time (CT) Layer Overhead

| Operation | Fast | CT | Overhead |
|-----------|------:|------:|--------:|
| Field Mul | 36 ns | 55 ns | 1.50× |
| Field Inverse | 3.0 μs | 14.2 μs | 4.80× |
| Point Add | 0.65 μs | 1.63 μs | 2.50× |
| Scalar Mul (k×P) | 130 μs | 322 μs | 2.49× |
| Generator Mul (k×G) | 7.6 μs | 310 μs | 40.8× |

*CT layer provides constant-time execution for side-channel resistance. Generator mul overhead is higher due to disabled precomputed table lookups (variable-time).*

### x86_64 / Linux (i5, Clang 19.1.7, AVX2, Release)

| Operation | Time |
|-----------|------:|
| Field Mul | 33 ns |
| Field Square | 32 ns |
| Field Add | 11 ns |
| Field Sub | 12 ns |
| Field Inverse | 5 μs |
| Point Add | 521 ns |
| Point Double | 278 ns |
| Point Scalar Mul | 110 μs |
| Generator Mul | 5 μs |
| Batch Inverse (n=100) | 140 ns |
| Batch Inverse (n=1000) | 92 ns |

### RISC-V 64-bit / Linux (Milk-V Mars, RVV, Clang 21.1.8, Release)

| Operation | Time |
|-----------|------:|
| Field Mul | 173 ns |
| Field Square | 160 ns |
| Field Add | 38 ns |
| Field Sub | 34 ns |
| Field Inverse | 17 μs |
| Point Add | 3 μs |
| Point Double | 1 μs |
| Point Scalar Mul | 621 μs |
| Generator Mul | 37 μs |
| Batch Inverse (n=100) | 695 ns |
| Batch Inverse (n=1000) | 547 ns |

*See [RISCV_OPTIMIZATIONS.md](RISCV_OPTIMIZATIONS.md) for optimization details.*

### ESP32-S3 / Embedded (Xtensa LX7 @ 240 MHz, ESP-IDF v5.5.1, -O3)

| Operation | Time |
|-----------|------:|
| Field Mul | 7,458 ns |
| Field Square | 7,592 ns |
| Field Add | 636 ns |
| Field Inv | 844 μs |
| Scalar × G (Generator Mul) | 2,483 μs |

*Portable C++ (no `__int128`, no assembly). All 35 library tests pass. See [examples/esp32_test/](examples/esp32_test/) for details.*

### ESP32-PICO-D4 / Embedded (Xtensa LX6 Dual Core @ 240 MHz, ESP-IDF v5.5.1, -O3)

| Operation | Time |
|-----------|------:|
| Field Mul | 6,993 ns |
| Field Square | 6,247 ns |
| Field Add | 985 ns |
| Field Inv | 609 μs |
| Scalar × G (Generator Mul) | 6,203 μs |
| CT Scalar × G | 44,810 μs |
| CT Add (complete) | 249,672 ns |
| CT Dbl | 87,113 ns |
| CT/Fast ratio | 6.5× |

*Portable C++ (no `__int128`, no assembly). All 35 self-tests + 8 CT tests pass. See [examples/esp32_test/](examples/esp32_test/) for details.*

### STM32F103ZET6 / Embedded (ARM Cortex-M3 @ 72 MHz, GCC 13.3.1, -O3)

| Operation | Time |
|-----------|------:|
| Field Mul | 15,331 ns |
| Field Square | 12,083 ns |
| Field Add | 4,139 ns |
| Field Inv | 1,645 μs |
| Scalar × G (Generator Mul) | 37,982 μs |

*ARM Cortex-M3 inline assembly (UMULL/ADDS/ADCS) for multiply/squaring/reduction. Portable C++ for field add/sub. All 35 library tests pass. See [examples/stm32_test/](examples/stm32_test/) for details.*

### Android ARM64 (RK3588, Cortex-A55/A76 @ 2.4 GHz, NDK r27 Clang 18, -O3)

| Operation | Time |
|-----------|------:|
| Field Mul | 85 ns |
| Field Square | 66 ns |
| Field Add | 18 ns |
| Field Sub | 16 ns |
| Field Inverse | 2,621 ns |
| Scalar Mul | 105 ns |
| Scalar Add | 12 ns |
| Point Add | 9,329 ns |
| Point Double | 8,711 ns |
| Fast Scalar × G (Generator Mul) | 7.6 μs |
| Fast Scalar × P (Non-Generator) | 77.6 μs |
| CT Scalar × G | 545 μs |
| CT ECDH | 545 μs |

*ARM64 inline assembly (MUL/UMULH) for field mul/sqr/add/sub/neg. ~5× faster than generic C++. All 12 Android tests pass. See [android/](android/) for details.*

### Embedded Cross-Platform Comparison

| Operation | ESP32-S3 LX7 (240 MHz) | ESP32 LX6 (240 MHz) | STM32F103 (72 MHz) |
|-----------|-------------------:|-------------------:|-------------------:|
| Field Mul | 7,458 ns | 6,993 ns | 15,331 ns |
| Field Square | 7,592 ns | 6,247 ns | 12,083 ns |
| Field Add | 636 ns | 985 ns | 4,139 ns |
| Field Inv | 844 μs | 609 μs | 1,645 μs |
| Scalar × G | 2,483 μs | 6,203 μs | 37,982 μs |

*Clock-Normalized = (STM32 time × 72) / (ESP32 time × 240). Values < 1.0 mean STM32 is faster per-clock.*

### CUDA (NVIDIA RTX 5060 Ti) — Kernel-Only

#### Core ECC Operations

| Operation | Time/Op | Throughput |
|-----------|---------|------------|
| Field Mul | 0.2 ns | 4,142 M/s |
| Field Add | 0.2 ns | 4,130 M/s |
| Field Inv | 10.2 ns | 98.35 M/s |
| Point Add | 1.6 ns | 619 M/s |
| Point Double | 0.8 ns | 1,282 M/s |
| Scalar Mul (P×k) | 225.8 ns | 4.43 M/s |
| Generator Mul (G×k) | 217.7 ns | 4.59 M/s |
| Affine Add (2M+1S+inv) | 0.4 ns | 2,532 M/s |
| Affine Lambda (2M+1S) | 0.6 ns | 1,654 M/s |
| Affine X-Only (1M+1S) | 0.4 ns | 2,328 M/s |
| Batch Inv (Montgomery) | 2.9 ns | 340 M/s |
| Jac→Affine (per-pt) | 14.9 ns | 66.9 M/s |

#### GPU Signature Operations (ECDSA + Schnorr)

| Operation | Time/Op | Throughput | Protocol |
|-----------|---------|------------|----------|
| **ECDSA Sign** | **204.8 ns** | **4.88 M/s** | RFC 6979 + low-S |
| **ECDSA Verify** | **410.1 ns** | **2.44 M/s** | Shamir + GLV |
| **ECDSA Sign+Recid** | **311.5 ns** | **3.21 M/s** | Recoverable (EIP-155) |
| **Schnorr Sign** | **273.4 ns** | **3.66 M/s** | BIP-340 |
| **Schnorr Verify** | **354.6 ns** | **2.82 M/s** | BIP-340 + GLV |

> **No other open-source GPU library provides secp256k1 ECDSA+Schnorr sign/verify.**
> This is the only CUDA+OpenCL+Metal implementation with full signature support.

*CUDA 12.0, sm_86;sm_89, batch=16K signatures, RTX 5060 Ti (36 SMs, 2602 MHz)*

### OpenCL (NVIDIA RTX 5060 Ti) — Kernel-Only

| Operation | Time/Op | Throughput |
|-----------|---------|------------|
| Field Mul | 0.2 ns | 4,137 M/s |
| Field Add | 0.2 ns | 4,124 M/s |
| Field Sqr | 0.2 ns | 5,985 M/s |
| Field Inv | 14.3 ns | 69.97 M/s |
| Point Add | 1.6 ns | 630.6 M/s |
| Point Double | 0.9 ns | 1,139 M/s |
| kG (Generator Mul) | 295.1 ns | 3.39 M/s |

*OpenCL 3.0 CUDA, Driver 580.126.09, PTX inline asm, batch=256K–1M*

### CUDA vs OpenCL — Kernel-Only Comparison (RTX 5060 Ti)

| Operation | CUDA | OpenCL | Faster |
|-----------|------|--------|--------|
| Field Mul | 0.2 ns | 0.2 ns | Tie |
| Field Add | 0.2 ns | 0.2 ns | Tie |
| Field Inv | 10.2 ns | 14.3 ns | **CUDA 1.40×** |
| Point Double | 0.8 ns | 0.9 ns | **CUDA 1.13×** |
| Point Add | 1.6 ns | 1.6 ns | Tie |
| kG (Generator Mul) | 217.7 ns | 295.1 ns | **CUDA 1.36×** |

> **Note:** Both measurements are kernel-only (no buffer allocation/copy overhead). CUDA uses local-variable optimization for zero pointer-aliasing overhead.

*Benchmarks: 2026-02-14, Linux x86_64, NVIDIA Driver 580.126.09*

### Apple Metal (Apple M3 Pro) — Kernel-Only

| Operation | Time/Op | Throughput |
|-----------|---------|------------|
| Field Mul | 1.9 ns | 527 M/s |
| Field Add | 1.0 ns | 990 M/s |
| Field Sub | 1.1 ns | 892 M/s |
| Field Sqr | 1.1 ns | 872 M/s |
| Field Inv | 106.4 ns | 9.40 M/s |
| Point Add | 10.1 ns | 98.6 M/s |
| Point Double | 5.1 ns | 196 M/s |
| Scalar Mul (P×k) | 2.94 μs | 0.34 M/s |
| Generator Mul (G×k) | 3.00 μs | 0.33 M/s |

*Metal 2.4, 8×32-bit Comba limbs, Apple M3 Pro (18 GPU cores, Unified Memory 18 GB)*

### Available Benchmark Targets

| Target | Description | Run Command |
|--------|-------------|-------------|
| `bench_comprehensive` | Full field/point/batch/signature benchmark suite | `./bench_comprehensive` |
| `bench_scalar_mul` | k×G and k×P with wNAF analysis | `./bench_scalar_mul` |
| `bench_ct` | Fast-vs-CT layer overhead comparison | `./bench_ct` |
| `bench_atomic_operations` | Individual ECC building block latencies | `./bench_atomic_operations` |
| `bench_field_52` | 4×64 vs 5×52 field representation comparison | `./bench_field_52` |
| `bench_field_26` | 4×64 vs 10×26 field representation comparison | `./bench_field_26` |
| `bench_field_mul_kernels` | BMI2 kernel micro-benchmark | `./bench_field_mul_kernels` |
| `bench_ecdsa_multiscalar` | k₁×G + k₂×Q (Shamir vs separate) | `./bench_ecdsa_multiscalar` |
| `bench_jsf_vs_shamir` | JSF vs Windowed Shamir comparison | `./bench_jsf_vs_shamir` |
| `bench_adaptive_glv` | GLV window size sweep (8–20) | `./bench_adaptive_glv` |
| `bench_glv_decomp_profile` | GLV decomposition analysis | `./bench_glv_decomp_profile` |
| `bench_comprehensive_riscv` | RISC-V optimized benchmark suite | `./bench_comprehensive_riscv` |

## 🏗️ Architecture

```
secp256k1-fast/
├── cpu/                 # CPU-optimized implementation
│   ├── include/         # Public headers
│   ├── src/            # Implementation
│   │   ├── field.cpp           # Field arithmetic
│   │   ├── scalar.cpp          # Scalar arithmetic
│   │   ├── point.cpp           # Point operations
│   │   ├── field_asm_x64.asm   # x64 assembly
│   │   ├── field_asm_x64_gas.S # x64 GAS syntax
│   │   └── field_asm_riscv64.S # RISC-V assembly
│   └── tests/          # Unit tests
├── cuda/               # CUDA GPU acceleration
│   ├── include/        # CUDA headers
│   ├── src/           # CUDA kernels
│   └── tests/         # CUDA tests
├── opencl/            # OpenCL GPU acceleration
│   ├── kernels/       # OpenCL kernel sources (.cl)
│   ├── include/       # OpenCL headers
│   ├── src/           # Host-side OpenCL code
│   └── tests/         # OpenCL tests
└── examples/
    ├── esp32_test/    # ESP32-S3 Xtensa LX7 port
    └── stm32_test/    # STM32F103ZET6 ARM Cortex-M3 port
```

## 🔬 Research Statement

This library explores the performance ceiling of secp256k1 across CPU architectures (x64, ARM64, RISC-V, Cortex-M, Xtensa) and GPUs (CUDA, OpenCL, Metal, ROCm). Zero external dependencies.

## 📚 Variant Overview

Internal 32-bit arithmetic variants (historical optimization stages):

| Variant | Description |
|---------|-------------|
| `secp256k1_32_fast` | Speed-first, variable-time |
| `secp256k1_32_hybrid_smart` | Mixed strategy experiments |
| `secp256k1_32_hybrid_final` | Stabilized hybrid arithmetic |
| `secp256k1_32_really_final` | Most mature 32-bit variant |

## 🪙 Supported Coins

All 27 secp256k1-based cryptocurrencies with native address generation (P2PKH, P2WPKH, P2TR, EIP-55):

| # | Coin | Ticker | Address Types | BIP-44 |
|---|------|--------|---------------|--------|
| 1 | **Bitcoin** | BTC | P2PKH, P2WPKH (Bech32), P2TR (Bech32m) | m/86'/0' |
| 2 | **Ethereum** | ETH | EIP-55 Checksum | m/44'/60' |
| 3 | **Litecoin** | LTC | P2PKH, P2WPKH | m/84'/2' |
| 4 | **Dogecoin** | DOGE | P2PKH | m/44'/3' |
| 5 | **Bitcoin Cash** | BCH | P2PKH | m/44'/145' |
| 6 | **Bitcoin SV** | BSV | P2PKH | m/44'/236' |
| 7 | **Zcash** | ZEC | P2PKH (transparent) | m/44'/133' |
| 8 | **Dash** | DASH | P2PKH | m/44'/5' |
| 9 | **DigiByte** | DGB | P2PKH, P2WPKH | m/44'/20' |
| 10 | **Namecoin** | NMC | P2PKH | m/44'/7' |
| 11 | **Peercoin** | PPC | P2PKH | m/44'/6' |
| 12 | **Vertcoin** | VTC | P2PKH, P2WPKH | m/44'/28' |
| 13 | **Viacoin** | VIA | P2PKH | m/44'/14' |
| 14 | **Groestlcoin** | GRS | P2PKH, P2WPKH | m/44'/17' |
| 15 | **Syscoin** | SYS | P2PKH | m/44'/57' |
| 16 | **BNB Smart Chain** | BNB | EIP-55 | m/44'/60' |
| 17 | **Polygon** | MATIC | EIP-55 | m/44'/60' |
| 18 | **Avalanche** | AVAX | EIP-55 (C-Chain) | m/44'/60' |
| 19 | **Fantom** | FTM | EIP-55 | m/44'/60' |
| 20 | **Arbitrum** | ARB | EIP-55 | m/44'/60' |
| 21 | **Optimism** | OP | EIP-55 | m/44'/60' |
| 22 | **Ravencoin** | RVN | P2PKH | m/44'/175' |
| 23 | **Flux** | FLUX | P2PKH | m/44'/19167' |
| 24 | **Qtum** | QTUM | P2PKH | m/44'/2301' |
| 25 | **Horizen** | ZEN | P2PKH | m/44'/121' |
| 26 | **Bitcoin Gold** | BTG | P2PKH | m/44'/156' |
| 27 | **Komodo** | KMD | P2PKH | m/44'/141' |

All EVM chains (ETH, BNB, MATIC, AVAX, FTM, ARB, OP) share the same address format (EIP-55 checksummed hex).

## 🚫 Scope

This is an ECC arithmetic library. It provides field/scalar/point operations, signature schemes (ECDSA, Schnorr, MuSig2, FROST, Adaptor), Pedersen commitments, Taproot, HD derivation (BIP-32/44), and 27-coin address generation.
It does not include key storage, wallet software, network protocols, or attack tools.

## ⚠️ API Stability

**C++ API**: Not yet stable. Breaking changes may occur in any minor release before **v4.0**. Core layers (field, scalar, point, ECDSA, Schnorr) have mature interfaces unlikely to change. Experimental layers (MuSig2, FROST, Adaptor, Pedersen, Taproot, HD, Coins) may see breaking changes.

**C ABI (`ufsecp`)**: Stable from v3.4.0. ABI version is tracked separately — minor version bumps add new functions without breaking existing ones. See [SUPPORTED_GUARANTEES.md](include/ufsecp/SUPPORTED_GUARANTEES.md) for tier details.

Pin your dependency version and review changelogs before upgrading.

## 📚 Documentation

- [Documentation Index](docs/README.md)
- [API Reference](docs/API_REFERENCE.md)
- [Build Guide](docs/BUILDING.md)
- [Benchmarks](docs/BENCHMARKS.md)
- [Threat Model](THREAT_MODEL.md)
- [Contributing](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

## 🧪 Testing

### Built-in Selftest

The library includes a comprehensive self-test (`Selftest()`) that runs **deterministic KAT vectors** covering all arithmetic operations. Every test/bench executable runs this selftest on startup.

### Three Modes

| Mode | Time | When | What |
|------|------|------|------|
| **smoke** | ~1-2s | App startup, embedded | Core KAT (10 scalar mul, field/scalar identities, point ops, batch inverse, boundary vectors) |
| **ci** | ~30-90s | Every push (CI) | Smoke + cross-checks, bilinearity, NAF/wNAF, batch sweeps, fast-vs-generic, algebraic stress |
| **stress** | ~10-60min | Nightly / manual | CI + 1000 random scalar muls, 500 field triples, 100 bilinearity pairs, batch inverse up to 8192 |

```cpp
#include "secp256k1/selftest.hpp"
using namespace secp256k1::fast;

// Legacy (runs ci mode):
Selftest(true);

// Explicit mode + seed:
Selftest(true, SelftestMode::smoke);              // Fast startup check
Selftest(true, SelftestMode::ci);                  // Full CI suite
Selftest(true, SelftestMode::stress, 0xDEADBEEF); // Nightly with custom seed
```

### Repro Bundle

On verbose output, selftest prints everything needed to reproduce a failure:

```
  Mode:     ci
  Seed:     0x53454350324b3147
  Compiler: Clang 17.0.6
  Platform: Linux x64
  Build:    Release
  ASM:      enabled
  Repro:    Selftest(true, SelftestMode::ci, 0x53454350324b3147)
```

### Sanitizer Builds

```bash
# ASan + UBSan (catches UB, out-of-bounds, use-after-free)
cmake --preset cpu-asan
cmake --build build/cpu-asan -j
ctest --test-dir build/cpu-asan --output-on-failure

# TSan (catches data races in multi-threaded code)
cmake --preset cpu-tsan
cmake --build build/cpu-tsan -j
ctest --test-dir build/cpu-tsan --output-on-failure
```

### Running Tests

```bash
# Build and run all tests (ci mode)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSECP256K1_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Platform Coverage Dashboard

| Platform | Backend | Compiler | Selftest CI | Stress | Notes |
|----------|---------|----------|-------------|--------|-------|
| Linux x64 | CPU | GCC 13 | ✅ CI | - | Debug + Release |
| Linux x64 | CPU | Clang 17 | ✅ CI | - | Debug + Release |
| Linux x64 | CPU | Clang 17 (ASan+UBSan) | ✅ CI | - | Sanitizer build |
| Linux x64 | CPU | Clang 17 (TSan) | ✅ CI | - | Thread sanitizer |
| Windows x64 | CPU | MSVC 2022 | ✅ CI | - | Release |
| macOS ARM64 | CPU + Metal | AppleClang | ✅ CI | - | Apple Silicon |
| macOS ARM64 | Metal GPU | AppleClang | ✅ CI | - | GPU shader tests |
| iOS ARM64 | CPU | Xcode | ✅ CI | - | Device + Simulator |
| Android ARM64 | CPU | NDK r27c | ✅ CI | - | arm64-v8a |
| WebAssembly | CPU | Emscripten | ✅ CI | - | Build + WASM benchmark |
| ROCm/HIP | CPU + GPU | ROCm 6.3 | ✅ CI | - | Compile + CPU test |

> Community-tested platforms: if you run selftest on a new platform, submit the log via PR and we'll add a row.

### Fuzz Testing

libFuzzer harnesses cover core arithmetic (`cpu/fuzz/`):

| Target | What it tests |
|--------|---------------|
| `fuzz_field` | add/sub round-trip, mul identity, square equivalence, inverse |
| `fuzz_scalar` | add/sub, mul identity, distributive law |
| `fuzz_point` | on-curve check, negate, compress round-trip, dbl vs add |

```bash
clang++ -fsanitize=fuzzer,address -O2 -std=c++20 \
  -I cpu/include cpu/fuzz/fuzz_field.cpp cpu/src/field.cpp cpu/src/field_asm.cpp \
  -o fuzz_field && ./fuzz_field -max_len=64 -runs=10000000
```

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/shrec/UltrafastSecp256k1.git
cd UltrafastSecp256k1
cmake -S . -B build-dev -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build-dev -j
```

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

### Open Source License

The library is free to use under AGPL-3.0 for open source projects. This means:
- ✅ You can use, modify, and distribute the code
- ✅ You must disclose your source code
- ✅ You must license your project under AGPL-3.0 or compatible license
- ✅ You must provide network access to your source code if you run it as a service

See [LICENSE](LICENSE) for full details.

### Commercial License

**For commercial/proprietary use without AGPL-3.0 obligations:**

If you want to use this library in a proprietary/closed-source product or service without disclosing your source code, please contact us for a commercial license.

📧 **Contact for commercial licensing:**
- Email: [payysoon@gmail.com](mailto:payysoon@gmail.com)
- GitHub: https://github.com/shrec/UltrafastSecp256k1

We offer flexible licensing options for commercial applications.

## 🙏 Acknowledgments

- Based on optimized secp256k1 implementations
- Inspired by Bitcoin Core's libsecp256k1
- RISC-V assembly contributions
- CUDA kernel optimizations

## 📧 Contact & Community

| Channel | Link |
|---------|------|
| Issues | [GitHub Issues](https://github.com/shrec/UltrafastSecp256k1/issues) |
| Discussions | [GitHub Discussions](https://github.com/shrec/UltrafastSecp256k1/discussions) |
| Wiki | [Documentation Wiki](https://github.com/shrec/UltrafastSecp256k1/wiki) |
| Benchmarks | [Live Dashboard](https://shrec.github.io/UltrafastSecp256k1/dev/bench/) |
| API Docs | [Doxygen](https://shrec.github.io/UltrafastSecp256k1/docs/) |
| Security | [Report Vulnerability](https://github.com/shrec/UltrafastSecp256k1/security/advisories/new) |
| Commercial | [payysoon@gmail.com](mailto:payysoon@gmail.com) |

## ☕ Support the Project

If you find this library useful, consider supporting development!

[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-ea4aaa.svg?logo=github)](https://github.com/sponsors/shrec)
[![PayPal](https://img.shields.io/badge/PayPal-Donate-blue.svg?logo=paypal)](https://paypal.me/IChkheidze)

---

**UltrafastSecp256k1** — The fastest open-source secp256k1 library. GPU-accelerated ECDSA & Schnorr signatures for Bitcoin, Ethereum, and 25+ blockchains.

<!-- SEO keywords (not rendered) -->
<!-- secp256k1 CUDA GPU ECDSA sign verify Schnorr BIP-340 Bitcoin Ethereum signature acceleration OpenCL Metal batch verification elliptic curve cryptography C++ high performance library blockchain cryptocurrency libsecp256k1 alternative GPU accelerated digital signatures NVIDIA AMD Apple Silicon embedded RISC-V ARM64 WebAssembly cross-platform multi-coin address generation BIP-32 BIP-44 HD wallet derivation key recovery EIP-155 RFC-6979 transaction signing fastest secp256k1 -->
