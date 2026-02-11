# Performance Benchmarks

Benchmark results for UltrafastSecp256k1 across different platforms.

---

## Summary

| Platform | Field Mul | Scalar Mul | Generator Mul |
|----------|-----------|------------|---------------|
| x86-64 (i5, AVX2) | 33 ns | 110 μs | 5 μs |
| RISC-V 64 (RVV) | 198 ns | 672 μs | 40 μs |
| CUDA (RTX 5060 Ti) | TBD | TBD | TBD |

---

## x86-64 Benchmarks

**Hardware:** Intel Core i5 (AVX2, BMI2, ADX)  
**OS:** Linux  
**Compiler:** Clang 19.1.7  
**Assembly:** x86-64 with BMI2/ADX intrinsics  
**SIMD:** AVX2

| Operation | Time | Notes |
|-----------|------|-------|
| Field Mul | 33 ns | Using mulx/adcx/adox |
| Field Square | 32 ns | Optimized squaring |
| Field Add | 11 ns | |
| Field Sub | 12 ns | |
| Field Inverse | 5 μs | Fermat's little theorem |
| Point Add | 521 ns | Jacobian coordinates |
| Point Double | 278 ns | |
| Point Scalar Mul | 110 μs | GLV + wNAF |
| Generator Mul | 5 μs | Precomputed tables |
| Batch Inverse (n=100) | 140 ns/elem | Montgomery's trick |
| Batch Inverse (n=1000) | 92 ns/elem | |

---

## RISC-V 64 Benchmarks

**Hardware:** RISC-V 64-bit (RV64GC + V extension)  
**OS:** Linux  
**Compiler:** Clang 21.1.8  
**Assembly:** RISC-V native assembly  
**SIMD:** RVV 1.0

| Operation | Time | Notes |
|-----------|------|-------|
| Field Mul | 198 ns | Optimized carry chain |
| Field Square | 177 ns | Dedicated squaring |
| Field Add | 34 ns | Branchless |
| Field Sub | 31 ns | Branchless |
| Field Inverse | 18 μs | |
| Point Add | 3 μs | |
| Point Double | 1 μs | |
| Point Scalar Mul | 672 μs | GLV + wNAF |
| Generator Mul | 40 μs | Precomputed tables |
| Batch Inverse (n=100) | 765 ns/elem | |
| Batch Inverse (n=1000) | 615 ns/elem | |

### RISC-V Optimization Gains

| Optimization | Speedup | Applied To |
|--------------|---------|------------|
| Native assembly | 2-3× | Field mul/square |
| Branchless algorithms | 1.2× | Field add/sub |
| Fast modular reduction | 1.5× | All field ops |
| RVV vectorization | 1.1× | Batch operations |
| Carry chain optimization | 1.3× | Multiplication |

---

## CUDA Benchmarks

**Hardware:** NVIDIA RTX 5060 Ti  
**CUDA:** 12.0+  
**Architecture:** sm_89 (Ada Lovelace)

| Operation | Time | Throughput | Notes |
|-----------|------|------------|-------|
| Field Mul (single) | TBD | - | |
| Field Mul (batch 1M) | TBD | TBD ops/s | |
| Scalar Mul (batch 1M) | TBD | TBD ops/s | |
| Hash160 (batch 1M) | TBD | TBD ops/s | |

### CUDA Configuration

```cpp
// Optimal settings for RTX 4090/5060 Ti
#define SECP256K1_CUDA_USE_HYBRID_MUL 1  // 32-bit hybrid (~10% faster)
#define SECP256K1_CUDA_USE_MONTGOMERY 0  // Standard domain (faster for search)
```

---

## Comparison with Other Libraries

### vs libsecp256k1 (Bitcoin Core)

| Operation | UltrafastSecp256k1 | libsecp256k1 | Speedup |
|-----------|-------------------|--------------|---------|
| Scalar Mul (x86-64) | 110 μs | ~150 μs | 1.36× |
| Generator Mul | 5 μs | ~8 μs | 1.6× |
| Batch verify | TBD | TBD | - |

### vs tiny-ecdsa

| Operation | UltrafastSecp256k1 | tiny-ecdsa | Speedup |
|-----------|-------------------|------------|---------|
| Scalar Mul | 110 μs | ~500 μs | 4.5× |
| Field Mul | 33 ns | ~200 ns | 6× |

---

## Benchmark Methodology

### CPU Benchmarks

1. **Warm-up:** 1 iteration discarded
2. **Measurement:** 3 iterations, take median
3. **Timer:** `std::chrono::high_resolution_clock`
4. **Compiler flags:** `-O3 -march=native`

### CUDA Benchmarks

1. **Warm-up:** 10 kernel launches discarded
2. **Measurement:** 100 launches, average
3. **Timer:** CUDA events
4. **Sync:** Full device synchronization between measurements

### Reproducibility

```bash
# Run CPU benchmark
./build/cpu/bench_comprehensive

# Run CUDA benchmark
./build/cuda/secp256k1_cuda_bench

# Results saved to: benchmark-<platform>-<date>.txt
```

---

## Optimization History

### RISC-V Timeline

| Date | Field Mul | Scalar Mul | Change |
|------|-----------|------------|--------|
| 2026-02-08 | 307 ns | 954 μs | Initial |
| 2026-02-09 | 205 ns | 676 μs | Carry optimization |
| 2026-02-10 | 198 ns | 672 μs | Square optimization |
| 2026-02-10 | 198 ns | 672 μs | **Current** |

### Key Optimizations Applied

1. **Branchless field operations** - Eliminates unpredictable branches
2. **Optimized carry propagation** - Reduces instruction count
3. **Dedicated squaring routine** - 25% fewer multiplications than generic mul
4. **GLV decomposition** - ~50% reduction in scalar bits
5. **wNAF encoding** - ~33% fewer point additions
6. **Precomputed tables** - Generator multiplication 10× faster

---

## Future Optimizations

### Planned

- [ ] ARM64 NEON assembly
- [ ] AVX-512 vectorization (x86-64)
- [ ] Multi-threaded batch operations
- [ ] OpenCL backend

### Experimental

- [ ] Montgomery domain for CUDA (mixed results)
- [ ] 8×32-bit limb representation
- [ ] Constant-time side-channel resistance

---

## Version

UltrafastSecp256k1 v1.0.0  
Benchmarks updated: 2026-02-11

