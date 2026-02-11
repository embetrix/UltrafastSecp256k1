# Documentation Index

Welcome to UltrafastSecp256k1 documentation.

## Quick Links

| Document | Description |
|----------|-------------|
| [API Reference](API_REFERENCE.md) | Complete function reference for CPU and CUDA |
| [Building](BUILDING.md) | Build instructions for all platforms |
| [Benchmarks](BENCHMARKS.md) | Performance measurements and comparisons |
| [RISC-V Optimizations](../RISCV_OPTIMIZATIONS.md) | RISC-V specific optimizations |

## Getting Started

### 1. Build the Library

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 2. Run Self-Test

```bash
./build/cpu/secp256k1_selftest
```

### 3. Use in Your Code

```cpp
#include <secp256k1/point.hpp>
#include <secp256k1/scalar.hpp>

using namespace secp256k1::fast;

int main() {
    // Generate public key from private key
    Scalar private_key = Scalar::from_hex("...");
    Point public_key = Point::generator().scalar_mul(private_key);
    
    auto compressed = public_key.to_compressed();
    // Use compressed[33] as your public key
    return 0;
}
```

## Architecture Overview

```
UltrafastSecp256k1/
├── cpu/                    # CPU implementation (x86-64, RISC-V)
│   ├── include/secp256k1/  # Public headers
│   │   ├── field.hpp       # Field element (mod p)
│   │   ├── scalar.hpp      # Scalar (mod n)
│   │   ├── point.hpp       # EC point operations
│   │   └── ...
│   └── src/                # Implementation
│       ├── field.cpp
│       ├── field_asm_riscv64.S
│       └── ...
│
├── cuda/                   # CUDA GPU implementation
│   ├── include/
│   │   ├── secp256k1.cuh   # Main GPU header
│   │   ├── hash160.cuh     # HASH160 computation
│   │   └── ...
│   └── src/
│
└── docs/                   # This documentation
```

## Performance Summary

| Platform | Scalar Multiplication | Notes |
|----------|----------------------|-------|
| x86-64 | ~110 μs | BMI2/ADX assembly |
| RISC-V | ~672 μs | RVV optimized |
| CUDA | TBD | Batch parallel |

## License

AGPL v3 - See [LICENSE](../LICENSE)

