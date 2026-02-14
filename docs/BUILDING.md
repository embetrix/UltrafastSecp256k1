# Building UltrafastSecp256k1

Complete build guide for all supported platforms.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Build Options](#build-options)
4. [Platform-Specific Instructions](#platform-specific-instructions)
   - [Linux x86-64](#linux-x86-64)
   - [Windows x86-64](#windows-x86-64)
   - [RISC-V 64](#risc-v-64)
   - [CUDA](#cuda)
5. [Cross-Compilation](#cross-compilation)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required

- **CMake** 3.18 or later
- **C++20 compiler**:
  - GCC 11+ (recommended for Linux)
  - Clang/LLVM 15+ (recommended, best optimization)
  - MSVC 2022+ (Windows, requires `-DSECP256K1_ALLOW_MSVC=ON`)
- **Ninja** (recommended) or Make

### Optional

- **CUDA Toolkit 12.0+** for GPU support
- **clang-19/21** for best RISC-V optimization

---

## Quick Start

### CPU Only (Default)

```bash
# Configure
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j

# Test
ctest --test-dir build --output-on-failure
```

### With CUDA

```bash
# Configure
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;86;89"

# Build
cmake --build build -j
```

---

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `SECP256K1_USE_ASM` | ON | Assembly optimizations (x64/RISC-V) |
| `SECP256K1_BUILD_CUDA` | OFF | Build CUDA GPU library |
| `SECP256K1_BUILD_OPENCL` | OFF | Build OpenCL GPU support |
| `SECP256K1_BUILD_TESTS` | ON | Build test suite |
| `SECP256K1_BUILD_BENCH` | ON | Build benchmarks |
| `SECP256K1_BUILD_EXAMPLES` | OFF | Build example programs |
| `SECP256K1_USE_LTO` | OFF | Link-Time Optimization |
| `SECP256K1_SPEED_FIRST` | OFF | Aggressive speed optimizations |
| `SECP256K1_ALLOW_MSVC` | OFF | Allow MSVC compiler |

### RISC-V Specific

| Option | Default | Description |
|--------|---------|-------------|
| `SECP256K1_RISCV_FAST_REDUCTION` | ON | Fast modular reduction |
| `SECP256K1_RISCV_USE_VECTOR` | ON | RVV vector extension |
| `SECP256K1_RISCV_USE_PREFETCH` | ON | Memory prefetch hints |

### CUDA Specific

| Option | Default | Description |
|--------|---------|-------------|
| `SECP256K1_CUDA_USE_MONTGOMERY` | OFF | Montgomery domain arithmetic |
| `SECP256K1_CUDA_LIMBS_32` | OFF | 8×32-bit limbs (experimental) |
| `CMAKE_CUDA_ARCHITECTURES` | 89 | Target GPU architectures |

---

## Platform-Specific Instructions

### Linux x86-64

#### Using Clang (Recommended)

```bash
# Install dependencies
sudo apt install cmake ninja-build clang-19 lld-19

# Configure with Clang
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-19 \
  -DCMAKE_CXX_COMPILER=clang++-19

# Build
cmake --build build -j$(nproc)
```

#### Using GCC

```bash
# Install dependencies
sudo apt install cmake ninja-build g++-11

# Configure with GCC
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-11 \
  -DCMAKE_CXX_COMPILER=g++-11

# Build
cmake --build build -j$(nproc)
```

---

### Windows x86-64

#### Using Clang/LLVM (Recommended)

```powershell
# Install LLVM from https://llvm.org/builds/
# Or via winget:
winget install LLVM.LLVM

# Configure
cmake -S . -B build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER=clang `
  -DCMAKE_CXX_COMPILER=clang++

# Build
cmake --build build -j
```

#### Using MSVC (Not Recommended)

```powershell
# Open Visual Studio Developer Command Prompt
# Then:
cmake -S . -B build -G "Visual Studio 17 2022" `
  -DSECP256K1_ALLOW_MSVC=ON

cmake --build build --config Release
```

> ⚠️ **Warning**: MSVC produces slower code compared to Clang/GCC.

---

### RISC-V 64

#### Native Build

```bash
# On RISC-V machine with Clang 19+
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-21 \
  -DCMAKE_CXX_COMPILER=clang++-21

cmake --build build -j$(nproc)
```

#### Cross-Compilation (from x86-64)

```bash
# Install toolchain
sudo apt install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu

# Configure for cross-compilation
cmake -S . -B build-riscv -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
  -DCMAKE_C_COMPILER=riscv64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=riscv64-linux-gnu-g++

cmake --build build-riscv -j$(nproc)
```

#### Expected Performance (RISC-V)

| Operation | Time |
|-----------|------|
| Field Mul | ~198 ns |
| Field Square | ~177 ns |
| Field Add | ~34 ns |
| Point Scalar Mul | ~672 μs |
| Generator Mul | ~40 μs |

---

### CUDA

#### Prerequisites

1. Install [CUDA Toolkit 12.0+](https://developer.nvidia.com/cuda-downloads)
2. Ensure `nvcc` is in PATH

#### Build

```bash
# Configure
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;86;89"

# Build
cmake --build build -j
```

#### GPU Architecture Reference

| Architecture | GPUs | Flag |
|--------------|------|------|
| sm_75 | RTX 2060-2080, T4 | 75 |
| sm_80 | A100 | 80 |
| sm_86 | RTX 3060-3090, A6000 | 86 |
| sm_89 | RTX 4060-4090, L4, L40 | 89 |
| sm_90 | H100 | 90 |

Example for RTX 4090:
```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=89 ...
```

---

## Cross-Compilation

### RISC-V from x86-64

See [RISC-V 64](#risc-v-64) section above.

### Windows from Linux

```bash
# Using MinGW-w64
sudo apt install mingw-w64

cmake -S . -B build-win -G Ninja \
  -DCMAKE_SYSTEM_NAME=Windows \
  -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc \
  -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++

cmake --build build-win -j
```

---

## Troubleshooting

### LTO Not Working on RISC-V

LTO requires `LLVMgold.so` plugin. If missing:

```bash
# Install LLVM with gold plugin
sudo apt install llvm-19-dev

# Or disable LTO
cmake -DSECP256K1_USE_LTO=OFF ...
```

### CUDA Compilation Errors

1. **"nvcc not found"**: Add CUDA to PATH
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   ```

2. **"unsupported architecture"**: Update `CMAKE_CUDA_ARCHITECTURES`:
   ```bash
   cmake -DCMAKE_CUDA_ARCHITECTURES=89 ...
   ```

### Assembly Errors on RISC-V

If you see "symbol already defined" errors:
```bash
# Rebuild clean
rm -rf build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### MSVC Link Errors

MSVC is not fully supported. Use Clang instead:
```powershell
# Install LLVM from https://llvm.org/builds/
cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ...
```

---

## Verification

After building, run tests to verify correctness:

```bash
# Run all tests
ctest --test-dir build --output-on-failure

# Run benchmarks
./build/cpu/bench_comprehensive
./build/cuda/secp256k1_cuda_bench  # If CUDA enabled
```

---

## Installation

```bash
# Install to system (Linux)
sudo cmake --install build --prefix /usr/local

# Install to custom location
cmake --install build --prefix /opt/secp256k1
```

### pkg-config

After installation, use pkg-config:

```bash
pkg-config --cflags --libs secp256k1-fast
```

### CMake Integration

```cmake
find_package(secp256k1-fast REQUIRED)
target_link_libraries(your_target PRIVATE secp256k1::fast-cpu)
```

---

## Version

UltrafastSecp256k1 v1.0.0

