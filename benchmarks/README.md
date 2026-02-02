# UltrafastSecp256k1 Benchmarks

Performance benchmarks across different platforms and configurations.

## 📊 Benchmark Results

### Directory Structure

```
benchmarks/
├── cpu/
│   ├── x86-64/
│   │   ├── windows/     # Windows x64 results
│   │   └── linux/       # Linux x64 results
│   ├── riscv64/
│   │   └── linux/       # RISC-V RV64GC (Milk-V Mars, etc.)
│   ├── arm64/
│   │   ├── linux/       # ARM64 Linux (RPi, etc.)
│   │   └── macos/       # Apple Silicon (M1/M2/M3)
│   └── esp32/
│       └── embedded/    # ESP32 (limited, core only)
├── gpu/
│   ├── cuda/
│   │   ├── rtx-40xx/    # RTX 4090, 4080, etc.
│   │   ├── rtx-30xx/    # RTX 3090, 3080, etc.
│   │   ├── rtx-20xx/    # RTX 2080 Ti, etc.
│   │   └── datacenter/  # A100, H100, V100
│   └── opencl/          # Future: AMD, Intel, etc.
└── comparison/          # Cross-platform comparisons
```

## 🚀 Running Benchmarks

### CPU Benchmarks

```bash
# Build with benchmarks
cmake -B build -DSECP256K1_BUILD_BENCH=ON
cmake --build build -j

# Run all CPU benchmarks
./build/cpu/bench/benchmark_field
./build/cpu/bench/benchmark_point
./build/cpu/bench/benchmark_scalar

# Save results
./build/cpu/bench/benchmark_field > benchmarks/cpu/x86-64/linux/field_$(date +%Y%m%d).txt
```

### GPU Benchmarks (CUDA)

```bash
# Build with CUDA
cmake -B build -DSECP256K1_BUILD_CUDA=ON -DSECP256K1_BUILD_BENCH=ON
cmake --build build -j

# Run GPU benchmarks
./build/cuda/bench/cuda_benchmark

# Save results
./build/cuda/bench/cuda_benchmark > benchmarks/gpu/cuda/rtx-4090/batch_$(date +%Y%m%d).txt
```

## 📈 Benchmark Format

Each benchmark file should include:

```
Platform: x86-64 / RISC-V / ARM64 / CUDA
CPU/GPU: Specific model
OS: Windows 11 / Linux 6.x / macOS 14
Compiler: GCC 13.2 / Clang 18 / MSVC 2022
Build: Release / -O3 / Assembly ON/OFF
Date: YYYY-MM-DD

=== Field Operations ===
Addition:        X ns/op
Multiplication:  X ns/op
Squaring:        X ns/op
Inversion:       X ns/op

=== Point Operations ===
Point Addition:      X µs/op
Point Doubling:      X µs/op
Point Multiply:      X µs/op
Batch Multiply (n):  X ms for n ops

=== Throughput ===
Operations/second:   X M ops/s
```

## 🎯 Submitting Benchmarks

If you run benchmarks on your hardware, please submit them!

1. Run the benchmark suite
2. Save results to appropriate directory
3. Include system information
4. Submit via Pull Request

**Template:**
```bash
# System info
uname -a
lscpu  # or cat /proc/cpuinfo
gcc --version  # or clang --version

# Run benchmarks
./build/cpu/bench/benchmark_field > results.txt
```

## 📊 Current Results

See individual platform directories for detailed results:
- [x86-64 Windows](cpu/x86-64/windows/)
- [x86-64 Linux](cpu/x86-64/linux/)
- [RISC-V Linux](cpu/riscv64/linux/)
- [ARM64 Linux](cpu/arm64/linux/)
- [CUDA RTX 4090](gpu/cuda/rtx-40xx/)

## 🏆 Leaderboards

### Fastest Field Multiplication
1. x86-64 + Assembly: ~8ns (AMD Zen 5)
2. RISC-V + Assembly: ~75ns (StarFive JH7110)
3. ARM64: TBD
4. Portable C++: ~25ns (x86-64)

### Highest GPU Throughput
1. RTX 4090: 8M ops/s (batch multiply)
2. RTX 3090: TBD
3. A100: TBD

**Contribute your results to update these leaderboards!**

---

For questions about benchmarking, open an issue or discussion on GitHub.
