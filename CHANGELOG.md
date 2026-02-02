# Changelog

All notable changes to secp256k1-fast will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public release
- CPU implementation with x86-64 and RISC-V assembly optimizations
- CUDA GPU acceleration for batch operations
- Comprehensive test suite
- Benchmark utilities
- Cross-platform CMake build system

### Platforms Supported
- x86-64 (Windows, Linux, macOS)
- RISC-V RV64GC (Linux)
- CUDA (NVIDIA GPUs)

## [1.0.0] - 2026-02-02

### Added
- Complete secp256k1 field arithmetic
- Point addition, doubling, and multiplication
- Scalar arithmetic
- GLV endomorphism optimization
- Assembly optimizations:
  - x86-64 BMI2/ADX (3-5× speedup)
  - RISC-V RV64GC (2-3× speedup)
  - RISC-V Vector Extension (RVV) support
- CUDA batch operations
- Memory-mapped database support
- Comprehensive documentation

### Performance
- x86-64 field multiplication: ~8ns (assembly)
- RISC-V field multiplication: ~75ns (assembly)
- CUDA batch throughput: 8M ops/s (RTX 4090)

---

**Legend:**
- `Added` - New features
- `Changed` - Changes in existing functionality
- `Deprecated` - Soon-to-be removed features
- `Removed` - Removed features
- `Fixed` - Bug fixes
- `Security` - Security fixes
