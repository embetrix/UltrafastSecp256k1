# Cross-Platform Test Matrix

> **Generated**: 2025 | **Library**: UltrafastSecp256k1 | **Total CTest Targets**: 22
>
> ყველა ტესტი უნდა იყოს იდენტური ყველა პლატფორმაზე. ნებისმიერი განსხვავება = **BUG**.

---

## Test Inventory (22 Tests)

| #  | Test Name               | Category            | Checks | Description                                                      |
|----|------------------------|---------------------|--------|------------------------------------------------------------------|
| 1  | selftest               | Core Selftest       | ~200   | Built-in self-test: field, scalar, point, generator consistency  |
| 2  | batch_add_affine       | Point Arithmetic    | ~50    | Batch affine addition correctness for sequential ECC search      |
| 3  | hash_accel             | Hashing             | ~80    | SHA-256, RIPEMD-160, Hash160 (SHA-NI accelerated where available)|
| 4  | field_52               | Field Arithmetic    | ~100   | 5×52-bit lazy reduction field implementation tests               |
| 5  | field_26               | Field Arithmetic    | ~100   | 10×26-bit field (32-bit platform path) implementation tests      |
| 6  | exhaustive             | Full Coverage       | ~500+  | Exhaustive small-order subgroup + enumeration tests              |
| 7  | comprehensive          | Full Coverage       | ~800+  | All arithmetic operations combined stress                        |
| 8  | bip340_vectors         | Standards Vectors   | ~30    | BIP-340 Schnorr signature official test vectors                  |
| 9  | bip32_vectors          | Standards Vectors   | ~40    | BIP-32 HD key derivation official test vectors                   |
| 10 | rfc6979_vectors        | Standards Vectors   | ~20    | RFC 6979 deterministic nonce official test vectors               |
| 11 | ecc_properties         | ECC Properties      | ~150   | Algebraic properties: associativity, commutativity, identity     |
| 12 | ct_sidechannel         | Constant-Time       | ~300   | Full CT layer: compare, select, cswap, scalar_mul CT paths      |
| 13 | ct_sidechannel_smoke   | Constant-Time       | ~100   | Smoke test: CT operations basic correctness                      |
| 14 | differential           | Differential Test   | ~200   | Differential testing: fast vs CT layer output equivalence        |
| 15 | ct_equivalence         | Constant-Time       | ~150   | CT scalar_mul ≡ fast scalar_mul bitwise equivalence              |
| 16 | diag_scalar_mul        | Diagnostics         | ~50    | Scalar multiplication step-by-step diagnostic comparison         |
| 17 | fault_injection        | Security Audit      | 610    | Fault injection simulation: bit-flips, coord corruption, GLV     |
| 18 | debug_invariants       | Security Audit      | 372    | Debug assertion verification: normalize, on_curve, scalar_valid  |
| 19 | fiat_crypto_vectors    | Golden Vectors      | 647    | Fiat-Crypto/Sage reference comparison: mul, sqr, inv, add, sub  |
| 20 | carry_propagation      | Boundary Stress     | 247    | Carry chain stress: all-ones, limb boundary, near-p, near-n     |
| 21 | cross_platform_kat     | KAT Equivalence     | 24     | Cross-platform Known Answer Test: field, scalar, point, ECDSA, Schnorr |
| 22 | abi_gate               | ABI Compatibility   | 12     | ABI version gate: compile-time macro validation                  |

---

## Platform Matrix

### Legend
- ✅ = All checks PASS
- ❌ = One or more checks FAIL
- ⚠️ = Partial (some tests skipped or known limitation)
- N/A = Not applicable / not targetable for this platform
- 🔲 = Not yet tested

### Test × Platform Status

| #  | Test Name             | x86-64 Win (Clang) | x86-64 Linux (Clang/GCC) | x86-64 macOS | ARM64 Linux | ARM64 macOS (Apple Si) | RISC-V 64 | WASM (Emscripten) | ESP32 (Xtensa) | STM32 (Cortex-M4) |
|----|----------------------|:-------------------:|:------------------------:|:------------:|:-----------:|:---------------------:|:---------:|:-----------------:|:--------------:|:-----------------:|
| 1  | selftest             | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | ✅        | 🔲                | 🔲             | 🔲                |
| 2  | batch_add_affine     | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | ✅        | 🔲                | 🔲             | 🔲                |
| 3  | hash_accel           | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | ✅        | 🔲                | 🔲             | 🔲                |
| 4  | field_52             | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | ✅        | 🔲                | N/A            | N/A               |
| 5  | field_26             | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | ✅        | 🔲                | ✅ ¹           | ✅ ¹              |
| 6  | exhaustive           | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | ✅        | 🔲                | 🔲             | 🔲                |
| 7  | comprehensive        | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | ✅        | 🔲                | 🔲             | 🔲                |
| 8  | bip340_vectors       | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | ✅        | 🔲                | 🔲             | 🔲                |
| 9  | bip32_vectors        | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | ✅        | 🔲                | 🔲             | 🔲                |
| 10 | rfc6979_vectors      | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | ✅        | 🔲                | 🔲             | 🔲                |
| 11 | ecc_properties       | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | ✅        | 🔲                | 🔲             | 🔲                |
| 12 | ct_sidechannel       | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | ✅        | 🔲                | 🔲             | 🔲                |
| 13 | ct_sidechannel_smoke | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | ✅        | 🔲                | 🔲             | 🔲                |
| 14 | differential         | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | ✅        | 🔲                | 🔲             | 🔲                |
| 15 | ct_equivalence       | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | ✅        | 🔲                | 🔲             | 🔲                |
| 16 | diag_scalar_mul      | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | ✅        | 🔲                | 🔲             | 🔲                |
| 17 | fault_injection      | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | 🔲        | 🔲                | 🔲             | 🔲                |
| 18 | debug_invariants     | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | 🔲        | 🔲                | 🔲             | 🔲                |
| 19 | fiat_crypto_vectors  | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | 🔲        | 🔲                | 🔲             | 🔲                |
| 20 | carry_propagation    | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | 🔲        | 🔲                | 🔲             | 🔲                |
| 21 | cross_platform_kat   | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | 🔲        | 🔲                | 🔲             | 🔲                |
| 22 | abi_gate             | ✅                  | ✅                        | 🔲           | 🔲          | 🔲                    | 🔲        | 🔲                | 🔲             | 🔲                |

> ¹ 32-bit platforms (ESP32, STM32) use field_26 only; field_52 requires 64-bit limbs.

---

## CI Coverage (Automated)

| Platform             | CI Workflow       | Trigger        | Status    |
|---------------------|-------------------|----------------|-----------|
| x86-64 Linux (GCC)  | ci.yml            | push/PR        | ✅ Active |
| x86-64 Linux (Clang) | ci.yml           | push/PR        | ✅ Active |
| x86-64 Windows (MSVC)| ci.yml           | push/PR        | ✅ Active |
| x86-64 Windows (Clang)| ci.yml          | push/PR        | ✅ Active |
| x86-64 macOS        | ci.yml            | push/PR        | ✅ Active |
| ARM64 Linux          | ci.yml (qemu)    | push/PR        | ✅ Active |
| RISC-V 64            | Manual / Cross   | manual         | ⚠️ Manual |
| WASM                 | —                 | —              | 🔲 Planned |
| ESP32                | —                 | —              | 🔲 Planned |
| STM32                | —                 | —              | 🔲 Planned |

---

## Verification Summary (Current Session — x86-64 Windows, Clang)

```
CTest Results: 22/22 passed, 0 failed

Individual check counts:
  selftest .................. ~200 checks
  batch_add_affine .......... ~50  checks
  hash_accel ................ ~80  checks
  field_52 .................. ~100 checks
  field_26 .................. ~100 checks
  exhaustive ................ ~500 checks
  comprehensive ............. ~800 checks
  bip340_vectors ............ ~30  checks
  bip32_vectors ............. ~40  checks
  rfc6979_vectors ........... ~20  checks
  ecc_properties ............ ~150 checks
  ct_sidechannel ............ ~300 checks
  ct_sidechannel_smoke ...... ~100 checks
  differential .............. ~200 checks
  ct_equivalence ............ ~150 checks
  diag_scalar_mul ........... ~50  checks
  fault_injection ........... 610  checks ✓
  debug_invariants .......... 372  checks ✓
  fiat_crypto_vectors ....... 647  checks ✓
  carry_propagation ......... 247  checks ✓
  cross_platform_kat ........ 24   checks ✓
  abi_gate .................. 12   checks ✓
  ─────────────────────────────────────────
  TOTAL (estimated):         ~4700+ individual assertions
```

---

## Test Categories

| Category            | Tests                                                        | Purpose                                    |
|--------------------|--------------------------------------------------------------|--------------------------------------------|
| Core Selftest      | selftest                                                     | Basic library self-validation on startup    |
| Field Arithmetic   | field_52, field_26, carry_propagation                        | Modular arithmetic correctness at all limb widths |
| Point Arithmetic   | batch_add_affine, ecc_properties, diag_scalar_mul            | Elliptic curve operations                  |
| Standards Vectors  | bip340_vectors, bip32_vectors, rfc6979_vectors               | Official standard compliance               |
| Golden Vectors     | fiat_crypto_vectors, cross_platform_kat                      | Deterministic correctness vs reference     |
| Constant-Time      | ct_sidechannel, ct_sidechannel_smoke, ct_equivalence, differential | Side-channel resistance verification |
| Security Audit     | fault_injection, debug_invariants                             | Fault tolerance & invariant enforcement    |
| Hashing            | hash_accel                                                    | Hash function correctness (SHA-NI, etc.)   |
| Full Coverage      | exhaustive, comprehensive                                     | Exhaustive enumeration + combined stress   |
| ABI Compatibility  | abi_gate                                                      | Version & ABI stability check              |

---

## Scripts (Audit Infrastructure)

| Script                          | Purpose                                           | Platform    |
|--------------------------------|---------------------------------------------------|-------------|
| scripts/verify_ct_disasm.sh    | Disassembly scan for CT branches                  | Linux       |
| scripts/valgrind_ct_check.sh   | Valgrind memcheck on CT paths                     | Linux       |
| scripts/ctgrind_validate.sh    | CTGRIND-style validation (secret-as-undefined)    | Linux       |
| scripts/generate_coverage.sh   | LLVM source-based code coverage                   | Linux/macOS |
| scripts/cross_compiler_ct_stress.sh | Multi-compiler CT verification              | Linux       |
| scripts/generate_selftest_report.sh | JSON self-test evidence report               | Any         |
| scripts/generate_dudect_badge.sh | Dudect timing badge generation                  | Linux       |
| scripts/cachegrind_ct_analysis.sh | Cache-line timing analysis                     | Linux       |
| scripts/perf_regression_check.sh | Benchmark regression tracking                   | Linux       |
| scripts/generate_self_audit_report.sh | Comprehensive audit evidence JSON          | Linux       |

---

## Platform-Specific Notes

### x86-64 (Primary)
- Assembly tier: Tier 3 (inline asm), Tier 2 (BMI2 intrinsics), Tier 1 (C++)
- SHA-NI acceleration available on supported CPUs
- Full CI matrix (Windows MSVC+Clang, Linux GCC+Clang, macOS)

### ARM64
- Uses generic C++ paths (no asm tier 3)
- CI via QEMU cross-compilation
- SHA-256 hardware acceleration via ARM CE where available

### RISC-V 64
- Custom assembly: field_asm52_riscv64.S (Tier 3)
- SLTU/carry chain bug fixes verified (see RISCV_FIX_SUMMARY.md)
- Manual cross-compilation + QEMU testing
- RVV (Vector Extension) support optional

### WASM (Emscripten) — Planned
- 32-bit path: field_26 (10×26-bit limbs)
- No inline assembly, pure C++ only
- KAT test should produce identical output

### ESP32 / STM32 — Planned
- 32-bit path: field_26
- No OS, bare-metal test harness needed
- KAT golden vectors are the acceptance criterion

---

## How to Run on a New Platform

```bash
# 1. Configure
cmake -S . -B build_<platform> -G Ninja -DCMAKE_BUILD_TYPE=Release

# 2. Build
cmake --build build_<platform> -j

# 3. Run ALL tests
ctest --test-dir build_<platform> --output-on-failure

# 4. Verify KAT equivalence (golden vectors must match exactly)
./build_<platform>/cpu/test_cross_platform_kat

# 5. Generate audit report
./scripts/generate_self_audit_report.sh build_<platform>
```

Expected: **22/22 tests PASS** with identical output on every platform.

---

> **ყველა პლატფორმაზე იდენტური შედეგი = სწორი იმპლემენტაცია.**
> **ნებისმიერი განსხვავება = ბაგი, რომელიც დაუყოვნებლივ უნდა გამოსწორდეს.**
