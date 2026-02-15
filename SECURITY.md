# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 3.0.x   | ✅ Active  |
| 2.x.x   | ⚠️ Critical fixes only |
| < 2.0   | ❌ Unsupported |

Security fixes apply to the latest release on the `main` branch.

---

## Reporting a Vulnerability

If you discover a potential security issue related to:

- Incorrect field or scalar arithmetic
- Point operation errors (addition, doubling, scalar multiplication)
- ECDSA signature forgery or invalid verification
- Schnorr signature forgery or invalid verification
- SHA-256 hash collisions or incorrect output
- Determinism violations (RFC 6979 nonce generation)
- Constant-time violations (timing side channels)
- Memory safety issues (buffer overflows, use-after-free)
- GPU kernel correctness issues (CUDA, ROCm, OpenCL)
- Undefined behavior affecting cryptographic correctness

**Do NOT open a public issue for suspected vulnerabilities.**

Instead, use [GitHub Security Advisories](https://github.com/shrec/UltrafastSecp256k1/security/advisories/new) to report privately, or contact the maintainer directly via GitHub.

We will investigate and respond within **72 hours**.

---

## Security Design

### Constant-Time Operations

v3.0.0 includes a constant-time layer (`ct::` namespace) providing:

- `ct::field_mul`, `ct::field_inv` — timing-safe field arithmetic
- `ct::scalar_mul` — timing-safe scalar multiplication
- `ct::point_add_complete`, `ct::point_dbl` — complete addition formulas

The CT layer uses no secret-dependent branches or memory access patterns. It carries a ~5–7× performance penalty relative to the optimized (variable-time) path.

**Note**: The default (non-CT) operations prioritize performance and are NOT constant-time. Use the `ct::` variants when processing secret keys or nonces.

### ECDSA & Schnorr

- ECDSA: Deterministic nonces via RFC 6979 (no random nonce generation needed)
- Schnorr: BIP-340 compliant with tagged hashing
- Both signature schemes include validation of inputs (point-on-curve, scalar range checks)

### Memory Handling

- No dynamic allocation in hot paths
- Sensitive data (private keys, nonces) should be zeroed by the caller after use
- Fixed-size POD types used throughout (no hidden copies)

---

## Scope

UltrafastSecp256k1 provides:

- Finite field arithmetic (𝔽ₚ for secp256k1 prime)
- Scalar arithmetic (mod n, curve order)
- Elliptic curve point operations (add, double, scalar multiply)
- ECDSA signatures (RFC 6979)
- Schnorr signatures (BIP-340)
- SHA-256 hashing
- GPU-accelerated batch operations (CUDA, ROCm, OpenCL)

**Out of scope**: Key management, wallet functionality, network protocols, and application-layer cryptographic protocols. Security responsibility for higher-level integrations remains with the integrating application.

---

## Acknowledgments

We appreciate responsible disclosure. Contributors who report valid security issues will be credited in the changelog (unless they prefer anonymity).

---

*UltrafastSecp256k1 v3.0.0 — Security Policy*
