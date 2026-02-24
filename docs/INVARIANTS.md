# Invariants

**UltrafastSecp256k1** — Complete Invariant Catalog

This document lists every mathematical, structural, and behavioral invariant that the library must maintain. Each invariant is either verified by existing tests or marked for future coverage.

---

## 1. Field Arithmetic Invariants (mod p)

**p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F**

| # | Invariant | Verified |
|---|-----------|----------|
| F1 | `normalize(a)` yields `0 ≤ a < p` for any input | ✅ test_field_audit |
| F2 | `add(a, b) == (a + b) mod p` | ✅ test_field_audit |
| F3 | `sub(a, b) == (a - b + p) mod p` | ✅ test_field_audit |
| F4 | `mul(a, b) == (a * b) mod p` | ✅ test_field_audit |
| F5 | `square(a) == mul(a, a)` | ✅ test_field_audit |
| F6 | `inv(a) * a == 1 mod p` for `a ≠ 0` | ✅ test_field_audit |
| F7 | `inv(0)` is undefined / returns zero | ✅ test_field_audit |
| F8 | `sqrt(a)² == a mod p` when `a` is a QR | ✅ test_field_audit |
| F9 | `sqrt(a)` returns nullopt when `a` is a QNR | ✅ test_field_audit |
| F10 | `negate(a) + a == 0 mod p` | ✅ test_field_audit |
| F11 | `from_bytes(to_bytes(a)) == a` for normalized `a` | ✅ test_field_audit |
| F12 | `from_limbs` interprets as little-endian uint64[4] | ✅ test_field_audit |
| F13 | `from_bytes` interprets as big-endian 32 bytes | ✅ test_field_audit |
| F14 | Commutativity: `add(a,b) == add(b,a)`, `mul(a,b) == mul(b,a)` | ✅ test_field_audit |
| F15 | Associativity: `add(add(a,b),c) == add(a,add(b,c))` | ✅ test_ecc_properties |
| F16 | Distributivity: `mul(a, add(b,c)) == add(mul(a,b), mul(a,c))` | ✅ test_ecc_properties |
| F17 | `field_select(0, a, b) == a`, `field_select(1, a, b) == b` (branchless) | ✅ test_field_audit |

---

## 2. Scalar Arithmetic Invariants (mod n)

**n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141**

| # | Invariant | Verified |
|---|-----------|----------|
| S1 | `scalar_add(a, b) == (a + b) mod n` | ✅ test_field_audit |
| S2 | `scalar_sub(a, b) == (a - b + n) mod n` | ✅ test_field_audit |
| S3 | `scalar_mul(a, b) == (a * b) mod n` | ✅ test_field_audit |
| S4 | `scalar_inv(a) * a == 1 mod n` for `a ≠ 0` | ✅ test_field_audit |
| S5 | `scalar_negate(a) + a == 0 mod n` | ✅ test_field_audit |
| S6 | `scalar_is_zero(0) == true` | ✅ test_field_audit |
| S7 | `scalar_is_zero(1) == false` | ✅ test_field_audit |
| S8 | `scalar_normalize(a)` yields `0 ≤ a < n` | ✅ test_field_audit |
| S9 | Low-S normalization: if `s > n/2`, replace with `n - s` | ✅ test_cross_libsecp256k1 |

---

## 3. Point / Group Invariants

**G = generator point of secp256k1**  
**n * G = O (point at infinity)**

| # | Invariant | Verified |
|---|-----------|----------|
| P1 | `G` is on curve: `G.y² == G.x³ + 7 mod p` | ✅ test_field_audit |
| P2 | `n * G == O` (generator order) | ✅ test_field_audit |
| P3 | `P + O == P` (identity element) | ✅ test_field_audit |
| P4 | `P + (-P) == O` (inverse) | ✅ test_field_audit |
| P5 | `(P + Q) + R == P + (Q + R)` (associativity) | ✅ test_ecc_properties |
| P6 | `P + Q == Q + P` (commutativity) | ✅ test_ecc_properties |
| P7 | `k * (P + Q) == k*P + k*Q` (distributivity) | ✅ test_ecc_properties |
| P8 | `(a + b) * G == a*G + b*G` (scalar addition homomorphism) | ✅ test_ecc_properties |
| P9 | `(a * b) * G == a * (b * G)` (scalar multiplication) | ✅ test_cross_libsecp256k1 |
| P10 | `to_affine(to_jacobian(P)) == P` | ✅ test_ecc_properties |
| P11 | `add_jacobian(P, Q) == add_affine(P, Q)` (consistency) | ✅ test_ecc_properties |
| P12 | `double_jacobian(P) == P + P` | ✅ test_field_audit |
| P13 | For any point P on curve: `P.y² == P.x³ + 7 mod p` | ✅ test_field_audit |
| P14 | Binary serialization round-trip: `deserialize(serialize(P)) == P` | ✅ test_fuzz_parsers |

---

## 4. GLV Endomorphism Invariants

| # | Invariant | Verified |
|---|-----------|----------|
| G1 | `phi(P) == lambda * P` where `lambda³ == 1 mod n` | ✅ test_comprehensive |
| G2 | `phi(phi(P)) + phi(P) + P == O` (endomorphism relation) | ✅ test_comprehensive |
| G3 | GLV decomposition: `k == k1 + k2 * lambda mod n` | ✅ test_comprehensive |
| G4 | `|k1|, |k2| < sqrt(n)` (balanced decomposition) | ✅ test_comprehensive |

---

## 5. ECDSA Invariants (RFC 6979)

| # | Invariant | Verified |
|---|-----------|----------|
| E1 | `verify(msg, sign(msg, sk), pk) == true` for valid (sk, pk) | ✅ test_rfc6979_vectors, test_cross_libsecp256k1 |
| E2 | Deterministic: `sign(msg, sk)` always produces same `(r, s)` | ✅ test_rfc6979_vectors |
| E3 | `r ∈ [1, n-1]` and `s ∈ [1, n-1]` | ✅ test_cross_libsecp256k1 |
| E4 | Low-S: `s ≤ n/2` enforced | ✅ test_cross_libsecp256k1 |
| E5 | DER encoding/decoding round-trip | ✅ test_fuzz_parsers |
| E6 | Signature with `sk = 0` or `sk ≥ n` fails | ✅ test_fuzz_address_bip32_ffi |
| E7 | Verify with wrong message returns false | ✅ test_cross_libsecp256k1 |
| E8 | Verify with wrong pubkey returns false | ✅ test_cross_libsecp256k1 |

---

## 6. Schnorr / BIP-340 Invariants

| # | Invariant | Verified |
|---|-----------|----------|
| B1 | `schnorr_verify(msg, schnorr_sign(msg, sk, aux), pk) == true` | ✅ test_bip340_vectors |
| B2 | All 15 official BIP-340 test vectors pass | ✅ test_bip340_vectors |
| B3 | Signature is 64 bytes: `(R.x[32] || s[32])` | ✅ test_bip340_vectors |
| B4 | `R` has even y-coordinate | ✅ test_bip340_vectors |
| B5 | Public key is x-only (32 bytes) | ✅ test_bip340_vectors |
| B6 | Sign with `sk = 0` fails | ✅ test_fuzz_address_bip32_ffi |

---

## 7. MuSig2 Invariants

| # | Invariant | Verified |
|---|-----------|----------|
| M1 | Aggregated signature verifies as standard BIP-340 Schnorr | ✅ test_musig2_frost |
| M2 | Key aggregation is deterministic for same pubkey set | ✅ test_musig2_frost |
| M3 | Nonce aggregation is deterministic for same inputs | ✅ test_musig2_frost |
| M4 | 2-of-2, 3-of-3, 5-of-5 scenarios produce valid sigs | ✅ test_musig2_frost |
| M5 | Invalid partial signature detected before aggregation | ✅ test_musig2_frost_advanced |
| M6 | Rogue-key attack: Wagner-style key manipulation detected | ✅ test_musig2_frost_advanced |
| M7 | Nonce reuse across different messages detected | ✅ test_musig2_frost_advanced |

---

## 8. FROST Invariants

| # | Invariant | Verified |
|---|-----------|----------|
| FR1 | t-of-n DKG produces consistent group public key | ✅ test_musig2_frost |
| FR2 | Signing shares reconstruct to group secret (Shamir) | ✅ test_musig2_frost |
| FR3 | Aggregated signature verifies as BIP-340 Schnorr | ✅ test_musig2_frost |
| FR4 | 2-of-3 threshold signing works with any 2 signers | ✅ test_musig2_frost |
| FR5 | 3-of-5 threshold signing works with any 3 signers | ✅ test_musig2_frost |
| FR6 | Lagrange coefficients: `Σ λ_i * s_i == s` (secret reconstruction) | ✅ test_musig2_frost |
| FR7 | Malicious share in DKG detected (commitment verification) | ✅ test_musig2_frost_advanced |
| FR8 | Invalid partial signature in signing detected | ✅ test_musig2_frost_advanced |
| FR9 | Below-threshold subset cannot produce valid signature | ✅ test_musig2_frost_advanced |

---

## 9. BIP-32 HD Derivation Invariants

| # | Invariant | Verified |
|---|-----------|----------|
| H1 | TV1-TV5 official vectors pass (90 checks) | ✅ test_bip32_vectors |
| H2 | `derive(master, "m") == master` | ✅ test_bip32_vectors |
| H3 | Hardened derivation: `child_privkey = parent_privkey + HMAC(parent_chaincode, 0x00||parent_privkey||index)` | ✅ test_bip32_vectors |
| H4 | Normal derivation: `child_privkey = parent_privkey + HMAC(parent_chaincode, parent_pubkey||index)` | ✅ test_bip32_vectors |
| H5 | Path parser: `"m/0/1'/2"` parsed correctly; invalid paths rejected | ✅ test_fuzz_address_bip32_ffi |
| H6 | Seed length must be 16–64 bytes | ✅ test_fuzz_address_bip32_ffi |
| H7 | Derivation is deterministic for same seed + path | ✅ test_bip32_vectors |

---

## 10. Address Generation Invariants

| # | Invariant | Verified |
|---|-----------|----------|
| A1 | P2PKH (Base58Check): `1...` prefix for mainnet | ✅ test_fuzz_address_bip32_ffi |
| A2 | P2WPKH (Bech32): `bc1q...` prefix for mainnet | ✅ test_fuzz_address_bip32_ffi |
| A3 | P2TR (Bech32m): `bc1p...` prefix for mainnet | ✅ test_fuzz_address_bip32_ffi |
| A4 | WIF round-trip: `decode(encode(sk)) == sk` | ✅ test_fuzz_address_bip32_ffi |
| A5 | NULL/invalid inputs return error codes, not crash | ✅ test_fuzz_address_bip32_ffi |
| A6 | Address from zero pubkey fails gracefully | ✅ test_fuzz_address_bip32_ffi |

---

## 11. C ABI (ufsecp) Invariants

| # | Invariant | Verified |
|---|-----------|----------|
| C1 | `ufsecp_context_create()` returns non-NULL | ✅ test_fuzz_address_bip32_ffi |
| C2 | `ufsecp_context_destroy(NULL)` is safe (no-op) | ✅ test_fuzz_address_bip32_ffi |
| C3 | All functions return `UFSECP_ERROR_NULL_ARGUMENT` for NULL pointers | ✅ test_fuzz_address_bip32_ffi |
| C4 | `ufsecp_last_error()` reflects last error code | ✅ test_fuzz_address_bip32_ffi |
| C5 | `ufsecp_error_string(code)` returns non-NULL for all defined codes | ✅ test_fuzz_address_bip32_ffi |
| C6 | `ufsecp_abi_version()` returns non-zero | ✅ test_fuzz_address_bip32_ffi |
| C7 | Thread: context is not thread-safe (documented); functions with separate contexts are safe | ⚠️ TSan CI |

---

## 12. Constant-Time Invariants

| # | Invariant | Verified |
|---|-----------|----------|
| CT1 | `ct::scalar_mul` execution time does not depend on scalar value | ✅ dudect (test_ct_sidechannel) |
| CT2 | `ct::ecdsa_sign` execution time does not depend on private key | ✅ dudect |
| CT3 | `ct::schnorr_sign` execution time does not depend on private key | ✅ dudect |
| CT4 | `ct::field_inv` execution time does not depend on input value | ✅ dudect |
| CT5 | No secret-dependent branches in CT code paths | ⚠️ Code review (no formal verification) |
| CT6 | No secret-dependent memory access patterns in CT code paths | ⚠️ Code review |

---

## 13. Batch / Performance Invariants

| # | Invariant | Verified |
|---|-----------|----------|
| BP1 | `batch_inverse(a[]) * a[i] == 1` for all non-zero `a[i]` | ✅ test_field_audit |
| BP2 | Batch verification result matches sequential verification | ✅ test_cross_libsecp256k1 |
| BP3 | Hamburg comb produces same result as double-and-add | ✅ test_field_audit |

---

## 14. Serialization / Parsing Invariants

| # | Invariant | Verified |
|---|-----------|----------|
| SP1 | DER parse → serialize round-trip identity | ✅ test_fuzz_parsers |
| SP2 | Compressed pubkey (33 bytes): `02/03 || x` round-trip | ✅ test_fuzz_parsers |
| SP3 | Uncompressed pubkey (65 bytes): `04 || x || y` round-trip | ✅ test_fuzz_parsers |
| SP4 | Invalid DER: truncated, wrong tag, bad length → error (no crash) | ✅ test_fuzz_parsers |
| SP5 | 10K random blobs: no crash on parse | ✅ test_fuzz_parsers |

---

## Summary

| Category | Total | Verified | Partial | Gap |
|----------|-------|----------|---------|-----|
| Field (F) | 17 | 17 | 0 | 0 |
| Scalar (S) | 9 | 9 | 0 | 0 |
| Point (P) | 14 | 14 | 0 | 0 |
| GLV (G) | 4 | 4 | 0 | 0 |
| ECDSA (E) | 8 | 8 | 0 | 0 |
| Schnorr (B) | 6 | 6 | 0 | 0 |
| MuSig2 (M) | 7 | 7 | 0 | 0 |
| FROST (FR) | 9 | 9 | 0 | 0 |
| BIP-32 (H) | 7 | 7 | 0 | 0 |
| Address (A) | 6 | 6 | 0 | 0 |
| C ABI (C) | 7 | 6 | 1 | 0 |
| CT (CT) | 6 | 4 | 2 | 0 |
| Batch (BP) | 3 | 3 | 0 | 0 |
| Parsing (SP) | 5 | 5 | 0 | 0 |
| **Total** | **108** | **105** | **3** | **0** |

The 3 partial items (CT5, CT6, C7) require formal verification or dedicated thread-safety testing infrastructure that is documented but not fully automated.
