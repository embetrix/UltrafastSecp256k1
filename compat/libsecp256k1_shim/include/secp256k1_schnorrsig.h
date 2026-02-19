/* ============================================================================
 * libsecp256k1-compatible Schnorr API — backed by UltrafastSecp256k1
 * ========================================================================== */
#ifndef SECP256K1_SCHNORRSIG_H
#define SECP256K1_SCHNORRSIG_H

#include "secp256k1.h"
#include "secp256k1_extrakeys.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Nonce function type for Schnorr ────────────────────────────────────── */
typedef int (*secp256k1_nonce_function_hardened)(
    unsigned char *nonce32,
    const unsigned char *msg, size_t msglen,
    const unsigned char *key32,
    const unsigned char *xonly_pk32,
    const unsigned char *algo, size_t algolen,
    void *data
);

SECP256K1_API const secp256k1_nonce_function_hardened secp256k1_nonce_function_bip340;

/* ── Extra params ───────────────────────────────────────────────────────── */
typedef struct secp256k1_schnorrsig_extraparams {
    unsigned char magic[4];
    secp256k1_nonce_function_hardened noncefp;
    void *ndata;
} secp256k1_schnorrsig_extraparams;

#define SECP256K1_SCHNORRSIG_EXTRAPARAMS_MAGIC { 0xda, 0x6f, 0xb3, 0x8c }
#define SECP256K1_SCHNORRSIG_EXTRAPARAMS_INIT { \
    SECP256K1_SCHNORRSIG_EXTRAPARAMS_MAGIC, \
    NULL, \
    NULL  \
}

/* ── Sign / Verify ──────────────────────────────────────────────────────── */
SECP256K1_API int secp256k1_schnorrsig_sign32(
    const secp256k1_context *ctx,
    unsigned char *sig64,
    const unsigned char *msg32,
    const secp256k1_keypair *keypair,
    const unsigned char *aux_rand32);

SECP256K1_API int secp256k1_schnorrsig_verify(
    const secp256k1_context *ctx,
    const unsigned char *sig64,
    const unsigned char *msg, size_t msglen,
    const secp256k1_xonly_pubkey *pubkey);

#ifdef __cplusplus
}
#endif

#endif /* SECP256K1_SCHNORRSIG_H */
