/* UltrafastSecp256k1 — JNI bridge (ufsecp stable C ABI v1).
 *
 * Each native method maps 1:1 to a ufsecp_* C function.
 * The Java class holds the opaque context pointer as a long.
 */

#include <jni.h>
#include <string.h>
#include "ufsecp.h"

/* Helper: throw Java exception on non-zero return code. Returns 0 on success. */
static int throw_on_err(JNIEnv *env, int rc, const char *op) {
    if (rc == 0) return 0;
    char msg[128];
    snprintf(msg, sizeof(msg), "ufsecp %s failed: %s (%d)", op, ufsecp_error_str(rc), rc);
    jclass cls = (*env)->FindClass(env, "com/ultrafast/ufsecp/UfsecpException");
    if (cls) (*env)->ThrowNew(env, cls, msg);
    return rc;
}

/* Helper: pin byte array, copy into fixed buf, unpin. Returns pinned pointer (must release). */
static jbyte* pin(JNIEnv *env, jbyteArray arr) {
    return (*env)->GetByteArrayElements(env, arr, NULL);
}
static void unpin(JNIEnv *env, jbyteArray arr, jbyte *ptr) {
    (*env)->ReleaseByteArrayElements(env, arr, ptr, JNI_ABORT);
}
static jbyteArray mk(JNIEnv *env, const uint8_t *data, int len) {
    jbyteArray r = (*env)->NewByteArray(env, len);
    if (r) (*env)->SetByteArrayRegion(env, r, 0, len, (const jbyte*)data);
    return r;
}

/* ── Context ─────────────────────────────────────────────────────────── */

JNIEXPORT jlong JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeCreate(JNIEnv *env, jclass clz) {
    (void)clz;
    ufsecp_ctx *ctx = NULL;
    int rc = ufsecp_ctx_create(&ctx);
    if (throw_on_err(env, rc, "ctx_create")) return 0;
    return (jlong)(uintptr_t)ctx;
}

JNIEXPORT void JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeDestroy(JNIEnv *env, jclass clz, jlong ptr) {
    (void)env; (void)clz;
    if (ptr) ufsecp_ctx_destroy((ufsecp_ctx*)(uintptr_t)ptr);
}

/* ── Key ops ─────────────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativePubkeyCreate(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray privkey) {
    (void)clz;
    jbyte *pk = pin(env, privkey);
    uint8_t out[33];
    int rc = ufsecp_pubkey_create((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)pk, out);
    unpin(env, privkey, pk);
    if (throw_on_err(env, rc, "pubkey_create")) return NULL;
    return mk(env, out, 33);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativePubkeyCreateUncompressed(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray privkey) {
    (void)clz;
    jbyte *pk = pin(env, privkey);
    uint8_t out[65];
    int rc = ufsecp_pubkey_create_uncompressed((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)pk, out);
    unpin(env, privkey, pk);
    if (throw_on_err(env, rc, "pubkey_create_uncompressed")) return NULL;
    return mk(env, out, 65);
}

JNIEXPORT jboolean JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeSeckeyVerify(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray privkey) {
    (void)clz;
    jbyte *pk = pin(env, privkey);
    int rc = ufsecp_seckey_verify((const ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)pk);
    unpin(env, privkey, pk);
    return rc == 0 ? JNI_TRUE : JNI_FALSE;
}

/* ── ECDSA ───────────────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeEcdsaSign(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray msgHash, jbyteArray privkey) {
    (void)clz;
    jbyte *msg = pin(env, msgHash);
    jbyte *pk  = pin(env, privkey);
    uint8_t sig[64];
    int rc = ufsecp_ecdsa_sign((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)msg, (const uint8_t*)pk, sig);
    unpin(env, privkey, pk);
    unpin(env, msgHash, msg);
    if (throw_on_err(env, rc, "ecdsa_sign")) return NULL;
    return mk(env, sig, 64);
}

JNIEXPORT jboolean JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeEcdsaVerify(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray msgHash, jbyteArray sig, jbyteArray pubkey) {
    (void)clz;
    jbyte *msg = pin(env, msgHash);
    jbyte *s   = pin(env, sig);
    jbyte *pk  = pin(env, pubkey);
    int rc = ufsecp_ecdsa_verify((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)msg, (const uint8_t*)s, (const uint8_t*)pk);
    unpin(env, pubkey, pk);
    unpin(env, sig, s);
    unpin(env, msgHash, msg);
    return rc == 0 ? JNI_TRUE : JNI_FALSE;
}

/* ── Schnorr ─────────────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeSchnorrSign(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray msg, jbyteArray privkey, jbyteArray auxRand) {
    (void)clz;
    jbyte *m  = pin(env, msg);
    jbyte *pk = pin(env, privkey);
    jbyte *ar = pin(env, auxRand);
    uint8_t sig[64];
    int rc = ufsecp_schnorr_sign((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)m, (const uint8_t*)pk, (const uint8_t*)ar, sig);
    unpin(env, auxRand, ar);
    unpin(env, privkey, pk);
    unpin(env, msg, m);
    if (throw_on_err(env, rc, "schnorr_sign")) return NULL;
    return mk(env, sig, 64);
}

JNIEXPORT jboolean JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeSchnorrVerify(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray msg, jbyteArray sig, jbyteArray pubkeyX) {
    (void)clz;
    jbyte *m  = pin(env, msg);
    jbyte *s  = pin(env, sig);
    jbyte *pk = pin(env, pubkeyX);
    int rc = ufsecp_schnorr_verify((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)m, (const uint8_t*)s, (const uint8_t*)pk);
    unpin(env, pubkeyX, pk);
    unpin(env, sig, s);
    unpin(env, msg, m);
    return rc == 0 ? JNI_TRUE : JNI_FALSE;
}

/* ── ECDH ────────────────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeEcdh(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray privkey, jbyteArray pubkey) {
    (void)clz;
    jbyte *pk  = pin(env, privkey);
    jbyte *pub = pin(env, pubkey);
    uint8_t out[32];
    int rc = ufsecp_ecdh((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)pk, (const uint8_t*)pub, out);
    unpin(env, pubkey, pub);
    unpin(env, privkey, pk);
    if (throw_on_err(env, rc, "ecdh")) return NULL;
    return mk(env, out, 32);
}

/* ── Hashing ─────────────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeSha256(
    JNIEnv *env, jclass clz, jbyteArray data) {
    (void)clz;
    jbyte *d = pin(env, data);
    jsize len = (*env)->GetArrayLength(env, data);
    uint8_t out[32];
    int rc = ufsecp_sha256((const uint8_t*)d, (size_t)len, out);
    unpin(env, data, d);
    if (throw_on_err(env, rc, "sha256")) return NULL;
    return mk(env, out, 32);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeHash160(
    JNIEnv *env, jclass clz, jbyteArray data) {
    (void)clz;
    jbyte *d = pin(env, data);
    jsize len = (*env)->GetArrayLength(env, data);
    uint8_t out[20];
    int rc = ufsecp_hash160((const uint8_t*)d, (size_t)len, out);
    unpin(env, data, d);
    if (throw_on_err(env, rc, "hash160")) return NULL;
    return mk(env, out, 20);
}

/* ── Addresses ───────────────────────────────────────────────────────── */

JNIEXPORT jstring JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeAddrP2pkh(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray pubkey, jint network) {
    (void)clz;
    jbyte *pk = pin(env, pubkey);
    uint8_t addr[128];
    size_t alen = 128;
    int rc = ufsecp_addr_p2pkh((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)pk, (int)network, addr, &alen);
    unpin(env, pubkey, pk);
    if (throw_on_err(env, rc, "addr_p2pkh")) return NULL;
    addr[alen] = '\0';
    return (*env)->NewStringUTF(env, (const char*)addr);
}

JNIEXPORT jstring JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeAddrP2wpkh(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray pubkey, jint network) {
    (void)clz;
    jbyte *pk = pin(env, pubkey);
    uint8_t addr[128];
    size_t alen = 128;
    int rc = ufsecp_addr_p2wpkh((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)pk, (int)network, addr, &alen);
    unpin(env, pubkey, pk);
    if (throw_on_err(env, rc, "addr_p2wpkh")) return NULL;
    addr[alen] = '\0';
    return (*env)->NewStringUTF(env, (const char*)addr);
}

JNIEXPORT jstring JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeAddrP2tr(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray xonly, jint network) {
    (void)clz;
    jbyte *pk = pin(env, xonly);
    uint8_t addr[128];
    size_t alen = 128;
    int rc = ufsecp_addr_p2tr((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)pk, (int)network, addr, &alen);
    unpin(env, xonly, pk);
    if (throw_on_err(env, rc, "addr_p2tr")) return NULL;
    addr[alen] = '\0';
    return (*env)->NewStringUTF(env, (const char*)addr);
}

/* ── WIF ─────────────────────────────────────────────────────────────── */

JNIEXPORT jstring JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeWifEncode(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray privkey, jboolean compressed, jint network) {
    (void)clz;
    jbyte *pk = pin(env, privkey);
    uint8_t wif[128];
    size_t wlen = 128;
    int rc = ufsecp_wif_encode((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)pk,
                               compressed ? 1 : 0, (int)network, wif, &wlen);
    unpin(env, privkey, pk);
    if (throw_on_err(env, rc, "wif_encode")) return NULL;
    wif[wlen] = '\0';
    return (*env)->NewStringUTF(env, (const char*)wif);
}

/* ── BIP-32 ──────────────────────────────────────────────────────────── */

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeBip32Master(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray seed) {
    (void)clz;
    jbyte *s = pin(env, seed);
    jsize slen = (*env)->GetArrayLength(env, seed);
    uint8_t key[82];
    int rc = ufsecp_bip32_master((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)s, (size_t)slen, key);
    unpin(env, seed, s);
    if (throw_on_err(env, rc, "bip32_master")) return NULL;
    return mk(env, key, 82);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeBip32Derive(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray parent, jint index) {
    (void)clz;
    jbyte *p = pin(env, parent);
    uint8_t child[82];
    int rc = ufsecp_bip32_derive((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)p, (uint32_t)index, child);
    unpin(env, parent, p);
    if (throw_on_err(env, rc, "bip32_derive")) return NULL;
    return mk(env, child, 82);
}

JNIEXPORT jbyteArray JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeBip32DerivePath(
    JNIEnv *env, jclass clz, jlong ctx, jbyteArray master, jstring path) {
    (void)clz;
    jbyte *m = pin(env, master);
    const char *p = (*env)->GetStringUTFChars(env, path, NULL);
    uint8_t key[82];
    int rc = ufsecp_bip32_derive_path((ufsecp_ctx*)(uintptr_t)ctx, (const uint8_t*)m, p, key);
    (*env)->ReleaseStringUTFChars(env, path, p);
    unpin(env, master, m);
    if (throw_on_err(env, rc, "bip32_derive_path")) return NULL;
    return mk(env, key, 82);
}

/* ── Version ─────────────────────────────────────────────────────────── */

JNIEXPORT jint JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeVersion(JNIEnv *env, jclass clz) {
    (void)env; (void)clz;
    return (jint)ufsecp_version();
}

JNIEXPORT jstring JNICALL Java_com_ultrafast_ufsecp_Ufsecp_nativeVersionString(JNIEnv *env, jclass clz) {
    (void)clz;
    return (*env)->NewStringUTF(env, ufsecp_version_string());
}
