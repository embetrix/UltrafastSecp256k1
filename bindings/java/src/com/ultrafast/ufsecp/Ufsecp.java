/**
 * UltrafastSecp256k1 — Java binding (ufsecp stable C ABI v1).
 *
 * High-performance secp256k1 ECC with dual-layer constant-time architecture.
 * Context-based API backed by JNI native bridge.
 *
 * Usage:
 *   try (Ufsecp ctx = Ufsecp.create()) {
 *       byte[] pub = ctx.pubkeyCreate(privkey);
 *   }
 */
package com.ultrafast.ufsecp;

public final class Ufsecp implements AutoCloseable {

    static {
        System.loadLibrary("ufsecp_jni");
    }

    // ── Error codes ──────────────────────────────────────────────────

    public static final int NET_MAINNET = 0;
    public static final int NET_TESTNET = 1;

    // ── Instance ─────────────────────────────────────────────────────

    private long ptr;

    private Ufsecp(long ptr) {
        this.ptr = ptr;
    }

    public static Ufsecp create() {
        long p = nativeCreate();
        return new Ufsecp(p);
    }

    @Override
    public void close() {
        if (ptr != 0) {
            nativeDestroy(ptr);
            ptr = 0;
        }
    }

    private void alive() {
        if (ptr == 0) throw new IllegalStateException("UfsecpContext already destroyed");
    }

    // ── Version ──────────────────────────────────────────────────────

    public static int version()         { return nativeVersion(); }
    public static String versionString(){ return nativeVersionString(); }

    // ── Key operations ───────────────────────────────────────────────

    public byte[] pubkeyCreate(byte[] privkey) {
        alive();
        return nativePubkeyCreate(ptr, privkey);
    }

    public byte[] pubkeyCreateUncompressed(byte[] privkey) {
        alive();
        return nativePubkeyCreateUncompressed(ptr, privkey);
    }

    public boolean seckeyVerify(byte[] privkey) {
        alive();
        return nativeSeckeyVerify(ptr, privkey);
    }

    // ── ECDSA ────────────────────────────────────────────────────────

    public byte[] ecdsaSign(byte[] msgHash, byte[] privkey) {
        alive();
        return nativeEcdsaSign(ptr, msgHash, privkey);
    }

    public boolean ecdsaVerify(byte[] msgHash, byte[] sig, byte[] pubkey) {
        alive();
        return nativeEcdsaVerify(ptr, msgHash, sig, pubkey);
    }

    // ── Schnorr ──────────────────────────────────────────────────────

    public byte[] schnorrSign(byte[] msg, byte[] privkey, byte[] auxRand) {
        alive();
        return nativeSchnorrSign(ptr, msg, privkey, auxRand);
    }

    public boolean schnorrVerify(byte[] msg, byte[] sig, byte[] pubkeyX) {
        alive();
        return nativeSchnorrVerify(ptr, msg, sig, pubkeyX);
    }

    // ── ECDH ─────────────────────────────────────────────────────────

    public byte[] ecdh(byte[] privkey, byte[] pubkey) {
        alive();
        return nativeEcdh(ptr, privkey, pubkey);
    }

    // ── Hashing ──────────────────────────────────────────────────────

    public static byte[] sha256(byte[] data)  { return nativeSha256(data); }
    public static byte[] hash160(byte[] data) { return nativeHash160(data); }

    // ── Addresses ────────────────────────────────────────────────────

    public String addrP2pkh(byte[] pubkey, int network) {
        alive();
        return nativeAddrP2pkh(ptr, pubkey, network);
    }

    public String addrP2wpkh(byte[] pubkey, int network) {
        alive();
        return nativeAddrP2wpkh(ptr, pubkey, network);
    }

    public String addrP2tr(byte[] xonly, int network) {
        alive();
        return nativeAddrP2tr(ptr, xonly, network);
    }

    // ── WIF ──────────────────────────────────────────────────────────

    public String wifEncode(byte[] privkey, boolean compressed, int network) {
        alive();
        return nativeWifEncode(ptr, privkey, compressed, network);
    }

    // ── BIP-32 ───────────────────────────────────────────────────────

    public byte[] bip32Master(byte[] seed) {
        alive();
        return nativeBip32Master(ptr, seed);
    }

    public byte[] bip32Derive(byte[] parent, int index) {
        alive();
        return nativeBip32Derive(ptr, parent, index);
    }

    public byte[] bip32DerivePath(byte[] master, String path) {
        alive();
        return nativeBip32DerivePath(ptr, master, path);
    }

    // ── Native declarations ──────────────────────────────────────────

    private static native long nativeCreate();
    private static native void nativeDestroy(long ptr);

    private static native int nativeVersion();
    private static native String nativeVersionString();

    private static native byte[] nativePubkeyCreate(long ctx, byte[] privkey);
    private static native byte[] nativePubkeyCreateUncompressed(long ctx, byte[] privkey);
    private static native boolean nativeSeckeyVerify(long ctx, byte[] privkey);

    private static native byte[] nativeEcdsaSign(long ctx, byte[] msgHash, byte[] privkey);
    private static native boolean nativeEcdsaVerify(long ctx, byte[] msgHash, byte[] sig, byte[] pubkey);

    private static native byte[] nativeSchnorrSign(long ctx, byte[] msg, byte[] privkey, byte[] auxRand);
    private static native boolean nativeSchnorrVerify(long ctx, byte[] msg, byte[] sig, byte[] pubkeyX);

    private static native byte[] nativeEcdh(long ctx, byte[] privkey, byte[] pubkey);

    private static native byte[] nativeSha256(byte[] data);
    private static native byte[] nativeHash160(byte[] data);

    private static native String nativeAddrP2pkh(long ctx, byte[] pubkey, int network);
    private static native String nativeAddrP2wpkh(long ctx, byte[] pubkey, int network);
    private static native String nativeAddrP2tr(long ctx, byte[] xonly, int network);

    private static native String nativeWifEncode(long ctx, byte[] privkey, boolean compressed, int network);

    private static native byte[] nativeBip32Master(long ctx, byte[] seed);
    private static native byte[] nativeBip32Derive(long ctx, byte[] parent, int index);
    private static native byte[] nativeBip32DerivePath(long ctx, byte[] master, String path);
}
