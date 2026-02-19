/**
 * UltrafastSecp256k1 — React Native iOS bridge (ufsecp stable C ABI v1).
 *
 * ObjC bridge that links against libufsecp.a (static) and exposes
 * methods to JS via RCT_EXPORT_METHOD.
 */

#import <React/RCTBridgeModule.h>
#import "ufsecp.h"

@interface RNUfsecp : NSObject <RCTBridgeModule>
@end

@implementation RNUfsecp

RCT_EXPORT_MODULE(Ufsecp)

/* ── Hex helpers ───────────────────────────────────────────────────── */

static NSData *hexToData(NSString *hex) {
    NSMutableData *d = [NSMutableData dataWithCapacity:hex.length / 2];
    unsigned char byte;
    char tmp[3] = {0};
    for (NSUInteger i = 0; i < hex.length; i += 2) {
        tmp[0] = [hex characterAtIndex:i];
        tmp[1] = [hex characterAtIndex:i + 1];
        byte = (unsigned char)strtol(tmp, NULL, 16);
        [d appendBytes:&byte length:1];
    }
    return d;
}

static NSString *dataToHex(const uint8_t *data, size_t len) {
    NSMutableString *s = [NSMutableString stringWithCapacity:len * 2];
    for (size_t i = 0; i < len; i++) [s appendFormat:@"%02x", data[i]];
    return s;
}

/* ── Context ───────────────────────────────────────────────────────── */

RCT_EXPORT_METHOD(create:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject) {
    ufsecp_ctx *ctx = NULL;
    int rc = ufsecp_ctx_create(&ctx);
    if (rc != 0) {
        reject(@"UFSECP", [NSString stringWithFormat:@"ctx_create failed: %d", rc], nil);
        return;
    }
    resolve(@((double)(uintptr_t)ctx));
}

RCT_EXPORT_METHOD(destroy:(double)handle resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject) {
    ufsecp_ctx *ctx = (ufsecp_ctx *)(uintptr_t)(long long)handle;
    if (ctx) ufsecp_ctx_destroy(ctx);
    resolve(nil);
}

/* ── Version ───────────────────────────────────────────────────────── */

RCT_EXPORT_METHOD(version:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject) {
    resolve(@(ufsecp_version()));
}

RCT_EXPORT_METHOD(versionString:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject) {
    resolve([NSString stringWithUTF8String:ufsecp_version_string()]);
}

/* ── Key ops ───────────────────────────────────────────────────────── */

RCT_EXPORT_METHOD(pubkeyCreate:(double)handle privkey:(NSString *)pkHex
                  resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject) {
    ufsecp_ctx *ctx = (ufsecp_ctx *)(uintptr_t)(long long)handle;
    NSData *pk = hexToData(pkHex);
    uint8_t out[33];
    int rc = ufsecp_pubkey_create(ctx, pk.bytes, out);
    if (rc != 0) { reject(@"UFSECP", @"pubkey_create failed", nil); return; }
    resolve(dataToHex(out, 33));
}

RCT_EXPORT_METHOD(pubkeyCreateUncompressed:(double)handle privkey:(NSString *)pkHex
                  resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject) {
    ufsecp_ctx *ctx = (ufsecp_ctx *)(uintptr_t)(long long)handle;
    NSData *pk = hexToData(pkHex);
    uint8_t out[65];
    int rc = ufsecp_pubkey_create_uncompressed(ctx, pk.bytes, out);
    if (rc != 0) { reject(@"UFSECP", @"pubkey_create_uncompressed failed", nil); return; }
    resolve(dataToHex(out, 65));
}

RCT_EXPORT_METHOD(seckeyVerify:(double)handle privkey:(NSString *)pkHex
                  resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject) {
    ufsecp_ctx *ctx = (ufsecp_ctx *)(uintptr_t)(long long)handle;
    NSData *pk = hexToData(pkHex);
    int rc = ufsecp_seckey_verify(ctx, pk.bytes);
    resolve(@(rc == 0));
}

/* ── ECDSA ─────────────────────────────────────────────────────────── */

RCT_EXPORT_METHOD(ecdsaSign:(double)handle msgHash:(NSString *)msgHex privkey:(NSString *)pkHex
                  resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject) {
    ufsecp_ctx *ctx = (ufsecp_ctx *)(uintptr_t)(long long)handle;
    NSData *msg = hexToData(msgHex);
    NSData *pk = hexToData(pkHex);
    uint8_t sig[64];
    int rc = ufsecp_ecdsa_sign(ctx, msg.bytes, pk.bytes, sig);
    if (rc != 0) { reject(@"UFSECP", @"ecdsa_sign failed", nil); return; }
    resolve(dataToHex(sig, 64));
}

RCT_EXPORT_METHOD(ecdsaVerify:(double)handle msgHash:(NSString *)msgHex sig:(NSString *)sigHex pubkey:(NSString *)pubHex
                  resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject) {
    ufsecp_ctx *ctx = (ufsecp_ctx *)(uintptr_t)(long long)handle;
    NSData *msg = hexToData(msgHex);
    NSData *sig = hexToData(sigHex);
    NSData *pub = hexToData(pubHex);
    int rc = ufsecp_ecdsa_verify(ctx, msg.bytes, sig.bytes, pub.bytes);
    resolve(@(rc == 0));
}

/* ── Schnorr ───────────────────────────────────────────────────────── */

RCT_EXPORT_METHOD(schnorrSign:(double)handle msg:(NSString *)msgHex privkey:(NSString *)pkHex auxRand:(NSString *)arHex
                  resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject) {
    ufsecp_ctx *ctx = (ufsecp_ctx *)(uintptr_t)(long long)handle;
    NSData *msg = hexToData(msgHex);
    NSData *pk = hexToData(pkHex);
    NSData *ar = hexToData(arHex);
    uint8_t sig[64];
    int rc = ufsecp_schnorr_sign(ctx, msg.bytes, pk.bytes, ar.bytes, sig);
    if (rc != 0) { reject(@"UFSECP", @"schnorr_sign failed", nil); return; }
    resolve(dataToHex(sig, 64));
}

RCT_EXPORT_METHOD(schnorrVerify:(double)handle msg:(NSString *)msgHex sig:(NSString *)sigHex pubkeyX:(NSString *)pxHex
                  resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject) {
    ufsecp_ctx *ctx = (ufsecp_ctx *)(uintptr_t)(long long)handle;
    NSData *msg = hexToData(msgHex);
    NSData *sig = hexToData(sigHex);
    NSData *px = hexToData(pxHex);
    int rc = ufsecp_schnorr_verify(ctx, msg.bytes, sig.bytes, px.bytes);
    resolve(@(rc == 0));
}

/* ── ECDH ──────────────────────────────────────────────────────────── */

RCT_EXPORT_METHOD(ecdh:(double)handle privkey:(NSString *)pkHex pubkey:(NSString *)pubHex
                  resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject) {
    ufsecp_ctx *ctx = (ufsecp_ctx *)(uintptr_t)(long long)handle;
    NSData *pk = hexToData(pkHex);
    NSData *pub = hexToData(pubHex);
    uint8_t out[32];
    int rc = ufsecp_ecdh(ctx, pk.bytes, pub.bytes, out);
    if (rc != 0) { reject(@"UFSECP", @"ecdh failed", nil); return; }
    resolve(dataToHex(out, 32));
}

/* ── Hashing ───────────────────────────────────────────────────────── */

RCT_EXPORT_METHOD(sha256:(NSString *)dataHex
                  resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject) {
    NSData *d = hexToData(dataHex);
    uint8_t out[32];
    int rc = ufsecp_sha256(d.bytes, d.length, out);
    if (rc != 0) { reject(@"UFSECP", @"sha256 failed", nil); return; }
    resolve(dataToHex(out, 32));
}

RCT_EXPORT_METHOD(hash160:(NSString *)dataHex
                  resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject) {
    NSData *d = hexToData(dataHex);
    uint8_t out[20];
    int rc = ufsecp_hash160(d.bytes, d.length, out);
    if (rc != 0) { reject(@"UFSECP", @"hash160 failed", nil); return; }
    resolve(dataToHex(out, 20));
}

/* ── Addresses ─────────────────────────────────────────────────────── */

RCT_EXPORT_METHOD(addrP2pkh:(double)handle pubkey:(NSString *)pubHex network:(int)network
                  resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject) {
    ufsecp_ctx *ctx = (ufsecp_ctx *)(uintptr_t)(long long)handle;
    NSData *pk = hexToData(pubHex);
    uint8_t addr[128]; size_t alen = 128;
    int rc = ufsecp_addr_p2pkh(ctx, pk.bytes, network, addr, &alen);
    if (rc != 0) { reject(@"UFSECP", @"addr_p2pkh failed", nil); return; }
    resolve([[NSString alloc] initWithBytes:addr length:alen encoding:NSUTF8StringEncoding]);
}

RCT_EXPORT_METHOD(addrP2wpkh:(double)handle pubkey:(NSString *)pubHex network:(int)network
                  resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject) {
    ufsecp_ctx *ctx = (ufsecp_ctx *)(uintptr_t)(long long)handle;
    NSData *pk = hexToData(pubHex);
    uint8_t addr[128]; size_t alen = 128;
    int rc = ufsecp_addr_p2wpkh(ctx, pk.bytes, network, addr, &alen);
    if (rc != 0) { reject(@"UFSECP", @"addr_p2wpkh failed", nil); return; }
    resolve([[NSString alloc] initWithBytes:addr length:alen encoding:NSUTF8StringEncoding]);
}

RCT_EXPORT_METHOD(addrP2tr:(double)handle xonly:(NSString *)xHex network:(int)network
                  resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject) {
    ufsecp_ctx *ctx = (ufsecp_ctx *)(uintptr_t)(long long)handle;
    NSData *x = hexToData(xHex);
    uint8_t addr[128]; size_t alen = 128;
    int rc = ufsecp_addr_p2tr(ctx, x.bytes, network, addr, &alen);
    if (rc != 0) { reject(@"UFSECP", @"addr_p2tr failed", nil); return; }
    resolve([[NSString alloc] initWithBytes:addr length:alen encoding:NSUTF8StringEncoding]);
}

/* ── WIF ───────────────────────────────────────────────────────────── */

RCT_EXPORT_METHOD(wifEncode:(double)handle privkey:(NSString *)pkHex compressed:(BOOL)compressed network:(int)network
                  resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject) {
    ufsecp_ctx *ctx = (ufsecp_ctx *)(uintptr_t)(long long)handle;
    NSData *pk = hexToData(pkHex);
    uint8_t wif[128]; size_t wlen = 128;
    int rc = ufsecp_wif_encode(ctx, pk.bytes, compressed ? 1 : 0, network, wif, &wlen);
    if (rc != 0) { reject(@"UFSECP", @"wif_encode failed", nil); return; }
    resolve([[NSString alloc] initWithBytes:wif length:wlen encoding:NSUTF8StringEncoding]);
}

/* ── BIP-32 ────────────────────────────────────────────────────────── */

RCT_EXPORT_METHOD(bip32Master:(double)handle seed:(NSString *)seedHex
                  resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject) {
    ufsecp_ctx *ctx = (ufsecp_ctx *)(uintptr_t)(long long)handle;
    NSData *seed = hexToData(seedHex);
    uint8_t key[82];
    int rc = ufsecp_bip32_master(ctx, seed.bytes, seed.length, key);
    if (rc != 0) { reject(@"UFSECP", @"bip32_master failed", nil); return; }
    resolve(dataToHex(key, 82));
}

RCT_EXPORT_METHOD(bip32Derive:(double)handle parent:(NSString *)parentHex index:(int)index
                  resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject) {
    ufsecp_ctx *ctx = (ufsecp_ctx *)(uintptr_t)(long long)handle;
    NSData *p = hexToData(parentHex);
    uint8_t child[82];
    int rc = ufsecp_bip32_derive(ctx, p.bytes, (uint32_t)index, child);
    if (rc != 0) { reject(@"UFSECP", @"bip32_derive failed", nil); return; }
    resolve(dataToHex(child, 82));
}

RCT_EXPORT_METHOD(bip32DerivePath:(double)handle master:(NSString *)masterHex path:(NSString *)pathStr
                  resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject) {
    ufsecp_ctx *ctx = (ufsecp_ctx *)(uintptr_t)(long long)handle;
    NSData *m = hexToData(masterHex);
    uint8_t key[82];
    int rc = ufsecp_bip32_derive_path(ctx, m.bytes, [pathStr UTF8String], key);
    if (rc != 0) { reject(@"UFSECP", @"bip32_derive_path failed", nil); return; }
    resolve(dataToHex(key, 82));
}

@end
