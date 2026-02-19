// ============================================================================
// shim_context.cpp — Context lifecycle (no-op for UltrafastSecp256k1)
// ============================================================================
#include "secp256k1.h"
#include <cstdlib>
#include <cstring>

// UltrafastSecp256k1 is stateless — contexts are opaque dummies.
// We allocate a small sentinel so null-checks in user code pass.

struct secp256k1_context_struct {
    unsigned int flags;
};

static secp256k1_context_struct g_static_ctx = { SECP256K1_CONTEXT_NONE };

extern "C" {

const secp256k1_context * const secp256k1_context_static = &g_static_ctx;

secp256k1_context *secp256k1_context_create(unsigned int flags) {
    (void)flags;
    auto *ctx = static_cast<secp256k1_context *>(std::malloc(sizeof(secp256k1_context)));
    if (ctx) ctx->flags = flags;
    return ctx;
}

secp256k1_context *secp256k1_context_clone(const secp256k1_context *ctx) {
    if (!ctx) return nullptr;
    auto *clone = static_cast<secp256k1_context *>(std::malloc(sizeof(secp256k1_context)));
    if (clone) std::memcpy(clone, ctx, sizeof(secp256k1_context));
    return clone;
}

void secp256k1_context_destroy(secp256k1_context *ctx) {
    if (ctx && ctx != &g_static_ctx) std::free(ctx);
}

int secp256k1_context_randomize(secp256k1_context *ctx, const unsigned char *seed32) {
    (void)ctx; (void)seed32;
    // UltrafastSecp256k1 does not use blinding — accepted as no-op.
    return 1;
}

void secp256k1_selftest(void) {
    // The underlying library has its own selftest; this is a compatibility stub.
}

} // extern "C"
