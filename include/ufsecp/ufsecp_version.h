/* ============================================================================
 * UltrafastSecp256k1 — Version & ABI Compatibility
 * ============================================================================
 * RULES:
 *   UFSECP_VERSION_MAJOR bump  →  ABI breaking (struct layout, removed funcs)
 *   UFSECP_VERSION_MINOR bump  →  ABI compatible (new funcs only)
 *   UFSECP_VERSION_PATCH bump  →  ABI compatible (bugfixes only)
 *   UFSECP_ABI_VERSION   bump  →  only on ABI-incompatible changes
 *
 * Clients should check:  ufsecp_abi_version() == expected_abi
 * ============================================================================ */

#ifndef UFSECP_VERSION_H
#define UFSECP_VERSION_H

#ifdef __cplusplus
extern "C" {
#endif

/* ── Compile-time version ─────────────────────────────────────────────────── */

#define UFSECP_VERSION_MAJOR   3
#define UFSECP_VERSION_MINOR   4
#define UFSECP_VERSION_PATCH   0

/** Packed: (major << 16) | (minor << 8) | patch.  Compare with >= for compat. */
#define UFSECP_VERSION_PACKED \
    ((UFSECP_VERSION_MAJOR << 16) | (UFSECP_VERSION_MINOR << 8) | UFSECP_VERSION_PATCH)

#define UFSECP_VERSION_STRING  "3.4.0"

/* ── ABI version (incremented ONLY on binary-incompatible changes) ────────── */

#define UFSECP_ABI_VERSION     1

/* ── Runtime queries ──────────────────────────────────────────────────────── */

#ifndef UFSECP_API
  #if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef UFSECP_BUILDING
      #define UFSECP_API __declspec(dllexport)
    #else
      #define UFSECP_API __declspec(dllimport)
    #endif
  #elif __GNUC__ >= 4
    #define UFSECP_API __attribute__((visibility("default")))
  #else
    #define UFSECP_API
  #endif
#endif

/** Return packed version at runtime (same as UFSECP_VERSION_PACKED). */
UFSECP_API unsigned int ufsecp_version(void);

/** Return ABI version at runtime (same as UFSECP_ABI_VERSION). */
UFSECP_API unsigned int ufsecp_abi_version(void);

/** Return human-readable version string, e.g. "3.3.0". */
UFSECP_API const char* ufsecp_version_string(void);

#ifdef __cplusplus
}
#endif

#endif /* UFSECP_VERSION_H */
