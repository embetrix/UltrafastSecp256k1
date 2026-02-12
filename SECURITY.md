# Security Policy

## Supported Versions

UltrafastSecp256k1 is under active development.  
Security fixes and updates apply to the latest commit on the `main` branch.

Older revisions may not receive patches.

---

## Reporting a Vulnerability

If you discover a potential security issue related to:

- Incorrect field arithmetic
- Scalar arithmetic inconsistencies
- Point operation errors
- Determinism or constant-time violations
- Undefined behavior affecting cryptographic correctness

Please report it responsibly.

Do NOT open a public issue for suspected vulnerabilities.

Instead, contact the maintainer directly via GitHub private message or email (if provided).

We will investigate and respond as soon as possible.

---

## Scope

UltrafastSecp256k1 provides elliptic curve and finite field primitives only.

This repository does not include key-search tooling or application-layer cryptographic protocols.

Security responsibility for higher-level integrations remains with the integrating application.
