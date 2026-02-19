"""
UltrafastSecp256k1 — Python bindings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
High-performance secp256k1 elliptic curve cryptography.

Usage::

    from ultrafast_secp256k1 import Secp256k1

    lib = Secp256k1()

    # Generate keypair
    privkey = bytes.fromhex("0000000000000000000000000000000000000000000000000000000000000001")
    pubkey = lib.ec_pubkey_create(privkey)

    # ECDSA sign/verify
    msg = b'\\x00' * 32
    sig = lib.ecdsa_sign(msg, privkey)
    assert lib.ecdsa_verify(msg, sig, pubkey)

    # Schnorr sign/verify
    pubkey_x = lib.schnorr_pubkey(privkey)
    aux = b'\\x00' * 32
    sig = lib.schnorr_sign(msg, privkey, aux)
    assert lib.schnorr_verify(msg, sig, pubkey_x)
"""

from ultrafast_secp256k1._ffi import Secp256k1

__all__ = ["Secp256k1"]
__version__ = "1.0.0"
