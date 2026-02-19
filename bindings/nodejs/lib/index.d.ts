/**
 * UltrafastSecp256k1 — TypeScript type definitions
 */

export declare const NETWORK_MAINNET: 0;
export declare const NETWORK_TESTNET: 1;

export declare class Secp256k1 {
    constructor(libPath?: string);

    version(): string;

    // Key operations
    ecPubkeyCreate(privkey: Buffer): Buffer;
    ecPubkeyCreateUncompressed(privkey: Buffer): Buffer;
    ecPubkeyParse(pubkey: Buffer): Buffer;
    ecSeckeyVerify(privkey: Buffer): boolean;
    ecPrivkeyNegate(privkey: Buffer): Buffer;
    ecPrivkeyTweakAdd(privkey: Buffer, tweak: Buffer): Buffer;
    ecPrivkeyTweakMul(privkey: Buffer, tweak: Buffer): Buffer;

    // ECDSA
    ecdsaSign(msgHash: Buffer, privkey: Buffer): Buffer;
    ecdsaVerify(msgHash: Buffer, sig: Buffer, pubkey: Buffer): boolean;
    ecdsaSerializeDer(sig: Buffer): Buffer;

    // Recovery
    ecdsaSignRecoverable(msgHash: Buffer, privkey: Buffer): { signature: Buffer; recoveryId: number };
    ecdsaRecover(msgHash: Buffer, sig: Buffer, recid: number): Buffer;

    // Schnorr (BIP-340)
    schnorrSign(msg: Buffer, privkey: Buffer, auxRand: Buffer): Buffer;
    schnorrVerify(msg: Buffer, sig: Buffer, pubkeyX: Buffer): boolean;
    schnorrPubkey(privkey: Buffer): Buffer;

    // ECDH
    ecdh(privkey: Buffer, pubkey: Buffer): Buffer;
    ecdhXonly(privkey: Buffer, pubkey: Buffer): Buffer;
    ecdhRaw(privkey: Buffer, pubkey: Buffer): Buffer;

    // Hashing
    sha256(data: Buffer): Buffer;
    hash160(data: Buffer): Buffer;
    taggedHash(tag: string, data: Buffer): Buffer;

    // Addresses
    addressP2PKH(pubkey: Buffer, network?: number): string;
    addressP2WPKH(pubkey: Buffer, network?: number): string;
    addressP2TR(internalKeyX: Buffer, network?: number): string;

    // WIF
    wifEncode(privkey: Buffer, compressed?: boolean, network?: number): string;
    wifDecode(wif: string): { privkey: Buffer; compressed: boolean; network: number };

    // BIP-32
    bip32MasterKey(seed: Buffer): Buffer;
    bip32DerivePath(masterKey: Buffer, path: string): Buffer;
    bip32GetPrivkey(key: Buffer): Buffer;
    bip32GetPubkey(key: Buffer): Buffer;

    // Taproot
    taprootOutputKey(internalKeyX: Buffer, merkleRoot?: Buffer | null): { outputKeyX: Buffer; parity: number };
    taprootTweakPrivkey(privkey: Buffer, merkleRoot?: Buffer | null): Buffer;
}
