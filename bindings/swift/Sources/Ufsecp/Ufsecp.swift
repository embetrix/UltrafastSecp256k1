/// UltrafastSecp256k1 — Swift binding (ufsecp stable C ABI v1).
///
/// High-performance secp256k1 ECC with dual-layer constant-time architecture.
/// Context-based API.
///
/// Usage:
///     let ctx = try UfsecpContext()
///     let pubkey = try ctx.pubkeyCreate(privkey: Data(repeating: 0, count: 31) + Data([0x01]))
///     ctx.destroy()

import Foundation
#if canImport(CUfsecp)
import CUfsecp
#endif

// MARK: - Error

public enum UfsecpErrorCode: Int32 {
    case ok = 0
    case nullArg = 1
    case badKey = 2
    case badPubkey = 3
    case badSig = 4
    case badInput = 5
    case verifyFail = 6
    case arith = 7
    case selftest = 8
    case `internal` = 9
    case bufTooSmall = 10
}

public struct UfsecpError: Error, CustomStringConvertible {
    public let operation: String
    public let code: UfsecpErrorCode

    public var description: String {
        "ufsecp \(operation) failed: \(code) (\(code.rawValue))"
    }
}

// MARK: - Result types

public enum Network: Int32 {
    case mainnet = 0
    case testnet = 1
}

public struct RecoverableSignature {
    public let signature: Data
    public let recoveryId: Int32
}

public struct TaprootOutputKeyResult {
    public let outputKeyX: Data
    public let parity: Int32
}

public struct WifDecoded {
    public let privkey: Data
    public let compressed: Bool
    public let network: Network
}

// MARK: - Context

public final class UfsecpContext {
    private var ctx: OpaquePointer?
    private var destroyed = false

    public init() throws {
        var ptr: OpaquePointer?
        let rc = ufsecp_ctx_create(&ptr)
        guard rc == 0, let p = ptr else {
            throw UfsecpError(operation: "ctx_create", code: UfsecpErrorCode(rawValue: rc) ?? .internal)
        }
        self.ctx = p
    }

    deinit { destroy() }

    public func destroy() {
        guard !destroyed, let c = ctx else { return }
        ufsecp_ctx_destroy(c)
        ctx = nil
        destroyed = true
    }

    // MARK: Version

    public static var version: UInt32 { ufsecp_version() }
    public static var abiVersion: UInt32 { ufsecp_abi_version() }
    public static var versionString: String { String(cString: ufsecp_version_string()) }

    public var lastError: Int32 { try! alive(); return ufsecp_last_error(ctx!) }
    public var lastErrorMsg: String { try! alive(); return String(cString: ufsecp_last_error_msg(ctx!)) }

    // MARK: Key Operations

    public func pubkeyCreate(privkey: Data) throws -> Data {
        try chk(privkey, 32, "privkey"); try alive()
        var out = [UInt8](repeating: 0, count: 33)
        try privkey.withUnsafeBytes { pk in
            try throwRC(ufsecp_pubkey_create(ctx!, pk.baseAddress!.assumingMemoryBound(to: UInt8.self), &out), "pubkey_create")
        }
        return Data(out)
    }

    public func pubkeyCreateUncompressed(privkey: Data) throws -> Data {
        try chk(privkey, 32, "privkey"); try alive()
        var out = [UInt8](repeating: 0, count: 65)
        try privkey.withUnsafeBytes { pk in
            try throwRC(ufsecp_pubkey_create_uncompressed(ctx!, pk.baseAddress!.assumingMemoryBound(to: UInt8.self), &out), "pubkey_create_uncompressed")
        }
        return Data(out)
    }

    public func pubkeyParse(pubkey: Data) throws -> Data {
        try alive()
        var out = [UInt8](repeating: 0, count: 33)
        try pubkey.withUnsafeBytes { pk in
            try throwRC(ufsecp_pubkey_parse(ctx!, pk.baseAddress!.assumingMemoryBound(to: UInt8.self), pubkey.count, &out), "pubkey_parse")
        }
        return Data(out)
    }

    public func pubkeyXonly(privkey: Data) throws -> Data {
        try chk(privkey, 32, "privkey"); try alive()
        var out = [UInt8](repeating: 0, count: 32)
        try privkey.withUnsafeBytes { pk in
            try throwRC(ufsecp_pubkey_xonly(ctx!, pk.baseAddress!.assumingMemoryBound(to: UInt8.self), &out), "pubkey_xonly")
        }
        return Data(out)
    }

    public func seckeyVerify(privkey: Data) throws -> Bool {
        try chk(privkey, 32, "privkey"); try alive()
        return privkey.withUnsafeBytes { pk in
            ufsecp_seckey_verify(ctx!, pk.baseAddress!.assumingMemoryBound(to: UInt8.self)) == 0
        }
    }

    public func seckeyNegate(privkey: Data) throws -> Data {
        try chk(privkey, 32, "privkey"); try alive()
        var buf = [UInt8](privkey)
        try throwRC(ufsecp_seckey_negate(ctx!, &buf), "seckey_negate")
        return Data(buf)
    }

    public func seckeyTweakAdd(privkey: Data, tweak: Data) throws -> Data {
        try chk(privkey, 32, "privkey"); try chk(tweak, 32, "tweak"); try alive()
        var buf = [UInt8](privkey)
        try tweak.withUnsafeBytes { tw in
            try throwRC(ufsecp_seckey_tweak_add(ctx!, &buf, tw.baseAddress!.assumingMemoryBound(to: UInt8.self)), "seckey_tweak_add")
        }
        return Data(buf)
    }

    public func seckeyTweakMul(privkey: Data, tweak: Data) throws -> Data {
        try chk(privkey, 32, "privkey"); try chk(tweak, 32, "tweak"); try alive()
        var buf = [UInt8](privkey)
        try tweak.withUnsafeBytes { tw in
            try throwRC(ufsecp_seckey_tweak_mul(ctx!, &buf, tw.baseAddress!.assumingMemoryBound(to: UInt8.self)), "seckey_tweak_mul")
        }
        return Data(buf)
    }

    // MARK: ECDSA

    public func ecdsaSign(msgHash: Data, privkey: Data) throws -> Data {
        try chk(msgHash, 32, "msgHash"); try chk(privkey, 32, "privkey"); try alive()
        var sig = [UInt8](repeating: 0, count: 64)
        try msgHash.withUnsafeBytes { msg in
            try privkey.withUnsafeBytes { pk in
                try throwRC(ufsecp_ecdsa_sign(ctx!,
                    msg.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    pk.baseAddress!.assumingMemoryBound(to: UInt8.self), &sig), "ecdsa_sign")
            }
        }
        return Data(sig)
    }

    public func ecdsaVerify(msgHash: Data, sig: Data, pubkey: Data) throws -> Bool {
        try chk(msgHash, 32, "msgHash"); try chk(sig, 64, "sig"); try chk(pubkey, 33, "pubkey"); try alive()
        return msgHash.withUnsafeBytes { msg in
            sig.withUnsafeBytes { s in
                pubkey.withUnsafeBytes { pk in
                    ufsecp_ecdsa_verify(ctx!,
                        msg.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        s.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        pk.baseAddress!.assumingMemoryBound(to: UInt8.self)) == 0
                }
            }
        }
    }

    // MARK: Schnorr

    public func schnorrSign(msg: Data, privkey: Data, auxRand: Data) throws -> Data {
        try chk(msg, 32, "msg"); try chk(privkey, 32, "privkey"); try chk(auxRand, 32, "auxRand"); try alive()
        var sig = [UInt8](repeating: 0, count: 64)
        try msg.withUnsafeBytes { m in
            try privkey.withUnsafeBytes { pk in
                try auxRand.withUnsafeBytes { ar in
                    try throwRC(ufsecp_schnorr_sign(ctx!,
                        m.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        ar.baseAddress!.assumingMemoryBound(to: UInt8.self), &sig), "schnorr_sign")
                }
            }
        }
        return Data(sig)
    }

    public func schnorrVerify(msg: Data, sig: Data, pubkeyX: Data) throws -> Bool {
        try chk(msg, 32, "msg"); try chk(sig, 64, "sig"); try chk(pubkeyX, 32, "pubkeyX"); try alive()
        return msg.withUnsafeBytes { m in
            sig.withUnsafeBytes { s in
                pubkeyX.withUnsafeBytes { pk in
                    ufsecp_schnorr_verify(ctx!,
                        m.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        s.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        pk.baseAddress!.assumingMemoryBound(to: UInt8.self)) == 0
                }
            }
        }
    }

    // MARK: ECDH

    public func ecdh(privkey: Data, pubkey: Data) throws -> Data {
        try chk(privkey, 32, "privkey"); try chk(pubkey, 33, "pubkey"); try alive()
        var out = [UInt8](repeating: 0, count: 32)
        try privkey.withUnsafeBytes { pk in
            try pubkey.withUnsafeBytes { pub in
                try throwRC(ufsecp_ecdh(ctx!,
                    pk.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    pub.baseAddress!.assumingMemoryBound(to: UInt8.self), &out), "ecdh")
            }
        }
        return Data(out)
    }

    // MARK: Hashing

    public static func sha256(_ data: Data) throws -> Data {
        var out = [UInt8](repeating: 0, count: 32)
        try data.withUnsafeBytes { d in
            try throwRC(ufsecp_sha256(d.baseAddress!.assumingMemoryBound(to: UInt8.self), data.count, &out), "sha256")
        }
        return Data(out)
    }

    public static func hash160(_ data: Data) throws -> Data {
        var out = [UInt8](repeating: 0, count: 20)
        try data.withUnsafeBytes { d in
            try throwRC(ufsecp_hash160(d.baseAddress!.assumingMemoryBound(to: UInt8.self), data.count, &out), "hash160")
        }
        return Data(out)
    }

    // MARK: Internal

    private func alive() throws {
        guard !destroyed else { throw UfsecpError(operation: "alive", code: .internal) }
    }

    private func chk(_ data: Data, _ expected: Int, _ name: String) throws {
        guard data.count == expected else {
            throw UfsecpError(operation: "\(name) size", code: .badInput)
        }
    }

    private static func throwRC(_ rc: Int32, _ op: String) throws {
        guard rc == 0 else {
            throw UfsecpError(operation: op, code: UfsecpErrorCode(rawValue: rc) ?? .internal)
        }
    }

    private func throwRC(_ rc: Int32, _ op: String) throws {
        guard rc == 0 else {
            throw UfsecpError(operation: op, code: UfsecpErrorCode(rawValue: rc) ?? .internal)
        }
    }
}
