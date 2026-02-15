// ============================================================================
// Coin Address — Implementation
// ============================================================================
// Dispatches to existing address.hpp functions with per-coin parameters.
// Ethereum/EVM addresses use Keccak-256 (delegated to ethereum.hpp).
// ============================================================================

#include "secp256k1/coins/coin_address.hpp"
#include "secp256k1/coins/ethereum.hpp"
#include "secp256k1/address.hpp"

namespace secp256k1::coins {

// ── Helpers ──────────────────────────────────────────────────────────────────

static Network to_network(bool testnet) {
    return testnet ? Network::Testnet : Network::Mainnet;
}

// ── Default Address ──────────────────────────────────────────────────────────

std::string coin_address(const fast::Point& pubkey,
                         const CoinParams& coin,
                         bool testnet) {
    switch (coin.default_encoding) {
        case AddressEncoding::BECH32:
            return coin_address_p2wpkh(pubkey, coin, testnet);
            
        case AddressEncoding::EIP55:
            return ethereum_address(pubkey);
            
        case AddressEncoding::BASE58CHECK:
        default:
            return coin_address_p2pkh(pubkey, coin, testnet);
    }
}

// ── P2PKH ────────────────────────────────────────────────────────────────────

std::string coin_address_p2pkh(const fast::Point& pubkey,
                               const CoinParams& coin,
                               bool testnet) {
    // Get hash160 of compressed pubkey
    auto compressed = pubkey.to_compressed();
    auto h160 = hash160(compressed.data(), compressed.size());
    
    // Build versioned payload: [version_byte] + [20-byte hash]
    std::uint8_t versioned[21];
    versioned[0] = testnet ? coin.p2pkh_version_test : coin.p2pkh_version;
    std::memcpy(versioned + 1, h160.data(), 20);
    
    return base58check_encode(versioned, 21);
}

// ── P2WPKH ───────────────────────────────────────────────────────────────────

std::string coin_address_p2wpkh(const fast::Point& pubkey,
                                const CoinParams& coin,
                                bool testnet) {
    // Check SegWit support
    if (!coin.features.supports_segwit) return {};
    
    const char* hrp = testnet ? coin.bech32_hrp_test : coin.bech32_hrp;
    if (!hrp) return {};
    
    // Get hash160 of compressed pubkey
    auto compressed = pubkey.to_compressed();
    auto h160 = hash160(compressed.data(), compressed.size());
    
    // Bech32 encode: witness version 0, 20-byte program
    return bech32_encode(hrp, 0, h160.data(), h160.size());
}

// ── P2TR ─────────────────────────────────────────────────────────────────────

std::string coin_address_p2tr(const fast::Point& internal_key,
                              const CoinParams& coin,
                              bool testnet) {
    // Check Taproot support
    if (!coin.features.supports_taproot) return {};
    
    const char* hrp = testnet ? coin.bech32_hrp_test : coin.bech32_hrp;
    if (!hrp) return {};
    
    // Use existing P2TR logic — get the x-only output key
    // For now, use the internal key directly (no script tree)
    auto compressed = internal_key.to_compressed();
    
    // x-only key = compressed[1..33]
    // Bech32m encode: witness version 1, 32-byte program
    return bech32_encode(hrp, 1, compressed.data() + 1, 32);
}

// ── WIF ──────────────────────────────────────────────────────────────────────

std::string coin_wif_encode(const fast::Scalar& private_key,
                            const CoinParams& coin,
                            bool compressed,
                            bool testnet) {
    auto key_bytes = private_key.to_bytes();
    
    // WIF format: [prefix] + [32-byte key] + [0x01 if compressed]
    std::uint8_t wif_data[34];
    wif_data[0] = testnet ? coin.wif_prefix_test : coin.wif_prefix;
    std::memcpy(wif_data + 1, key_bytes.data(), 32);
    
    std::size_t len = 33;
    if (compressed) {
        wif_data[33] = 0x01;
        len = 34;
    }
    
    return base58check_encode(wif_data, len);
}

// ── Full Key Generation ──────────────────────────────────────────────────────

CoinKeyPair coin_derive(const fast::Scalar& private_key,
                        const CoinParams& coin,
                        bool testnet,
                        const CurveContext* ctx) {
    CoinKeyPair result;
    result.private_key = private_key;
    
    // Derive public key (with optional custom generator)
    result.public_key = derive_public_key(private_key, ctx);
    
    // Generate address in default format
    result.address = coin_address(result.public_key, coin, testnet);
    
    // WIF encode (skip for EVM coins — they use raw hex keys)
    if (!coin.features.uses_evm) {
        result.wif = coin_wif_encode(private_key, coin, true, testnet);
    }
    
    return result;
}

} // namespace secp256k1::coins
