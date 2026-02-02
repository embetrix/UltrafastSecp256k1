#ifndef D8C0921F_52E2_4C1D_9D0D_8F7B0FFB09E8
#define D8C0921F_52E2_4C1D_9D0D_8F7B0FFB09E8

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <stdexcept>

#ifndef SECP256K1_HAVE_ROCKSDB
#define SECP256K1_HAVE_ROCKSDB 0
#endif

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"

namespace secp256k1::fast {

#if SECP256K1_HAVE_ROCKSDB

class KeyValueStore {
public:
    struct Config {
        std::string path;
        bool create_if_missing{true};
        bool error_if_exists{false};
        bool use_direct_io{false};
        std::size_t write_buffer_mb{64};
    };

    struct Entry {
        FieldElement x;
        Scalar private_key;
    };

    explicit KeyValueStore(const Config& config);
    ~KeyValueStore();

    KeyValueStore(const KeyValueStore&) = delete;
    KeyValueStore& operator=(const KeyValueStore&) = delete;

    KeyValueStore(KeyValueStore&&) noexcept;
    KeyValueStore& operator=(KeyValueStore&&) noexcept;

    void put(const FieldElement& x_coord, const Scalar& private_key);
    [[nodiscard]] std::optional<Scalar> get(const FieldElement& x_coord) const;
    [[nodiscard]] bool contains(const FieldElement& x_coord) const;
    void flush();

    [[nodiscard]] std::optional<Entry> next();
    void reset_iteration();

private:
    class Impl;
    Impl* impl_{nullptr};
};

#else

class KeyValueStore {
public:
    struct Config {
        std::string path;
    };

    struct Entry {
        FieldElement x;
        Scalar private_key;
    };

    explicit KeyValueStore(const Config&) { raise(); }
    ~KeyValueStore() = default;

    KeyValueStore(const KeyValueStore&) = delete;
    KeyValueStore& operator=(const KeyValueStore&) = delete;

    KeyValueStore(KeyValueStore&&) noexcept = default;
    KeyValueStore& operator=(KeyValueStore&&) noexcept = default;

    void put(const FieldElement&, const Scalar&) { raise(); }
    [[nodiscard]] std::optional<Scalar> get(const FieldElement&) const { raise(); return std::nullopt; }
    [[nodiscard]] bool contains(const FieldElement&) const { raise(); return false; }
    void flush() { raise(); }
    [[nodiscard]] std::optional<Entry> next() { raise(); return std::nullopt; }
    void reset_iteration() { raise(); }

private:
    [[noreturn]] static void raise() {
        throw std::logic_error("RocksDB support is disabled in this build");
    }
};

#endif

} // namespace secp256k1::fast


#endif /* D8C0921F_52E2_4C1D_9D0D_8F7B0FFB09E8 */
