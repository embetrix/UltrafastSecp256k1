#include "secp256k1/storage.hpp"

#if SECP256K1_HAVE_ROCKSDB

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/status.h>
#include <rocksdb/write_batch.h>
#include <rocksdb/slice.h>

namespace secp256k1::fast {

namespace {

[[nodiscard]] std::string make_key(const FieldElement& x) {
    auto bytes = x.to_bytes();
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

[[nodiscard]] std::string encode_scalar_minimal(const Scalar& scalar) {
    auto bytes = scalar.to_bytes();
    std::size_t first_non_zero = 0;
    while (first_non_zero + 1 < bytes.size() && bytes[first_non_zero] == 0) {
        ++first_non_zero;
    }
    const auto* data = bytes.data() + first_non_zero;
    const std::size_t length = bytes.size() - first_non_zero;
    return std::string(reinterpret_cast<const char*>(data), length);
}

template <typename SliceLike>
[[nodiscard]] Scalar decode_scalar_slice(const SliceLike& value) {
    const std::size_t size = static_cast<std::size_t>(value.size());
    if (size == 0 || size > 32) {
        throw std::runtime_error("Invalid scalar size stored in RocksDB");
    }
    std::array<std::uint8_t, 32> bytes{};
    std::memcpy(bytes.data() + (32 - size), value.data(), size);
    return Scalar::from_bytes(bytes);
}

[[nodiscard]] FieldElement decode_field_slice(const rocksdb::Slice& value) {
    if (value.size() != 32) {
        throw std::runtime_error("Invalid field element size stored in RocksDB");
    }
    std::array<std::uint8_t, 32> bytes{};
    std::memcpy(bytes.data(), value.data(), 32);
    return FieldElement::from_bytes(bytes);
}

[[nodiscard]] rocksdb::Options build_options(const KeyValueStore::Config& config) {
    rocksdb::Options options;
    options.create_if_missing = config.create_if_missing;
    options.error_if_exists = config.error_if_exists;
    options.compression = rocksdb::kNoCompression;
    options.write_buffer_size = static_cast<size_t>(config.write_buffer_mb) * 1024ULL * 1024ULL;
    options.use_direct_reads = config.use_direct_io;
    options.use_direct_io_for_flush_and_compaction = config.use_direct_io;
    options.IncreaseParallelism(std::max(1, static_cast<int>(std::thread::hardware_concurrency())));
    options.OptimizeLevelStyleCompaction();
    return options;
}

} // namespace

class KeyValueStore::Impl {
public:
    explicit Impl(const Config& config) {
        rocksdb::Options options = build_options(config);
        rocksdb::DB* raw_db = nullptr;
        auto status = rocksdb::DB::Open(options, config.path, &raw_db);
        if (!status.ok()) {
            throw std::runtime_error("Failed to open RocksDB: " + status.ToString());
        }
        db_.reset(raw_db);
        refresh_iterator_unlocked();
    }

    void put(const FieldElement& key, const Scalar& value) {
        const auto key_bytes = make_key(key);
        const auto value_bytes = encode_scalar_minimal(value);
        rocksdb::WriteOptions write_options;
        write_options.disableWAL = false;
        const auto status = db_->Put(write_options, key_bytes, value_bytes);
        if (!status.ok()) {
            throw std::runtime_error("RocksDB Put failed: " + status.ToString());
        }
        std::lock_guard<std::mutex> lock(iter_mutex_);
        needs_refresh_ = true;
    }

    [[nodiscard]] std::optional<Scalar> get(const FieldElement& key) const {
        const auto key_bytes = make_key(key);
        rocksdb::PinnableSlice slice;
        rocksdb::ReadOptions read_options;
        const auto status = db_->Get(read_options, db_->DefaultColumnFamily(), key_bytes, &slice);
        if (status.IsNotFound()) {
            return std::nullopt;
        }
        if (!status.ok()) {
            throw std::runtime_error("RocksDB Get failed: " + status.ToString());
        }
        return decode_scalar_slice(slice);
    }

    [[nodiscard]] bool contains(const FieldElement& key) const {
        const auto key_bytes = make_key(key);
        std::string value;
        rocksdb::ReadOptions read_options;
        read_options.fill_cache = false;
        const auto status = db_->Get(read_options, db_->DefaultColumnFamily(), key_bytes, &value);
        if (status.IsNotFound()) {
            return false;
        }
        if (!status.ok()) {
            throw std::runtime_error("RocksDB Get failed: " + status.ToString());
        }
        return true;
    }

    void flush() {
        rocksdb::FlushOptions flush_options;
        const auto status = db_->Flush(flush_options);
        if (!status.ok()) {
            throw std::runtime_error("RocksDB Flush failed: " + status.ToString());
        }
    }

    [[nodiscard]] std::optional<KeyValueStore::Entry> next() {
        std::lock_guard<std::mutex> lock(iter_mutex_);
        ensure_iterator_unlocked();
        if (!iterator_ || !iterator_->Valid()) {
            return std::nullopt;
        }
        KeyValueStore::Entry entry{
            decode_field_slice(iterator_->key()),
            decode_scalar_slice(iterator_->value())
        };
        iterator_->Next();
        const auto status = iterator_->status();
        if (!status.ok()) {
            throw std::runtime_error("RocksDB iterator error: " + status.ToString());
        }
        return entry;
    }

    void reset_iteration() {
        std::lock_guard<std::mutex> lock(iter_mutex_);
        refresh_iterator_unlocked();
    }

private:
    std::unique_ptr<rocksdb::DB> db_{};
    mutable std::mutex iter_mutex_{};
    std::unique_ptr<rocksdb::Iterator> iterator_{};
    rocksdb::ReadOptions iterator_read_options_{};
    bool needs_refresh_{false};

    void refresh_iterator_unlocked() {
        iterator_read_options_.pin_data = true;
        iterator_read_options_.total_order_seek = true;
        iterator_read_options_.fill_cache = false;
        iterator_.reset(db_->NewIterator(iterator_read_options_));
        if (iterator_) {
            iterator_->SeekToFirst();
        }
        needs_refresh_ = false;
    }

    void ensure_iterator_unlocked() {
        if (needs_refresh_ || !iterator_) {
            refresh_iterator_unlocked();
        }
        if (iterator_ && !iterator_->status().ok()) {
            refresh_iterator_unlocked();
        }
    }
};

KeyValueStore::KeyValueStore(const Config& config) : impl_(new Impl(config)) {}

KeyValueStore::~KeyValueStore() {
    delete impl_;
    impl_ = nullptr;
}

KeyValueStore::KeyValueStore(KeyValueStore&& other) noexcept : impl_(std::exchange(other.impl_, nullptr)) {}

KeyValueStore& KeyValueStore::operator=(KeyValueStore&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = std::exchange(other.impl_, nullptr);
    }
    return *this;
}

void KeyValueStore::put(const FieldElement& x_coord, const Scalar& private_key) {
    impl_->put(x_coord, private_key);
}

std::optional<Scalar> KeyValueStore::get(const FieldElement& x_coord) const {
    return impl_->get(x_coord);
}

bool KeyValueStore::contains(const FieldElement& x_coord) const {
    return impl_->contains(x_coord);
}

void KeyValueStore::flush() {
    impl_->flush();
}

std::optional<KeyValueStore::Entry> KeyValueStore::next() {
    return impl_->next();
}

void KeyValueStore::reset_iteration() {
    impl_->reset_iteration();
}

} // namespace secp256k1::fast

#endif // SECP256K1_HAVE_ROCKSDB
