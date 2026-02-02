#pragma once

#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/write_batch.h>
#include <rocksdb/slice.h>
#include <rocksdb/statistics.h>
#include <memory>
#include <string>
#include <stdexcept>

namespace secp256k1 {
namespace fast {

/**
 * Developer-friendly RocksDB wrapper with performance-oriented configuration
 * 
 * Features:
 * - Snappy compression (lightweight, fast)
 * - Optimized for sequential writes (generator mode)
 * - Optimized for random reads (lookup mode)
 * - Pre-configured for maximum performance
 * - Simple API - no boilerplate configuration needed
 * 
 * Usage:
 *   // For generator (write-heavy workload)
 *   RocksDBWrapper db("./pubkeys.db", RocksDBWrapper::Mode::GENERATOR);
 *   db.put(key, value);
 *   db.flush();
 * 
 *   // For lookup (read-heavy workload)
 *   RocksDBWrapper db("./pubkeys.db", RocksDBWrapper::Mode::LOOKUP);
 *   std::string value;
 *   if (db.get(key, value)) { ... }
 */
class RocksDBWrapper {
public:
    enum class Mode {
        GENERATOR,  // Optimized for sequential writes (batch inserts)
        LOOKUP      // Optimized for random reads (point lookups)
    };

    /**
     * Open or create RocksDB database with optimized settings
     * 
     * @param db_path Path to database directory
     * @param mode Operation mode (GENERATOR or LOOKUP)
     * @param create_if_missing Create database if it doesn't exist (default: true)
     * @throws std::runtime_error if database cannot be opened
     */
    explicit RocksDBWrapper(
        const std::string& db_path,
        Mode mode = Mode::LOOKUP,
        bool create_if_missing = true
    );

    ~RocksDBWrapper();

    // Disable copy (RocksDB objects cannot be copied)
    RocksDBWrapper(const RocksDBWrapper&) = delete;
    RocksDBWrapper& operator=(const RocksDBWrapper&) = delete;

    // Allow move
    RocksDBWrapper(RocksDBWrapper&&) noexcept = default;
    RocksDBWrapper& operator=(RocksDBWrapper&&) noexcept = default;

    /**
     * Put key-value pair into database
     * 
     * @param key Key as string or bytes
     * @param value Value as string or bytes
     * @throws std::runtime_error on write failure
     */
    void put(const std::string& key, const std::string& value);
    void put(const rocksdb::Slice& key, const rocksdb::Slice& value);

    /**
     * Get value by key
     * 
     * @param key Key to lookup
     * @param value Output parameter for retrieved value
     * @return true if key exists, false otherwise
     * @throws std::runtime_error on read failure
     */
    bool get(const std::string& key, std::string& value) const;
    bool get(const rocksdb::Slice& key, std::string& value) const;

    /**
     * Check if key exists in database
     * 
     * @param key Key to check
     * @return true if key exists, false otherwise
     */
    bool exists(const std::string& key) const;
    bool exists(const rocksdb::Slice& key) const;

    /**
     * Delete key from database
     * 
     * @param key Key to delete
     * @throws std::runtime_error on delete failure
     */
    void del(const std::string& key);
    void del(const rocksdb::Slice& key);

    /**
     * Batch write operations (for GENERATOR mode)
     * 
     * Usage:
     *   auto batch = db.create_batch();
     *   for (auto& [k, v] : data) {
     *       batch.put(k, v);
     *   }
     *   db.write_batch(batch);
     */
    class WriteBatch {
    public:
        WriteBatch() = default;
        
        void put(const std::string& key, const std::string& value) {
            batch_.Put(key, value);
        }
        
        void put(const rocksdb::Slice& key, const rocksdb::Slice& value) {
            batch_.Put(key, value);
        }
        
        void del(const std::string& key) {
            batch_.Delete(key);
        }
        
        void del(const rocksdb::Slice& key) {
            batch_.Delete(key);
        }
        
        void clear() {
            batch_.Clear();
        }
        
        size_t count() const {
            return batch_.Count();
        }

    private:
        friend class RocksDBWrapper;
        rocksdb::WriteBatch batch_;
    };

    WriteBatch create_batch() const {
        return WriteBatch();
    }

    void write_batch(WriteBatch& batch);

    /**
     * Flush all buffered writes to disk (for GENERATOR mode)
     * Ensures data durability
     */
    void flush();

    /**
     * Compact database to optimize storage and read performance
     * Call after bulk inserts in GENERATOR mode
     */
    void compact();

    /**
     * Get database statistics (hits, misses, bytes read/written)
     */
    std::string get_stats() const;

    /**
     * Get approximate database size in bytes
     */
    uint64_t get_size() const;

    /**
     * Get approximate number of keys
     */
    uint64_t get_key_count() const;

    /**
     * Get database path
     */
    const std::string& get_path() const { return db_path_; }

    /**
     * Get operation mode
     */
    Mode get_mode() const { return mode_; }

    /**
     * Check if database is open
     */
    bool is_open() const { return db_ != nullptr; }

private:
    void configure_generator_options(rocksdb::Options& options);
    void configure_lookup_options(rocksdb::Options& options);
    void configure_common_options(rocksdb::Options& options);

    std::string db_path_;
    Mode mode_;
    std::unique_ptr<rocksdb::DB> db_;
    std::shared_ptr<rocksdb::Statistics> stats_;
};

} // namespace fast
} // namespace secp256k1
