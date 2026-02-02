#pragma once

#include "secp256k1/point.hpp"
#include "secp256k1/field.hpp"
#include <string>
#include <cstdint>
#include <fstream>
#include <vector>
#include <functional>

namespace secp256k1 {

// Standardized Sorted ECC Database
// Format: Simple array of 40-byte entries
//   - Bytes 0-31:  X coordinate as native x86/x64 uint64_t[4] array (exactly as in Point::x.limbs)
//   - Bytes 32-39: j value as native x86/x64 uint64_t
// 
// Endianness: Native x86/x64 (little-endian) - NO conversions needed!
//   Memory layout matches Point::x.limbs exactly: limbs[0]=LSW ... limbs[3]=MSW
//   File format = memcpy(x_limbs, &point.x.limbs[0], 32); memcpy(j, &j_val, 8);
//
// Sorted by X coordinate (limbs[3] > [2] > [1] > [0] - MSB first comparison)
class SortedEccDB {
public:
    static constexpr size_t ENTRY_SIZE = 40;

    struct Entry 
    {
        uint64_t x_limbs[4];  // X coordinate (native x86/x64, matches Point::x.limbs memory layout)
        uint64_t j;           // j value (native x86/x64)
        
        // Convert from Point (affine X coordinate)
        static Entry from_point(const secp256k1::fast::FieldElement& x_affine, uint64_t j_value);
        
        // Compare entries for sorting (by X coordinate, MSB first)
        bool operator<(const Entry& other) const;
        bool operator>(const Entry& other) const;
        bool operator==(const Entry& other) const;
    };

    // Constructor
    explicit SortedEccDB(const std::string& path, 
                        const std::string& chunk_dir = ".", 
                        const std::string& output_dir = ".");
    
    // Generate database from j=1 to j=count (multi-threaded)
    // num_threads=0 means auto-detect (uses hardware_concurrency)
    static void generate(const std::string& path, uint64_t count, 
                        const std::string& chunk_dir = ".",
                        uint64_t start_j = 1,
                        size_t num_threads = 0,
                        std::function<void(uint64_t, uint64_t)> progress_callback = nullptr);
    
    // Merge sorted chunks from chunk_dir into a single sorted database at path
    // chunks must be named <filename>.chunk.<index>
    static void merge(const std::string& path, 
                     const std::string& chunk_dir,
                     std::function<void(uint64_t, uint64_t)> progress_callback = nullptr);

    // Sort database using external merge sort (multi-threaded chunk sorting)
    // memory_limit_mb: maximum memory per chunk
    // num_threads=0 means auto-detect
    static void sort(const std::string& path, 
                    size_t memory_limit_mb = 1024,
                    size_t num_threads = 0,
                    std::function<void(const std::string&)> progress_callback = nullptr);
    
    // Validate database is correctly sorted
    struct ValidationResult {
        bool is_sorted;
        uint64_t total_checked;
        uint64_t sort_errors;
        uint64_t first_error_index;
    };
    ValidationResult validate_sort() const;
    
    // Validate all j=1..count exist in database
    struct CompletenessResult {
        bool is_complete;
        uint64_t total_checked;
        uint64_t missing_count;
        std::vector<uint64_t> missing_j_samples; // First 100 missing
    };
    CompletenessResult validate_completeness(uint64_t expected_count) const;
    
    // Verify database integrity: generate all j*G points and lookup in database
    // Returns true if ALL j=1..entry_count found with correct j values
    struct VerifyResult {
        bool is_valid;
        uint64_t total_tested;
        uint64_t found_correct;
        uint64_t not_found;
        uint64_t value_mismatch;
        double elapsed_seconds;
    };
    
    // Verify sorted database (sorted by X, binary search lookup)
    VerifyResult verify_sorted(size_t num_threads = 1, 
                               std::function<void(uint64_t, uint64_t)> progress_callback = nullptr) const;
    
    // Verify unsorted database (j=1..N sequential, direct index lookup)
    VerifyResult verify_unsorted(size_t num_threads = 1,
                                 std::function<void(uint64_t, uint64_t)> progress_callback = nullptr) const;
    
    // Auto-detect and verify (checks first entry to determine if sorted)
    VerifyResult verify(size_t num_threads = 1, 
                       std::function<void(uint64_t, uint64_t)> progress_callback = nullptr) const;
    
    // Lookup X coordinate in database
    bool lookup(const uint64_t x_limbs[4], uint64_t* out_j = nullptr) const;
    bool lookup(const secp256k1::fast::FieldElement& x, uint64_t* out_j = nullptr) const;
    
    // Get database info
    uint64_t get_entry_count() const { return entry_count_; }
    std::string get_path() const { return path_; }

    // Load prefix index file for faster lookups
    void load_index(const std::string& idx_path);

    // Destructor to clean up resources (memory maps)
    ~SortedEccDB();
    
private:
    std::string path_;
    uint64_t entry_count_;
    mutable std::ifstream file_;
    
    // Memory Mapping Support (Windows)
    void* map_addr_ = nullptr;      // Pointer to mapped memory (const Entry*)
    void* map_handle_ = nullptr;    // HANDLE to file mapping
    void* file_handle_ = nullptr;   // HANDLE to file
    
    // Index support
    std::vector<uint64_t> index_table_;
    bool has_index_ = false;
    
    // Format detection (X-only or full entry)
    bool is_xonly_ = false;         // true for 32-byte X-only format, false for 40-byte full format
    size_t record_size_ = ENTRY_SIZE; // 32 or 40 bytes per record
    size_t header_size_ = 0;        // Header size if present (e.g., 16 bytes for "SOTSDBA" magic)

    // Working directories for temporary files
    std::string chunk_dir_;    // Directory for temporary chunks during generation/sorting
    std::string output_dir_;   // Directory for final output files
    
    void init();
    bool binary_search(const uint64_t x_limbs[4], uint64_t* out_j) const;
    
    // Helper: Generate range [start_j, start_j+count) and write to file
    // Returns number of entries written
    static uint64_t range_gen(const std::string& output_file,
                             uint64_t start_j, 
                             uint64_t count,
                             std::function<void(uint64_t, uint64_t)> progress_callback = nullptr);
};

} // namespace secp256k1
