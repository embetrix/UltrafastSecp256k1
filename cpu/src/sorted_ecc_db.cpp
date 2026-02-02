#include "secp256k1/sorted_ecc_db.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/precompute.hpp"
#include "platform_compat.h"
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <filesystem>
#include <future>
#include <tuple>
#include <memory>
#include <algorithm>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#undef min
#undef max
#endif

namespace secp256k1 {
// Safe large-file seek+read helper for 64-bit offsets
static inline bool safe_seek_read(std::ifstream &file, uint64_t index, size_t entry_size, void *buffer) {
    if (!file.good()) file.clear();
    const uint64_t off64 = index * static_cast<uint64_t>(entry_size);
    const std::streamoff offset = static_cast<std::streamoff>(off64);
    file.seekg(offset, std::ios::beg);
    if (file.fail()) {
        file.clear();
        file.seekg(offset, std::ios::beg);
        if (file.fail()) return false;
    }
    file.read(reinterpret_cast<char*>(buffer), static_cast<std::streamsize>(entry_size));
    if (file.fail() || file.gcount() != static_cast<std::streamsize>(entry_size)) return false;
    return true;
}
namespace fs = std::filesystem;

// Entry comparison operators - compare X coordinates MSB first
// x_limbs[0] = LSW, x_limbs[3] = MSW (native x86/x64 layout)
// For sorting: compare limbs[3] first (MSB), then [2], [1], [0] (LSB)

bool SortedEccDB::Entry::operator<(const Entry& other) const {
    // Compare MSB to LSB
    if (x_limbs[3] != other.x_limbs[3]) return x_limbs[3] < other.x_limbs[3];
    if (x_limbs[2] != other.x_limbs[2]) return x_limbs[2] < other.x_limbs[2];
    if (x_limbs[1] != other.x_limbs[1]) return x_limbs[1] < other.x_limbs[1];
    return x_limbs[0] < other.x_limbs[0];
}

bool SortedEccDB::Entry::operator>(const Entry& other) const {
    // Compare MSB to LSB
    if (x_limbs[3] != other.x_limbs[3]) return x_limbs[3] > other.x_limbs[3];
    if (x_limbs[2] != other.x_limbs[2]) return x_limbs[2] > other.x_limbs[2];
    if (x_limbs[1] != other.x_limbs[1]) return x_limbs[1] > other.x_limbs[1];
    return x_limbs[0] > other.x_limbs[0];
}

bool SortedEccDB::Entry::operator==(const Entry& other) const {
    return x_limbs[3] == other.x_limbs[3] &&
           x_limbs[2] == other.x_limbs[2] &&
           x_limbs[1] == other.x_limbs[1] &&
           x_limbs[0] == other.x_limbs[0];
}

// Convert FieldElement to Entry - native x86/x64, zero conversions!
SortedEccDB::Entry SortedEccDB::Entry::from_point(
    const fast::FieldElement& x_affine, 
    uint64_t j_value) 
{
    Entry e;
    // Direct memcpy - FieldElement::limbs() returns const std::array&
    std::memcpy(e.x_limbs, x_affine.limbs().data(), 32);
    e.j = j_value;
    return e;
}

// Constructor - open file and calculate entry count
SortedEccDB::SortedEccDB(const std::string& path, 
                        const std::string& chunk_dir, 
                        const std::string& output_dir) 
    : path_(path), chunk_dir_(chunk_dir), output_dir_(output_dir) {
    init();
}

SortedEccDB::~SortedEccDB() {
#ifdef _WIN32
    if (map_addr_) UnmapViewOfFile(map_addr_);
    if (map_handle_) CloseHandle(static_cast<HANDLE>(map_handle_));
    if (file_handle_) CloseHandle(static_cast<HANDLE>(file_handle_));
#else
    if (map_addr_) {
        munmap(map_addr_, entry_count_ * ENTRY_SIZE);
    }
    if (file_handle_) {
        close((int)(intptr_t)file_handle_);
    }
#endif
}

void SortedEccDB::init() {
    // Get file size using filesystem (more reliable)
    if (!fs::exists(path_)) {
        throw std::runtime_error("Database file does not exist: " + path_);
    }
    
    uint64_t file_size = fs::file_size(path_);
    
    // Auto-detect format based on file extension
    is_xonly_ = (path_.find(".bin") != std::string::npos || 
                 path_.find("_xonly") != std::string::npos);
    
    record_size_ = is_xonly_ ? 32 : ENTRY_SIZE; // 32 for X-only, 40 for full
    
    // Check for 16-byte header (magic "SOTSDBA\0" + 8 bytes)
    header_size_ = 0;
    std::ifstream check_file(path_, std::ios::binary);
    if (check_file) {
        char magic[8];
        check_file.read(magic, 8);
        if (check_file.gcount() == 8 && 
            std::string(magic, 7) == "SOTSDBA") {
            header_size_ = 16;
            fprintf(stderr, "[SortedEccDB] Detected 16-byte header, skipping...\n");
        }
        check_file.close();
    }
    
    // Calculate entry count based on detected format (skip header if present)
    uint64_t data_size = file_size - header_size_;
    if (data_size % record_size_ != 0) {
        std::string err = "Invalid database file size (not multiple of " + 
                         std::to_string(record_size_) + " bytes)";
        throw std::runtime_error(err);
    }
    
    entry_count_ = data_size / record_size_;
    
    // Log detected format
    if (is_xonly_) {
        fprintf(stderr, "[SortedEccDB] Detected X-only format: %s (%lu entries, 32 bytes/record)\n", 
                path_.c_str(), entry_count_);
    } else {
        fprintf(stderr, "[SortedEccDB] Detected full format: %s (%lu entries, 40 bytes/record)\n", 
                path_.c_str(), entry_count_);
    }
    
    // Open file for reading
    file_.open(path_, std::ios::in | std::ios::binary);
    if (!file_) {
        throw std::runtime_error("Failed to open database: " + path_);
    }

    // Try to memory map the file for faster access
#ifdef _WIN32
    HANDLE hFile = CreateFileA(path_.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile != INVALID_HANDLE_VALUE) {
        HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (hMap) {
            void* data = MapViewOfFile(hMap, FILE_MAP_READ, header_size_, 0, 0);
            if (data) {
                file_handle_ = hFile;
                map_handle_ = hMap;
                map_addr_ = data;
            } else {
                CloseHandle(hMap);
                CloseHandle(hFile);
            }
        } else {
            CloseHandle(hFile);
        }
    }
#else
    int fd = open(path_.c_str(), O_RDONLY);
    if (fd != -1) {
        // Map the data portion (skip header if present)
        void* data = mmap(NULL, entry_count_ * record_size_, PROT_READ, MAP_SHARED, fd, header_size_);
        if (data != MAP_FAILED) {
            file_handle_ = (void*)(intptr_t)fd;
            map_addr_ = data;
            // map_handle_ is unused on Linux
        } else {
            close(fd);
        }
    }
#endif
}

// Private helper: Generate range [start_j, start_j+count) and write to file
// Uses batch inverse optimization (1024 points at a time)
// Returns number of entries written
uint64_t SortedEccDB::range_gen(const std::string& output_file,
                               uint64_t start_j, 
                               uint64_t count,
                               std::function<void(uint64_t, uint64_t)> progress_callback)
{
    if (count == 0) return 0;
    
    // Ensure precompute table ready
    fast::ensure_fixed_base_ready();
    
    // Start from j=start_j
    fast::Point p = fast::Point::generator();
    if (start_j > 1) {
        p = p.scalar_mul(fast::Scalar::from_uint64(start_j));
    }
    
    // Batch processing constants
    constexpr size_t BATCH_SIZE = 1024;
    
    // Vectors for batch inverse (thread-safe, heap allocated)
    std::vector<fast::Point> batch_points(BATCH_SIZE);
    std::vector<fast::FieldElement> batch_z(BATCH_SIZE);
    std::vector<uint64_t> batch_j(BATCH_SIZE);
    
    // Output buffer - accumulate ALL entries to sort before writing
    std::vector<Entry> output_buffer;
    output_buffer.reserve(count);
    
    size_t batch_idx = 0;
    
    for (uint64_t j = start_j; j < start_j + count; ++j) {
        // Save point and its Z coordinate
        batch_points[batch_idx] = p;
        batch_z[batch_idx] = p.z();
        batch_j[batch_idx] = j;
        batch_idx++;
        
        p.next_inplace();  // Advance to next point
        
        // Process batch when full or at end
        if (batch_idx == BATCH_SIZE || j == start_j + count - 1) {
            // Batch inverse Z coordinates (in-place)
            fast::fe_batch_inverse(batch_z.data(), batch_idx);
            
            // Convert each point to affine and create entries
            for (size_t i = 0; i < batch_idx; ++i) {
                // Compute affine X = X_jacobian * Z^(-2)
                // batch_z now contains Z^(-1) after fe_batch_inverse
                fast::FieldElement z_inv_sq = batch_z[i].square();
                fast::FieldElement x_affine = batch_points[i].X() * z_inv_sq;
                
                // Create entry
                output_buffer.push_back(Entry::from_point(x_affine, batch_j[i]));
            }
            
            if (progress_callback && output_buffer.size() % 100000 == 0) {
                progress_callback(output_buffer.size(), count);
            }
            
            batch_idx = 0;
        }
    }
    
    // Sort the chunk in memory
    std::sort(output_buffer.begin(), output_buffer.end());
    
    // Write sorted chunk to disk
    std::ofstream out(output_file, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Failed to create output file: " + output_file);
    }
    
    out.write(reinterpret_cast<const char*>(output_buffer.data()), 
             output_buffer.size() * ENTRY_SIZE);
             
    if (!out) {
        throw std::runtime_error("Write error during range generation");
    }
    
    if (progress_callback) {
        progress_callback(count, count);
    }
    
    return output_buffer.size();
}

// Generate database from j=1 to j=count (multi-threaded)
void SortedEccDB::generate(const std::string& path, uint64_t count, 
                          const std::string& chunk_dir,
                          uint64_t start_j,
                          size_t num_threads,
                          std::function<void(uint64_t, uint64_t)> progress_callback)
{
    // Auto-detect thread count
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1;
    }
    
    // Define chunk size: 256MB per chunk (approx 6.7 million entries)
    // This ensures efficient in-memory sorting and manageable file sizes
    constexpr size_t CHUNK_MEMORY_LIMIT = 256 * 1024 * 1024; 
    const uint64_t entries_per_chunk = CHUNK_MEMORY_LIMIT / ENTRY_SIZE;
    const uint64_t total_chunks = (count + entries_per_chunk - 1) / entries_per_chunk;
    
    // Use provided chunk directory
    fs::path chunk_dir_path(chunk_dir);
    if (chunk_dir.empty()) chunk_dir_path = ".";
    
    // Ensure chunk directory exists
    if (!fs::exists(chunk_dir_path)) {
        fs::create_directories(chunk_dir_path);
    }
    
    // Create chunks
    std::atomic<uint64_t> next_chunk_idx{0};
    std::atomic<uint64_t> total_generated{0};
    std::mutex progress_mutex;
    
    std::vector<std::thread> threads;
    
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            while (true) {
                uint64_t chunk_idx = next_chunk_idx.fetch_add(1);
                if (chunk_idx >= total_chunks) break;
                
                uint64_t chunk_start_j = start_j + chunk_idx * entries_per_chunk;
                uint64_t chunk_count = std::min(entries_per_chunk, count - (chunk_start_j - start_j));
                
                // Naming convention: <filename>.chunk.<index>
                // e.g. database.dat.chunk.0
                std::string chunk_filename = fs::path(path).filename().string() + ".chunk." + std::to_string(chunk_idx);
                std::string chunk_path = (chunk_dir_path / chunk_filename).string();
                
                // Generate and sort this chunk
                range_gen(chunk_path, chunk_start_j, chunk_count, nullptr);
                
                // Update progress
                uint64_t current_total = total_generated.fetch_add(chunk_count) + chunk_count;
                if (progress_callback) {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    progress_callback(current_total, count);
                }
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // NOTE: Merging is done separately to perform K-way merge sort
    // The chunks are left in chunk_dir for the merge step
    
    if (progress_callback) {
        progress_callback(count, count);
    }
}

// Merge sorted chunks into final database
void SortedEccDB::merge(const std::string& path, 
                       const std::string& chunk_dir,
                       std::function<void(uint64_t, uint64_t)> progress_callback)
{
    fs::path output_path(path);
    fs::path chunk_dir_path(chunk_dir);
    std::string base_filename = output_path.filename().string();
    
    // Find all chunk files
    std::vector<std::string> chunk_files;
    if (fs::exists(chunk_dir_path)) {
        for (const auto& entry : fs::directory_iterator(chunk_dir_path)) {
            std::string filename = entry.path().filename().string();
            // Check if file matches pattern: <base_filename>.chunk.<index>
            if (filename.find(base_filename + ".chunk.") == 0) {
                chunk_files.push_back(entry.path().string());
            }
        }
    }
    
    if (chunk_files.empty()) {
        throw std::runtime_error("No chunk files found in " + chunk_dir);
    }
    
    // Calculate total size for progress
    uint64_t total_entries = 0;
    for (const auto& file : chunk_files) {
        total_entries += fs::file_size(file) / ENTRY_SIZE;
    }
    
    // K-way merge using min-heap
    struct ChunkReader {
        std::ifstream stream;
        Entry current;
        bool valid;
        size_t chunk_id;
        
        ChunkReader(const std::string& file, size_t id) : valid(false), chunk_id(id) {
            stream.open(file, std::ios::binary);
            if (stream) {
                read_next();
            }
        }
        
        bool read_next() {
            if (stream.read(reinterpret_cast<char*>(&current), ENTRY_SIZE)) {
                valid = true;
                return true;
            }
            valid = false;
            return false;
        }
        
        // Comparison for min-heap (invert for max-heap behavior)
        bool operator>(const ChunkReader& other) const {
            return current > other.current;
        }
    };
    
    std::vector<std::unique_ptr<ChunkReader>> readers;
    readers.reserve(chunk_files.size());
    for (size_t i = 0; i < chunk_files.size(); ++i) {
        readers.push_back(std::make_unique<ChunkReader>(chunk_files[i], i));
    }
    
    // Min-heap of chunk readers
    auto cmp = [](const ChunkReader* a, const ChunkReader* b) {
        return *a > *b;
    };
    std::priority_queue<ChunkReader*, std::vector<ChunkReader*>, decltype(cmp)> min_heap(cmp);
    
    for (const auto& reader : readers) {
        if (reader->valid) {
            min_heap.push(reader.get());
        }
    }
    
    // Create output file
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Failed to create output file: " + path);
    }
    
    // Buffer for writing
    const size_t WRITE_BUFFER_SIZE = 1024 * 1024; // 1MB buffer (approx 26k entries)
    std::vector<Entry> write_buffer;
    write_buffer.reserve(WRITE_BUFFER_SIZE);
    
    uint64_t merged_count = 0;
    
    while (!min_heap.empty()) {
        ChunkReader* reader = min_heap.top();
        min_heap.pop();
        
        write_buffer.push_back(reader->current);
        
        if (write_buffer.size() >= WRITE_BUFFER_SIZE) {
            out.write(reinterpret_cast<const char*>(write_buffer.data()),
                     write_buffer.size() * ENTRY_SIZE);
            merged_count += write_buffer.size();
            write_buffer.clear();
            
            if (progress_callback) {
                progress_callback(merged_count, total_entries);
            }
        }
        
        if (reader->read_next()) {
            min_heap.push(reader);
        }
    }
    
    // Write remaining buffer
    if (!write_buffer.empty()) {
        out.write(reinterpret_cast<const char*>(write_buffer.data()),
                 write_buffer.size() * ENTRY_SIZE);
        merged_count += write_buffer.size();
    }
    
    out.close();
    
    // Clean up chunk files
    readers.clear(); // Close files
    for (const auto& file : chunk_files) {
        fs::remove(file);
    }
    
    if (progress_callback) {
        progress_callback(merged_count, total_entries);
    }
}

// Sort database in-place using std::sort (for small-medium databases)
// For huge databases (>10GB), use external merge sort (multi-threaded)
void SortedEccDB::sort(const std::string& path, 
                      size_t memory_limit_mb,
                      size_t num_threads,
                      std::function<void(const std::string&)> progress_callback)
{
    // Auto-detect thread count
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1;
    }
    
    // Get file size using filesystem (more reliable than tellg)
    if (!fs::exists(path)) {
        throw std::runtime_error("Database file does not exist: " + path);
    }
    
    uint64_t file_size = fs::file_size(path);
    
    if (file_size % ENTRY_SIZE != 0) {
        throw std::runtime_error("Invalid database file size");
    }
    
    uint64_t entry_count = file_size / ENTRY_SIZE;
    uint64_t memory_bytes = static_cast<uint64_t>(memory_limit_mb) * 1024 * 1024;
    uint64_t entries_fit_memory = memory_bytes / ENTRY_SIZE;
    
    // Open file for reading
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open database for sorting: " + path);
    }
    
    if (entry_count <= entries_fit_memory) {
        // Small database - load all into memory and sort
        if (progress_callback) {
            progress_callback("Loading " + std::to_string(entry_count) + " entries into memory...");
        }
        
        std::vector<Entry> entries(entry_count);
        in.read(reinterpret_cast<char*>(entries.data()), file_size);
        in.close();
        
        if (!in) {
            throw std::runtime_error("Read error during sort");
        }
        
        if (progress_callback) {
            progress_callback("Sorting in memory...");
        }
        
        // Sort using native x86/x64 comparison (MSB first)
        std::sort(entries.begin(), entries.end());
        
        if (progress_callback) {
            progress_callback("Writing sorted database...");
        }
        
        // Write back
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("Failed to write sorted database");
        }
        
        out.write(reinterpret_cast<const char*>(entries.data()), file_size);
        
        if (!out) {
            throw std::runtime_error("Write error during sort");
        }
        
        if (progress_callback) {
            progress_callback("Sort complete");
        }
        return;
    }
    
    // External merge sort for large databases
    if (progress_callback) {
        progress_callback("Starting external merge sort for " + 
                        std::to_string(entry_count) + " entries...");
    }
    
    // Reset stream to beginning for reading chunks
    in.seekg(0, std::ios::beg);
    
    // Stage 1: Create sorted chunks
    std::vector<std::string> chunk_files;
    uint64_t chunk_count = (entry_count + entries_fit_memory - 1) / entries_fit_memory;
    
    for (uint64_t chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
        uint64_t chunk_entries = std::min(entries_fit_memory, 
                                         entry_count - chunk_idx * entries_fit_memory);
        
        if (progress_callback) {
            progress_callback("Sorting chunk " + std::to_string(chunk_idx + 1) + 
                            "/" + std::to_string(chunk_count) + 
                            " (" + std::to_string(chunk_entries) + " entries)");
        }
        
        std::vector<Entry> chunk(chunk_entries);
        in.read(reinterpret_cast<char*>(chunk.data()), chunk_entries * ENTRY_SIZE);
        
        if (!in && !in.eof()) {
            throw std::runtime_error("Read error during chunk creation");
        }
        
        std::sort(chunk.begin(), chunk.end());
        
        std::string chunk_file = path + ".sortchunk" + std::to_string(chunk_idx);
        chunk_files.push_back(chunk_file);
        
        std::ofstream chunk_out(chunk_file, std::ios::binary);
        chunk_out.write(reinterpret_cast<const char*>(chunk.data()), 
                       chunk_entries * ENTRY_SIZE);
        chunk_out.close();
        
        if (!chunk_out) {
            throw std::runtime_error("Write error during chunk creation");
        }
    }
    in.close();
    
    if (progress_callback) {
        progress_callback("Merging " + std::to_string(chunk_files.size()) + " sorted chunks...");
    }
    
    // Stage 2: N-way merge using min-heap
    struct ChunkReader {
        std::ifstream stream;
        Entry current;
        bool valid;
        size_t chunk_id;
        
        ChunkReader(const std::string& file, size_t id) : valid(false), chunk_id(id) {
            stream.open(file, std::ios::binary);
            if (stream) {
                read_next();
            }
        }
        
        bool read_next() {
            if (stream.read(reinterpret_cast<char*>(&current), ENTRY_SIZE)) {
                valid = true;
                return true;
            }
            valid = false;
            return false;
        }
        
        // Comparison for min-heap (invert for max-heap behavior)
        bool operator>(const ChunkReader& other) const {
            return current > other.current;
        }
    };
    
    std::vector<ChunkReader> readers;
    readers.reserve(chunk_files.size());
    for (size_t i = 0; i < chunk_files.size(); ++i) {
        readers.emplace_back(chunk_files[i], i);
    }
    
    // Min-heap of chunk readers
    auto cmp = [](const ChunkReader* a, const ChunkReader* b) {
        return *a > *b;
    };
    std::priority_queue<ChunkReader*, std::vector<ChunkReader*>, decltype(cmp)> min_heap(cmp);
    
    for (auto& reader : readers) {
        if (reader.valid) {
            min_heap.push(&reader);
        }
    }
    
    // Write merged output
    std::string temp_output = path + ".sorted";
    std::ofstream out(temp_output, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to create sorted output file");
    }
    
    const size_t WRITE_BUFFER_SIZE = 4096;
    std::vector<Entry> write_buffer;
    write_buffer.reserve(WRITE_BUFFER_SIZE);
    
    uint64_t merged_count = 0;
    while (!min_heap.empty()) {
        ChunkReader* reader = min_heap.top();
        min_heap.pop();
        
        write_buffer.push_back(reader->current);
        
        if (write_buffer.size() == WRITE_BUFFER_SIZE) {
            out.write(reinterpret_cast<const char*>(write_buffer.data()),
                     write_buffer.size() * ENTRY_SIZE);
            merged_count += write_buffer.size();
            write_buffer.clear();
            
            if (progress_callback && merged_count % 1000000 == 0) {
                progress_callback("Merged " + std::to_string(merged_count) + 
                                "/" + std::to_string(entry_count) + " entries");
            }
        }
        
        if (reader->read_next()) {
            min_heap.push(reader);
        }
    }
    
    // Write remaining buffer
    if (!write_buffer.empty()) {
        out.write(reinterpret_cast<const char*>(write_buffer.data()),
                 write_buffer.size() * ENTRY_SIZE);
        merged_count += write_buffer.size();
    }
    
    if (!out) {
        throw std::runtime_error("Write error during merge");
    }
    out.close();
    
    // Close chunk readers and delete chunk files
    readers.clear();
    for (const auto& chunk_file : chunk_files) {
        fs::remove(chunk_file);
    }
    
    // Replace original with sorted version
    fs::remove(path);
    fs::rename(temp_output, path);
    
    if (progress_callback) {
        progress_callback("Sort complete: " + std::to_string(merged_count) + " entries");
    }
}

// Load prefix index file
void SortedEccDB::load_index(const std::string& idx_path) {
    if (!fs::exists(idx_path)) {
        throw std::runtime_error("Index file does not exist: " + idx_path);
    }
    
    uint64_t file_size = fs::file_size(idx_path);
    size_t expected_size = (16777216 + 1) * sizeof(uint64_t); // (2^24 + 1) * 8
    
    if (file_size != expected_size) {
        throw std::runtime_error("Invalid index file size");
    }
    
    std::ifstream idx_file(idx_path, std::ios::binary);
    if (!idx_file) {
        throw std::runtime_error("Failed to open index file");
    }
    
    index_table_.resize(16777216 + 1);
    idx_file.read(reinterpret_cast<char*>(index_table_.data()), expected_size);
    
    if (!idx_file) {
        throw std::runtime_error("Failed to read index file");
    }
    
    has_index_ = true;
    // fprintf(stderr, "[INFO] Loaded prefix index (%llu MB)\n", (unsigned long long)(file_size / 1024 / 1024));
}



// Binary search implementation - native x86/x64, no endian conversions!
bool SortedEccDB::binary_search(const uint64_t x_limbs[4], uint64_t* out_j) const {
    // Fast path: Memory Mapped I/O
    if (map_addr_) {
        const Entry* entries = static_cast<const Entry*>(map_addr_);
        
        uint64_t left = 0;
        uint64_t right = entry_count_;
        
        // Use index if available
        if (has_index_) {
            uint32_t prefix = static_cast<uint32_t>(x_limbs[3] >> 40);
            if (prefix < index_table_.size() - 1) {
                left = index_table_[prefix];
                right = index_table_[prefix + 1];
                if (left > entry_count_) left = entry_count_;
                if (right > entry_count_) right = entry_count_;
            }
        }
        
        Entry target;
        std::memcpy(target.x_limbs, x_limbs, 32);
        
        const Entry* begin = entries + left;
        const Entry* end = entries + right;
        
        // std::lower_bound is extremely fast on memory mapped pointers
        // OPTIMIZATION: Copy the small range to stack/heap to force sequential page-in
        // This avoids random page faults during binary search which can be slower on rotating disks
        uint64_t count = right - left;
        
        // If range is small enough for stack (e.g. < 2048 entries = 80KB)
        if (count < 2048) {
            static thread_local std::vector<Entry> stack_buf;
            if (stack_buf.capacity() < 2048) stack_buf.reserve(2048);
            if (stack_buf.size() < count) stack_buf.resize(count);
            
            const Entry* src = entries + left;
            
            // Sequential copy forces OS to read pages efficiently
            std::memcpy(stack_buf.data(), src, count * sizeof(Entry));
            
            Entry* buf_begin = stack_buf.data();
            Entry* buf_end = stack_buf.data() + count;
            
            auto it = std::lower_bound(buf_begin, buf_end, target);
            if (it != buf_end && *it == target) {
                if (out_j) *out_j = it->j;
                return true;
            }
            return false;
        }
        
        // Fallback for larger ranges (direct mapped search)
        auto it = std::lower_bound(begin, end, target);
        
        if (it != end && *it == target) {
            if (out_j) *out_j = it->j;
            return true;
        }
        return false;
    }

    uint64_t left = 0;
    uint64_t right = entry_count_;
    
    // Use index if available to narrow search range
    if (has_index_) {
        // Extract top 3 bytes (24 bits) from MSW (x_limbs[3])
        uint32_t prefix = static_cast<uint32_t>(x_limbs[3] >> 40);
        
        if (prefix < index_table_.size() - 1) {
            left = index_table_[prefix];
            right = index_table_[prefix + 1];
            
            // Safety check
            if (left > entry_count_) left = entry_count_;
            if (right > entry_count_) right = entry_count_;
        }
    }

    // OPTIMIZATION: If range is small (e.g. < 4096 entries), read entire block into RAM
    // This reduces disk I/O from log2(N) seeks to just 1 seek + 1 read.
    // 4096 entries * 40 bytes = 160 KB (very cheap for RAM)
    if (right - left <= 4096 && right > left) {
        uint64_t count = right - left;
        static thread_local std::vector<Entry> block;
        if (block.capacity() < 4096) block.reserve(4096);
        if (block.size() < count) block.resize(count);
        // Resize once per thread; reuse buffer storage across lookups
        
        // Manual seek and read for the block
        const uint64_t off64 = left * static_cast<uint64_t>(record_size_) + header_size_;
        const std::streamoff offset = static_cast<std::streamoff>(off64);
        
        if (!file_.good()) file_.clear();
        file_.seekg(offset, std::ios::beg);
        
        size_t bytes_to_read = count * record_size_;
        if (is_xonly_) {
            // For X-only format, read 32-byte blocks and fill Entry structures
            std::vector<uint8_t> raw_buffer(bytes_to_read);
            file_.read(reinterpret_cast<char*>(raw_buffer.data()), bytes_to_read);
            if (!file_ || file_.gcount() != static_cast<std::streamsize>(bytes_to_read)) {
                return false;
            }
            // Convert to Entry format (X + j=0)
            for (size_t i = 0; i < count; ++i) {
                std::memcpy(block[i].x_limbs, raw_buffer.data() + i * 32, 32);
                block[i].j = 0;
            }
        } else {
            file_.read(reinterpret_cast<char*>(block.data()), bytes_to_read);
            if (!file_ || file_.gcount() != static_cast<std::streamsize>(bytes_to_read)) {
                return false;
            }
        }
        
        // Binary search in memory
        Entry target;
        std::memcpy(target.x_limbs, x_limbs, 32);
        
        // Use pointers directly to avoid vector size issues
        Entry* block_begin = block.data();
        Entry* block_end = block.data() + count;
        
        auto it = std::lower_bound(block_begin, block_end, target);
        if (it != block_end && *it == target) {
            if (out_j) *out_j = it->j;
            return true;
        }
        return false;
    }

    // Fallback to disk-based binary search for large ranges
    while (left < right) {
        uint64_t mid = left + (right - left) / 2;
        
        // Read entry at mid - direct memcpy, native format
        const uint64_t mid64 = static_cast<uint64_t>(mid);
        const uint64_t off64 = mid64 * static_cast<uint64_t>(record_size_) + header_size_;
        std::streampos offset = static_cast<std::streampos>(static_cast<long long>(off64));
        // Clear any previous fail/eof flags before large-file seek
        if (!file_.good()) file_.clear();
        file_.seekg(offset, std::ios::beg);
        Entry current;
        if (is_xonly_) {
            // Read only X coordinate (32 bytes), j is unknown for X-only databases
            file_.read(reinterpret_cast<char*>(&current.x_limbs), 32);
            current.j = 0; // X-only format doesn't have j values
        } else {
            file_.read(reinterpret_cast<char*>(&current), ENTRY_SIZE);
        }
        
        if (!file_) {
            // fprintf(stderr, "[BS] read fail mid=%llu left=%llu right=%llu offset=%lld eof=%d fail=%d\n",
            //     (unsigned long long)mid, (unsigned long long)left, (unsigned long long)right, (long long)offset,
            //     file_.eof(), file_.fail());
            return false;
        }
        // if (debug_steps < 6) {
        //     fprintf(stderr, "[BS] mid=%llu L=%llu R=%llu curX=%016llx%016llx%016llx%016llx tgtX=%016llx%016llx%016llx%016llx\n",
        //         (unsigned long long)mid, (unsigned long long)left, (unsigned long long)right,
        //         (unsigned long long)current.x_limbs[3], (unsigned long long)current.x_limbs[2],
        //         (unsigned long long)current.x_limbs[1], (unsigned long long)current.x_limbs[0],
        //         (unsigned long long)x_limbs[3], (unsigned long long)x_limbs[2],
        //         (unsigned long long)x_limbs[1], (unsigned long long)x_limbs[0]);
        //     debug_steps++;
        // }
        
        // Compare X coordinates MSB first: limbs[3] → [2] → [1] → [0]
        if (current.x_limbs[3] != x_limbs[3]) {
            if (current.x_limbs[3] < x_limbs[3]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        } else if (current.x_limbs[2] != x_limbs[2]) {
            if (current.x_limbs[2] < x_limbs[2]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        } else if (current.x_limbs[1] != x_limbs[1]) {
            if (current.x_limbs[1] < x_limbs[1]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        } else if (current.x_limbs[0] != x_limbs[0]) {
            if (current.x_limbs[0] < x_limbs[0]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        } else {
            // Found exact match!
            if (out_j) *out_j = current.j;
            return true;
        }
    }
    
    return false;
}

// Lookup by limbs array
bool SortedEccDB::lookup(const uint64_t x_limbs[4], uint64_t* out_j) const {
    // NOTE: binary_search uses file_ which is NOT thread-safe!
    // For multi-threaded usage, each thread needs its own file handle
    return binary_search(x_limbs, out_j);
}

// Lookup by FieldElement
bool SortedEccDB::lookup(const fast::FieldElement& x, uint64_t* out_j) const {
    return binary_search(x.limbs().data(), out_j);
}

// Validate database is sorted
SortedEccDB::ValidationResult SortedEccDB::validate_sort() const {
    ValidationResult result;
    result.is_sorted = true;
    result.total_checked = 0;
    result.sort_errors = 0;
    result.first_error_index = 0;
    
    if (entry_count_ == 0) return result;
    
    std::ifstream file(path_, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open database for validation");
    }
    
    Entry prev, current;
    file.read(reinterpret_cast<char*>(&prev), ENTRY_SIZE);
    result.total_checked = 1;
    
    for (uint64_t i = 1; i < entry_count_; ++i) {
        file.read(reinterpret_cast<char*>(&current), ENTRY_SIZE);
        
        // Check: prev < current (strict ascending order)
        if (!(prev < current)) {
            result.is_sorted = false;
            result.sort_errors++;
            if (result.sort_errors == 1) {
                result.first_error_index = i;
            }
        }
        
        prev = current;
        result.total_checked++;
    }
    
    return result;
}

// Verify UNSORTED database: j values are sequential (j=entry_index+1)
SortedEccDB::VerifyResult SortedEccDB::verify_unsorted(
    size_t num_threads,
    std::function<void(uint64_t, uint64_t)> progress_callback) const
{
    VerifyResult result{};
    result.total_tested = entry_count_;
    
    if (entry_count_ == 0) {
        result.is_valid = true;
        return result;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (num_threads <= 1) {
        // Single-threaded: read file sequentially
        std::ifstream file(path_, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open database");
        }
        
        fast::Point p = fast::Point::generator();
        
        for (uint64_t j = 1; j <= entry_count_; ++j) {
            // Read entry at index (j-1)
            Entry entry;
            file.read(reinterpret_cast<char*>(&entry), ENTRY_SIZE);
            
            // Get expected X coordinate from j*G
            fast::FieldElement z_inv = p.z().inverse();
            fast::FieldElement z_inv2 = z_inv.square();
            fast::FieldElement x_affine = p.x() * z_inv2;
            uint64_t x_limbs[4];
            std::memcpy(x_limbs, x_affine.limbs().data(), 32);
            
            // Check: entry.x_limbs == x_limbs && entry.j == j
            bool x_match = (entry.x_limbs[0] == x_limbs[0] &&
                           entry.x_limbs[1] == x_limbs[1] &&
                           entry.x_limbs[2] == x_limbs[2] &&
                           entry.x_limbs[3] == x_limbs[3]);
            
            if (!x_match) {
                result.not_found++;
            } else if (entry.j != j) {
                result.value_mismatch++;
            } else {
                result.found_correct++;
            }
            
            if (progress_callback && (j % 100000 == 0 || j == entry_count_)) {
                progress_callback(j, entry_count_);
            }
            
            p = p.next();
        }
    } else {
        // Multi-threaded
        std::atomic<uint64_t> found_correct{0};
        std::atomic<uint64_t> not_found{0};
        std::atomic<uint64_t> value_mismatch{0};
        std::atomic<uint64_t> completed{0};
        
        std::mutex progress_mutex;
        auto last_progress = std::chrono::high_resolution_clock::now();
        
        auto worker = [&](uint64_t start_j, uint64_t count) {
            std::ifstream file(path_, std::ios::binary);
            if (!file) return;
            
            // Seek to start position
            file.seekg((start_j - 1) * ENTRY_SIZE);
            
            fast::Point p = fast::Point::generator();
            if (start_j > 1) {
                fast::Scalar s = fast::Scalar::from_uint64(start_j - 1);
                p = p.scalar_mul(s);
            }
            
            for (uint64_t i = 0; i < count; ++i) {
                uint64_t j = start_j + i;
                
                Entry entry;
                file.read(reinterpret_cast<char*>(&entry), ENTRY_SIZE);
                
                fast::FieldElement z_inv = p.z().inverse();
                fast::FieldElement z_inv2 = z_inv.square();
                fast::FieldElement x_affine = p.x() * z_inv2;
                uint64_t x_limbs[4];
                std::memcpy(x_limbs, x_affine.limbs().data(), 32);
                
                bool x_match = (entry.x_limbs[0] == x_limbs[0] &&
                               entry.x_limbs[1] == x_limbs[1] &&
                               entry.x_limbs[2] == x_limbs[2] &&
                               entry.x_limbs[3] == x_limbs[3]);
                
                if (!x_match) {
                    not_found++;
                } else if (entry.j != j) {
                    value_mismatch++;
                } else {
                    found_correct++;
                }
                
                completed++;
                
                if (progress_callback) {
                    auto now = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_progress).count();
                    
                    if (elapsed > 5000 || completed.load() == entry_count_) {
                        std::lock_guard<std::mutex> lock(progress_mutex);
                        auto current_time = std::chrono::high_resolution_clock::now();
                        auto current_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_progress).count();
                        
                        if (current_elapsed > 5000 || completed.load() == entry_count_) {
                            progress_callback(completed.load(), entry_count_);
                            last_progress = current_time;
                        }
                    }
                }
                
                p = p.next();
            }
        };
        
        std::vector<std::thread> threads;
        uint64_t chunk_size = (entry_count_ + num_threads - 1) / num_threads;
        
        for (size_t t = 0; t < num_threads; ++t) {
            uint64_t start_j = 1 + t * chunk_size;
            if (start_j > entry_count_) break;
            
            uint64_t count = std::min(chunk_size, entry_count_ - start_j + 1);
            threads.emplace_back(worker, start_j, count);
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        result.found_correct = found_correct.load();
        result.not_found = not_found.load();
        result.value_mismatch = value_mismatch.load();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    result.is_valid = (result.found_correct == entry_count_);
    
    return result;
}

// Verify SORTED database: Generate points sequentially and Binary Search in DB
SortedEccDB::VerifyResult SortedEccDB::verify_sorted(
    size_t num_threads,
    std::function<void(uint64_t, uint64_t)> progress_callback) const 
{
    VerifyResult result{};
    result.total_tested = entry_count_;
    
    if (entry_count_ == 0) {
        result.is_valid = true;
        return result;
    }
    
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Memory Map the file for fast random access
    struct ScopedMap {
        HANDLE hFile = INVALID_HANDLE_VALUE;
        HANDLE hMap = NULL;
        const Entry* data = nullptr;
        uint64_t size = 0;

        ScopedMap(const std::string& path) {
            hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
            if (hFile == INVALID_HANDLE_VALUE) {
#ifdef _WIN32
                fprintf(stderr, "[WARN] CreateFileA failed for %s\n", path.c_str());
#else
                // Linux: use real POSIX mmap
                int fd = open(path.c_str(), O_RDONLY);
                if (fd == -1) {
                    fprintf(stderr, "[WARN] open failed for %s: %s\n", path.c_str(), strerror(errno));
                    return;
                }

                struct stat st;
                if (fstat(fd, &st) == -1) {
                    close(fd); 
                    return;
                }
                size = st.st_size;

                void* mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
                close(fd); // Can close fd immediately after mmap
                
                if (mapped == MAP_FAILED) {
                    fprintf(stderr, "[WARN] mmap failed: %s\n", strerror(errno));
                    return;
                }

                data = static_cast<const Entry*>(mapped);
#endif
                return;
            }

            LARGE_INTEGER fileSize;
            if (!GetFileSizeEx(hFile, &fileSize)) {
                CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE;
                return;
            }
            size = fileSize.QuadPart;

            hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
            if (!hMap) {
                fprintf(stderr, "[WARN] CreateFileMappingA failed (Error: %lu)\n", GetLastError());
                CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE;
                return;
            }

            data = static_cast<const Entry*>(MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0));
            if (!data) {
                fprintf(stderr, "[WARN] MapViewOfFile failed (Error: %lu). Falling back to slow disk I/O.\n", GetLastError());
            }
        }

        ~ScopedMap() {
#ifdef _WIN32
            if (data) UnmapViewOfFile(data);
            if (hMap) CloseHandle(hMap);
            if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
#else
            if (data) munmap(const_cast<void*>(static_cast<const void*>(data)), size);
#endif
        }
    };

    ScopedMap map(path_);
    
    // If mapping failed or file is too large (> 64GB) to be efficiently random-accessed on consumer hardware,
    // switch to "Stream Verification" (External Sort) strategy.
    // Random access on >RAM datasets is extremely slow (thrashing).
    // 64GB is a generous threshold; usually 32GB is the limit for common dev machines.
    bool use_stream_verification = (!map.data) || (entry_count_ * ENTRY_SIZE > 64ULL * 1024 * 1024 * 1024);

    if (use_stream_verification) {
        if (map.data) {
             // Unmap to free resources for external sort
             // ScopedMap destructor handles this when it goes out of scope, but we can force it?
             // No, just let it be. But we should print a message.
             fprintf(stderr, "[INFO] Database size (%llu GB) exceeds RAM threshold. Switching to Stream Verification (External Sort).\n", 
                (unsigned long long)(entry_count_ * ENTRY_SIZE / 1024 / 1024 / 1024));
        } else {
             fprintf(stderr, "[INFO] Memory mapping failed. Switching to Stream Verification (External Sort).\n");
        }

        // Create temp directory for verification chunks
        std::string temp_dir = fs::path(path_).parent_path().string() + "/verify_temp_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        fs::create_directories(temp_dir);

        // 1. Generate chunks (j=1..N) to temp_dir
        // Reuse generate() logic but we need to call it carefully.
        // generate() creates chunks in chunk_dir.
        // We can call generate() with a dummy path?
        // generate(path, count, chunk_dir, ...)
        // The 'path' argument is only used to derive chunk filenames: fs::path(path).filename() + ".chunk.X"
        // So we can pass "verify.dat" as path.
        
        std::atomic<uint64_t> total_verified{0};
        std::atomic<uint64_t> total_mismatch{0};
        
        // Step 1: Generate Chunks
        // We use the same number of threads as requested
        generate("verify_dummy.dat", entry_count_, temp_dir, 1, num_threads, 
            [&](uint64_t current, uint64_t total) {
                if (progress_callback) progress_callback(current / 2, total); // 0-50% progress
            });

        // Step 2: Merge-Compare
        // We open all chunks and the main DB, and compare streams.
        
        // Find chunk files
        std::vector<std::string> chunk_files;
        for (const auto& entry : fs::directory_iterator(temp_dir)) {
            if (entry.path().filename().string().find("verify_dummy.dat.chunk.") == 0) {
                chunk_files.push_back(entry.path().string());
            }
        }
        
        // K-way merge reader
        struct ChunkReader {
            std::ifstream stream;
            Entry current;
            bool valid;
            
            ChunkReader(const std::string& file) : valid(false) {
                stream.open(file, std::ios::binary);
                if (stream) read_next();
            }
            
            bool read_next() {
                if (stream.read(reinterpret_cast<char*>(&current), ENTRY_SIZE)) {
                    valid = true;
                    return true;
                }
                valid = false;
                return false;
            }
            
            bool operator>(const ChunkReader& other) const {
                return current > other.current;
            }
        };
        
        std::vector<std::unique_ptr<ChunkReader>> readers;
        for (const auto& f : chunk_files) {
            readers.push_back(std::make_unique<ChunkReader>(f));
        }
        
        auto cmp = [](const ChunkReader* a, const ChunkReader* b) {
            return *a > *b;
        };
        std::priority_queue<ChunkReader*, std::vector<ChunkReader*>, decltype(cmp)> min_heap(cmp);
        
        for (const auto& r : readers) {
            if (r->valid) min_heap.push(r.get());
        }
        
        // Open main DB for sequential reading
        std::ifstream db_file(path_, std::ios::binary);
        // Use large buffer for DB file
        std::vector<char> db_buffer(16 * 1024 * 1024); // 16MB buffer
        db_file.rdbuf()->pubsetbuf(db_buffer.data(), db_buffer.size());
        
        Entry db_entry;
        uint64_t processed = 0;
        
        while (!min_heap.empty() && db_file.read(reinterpret_cast<char*>(&db_entry), ENTRY_SIZE)) {
            ChunkReader* reader = min_heap.top();
            min_heap.pop();
            
            // Compare reader->current with db_entry
            if (!(reader->current == db_entry)) {
                total_mismatch++;
                // If mismatch, we might be out of sync?
                // But since both are sorted streams, a mismatch means the DB is wrong or generated wrong.
                // We continue?
            } else {
                total_verified++;
            }
            
            if (reader->read_next()) {
                min_heap.push(reader);
            }
            
            processed++;
            if (progress_callback && processed % 1000000 == 0) {
                progress_callback(entry_count_ / 2 + processed / 2, entry_count_); // 50-100% progress
            }
        }
        
        // Check if any stream has leftovers
        if (!min_heap.empty() || db_file.peek() != EOF) {
            total_mismatch++; // Size mismatch effectively
        }
        
        // Cleanup
        readers.clear();
        fs::remove_all(temp_dir);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
        result.found_correct = total_verified;
        result.value_mismatch = total_mismatch;
        result.is_valid = (total_mismatch == 0 && total_verified == entry_count_);
        
        return result;
    }
    
    std::atomic<uint64_t> total_found{0};
    std::atomic<uint64_t> total_not_found{0};
    std::atomic<uint64_t> total_mismatch{0};
    std::atomic<uint64_t> total_processed{0};
    std::mutex progress_mutex;

    // Helper: Range Check
    auto range_check = [&](uint64_t range_start_j, uint64_t range_count) {
        uint64_t local_found = 0;
        uint64_t local_not_found = 0;
        uint64_t local_mismatch = 0;

        // Fallback file stream if mapping failed
        std::unique_ptr<std::ifstream> file_ptr;
        if (!map.data) {
             file_ptr = std::make_unique<std::ifstream>(path_, std::ios::binary);
             if (!*file_ptr) return;
        }

        fast::ensure_fixed_base_ready();

        fast::Point p = fast::Point::generator();
        if (range_start_j > 1) {
            p = p.scalar_mul(fast::Scalar::from_uint64(range_start_j));
        }

        constexpr size_t BATCH_SIZE = 4096;
        std::vector<fast::Point> batch_points(BATCH_SIZE);
        std::vector<fast::FieldElement> batch_z(BATCH_SIZE);
        std::vector<uint64_t> batch_j(BATCH_SIZE);
        
        // Query structure for sorting to improve locality
        struct Query {
            uint64_t x_limbs[4];
            uint64_t j;
            
            bool operator<(const Query& other) const {
                if (x_limbs[3] != other.x_limbs[3]) return x_limbs[3] < other.x_limbs[3];
                if (x_limbs[2] != other.x_limbs[2]) return x_limbs[2] < other.x_limbs[2];
                if (x_limbs[1] != other.x_limbs[1]) return x_limbs[1] < other.x_limbs[1];
                return x_limbs[0] < other.x_limbs[0];
            }
        };
        std::vector<Query> queries;
        queries.reserve(BATCH_SIZE);

        size_t batch_idx = 0;
        
        auto lookup_impl = [&](const uint64_t x_limbs[4], uint64_t* out_j) -> bool {
            if (map.data) {
                // Fast in-memory binary search
                Entry target;
                std::memcpy(target.x_limbs, x_limbs, 32);
                
                const Entry* begin = map.data;
                const Entry* end = map.data + entry_count_;
                const Entry* it = std::lower_bound(begin, end, target);
                
                if (it != end && *it == target) {
                    if (out_j) *out_j = it->j;
                    return true;
                }
                return false;
            } else {
                // Disk-based binary search (slow fallback)
                std::ifstream& file = *file_ptr;
                uint64_t left = 0;
                uint64_t right = entry_count_;
                
                while (left < right) {
                    uint64_t mid = left + (right - left) / 2;
                    Entry current;
                    if (!safe_seek_read(file, mid, ENTRY_SIZE, &current)) {
                        file.clear();
                        return false;
                    }
                    
                    if (current.x_limbs[3] != x_limbs[3]) {
                        if (current.x_limbs[3] < x_limbs[3]) left = mid + 1;
                        else right = mid;
                    } else if (current.x_limbs[2] != x_limbs[2]) {
                        if (current.x_limbs[2] < x_limbs[2]) left = mid + 1;
                        else right = mid;
                    } else if (current.x_limbs[1] != x_limbs[1]) {
                        if (current.x_limbs[1] < x_limbs[1]) left = mid + 1;
                        else right = mid;
                    } else if (current.x_limbs[0] != x_limbs[0]) {
                        if (current.x_limbs[0] < x_limbs[0]) left = mid + 1;
                        else right = mid;
                    } else {
                        if (out_j) *out_j = current.j;
                        return true;
                    }
                }
                return false;
            }
        };

        for (uint64_t j = range_start_j; j < range_start_j + range_count; ++j) {
            batch_points[batch_idx] = p;
            batch_z[batch_idx] = p.z();
            batch_j[batch_idx] = j;
            batch_idx++;
            
            p.next_inplace();

            if (batch_idx == BATCH_SIZE || j == range_start_j + range_count - 1) {
                fast::fe_batch_inverse(batch_z.data(), batch_idx);
                
                queries.clear();
                for (size_t i = 0; i < batch_idx; ++i) {
                    fast::FieldElement z_inv_sq = batch_z[i].square();
                    fast::FieldElement x_affine = batch_points[i].X() * z_inv_sq;
                    
                    Query q;
                    std::memcpy(q.x_limbs, x_affine.limbs().data(), 32);
                    q.j = batch_j[i];
                    queries.push_back(q);
                }
                
                // Sort queries to improve memory locality
                std::sort(queries.begin(), queries.end());

                for (const auto& q : queries) {
                    uint64_t found_j_val = 0;
                    if (lookup_impl(q.x_limbs, &found_j_val)) {
                        if (found_j_val == q.j) {
                            local_found++;
                        } else {
                            local_mismatch++;
                        }
                    } else {
                        local_not_found++;
                    }
                }
                
                uint64_t current = total_processed.fetch_add(batch_idx) + batch_idx;
                if (progress_callback) {
                     if (current % 100000 < BATCH_SIZE) {
                        std::lock_guard<std::mutex> lock(progress_mutex);
                        progress_callback(current, entry_count_);
                     }
                }

                batch_idx = 0;
            }
        }
        total_found += local_found;
        total_not_found += local_not_found;
        total_mismatch += local_mismatch;
    };

    // Launch threads
    std::vector<std::future<void>> futures;
    uint64_t chunk_size = (entry_count_ + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
        uint64_t start_j = 1 + t * chunk_size;
        if (start_j > entry_count_) break;
        uint64_t count = std::min(chunk_size, entry_count_ - start_j + 1);
        
        futures.push_back(std::async(std::launch::async, range_check, start_j, count));
    }

    for (auto& f : futures) {
        f.wait();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    
    result.found_correct = total_found;
    result.not_found = total_not_found;
    result.value_mismatch = total_mismatch;
    result.is_valid = (result.found_correct == entry_count_);
    
    return result;
}

// Auto-detect and verify: check if database is sorted or unsorted
SortedEccDB::VerifyResult SortedEccDB::verify(
    size_t num_threads,
    std::function<void(uint64_t, uint64_t)> progress_callback) const
{
    if (entry_count_ == 0) {
        VerifyResult result{};
        result.is_valid = true;
        return result;
    }
    
    // Read first entry
    std::ifstream file(path_, std::ios::binary);
    Entry first;
    file.read(reinterpret_cast<char*>(&first), ENTRY_SIZE);
    
    // Check if j=1 (unsorted) or j!=1 (sorted)
    if (first.j == 1) {
        // Likely unsorted (sequential j values)
        return verify_unsorted(num_threads, progress_callback);
    } else {
        // Likely sorted by X
        return verify_sorted(num_threads, progress_callback);
    }
}

} // namespace secp256k1


