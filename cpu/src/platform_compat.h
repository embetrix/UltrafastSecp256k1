#ifndef SECP256K1_PLATFORM_COMPAT_H
#define SECP256K1_PLATFORM_COMPAT_H

// Platform compatibility layer for Windows types and intrinsics on Linux/Unix

#ifndef _WIN32
// Linux/Unix platform - define Windows types and map to POSIX equivalents

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>

// Windows handle types
typedef int HANDLE;
#define INVALID_HANDLE_VALUE (-1)
#ifndef NULL
#define NULL 0
#endif

// Windows file constants
#define GENERIC_READ    0x80000000
#define FILE_SHARE_READ 0x00000001
#define OPEN_EXISTING   3
#define FILE_ATTRIBUTE_NORMAL 0x80
#define PAGE_READONLY   0x02
#define FILE_MAP_READ   0x0004

// Stub structures for compatibility
struct LARGE_INTEGER {
    int64_t QuadPart;
};

// No-op functions for Windows API on Linux (actual implementation uses POSIX)
inline HANDLE CreateFileA(const char*, uint32_t, uint32_t, void*, uint32_t, uint32_t, HANDLE) { return INVALID_HANDLE_VALUE; }
inline bool GetFileSizeEx(HANDLE, LARGE_INTEGER*) { return false; }
inline HANDLE CreateFileMappingA(HANDLE, void*, uint32_t, uint32_t, uint32_t, const char*) { return 0; }
inline void* MapViewOfFile(HANDLE, uint32_t, uint32_t, uint32_t, size_t) { return nullptr; }
inline bool UnmapViewOfFile(const void*) { return true; }
inline bool CloseHandle(HANDLE) { return true; }
inline uint32_t GetLastError() { return 0; }

#endif // !_WIN32

// Cross-platform intrinsics compatibility
#if defined(__GNUC__) && !defined(_MSC_VER)
// GCC/Clang intrinsics expect unsigned long long*, but we use uint64_t*
// Define wrapper macros to avoid explicit casts everywhere

#include <x86intrin.h>

// Map uint64_t* to unsigned long long* for ADX intrinsics
#define COMPAT_ADDCARRY_U64(carry, a, b, out) \
    _addcarry_u64(carry, a, b, reinterpret_cast<unsigned long long*>(out))

#define COMPAT_SUBBORROW_U64(borrow, a, b, out) \
    _subborrow_u64(borrow, a, b, reinterpret_cast<unsigned long long*>(out))

#else
// MSVC - use intrinsics directly
#define COMPAT_ADDCARRY_U64(carry, a, b, out) _addcarry_u64(carry, a, b, out)
#define COMPAT_SUBBORROW_U64(borrow, a, b, out) _subborrow_u64(borrow, a, b, out)
#endif

#endif // SECP256K1_PLATFORM_COMPAT_H
