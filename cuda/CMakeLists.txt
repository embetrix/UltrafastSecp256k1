cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES OFF)
project(Secp256k1Cuda LANGUAGES CXX CUDA)

option(SECP256K1_CUDA_USE_MONTGOMERY "Use Montgomery field arithmetic backend in CUDA" OFF)
option(SECP256K1_CUDA_LIMBS_32 "Use 8x32-bit limbs for field arithmetic" OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Set CUDA architecture (auto-detect or target specific)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

include_directories(include)

# Library target
add_library(secp256k1_cuda_lib STATIC
    src/secp256k1.cu
)
target_include_directories(secp256k1_cuda_lib PUBLIC include)

if(SECP256K1_CUDA_USE_MONTGOMERY)
    target_compile_definitions(secp256k1_cuda_lib PUBLIC SECP256K1_CUDA_USE_MONTGOMERY=1)
endif()

if(SECP256K1_CUDA_LIMBS_32)
    target_compile_definitions(secp256k1_cuda_lib PUBLIC SECP256K1_CUDA_LIMBS_32=1)
endif()

target_compile_options(secp256k1_cuda_lib PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
)
set_target_properties(secp256k1_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Test suite with 30 vector tests
add_executable(secp256k1_cuda_test
    src/test_suite.cu
)
target_link_libraries(secp256k1_cuda_test PRIVATE secp256k1_cuda_lib)
