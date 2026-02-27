cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(flash_attention LANGUAGES CXX CUDA)

# ── C++ / CUDA standards ────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── GPU architecture detection ──────────────────────────────────────────
# Default to SM 8.0 (A100).  Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# ── Compiler flags ──────────────────────────────────────────────────────
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3")

# ── Find packages ──────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── Header-only includes ───────────────────────────────────────────────
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ── Static library with all kernels ────────────────────────────────────
add_library(flash_attn_kernels STATIC
    src/naive_attention.cu
    src/flash_attn_fwd.cu
    src/flash_attn_bwd.cu
)
target_include_directories(flash_attn_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(flash_attn_kernels PUBLIC CUDA::cudart)
set_target_properties(flash_attn_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ── Optional: standalone test binary (no PyTorch dependency) ───────────
option(BUILD_STANDALONE_TEST "Build standalone CUDA test binary" OFF)
if(BUILD_STANDALONE_TEST)
    add_executable(test_flash_attn tests/test_standalone.cu)
    target_link_libraries(test_flash_attn flash_attn_kernels)
endif()
