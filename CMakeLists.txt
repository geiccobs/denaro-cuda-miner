cmake_minimum_required(VERSION 3.24)
project(cuda)

if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
else()
    message(FATAL_ERROR "CUDA is not supported")
endif()

find_package(CURL REQUIRED)
find_package(json-c REQUIRED)

add_executable(cuda main.cu requests.cu kernel/kernel.cu crypto/base58.cu crypto/sha256.cu crypto/hex.cu)
set_property(TARGET cuda PROPERTY CUDA_STANDARD 11)
set_property(TARGET cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cuda PROPERTY CUDA_ARCHITECTURES native)
set_property(TARGET cuda PROPERTY CUDA_NVCC_FLAGS -use_fast_math -O3)

target_link_libraries(cuda PRIVATE CURL::libcurl json-c::json-c)