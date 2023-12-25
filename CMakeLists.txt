cmake_minimum_required(VERSION 3.23)
project(cuda)

find_package(CURL REQUIRED)
find_package(json-c REQUIRED)

enable_language(CUDA)

add_executable(cuda main.cu requests.cu kernel/kernel.cu crypto/base58.cu crypto/sha256.cu crypto/hex.cu)
set_property(TARGET cuda PROPERTY CUDA_STANDARD 11)
set_property(TARGET cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cuda PROPERTY CUDA_ARCHITECTURES native)
target_link_libraries(cuda PRIVATE CURL::libcurl json-c::json-c)