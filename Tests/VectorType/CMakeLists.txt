cmake_minimum_required(VERSION 3.12)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_86,code=sm_86")
find_package(CUDA REQUIRED)
cuda_add_executable(load_data load_data.cu)
