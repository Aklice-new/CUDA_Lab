cmake_minimum_required(VERSION 3.12)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_86,code=sm_86 ")# -g -G")
find_package(CUDA REQUIRED)

cuda_add_executable(sgemm sgemm.cu )
target_link_libraries(sgemm ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_cusparse_LIBRARY})