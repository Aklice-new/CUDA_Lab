cmake_minimum_required(VERSION 3.12)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_86,code=sm_86")# -g -G")
find_package(CUDA REQUIRED)


macro(add_cuda_example file_name)
    cuda_add_executable(${file_name} ${file_name}.cu )
    target_link_libraries(${file_name} ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_cusparse_LIBRARY})    
endmacro()
add_cuda_example(gemv_n32)
add_cuda_example(gemv_n16)
add_cuda_example(gemv_n128)