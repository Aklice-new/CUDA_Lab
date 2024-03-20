#include <cstdlib>
#include <cuda.h>
#include <iostream>

#define MAX_SHARED_MEMORY_SIZE 12288

__global__ void static_memory_kernel(float *xx)
{
    __shared__ float s_a[MAX_SHARED_MEMORY_SIZE]; // 4 * 1024
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    s_a[tid] = tid;
    __syncthreads();
    xx[tid] = s_a[tid];
}

__global__ void dynamic_memory_kernel(float *xx)
{
    extern __shared__ float s_a[];
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    s_a[tid] = threadIdx.x;
    __syncthreads();
    xx[tid] = s_a[tid];
}
int main()
{
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Max shared memory per block " << prop.sharedMemPerBlock << std::endl;
    std::cout << "Max thread per block " << prop.maxThreadsPerBlock << std::endl;
    dim3 threadPerBlock = prop.maxThreadsPerBlock;
    dim3 blockPerGrid = (MAX_SHARED_MEMORY_SIZE + threadPerBlock.x - 1) / threadPerBlock.x;
    float *d_array, *h_array;
    h_array = (float *)std::malloc(MAX_SHARED_MEMORY_SIZE * sizeof(float));
    cudaMalloc((void **)&d_array, MAX_SHARED_MEMORY_SIZE * sizeof(float));
    static_memory_kernel<<<blockPerGrid, threadPerBlock>>>(d_array);
    cudaDeviceSynchronize();
    dynamic_memory_kernel<<<blockPerGrid, threadPerBlock, 4 * MAX_SHARED_MEMORY_SIZE>>>(d_array);
    cudaDeviceSynchronize();
    cudaMemcpy(h_array, d_array, MAX_SHARED_MEMORY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < MAX_SHARED_MEMORY_SIZE; i++)
    {
        std::cout << "i " << i << " " << h_array[i] << std::endl;
    }
}