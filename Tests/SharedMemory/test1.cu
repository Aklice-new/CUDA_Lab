/**
 * @file test1.cu
 * @author your name (you@domain.com)
 * @brief 为了测试动态共享内存和静态共享内存的性能差异，测试了不同程序的耗时
 * @version 0.1
 * @date 2024-02-28
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <cuda.h>
#include <stdio.h>

#define ARRAY_SIZE 1024

__global__ void test_static_shared_memory(float *a, float *b)
{
    __shared__ float shared_a[ARRAY_SIZE];
    __shared__ float shared_b[ARRAY_SIZE];

    const int tid = threadIdx.x;
    const int id = threadIdx.x + blockDim.x * blockIdx.x;

    shared_a[threadIdx.x] = a[threadIdx.x + blockIdx.x * blockDim.x];
    shared_b[threadIdx.x] = b[threadIdx.x + blockIdx.x * blockDim.x];
    for (int i = 0; i < 3000; i++)
    {
        for (int j = 0; j < 10000; j++)
        {
            shared_a[threadIdx.x] = shared_a[threadIdx.x] + shared_b[threadIdx.x];
        }
    }

    a[threadIdx.x + blockIdx.x * blockDim.x] = shared_a[threadIdx.x];
}

__global__ void test_dynamic_shared_memory(float *a, float *b)
{
    extern __shared__ float shared_a[];
    extern __shared__ float shared_b[];

    const int tid = threadIdx.x;
    const int id = threadIdx.x + blockDim.x * blockIdx.x;

    shared_a[tid] = 1544514;
    shared_b[tid] = 998244353;
    // printf("shared_a[0] is %f, shared_b[0] is %f \n", shared_a[0], shared_b[0]);
    // __syncthreads();
    a[id] = shared_a[tid];
    b[id] = shared_b[tid];
    // if (id == 0)
    // {
    //     for (int i = 0; i < 1024; i++)
    //     {
    //         printf("shared_a address is %x, shared_b[ address is %f \n", &shared_a[i], &shared_b[i]);
    //     }
    //     // printf("shared_a[0] is %f, shared_b[0] is %f \n", shared_a[0], shared_b[0]);
    //     // printf("Address shared_A[0] is %x, shared_B[0] is %x \n", &shared_a[0], &shared_b[0]);
    // }

    // printf("Address shared_A[0] is %x, shared_B[0] is %x \n", &shared_a[tid], &shared_b[tid]);
    // shared_a[threadIdx.x] = a[threadIdx.x + blockIdx.x * blockDim.x];
    // shared_b[threadIdx.x] = b[threadIdx.x + blockIdx.x * blockDim.x];

    // for (int i = 0; i < 3000; i++)
    // {
    //     for (int j = 0; j < 10000; j++)
    //     {
    //         shared_a[threadIdx.x] = shared_a[threadIdx.x] + shared_b[threadIdx.x];
    //     }
    // }

    // a[threadIdx.x + blockIdx.x * blockDim.x] = shared_a[threadIdx.x];
}

int main()
{
    float *h_a, *h_b;
    float *d_a, *d_b;
    int size = ARRAY_SIZE * sizeof(float);
    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        h_a[i] = i;
        h_b[i] = i;
    }
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    dim3 block(64);
    dim3 grid(ARRAY_SIZE / block.x);

    // for warm up
    for (int i = 0; i < 20; i++)
    {
        test_static_shared_memory<<<grid, block>>>(d_a, d_b);
    }
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    // test dynamic shared memory
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    float time_sum2 = 0;
    for (int i = 0; i < 1; i++)
    {
        cudaEventRecord(start2, 0);
        // shared memory size: 2 * ARRAY_SIZE * sizeof(float) = 2 * 1024 * 4 = 8KB
        test_dynamic_shared_memory<<<grid, block, 2 * ARRAY_SIZE * sizeof(float)>>>(d_a, d_b);
        cudaEventRecord(stop2, 0);
        cudaDeviceSynchronize();
        cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
        printf("h_a[0] is %f, h_b[0] is %f\n", h_a[0], h_b[0]);
        float time2;
        cudaEventElapsedTime(&time2, start2, stop2);
        time_sum2 += time2;
    }

    printf("Dynamic shared memory time: %f\n", time_sum2 / 10);

    // test static shared memory
    float time_sum1 = 0;
    for (int i = 0; i < 10; i++)
    {
        cudaEventRecord(start1, 0);
        test_static_shared_memory<<<grid, block>>>(d_a, d_b);
        cudaEventRecord(stop1, 0);
        cudaDeviceSynchronize();
        float time1;
        cudaEventElapsedTime(&time1, start1, stop1);
        time_sum1 += time1;
    }
    printf("Static shared memory time: %f\n", time_sum1 / 10);
}
