/**
 * @file reduction.cu
 * @author aklice
 * @brief
  展开for循环，交给编译器去优化
 * @version 0.1
 * @date 2024-03-28
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstdlib>

#define THREAD_PER_BLOCK 256
#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))
#define checkCudaErrors(func)                                                  \
  {                                                                            \
    cudaError_t err = (func);                                                  \
    if (err != cudaSuccess)                                                    \
      printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  }

/**
 *  这里简单说一下cuda中的volatile关键词：
    首先加了volatile关键词之后，就可以禁止编译器对代码的优化，
    有时候可能会将共享内存优化为寄存器来优化读取，这种优化会导致一些问题，
    如果a线程中的shm被优化为了register，那么它就只属于a线程，如果b线程这个时候进行了修改
    此时，b的修改对a来说就不可见，会导致错误。
    所以使用volatile的一般情况是：在使用共享内存的过程中，如果这些shm会被除当前线程之外的
    线程修改，同时不加内存栅栏或者同步等操作，这时候就需要使用volatile关键词，禁止编译器的优化
 */
template <unsigned int BLOCK_SIZE>
__device__ void warpReduce(volatile float* sdata, int tid) {
  if (BLOCK_SIZE >= 64) sdata[tid] += sdata[tid + 32];
  if (BLOCK_SIZE >= 32) sdata[tid] += sdata[tid + 16];
  if (BLOCK_SIZE >= 16) sdata[tid] += sdata[tid + 8];
  if (BLOCK_SIZE >= 8) sdata[tid] += sdata[tid + 4];
  if (BLOCK_SIZE >= 4) sdata[tid] += sdata[tid + 2];
  if (BLOCK_SIZE >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int BLOCK_SIZE>
__global__ void reduction_kernel0(float* A, float* out) {
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int tid = bx * blockDim.x * 2 + tx;  // 这里 blockDim.x * 2 = NUM_PER_BLOCK
  __shared__ float A_s[THREAD_PER_BLOCK];
  A_s[tx] = A[tid] + A[tid + blockDim.x];
  __syncthreads();

  if (BLOCK_SIZE >= 512) {
    if (tx < 256) {
      A_s[tx] += A_s[tx + 256];
    }
    __syncthreads();
  }
  if (BLOCK_SIZE >= 256) {
    if (tx < 128) {
      A_s[tx] += A_s[tx + 128];
    }
    __syncthreads();
  }
  if (BLOCK_SIZE >= 128) {
    if (tx < 64) {
      A_s[tx] += A_s[tx + 64];
    }
    __syncthreads();
  }
  if (tx < 32) {
    warpReduce<BLOCK_SIZE>(A_s, tx);
  }
  if (tx == 0) {
    out[bx] = A_s[tx];
  }
}

int main() {
  const int N = 32 * 1024 * 1024;
  float* h_A;
  h_A = (float*)malloc(N * sizeof(float));
  float* d_A;
  checkCudaErrors(cudaMalloc((void**)&d_A, N * sizeof(float)));
  int NUM_PER_BLOCK = 2 * THREAD_PER_BLOCK;
  int block_num = N / NUM_PER_BLOCK;
  float* d_out;
  checkCudaErrors(cudaMalloc((void**)&d_out, block_num * sizeof(float)));
  float *h_out, *res;
  h_out = (float*)malloc(block_num * sizeof(float));
  res = (float*)malloc(block_num * sizeof(float));

  for (int i = 0; i < N; i++) {
    h_A[i] = 1;
  }
  for (int i = 0; i < block_num; i++) {
    int sum = 0;
    for (int j = 0; j < NUM_PER_BLOCK; j++) {
      sum += h_A[i * NUM_PER_BLOCK + j];
    }
    res[i] = sum;
  }
  checkCudaErrors(
      cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_out, h_out, block_num * sizeof(float),
                             cudaMemcpyHostToDevice));

  dim3 threads_per_block = THREAD_PER_BLOCK;
  dim3 block_per_grid = CEIL_DIV(N, NUM_PER_BLOCK);
  int nIter = 1000;
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start, 0));
  for (int i = 0; i < nIter; i++) {
    reduction_kernel0<THREAD_PER_BLOCK>
        <<<block_per_grid, threads_per_block>>>(d_A, d_out);
  }
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  float msec = 0.f;
  checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
  checkCudaErrors(cudaMemcpy(h_out, d_out, block_num * sizeof(float),
                             cudaMemcpyDeviceToHost));

  float eps = 1.e-6;  // machine zero
  bool flag = true;
  for (int i = 0; i < block_num; i++) {
    if (fabs(h_out[i] - res[i]) > eps) {
      printf("Error in  %d  Answer = %f, Res = %f.\n", i, h_out[i], res[i]);
      flag = false;
      break;
    }
  }
  if (flag) {
    printf("Calculate Successed!\n");
  }
  int bytes = N * sizeof(float);
  float bandwidth = bytes / (msec * 1e6 / nIter);

  printf("Bandwidth is %.3f\n", bandwidth);

  cudaFree(d_A);
  cudaFree(d_out);
  free(h_A);
  free(h_out);
  free(res);
  return 0;
}