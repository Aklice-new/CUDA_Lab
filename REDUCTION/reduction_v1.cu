/**
 * @file reduction.cu
 * @author aklice
 * @brief 这个内容是关于reduction的实现
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

__global__ void reduction_kernel0(float* A, float* out) {
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int tid = bx * blockDim.x + tx;
  __shared__ float A_s[THREAD_PER_BLOCK];
  A_s[tx] = A[tid];
  __syncthreads();
  for (int stride = 1; stride <= THREAD_PER_BLOCK; stride *= 2) {
    if (tx % (2 * stride) == 0) {
      A_s[tx] += A_s[tx + stride];
    }
    __syncthreads();
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
  int block_num = N / THREAD_PER_BLOCK;
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
    for (int j = 0; j < THREAD_PER_BLOCK; j++) {
      sum += h_A[i * THREAD_PER_BLOCK + j];
    }
    res[i] = sum;
  }
  checkCudaErrors(
      cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_out, h_out, block_num * sizeof(float),
                             cudaMemcpyHostToDevice));

  dim3 threads_per_block = THREAD_PER_BLOCK;
  dim3 block_per_grid = CEIL_DIV(N, THREAD_PER_BLOCK);
  int nIter = 1000;
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start, 0));
  for (int i = 0; i < nIter; i++) {
    reduction_kernel0<<<block_per_grid, threads_per_block>>>(d_A, d_out);
  }
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  float msec = 0.f;
  checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
  checkCudaErrors(cudaMemcpy(h_out, d_out, block_num * sizeof(float),
                             cudaMemcpyDeviceToHost));

  float eps = 1.e-6;  // machine zero
  for (int i = 0; i < block_num; i++) {
    if (fabs(h_out[i] - res[i]) > eps) {
      printf("Error in  %d  Answer = %f, Res = %f.\n", i, h_out[i], res[i]);
      break;
    }
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