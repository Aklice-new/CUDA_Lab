/**
 * @file reduction.cu
 * @author aklice
 * @brief
  warp_shuffle指令进行优化
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
#define NUM_PER_THREAD 2
#define WARP_SIZE 32
#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))
#define checkCudaErrors(func)                                                  \
  {                                                                            \
    cudaError_t err = (func);                                                  \
    if (err != cudaSuccess)                                                    \
      printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  }

template <unsigned int BLOCK_SIZE>
__device__ __forceinline__ float warpReduceSum(float sum) {
  // __shfl_down_sync(mask, val, offset) warp 内高通道的值传递到低通道
  if (BLOCK_SIZE >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16);
  if (BLOCK_SIZE >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);
  if (BLOCK_SIZE >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);
  if (BLOCK_SIZE >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);
  if (BLOCK_SIZE >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);
  return sum;
}

template <unsigned int BLOCK_SIZE>
__global__ void reduction_kernel0(float* A, float* out) {
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int tid = bx * blockDim.x * NUM_PER_THREAD +
            tx;   // 这里 blockDim.x * 2 = NUM_PER_BLOCK
  float sum = 0;  // 记录当前线程块的和
  // 先计算一下每个线程负责的数据的规约
  for (int i = 0; i < NUM_PER_THREAD; i++) {
    sum += A[tid + i * blockDim.x];
  }
  // 接下来就是BLOCK_SIZE个线程的规约操作
  // 我们先将BLOCK 分成多个warp来看待，首先计算warp内的规约
  __shared__ float warpSum[WARP_SIZE];
  int laneId = tx % WARP_SIZE;
  int warpId = tx / WARP_SIZE;
  // 计算该线程所在的warp的规约
  // 在所有的warp中进行规约操作
  sum = warpReduceSum<BLOCK_SIZE>(sum);
  // 将结果合并warp sum中
  if (laneId == 0) {
    warpSum[warpId] = sum;
  }
  // 通过blockDim.x / WARP_SIZE进行计算，得到warp_nums
  // 然后让一些线程在负责对应warp的计算
  // 所以这里有两步shuffle操作
  sum = (tx < blockDim.x / WARP_SIZE) ? warpSum[laneId] : 0;
  // 将warp中的所有结果保存在warp 0中
  if (warpId == 0) {
    sum = warpReduceSum<BLOCK_SIZE / WARP_SIZE>(sum);
  }
  // 将该block内的结果保存至最终结果
  if (tx == 0) {
    out[bx] = sum;
  }
}

int main() {
  const int N = 32 * 1024 * 1024;
  float* h_A;
  h_A = (float*)malloc(N * sizeof(float));
  float* d_A;
  checkCudaErrors(cudaMalloc((void**)&d_A, N * sizeof(float)));
  int NUM_PER_BLOCK = NUM_PER_THREAD * THREAD_PER_BLOCK;
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